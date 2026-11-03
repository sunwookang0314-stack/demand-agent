"""API 없이 경보 원인 도구·가드레일을 시험한다(가상 데이터). 실행: PYTHONPATH=src python tests/test_cause.py"""
import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path

opened = []  # 이 프로세스가 연 파일 전부(agent를 불러오기 전부터 센다)
sys.addaudithook(lambda event, args: event == "open" and opened.append(str(args[0])))

import pandas as pd  # noqa: E402
from agents import RunContextWrapper, set_tracing_disabled  # noqa: E402
from agents.testing import ScriptedModel, assistant_message, function_call  # noqa: E402
from agents.tool_context import ToolContext  # noqa: E402

import agent  # noqa: E402

set_tracing_disabled(True)  # 시험 중에 실행 기록을 OpenAI로 보내지 않는다
n = 0


def check(cond, msg=""):
    global n
    assert cond, msg
    n += 1


async def call(tool, ledger, **args):
    """SDK가 도구를 부를 때와 같은 길로 부른다(원장 기록이 도구 감싸개 안에서 일어나는지까지 본다)."""
    a = json.dumps(args, ensure_ascii=False)
    return await tool.on_invoke_tool(ToolContext(ledger, tool_name=tool.name, tool_call_id="t", tool_arguments=a), a)


async def guard(ledger, **v):
    r = await agent.cause_checks.run(RunContextWrapper(ledger), agent.cause_agent, agent.CauseVerdict(**v))
    return r.output


def verdict(cause, ids, reason, estimate="없음"):
    return json.dumps({"cause": cause, "memo_ids": ids, "estimate": estimate, "reason": reason}, ensure_ascii=False)


async def main():
    # 기존 에이전트는 그대로, 원인 에이전트는 새 도구 2개만
    check([t.name for t in agent.agent.tools] == ["get_forecast", "get_accuracy", "calculate", "publish_plan"])
    check([t.name for t in agent.cause_agent.tools] == ["get_alert", "search_memos"])

    # 경보 도구: 도구 출력 숫자가 원장에 들어가고, 원장 밖 숫자는 막힌다
    led = agent.CauseLedger()
    out = await call(agent.get_alert, led, alert_id="A059")  # tune 구간 경보(시험에 eval 경보를 쓰지 않는다)
    check("C12" in out and "102.0% 많음" in out and "2019-05: 발주 7,304톤, 기대치 3,616톤" in out, out)
    check({(102.0, "%"), (7304.0, "톤"), (3616.0, "톤"), (92249.0, "톤")} <= led.numbers and led.memo_until == "2019-06")
    check(agent.unverified(led, "C12 발주가 기대치 3,616톤보다 102.0% 많은 7,304톤이다.") == [])
    check(agent.unverified(led, "기대치의 2배다.") and agent.unverified(led, "약 7,300톤이다."))
    check((await call(agent.get_alert, agent.CauseLedger(), alert_id="A999")).startswith("경보 없음"))

    # 메모 도구: 경보가 뜬 달 뒤 메모는 주지 않고, 돌려준 id만 원장에 들어간다
    out = await call(agent.search_memos, led, customer="C12", from_month="2018-05", to_month="2020-12")
    shown = set(re.findall(r"M\d+", out))
    memos = pd.read_csv(agent.SYN / "memos.csv", dtype=str)
    want = memos[(memos["customer"] == "C12") & (memos["date"].str[:7].between("2018-05", "2019-06"))]
    check(shown == set(want["memo_id"]) == led.memo_ids and "2019-06 뒤 메모는 뺌" in out, (shown, out))
    check(not (set(memos.loc[(memos["customer"] == "C12") & (memos["date"] > "2019-07"), "memo_id"]) & shown))
    check((await call(agent.search_memos, led, customer="C12", from_month="5월", to_month="2019-06")).startswith("검색 실패"))

    # 가짜 memo_id 차단: 없는 id, 있지만 이번에 돌려주지 않은 id, 이유 문장 속 id, 메모 없는 단정
    real = sorted(led.memo_ids)[0]
    other = memos.loc[memos["customer"] != "C12", "memo_id"].iloc[0]  # 메모 파일엔 있지만 이번 실행에서 안 돌려준 것
    ok = dict(cause="재고조정", memo_ids=[real], estimate="없음", reason=f"{real} 메모가 재고를 줄인다고 했다.")
    check(not (await guard(led, **ok)).tripwire_triggered)
    for bad, key in [(dict(memo_ids=["M999"]), "fake_memo_ids"), (dict(memo_ids=[other]), "fake_memo_ids"),
                     (dict(reason=f"{other}의 메모를 보면 재고를 줄인다."), "fake_memo_ids"),
                     (dict(memo_ids=[]), "no_memo"), (dict(reason="기대치보다 30.0% 적어 재고조정이다."), "unverified")]:
        r = await guard(led, **(ok | bad))
        check(r.tripwire_triggered and r.output_info[key], (bad, r.output_info))
    hold = dict(cause="판단 보류", memo_ids=[], estimate="재고조정", reason="같은 달 경보가 많아 재고조정일 수 있으나 근거 없는 추정이다.")
    check(not (await guard(led, **hold)).tripwire_triggered)  # 패턴 추정은 판단 보류 + estimate로는 낼 수 있다

    # judge 전체 흐름(SDK 실행 + 가드레일 + 재시도)을 대본 모델로: 1차에 가짜 id -> 막힘 -> 2차에 돌려받은 id로 통과
    model = ScriptedModel([
        [function_call("get_alert", {"alert_id": "A059"}, call_id="c1")],
        [function_call("search_memos", {"customer": "C12", "from_month": "2018-05", "to_month": "2019-06"}, call_id="c2")],
        [assistant_message(verdict("재고조정", ["M999"], "M999 메모가 재고를 줄인다고 했다."))],
        [function_call("search_memos", {"customer": "C12", "from_month": "2018-05", "to_month": "2019-06"}, call_id="c3")],
        [assistant_message(verdict("판단 보류", [real], f"{real} 메모는 원인을 말하지 않는다."))],
    ])
    agent.cause_agent = agent.cause_agent.clone(model=model)
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.jsonl"
        e = await agent.judge("A059", log=log)
        check(e["attempts"] == 2 and e["blocked"][0]["fake_memo_ids"] == ["M999"] and e["answer"]["memo_ids"] == [real], e)
        check("M999" in str(model.calls[3].input) and "M999" not in str(model.calls[0].input))  # 재시도에 막힌 이유를 알려 줬다
        check(json.loads(log.read_text(encoding="utf-8"))["alert_id"] == "A059")
    model.assert_complete()

    # 경보 표의 gap_pct를 도구의 기대치 식이 그대로 재현하는가(inventory_alarm.gaps()와 식이 갈라지면 실패)
    alerts = pd.read_csv(agent.ALERTS, dtype=str)
    for aid in alerts["alert_id"]:
        lines = agent.alert_text(agent.CauseLedger(), aid).splitlines()
        head = float(re.search(r"기대치보다 ([\d.]+)%", lines[0]).group(1))
        last = float(re.search(r"\(([\d.]+)% ", lines[-1]).group(1))
        assert abs(head - last) <= 0.1, (aid, lines[0], lines[-1])
    check(True)

    # 정답 폴더는 한 번도 열지 않았다(대조군: 경보·발주·시장·메모 파일은 열었다)
    parts = [Path(p).parts for p in opened]
    check(not any("truth" in p for p in parts), [p for p in opened if "truth" in p])
    for f in ("alerts.csv", "orders.csv", "market.csv", "memos.csv"):
        check(any(p[-1] == f for p in parts), f)


asyncio.run(main())
print("ok", n, "checks")
