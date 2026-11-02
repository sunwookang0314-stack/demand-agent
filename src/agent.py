"""수요계획 담당자용 에이전트 (OpenAI Agents SDK)

도구 4개: get_forecast(고정해 둔 2026-10~12 예측), get_accuracy(백테스트·과거 구간 오차), calculate(도구 숫자끼리 합·차),
publish_plan(계획 확정, 사람 승인 필요).
숫자 원장 가드레일: 최종 답에 나온 숫자가 이번 실행에서 도구가 돌려준 숫자나 질문에 있던 숫자 중에 없으면 답을 막는다.
그래서 계산·반올림·단위 변환은 에이전트가 하지 않고 도구가 해서 준다.
한계: 숫자의 출처는 막지만 의미는 막지 못한다(10월 값을 11월 값이라 쓰기, 부호 뒤집기는 통과한다).
"""
import asyncio
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from agents import (Agent, GuardrailFunctionOutput, OutputGuardrailTripwireTriggered, RunContextWrapper, Runner,
                    function_tool, output_guardrail)
from agents import __version__ as sdk_version
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
PRODUCTS = ("ABS", "PC")
COUNTRY = {"BR": "브라질", "CN": "중국", "DE": "독일", "HK": "홍콩", "ID": "인도네시아", "IN": "인도", "JP": "일본",
           "MX": "멕시코", "MY": "말레이시아", "PL": "폴란드", "TH": "태국", "TR": "튀르키예", "TW": "대만", "US": "미국",
           "VN": "베트남"}


@dataclass
class Ledger:
    numbers: set = field(default_factory=set)  # 답에 써도 되는 (값, 단위): 도구 결과 + 질문
    tool: set = field(default_factory=set)  # 그중 도구가 준 것
    published: list = field(default_factory=list)  # 이번 질문에서 처리한 확정 요청(재시도 때 다시 묻지 않게)


# ---- 숫자 원장 ----
_UNIT = r"(?:톤|%p|%포인트|%|퍼센트|프로|포인트|배)"
_DATE = re.compile(rf"\d{{4}}-\d{{2}}(?:-\d{{2}})?(?!\d|\s*{_UNIT})|\d{{4}}년|\d{{1,2}}월|\d{{1,2}}일"
                   rf"|(?<![\d,.])(?:19|20)\d{{2}}(?![\d,.]|\s*(?:만|천|억|{_UNIT}))")
_NUM = re.compile(rf"(\d[\d,]*(?:\.\d+)?)\s*(만|천|억)?\s*({_UNIT})?")
_UNIT_NAME = {"%": "%", "퍼센트": "%", "프로": "%", "%p": "%p", "%포인트": "%p", "포인트": "%p", "톤": "톤", "배": "배"}
# '오만 칠천 톤', '수천 톤'. 수가 앞에 없는 '만 톤 단위' 같은 표현은 뺀다
_KOREAN_QTY = re.compile(r"(?<![가-힣])[일이삼사오육칠팔구수몇][일이삼사오육칠팔구십백천만억\s]*톤")


def numbers(text, checked_only=False):
    """글 속 (값, 단위) 쌍. 부호는 무시하고 날짜는 뺀다.
    checked_only면 검사 대상만: 단위·만천억이 붙었거나 소수점·천 단위 쉼표가 있거나 32 이상(그 밖의 작은 정수는 '3개월' 같은 셈)."""
    text = _DATE.sub(" ", unicodedata.normalize("NFKC", text))
    out = set()
    for m in _NUM.finditer(text):
        tok, mult, unit = m.group(1).rstrip(","), m.group(2), _UNIT_NAME.get(m.group(3), "")
        v = round(float(tok.replace(",", "")) * {"만": 1e4, "천": 1e3, "억": 1e8}.get(mult, 1), 6)
        if not checked_only or unit or mult or "." in tok or "," in tok or v >= 32:
            out.add((v, unit))
    return out


def unverified(ledger, text):
    """원장에 없는 숫자. 단위가 없는 숫자(범위 앞쪽 등)는 같은 값이 어떤 단위로든 있으면 통과."""
    values = {v for v, _ in ledger.numbers}
    bad = [(v, u) for v, u in numbers(text, checked_only=True)
           if not ((v, u) in ledger.numbers if u else v in values)]
    bad += [(m.group(), "한글 수량") for m in _KOREAN_QTY.finditer(unicodedata.normalize("NFKC", text))]
    return sorted(bad, key=str)


def record(ledger, text):
    """도구가 돌려준 글의 숫자를 원장에 넣는다."""
    n = numbers(text)
    ledger.numbers |= n
    ledger.tool |= n
    return text


def note_question(ledger, text):
    """질문에 있던 숫자는 답에 다시 써도 되지만 도구 숫자로 치지는 않는다."""
    ledger.numbers |= numbers(text)


def _t(x):
    return f"{x:,.0f}톤"


def _p(x):
    return f"{x * 100:.1f}%"


# ---- 도구 본문 (SDK 없이도 부를 수 있게 일반 함수로 둔다) ----
def forecast_text(product):
    fc = pd.read_csv(OUT / "forecast_2026Q4.csv", parse_dates=["ds", "cutoff"])
    fc = fc[fc["unique_id"].str.startswith(product + "|")]
    c = pd.read_csv(ROOT / "data" / "customs.csv", parse_dates=["month"])
    act = c[c["product"] == product].groupby("month")["exp_kg"].sum() / 1000
    lad = pd.read_csv(OUT / "ladder.csv", dtype={"h": str})
    lad = lad[(lad["product"] == product) & (lad["model"] == "Combo")].set_index("h")
    tot = fc.groupby("ds")[["Combo", "AutoETS", "LGBM"]].sum()
    lines = [f"{product} 월별 수출 예측(관세청 15개국 합계, 기준 시점 {fc['cutoff'].iloc[0]:%Y-%m}, "
             "최종 예측 = AutoETS와 LightGBM의 평균)"]
    # 범위 끝값·직전 달 대비 차이도 여기서 계산해 준다(새 질문 평가에서 에이전트가 이걸 직접 계산하다 가드레일에 막혔다)
    prev_name, prev = f"직전 달 실적({act.index[-1]:%Y-%m})", act.iloc[-1]
    for h, (ds, r) in enumerate(tot.iterrows(), 1):
        c, last, e = r["Combo"], act[ds - pd.DateOffset(years=1)], lad.loc[str(h), "wape_total"]
        lines.append(f"- {ds:%Y-%m}: {_t(c)} (AutoETS {_t(r['AutoETS'])}, LightGBM {_t(r['LGBM'])}). "
                     f"작년 같은 달 실적 {_t(last)}보다 {_p(abs(c / last - 1))} {'많음' if c >= last else '적음'}. "
                     f"{prev_name} {_t(prev)}보다 {_t(abs(c - prev))} {'많음' if c >= prev else '적음'}. "
                     f"{h}개월 앞 예측의 평가 구간 평균 오차 {_p(e)}를 적용한 폭 ±{_t(c * e)}, "
                     f"즉 {_t(c * (1 - e))}~{_t(c * (1 + e))}(신뢰구간 아님)")
        prev_name, prev = f"{ds:%m}월 예측", c
    total = tot["Combo"].sum()
    top = fc.groupby("unique_id")["Combo"].sum().nlargest(5)
    lines += [f"- 3개월 합계: {_t(total)}",
              "- 3개월 합계 상위 5개국: " + ", ".join(
                  f"{COUNTRY[u.split('|')[1]]} {_t(v)}({_p(v / total)})" for u, v in top.items()),
              f"- 국가별 숫자는 합계보다 덜 정확하다(평가 구간 국가별 WAPE {_p(lad.loc['1-3', 'wape_country'])}).",
              "주의: 고객사 발주가 아니라 관세청 수출 통계를 대신 쓴 공개 데이터 예측이다. "
              "15개국 합계라 전체 수출의 일부다(ABS는 2024년 기준 66%, PC는 확인 안 함)."]
    return "\n".join(lines)


def _naive_vs_combo_ci(preds, product, naive, reps=4000, block=3, seed=0):
    """(Combo 오차 - 나이브 오차)의 95% 구간. 이웃 기준 시점끼리 오차가 닮아서 연속 3개씩 묶어 다시 뽑는다
    (원형 이동 블록: 시작점은 아무 데나, 끝을 넘으면 처음으로 이어 붙인다)."""
    p = preds[preds["unique_id"].str.startswith(product + "|")]
    t = p.groupby(["cutoff", "ds"])[["y", "Combo", naive]].sum()
    t["ec"], t["en"] = (t["Combo"] - t["y"]).abs(), (t[naive] - t["y"]).abs()
    s = t.groupby("cutoff")[["y", "ec", "en"]].sum()
    y, ec, en = s["y"].to_numpy(), s["ec"].to_numpy(), s["en"].to_numpy()
    n = len(y)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(reps):
        starts = rng.integers(n, size=math.ceil(n / block))
        idx = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        d.append((ec[idx].sum() - en[idx].sum()) / y[idx].sum())
    return np.percentile(d, [2.5, 97.5])


def accuracy_text(product):
    lad = pd.read_csv(OUT / "ladder.csv", dtype={"h": str})
    hold = pd.read_csv(OUT / "holdout_past.csv")
    parts = [("평가 구간(목표월 2024-08~2026-09, 기준 시점 24개)", lad[lad["h"] == "1-3"], "backtest_preds.csv"),
             ("과거 구간(목표월 2019-01~2021-12, 규칙을 고정한 뒤 처음 채점)", hold[hold["scope"] == "all"],
              "holdout_past_preds.csv")]
    lines = [f"{product} 예측 정확도(1~3개월 앞, 제품 합계 WAPE = 오차 절댓값 합 / 실적 합)"]
    for name, d, preds_file in parts:
        d = d[d["product"] == product].set_index("model")
        naive = min(["Naive", "SeasonalNaive"], key=lambda m: d.loc[m, "wape_total"])
        c, n = d.loc["Combo", "wape_total"], d.loc[naive, "wape_total"]
        lo, hi = _naive_vs_combo_ci(pd.read_csv(OUT / preds_file), product, naive)
        if min(abs(lo), abs(hi)) <= 0.002:
            sure = "구간 끝이 0에서 0.2%p 안이라 경계선, 차이를 단정하지 않음"
        elif hi < 0 or lo > 0:
            sure = "0을 포함하지 않아 차이가 확인됨"
        else:
            sure = "0을 포함해 차이를 확정할 수 없음"
        lines.append(f"- {name}: 최종 예측 {_p(c)}, 나이브({'직전 달' if naive == 'Naive' else '작년 같은 달'}) {_p(n)}, "
                     f"차이 {_p(abs(c - n))}p {'낮음' if c <= n else '높음'}"
                     f"(95% 구간 {lo * 100:.2f}%p~{hi * 100:.2f}%p, {sure}). "
                     f"국가별 WAPE {_p(d.loc['Combo', 'wape_country'])}, "
                     f"편향 {_p(abs(d.loc['Combo', 'bias']))} {'과대' if d.loc['Combo', 'bias'] >= 0 else '과소'}예측. "
                     f"단독 모델: AutoETS {_p(d.loc['AutoETS', 'wape_total'])}, LightGBM {_p(d.loc['LGBM', 'wape_total'])}")
    lines += ["- 최종 규칙(평균)은 평가 구간 결과를 본 뒤 정했다. 평가 구간 숫자는 새 자료 성적이 아니고, "
              "과거 구간이 규칙을 고정한 뒤 처음 채점한 구간이다.",
              "- 모델 순위는 시기마다 뒤집혔다. 그래서 하나를 고르지 않고 평균을 쓴다.",
              "- 외부 지표(원료 가격, 재고, 수요 산업 생산, 환율)는 쓰지 않는다. 지표 12개를 제품별로 18번 시험해 "
              "정보 없는 랜덤 시계열 20개 중 최고보다 나은 경우가 1번뿐(우연으로도 나올 수 있는 수준)이었고, "
              "그 1개도 평가 구간에서 오차를 키웠다."]
    return "\n".join(lines)


def plan_months():
    return sorted(pd.read_csv(OUT / "forecast_2026Q4.csv", parse_dates=["ds"])["ds"].dt.strftime("%Y-%m").unique())


def publish(product, month, qty_t, reason, path=OUT / "approved_plans.csv"):
    """계획 수량을 확정 파일에 한 줄 더한다. 검증과 메시지를 다 만든 뒤 마지막에 기록한다(기록 뒤 오류로 중복 줄이 생기지 않게).
    에이전트에선 담당자가 승인한 뒤에만 실행된다."""
    fc = pd.read_csv(OUT / "forecast_2026Q4.csv", parse_dates=["ds"])
    try:
        qty = float(qty_t)
    except (TypeError, ValueError):
        qty = math.nan
    reason = str(reason).strip()
    months = plan_months()
    if product not in PRODUCTS or month not in months:
        return f"확정 실패: 제품은 {', '.join(PRODUCTS)}, 월은 {', '.join(months)} 중 하나여야 한다."
    if not math.isfinite(qty) or qty <= 0:
        return "확정 실패: 수량은 0보다 큰 숫자(톤)여야 한다."
    if not reason:
        return "확정 실패: 사유를 적어야 한다."
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        if (old["product"].eq(product) & old["month"].eq(month)).any():
            return f"확정 실패: {product} {month}은 이미 확정돼 있다. 바꾸려면 담당자가 파일에서 이전 줄을 지운 뒤 다시 확정한다."
    f = fc[fc["unique_id"].str.startswith(product + "|") & (fc["ds"] == pd.Timestamp(month + "-01"))]["Combo"].sum()
    gap = qty / f - 1
    msg = (f"담당자 승인을 받아 기록했다: {product} {month} 계획 {_t(qty)}(예측 {_t(f)}보다 {_p(abs(gap))} {'많음' if gap >= 0 else '적음'}), "
           f"사유: {reason}" + (" / 예측과 50% 넘게 차이 나니 사유를 다시 확인할 것" if abs(gap) > 0.5 else ""))
    safe = "'" + reason if reason[0] in "=+-@" else reason  # 엑셀에서 열 때 수식으로 실행되지 않게
    pd.DataFrame([{"saved_at": datetime.now().isoformat(timespec="seconds"), "product": product, "month": month,
                   "plan_t": qty, "forecast_t": round(f), "reason": safe}]).to_csv(
        path, mode="a", header=not path.exists(), index=False)
    return msg


def calc(ledger, a, op, b, unit):
    """원장 숫자 두 개를 더하거나 뺀다. 두 숫자가 원장에 그 단위로 있어야 하고(톤-톤, %-%, %p-%p),
    결과만 원장에 넣는다. 질문에서 온 숫자가 섞이면 결과에 표시한다."""
    a, b = float(a), float(b)
    out = {"톤": "톤", "%p": "%p", "%": "%p" if op == "-" else None}[unit]
    if out is None:
        return "계산 거절: 퍼센트끼리 더하는 건 의미가 없다."
    missing = [x for x in (a, b) if (round(abs(x), 6), unit) not in ledger.numbers]
    if missing:
        return f"계산 거절: {missing}는 이번 대화에 {unit} 단위로 나온 숫자가 아니다."
    asked = [x for x in (a, b) if (round(abs(x), 6), unit) not in ledger.tool]

    def fmt(x, u):
        return f"{x:,.0f}톤" if u == "톤" else f"{x:g}{u}"

    res = fmt(a + b if op == "+" else a - b, out)
    record(ledger, res)
    return f"{fmt(a, unit)} {op} {fmt(b, unit)} = {res}" + (f" (질문에서 준 숫자 {asked} 포함)" if asked else "")


# ---- 에이전트 (OpenAI Agents SDK) ----
load_dotenv(ROOT / ".env")


@function_tool
def get_forecast(ctx: RunContextWrapper[Ledger], product: Literal["ABS", "PC"]) -> str:
    """고정해 둔 2026-10~12 월별 수출 예측(톤)과 작년 같은 달 대비 증감, 과거 오차로 낸 폭, 상위 5개국을 돌려준다.

    Args:
        product: 제품. ABS 또는 PC.
    """
    return record(ctx.context, forecast_text(product))


@function_tool
def get_accuracy(ctx: RunContextWrapper[Ledger], product: Literal["ABS", "PC"]) -> str:
    """예측이 얼마나 맞았는지(평가 구간·과거 구간 오차, 나이브 대비, 95% 구간, 편향)와 한계, 그리고 예측 방법
    (평균을 쓰는 이유, 환율·원료 가격 같은 외부 지표를 쓰지 않는 이유)을 돌려준다.

    Args:
        product: 제품. ABS 또는 PC.
    """
    return record(ctx.context, accuracy_text(product))


@function_tool
def calculate(ctx: RunContextWrapper[Ledger], a: float, op: Literal["+", "-"], b: float,
              unit: Literal["톤", "%", "%p"]) -> str:
    """도구가 준 같은 단위 숫자 두 개의 합이나 차를 계산한다(예: 10월 예측 - 11월 예측). 직접 계산하지 말고 이 도구를 쓴다.

    Args:
        a: 첫 숫자(도구가 준 값, 쉼표 없이).
        op: + 또는 -.
        b: 둘째 숫자(도구가 준 값, 쉼표 없이).
        unit: 두 숫자의 단위(둘이 같아야 한다). %끼리 빼면 결과는 %p.
    """
    return calc(ctx.context, a, op, b, unit)


async def _valid_plan(ctx, params, call_id):
    """제품·월이 맞는 요청만 담당자에게 묻는다. 틀린 요청은 묻지 않고 도구가 바로 '확정 실패'를 돌려준다."""
    return params.get("product") in PRODUCTS and params.get("month") in plan_months()


@function_tool(needs_approval=_valid_plan)
def publish_plan(ctx: RunContextWrapper[Ledger], product: Literal["ABS", "PC"], month: str, qty_t: float,
                 reason: str) -> str:
    """월 계획 수량을 확정 파일에 기록한다. 담당자가 승인해야 실행된다.

    Args:
        product: 제품. ABS 또는 PC.
        month: 확정할 달. YYYY-MM 형식(예: 2026-11).
        qty_t: 계획 수량(톤).
        reason: 확정 사유. 예측과 다르게 잡으면 왜 다른지.
    """
    msg = record(ctx.context, publish(product, month, qty_t, reason))
    ctx.context.published.append(msg)
    return msg


@output_guardrail
async def numbers_from_tools(ctx: RunContextWrapper[Ledger], agent: Agent, output: str) -> GuardrailFunctionOutput:
    bad = unverified(ctx.context, output)
    return GuardrailFunctionOutput(output_info={"unverified": bad}, tripwire_triggered=bool(bad))


INSTRUCTIONS = """너는 롯데케미칼 첨단소재(ABS·PC) 수요계획 담당자를 돕는 에이전트다.
규칙:
1. 숫자는 반드시 도구를 불러서 얻고, 도구가 준 표기 그대로 옮긴다. 반올림·단위 변환(예: 만 톤)은 하지 않는다.
   합이나 차가 필요하면 직접 계산하지 말고 calculate를 부른다. 범위 끝값이나 설명 중에 곁들이는 차이도 마찬가지다.
   그 밖에 도구로 얻을 수 없는 숫자는
   "도구에 없는 숫자라 답하지 않는다"고 말한다.
2. 예측을 말할 때는 get_accuracy로 얼마나 믿을 만한지와 한계도 함께 말한다. 예측 방법이나 외부 지표에 대한 질문도
   get_accuracy를 불러 답한다. 정확도·방법 문장은 줄이거나 바꿔 말하지 말고 도구 문장을 그대로 인용한다
   ('20개 중 최고', '경계선' 같은 조건을 빼면 뜻이 바뀐다).
3. 계획을 확정해 달라는 요청이면 publish_plan을 부른다. 담당자 승인 여부는 도구가 돌아온 시점에 이미 정해져 있다.
   도구 결과 문장의 첫머리('담당자 승인을 받아 기록했다' 또는 '확정 실패')를 그대로 옮기고,
   거절됐으면 '담당자가 거절해 확정하지 않았다'고 말한다. '승인이 필요하다'처럼 미래형으로 쓰지 않는다.
4. 한국어로 짧게 답한다. 근거 없는 단정은 하지 않는다."""

MODEL = "gpt-5.6-luna"
agent = Agent[Ledger](name="수요계획 에이전트", instructions=INSTRUCTIONS, model=MODEL,
                      tools=[get_forecast, get_accuracy, calculate, publish_plan],
                      output_guardrails=[numbers_from_tools])


def ask_human(item):
    print(f"\n[승인 요청] {item.tool_name} {item.arguments}")
    return input("승인할까요? (y/n) ").strip().lower() == "y"


async def ask(question, approve=ask_human, log=OUT / "agent_log.jsonl"):
    """질문 하나를 끝까지 처리한다. 승인 대기에서 멈추면 approve로 묻고 이어 간다.
    가드레일에 걸리면 걸린 숫자를 알려 주고 한 번만 다시 시킨다. 이미 담당자에게 물은 확정 요청은 재시도에서 다시 묻지 않는다.
    오류가 나도 기록은 남긴다(키가 섞일 수 있어 오류 메시지는 빼고 종류만)."""
    entry = {"at": datetime.now().isoformat(timespec="seconds"), "model": MODEL, "sdk": sdk_version,
             "question": question, "approvals": [], "blocked": [], "answer": None}
    ledger = Ledger()
    note_question(ledger, question)
    runner, prompt = agent, question
    try:
        for attempt in (1, 2):
            try:
                result = await Runner.run(runner, prompt, context=ledger)
                while result.interruptions:
                    state = result.to_state()
                    for item in result.interruptions:
                        ok = approve(item)
                        entry["approvals"].append({"tool": item.tool_name, "args": item.arguments, "approved": ok})
                        if ok:
                            state.approve(item)
                        else:
                            state.reject(item, rejection_message="담당자가 거절해 확정하지 않았다.")
                            ledger.published.append(f"담당자가 거절해 확정하지 않았다: {item.arguments}")
                    result = await Runner.run(runner, state)
                entry |= {"answer": result.final_output, "attempts": attempt,
                          "tools": [i.raw_item.name for i in result.new_items if i.type == "tool_call_item"]}
                break
            except OutputGuardrailTripwireTriggered as e:
                bad = e.guardrail_result.output.output_info["unverified"]
                entry["blocked"].append({"answer": e.guardrail_result.agent_output, "unverified": [str(b) for b in bad]})
                note = ""
                if ledger.published:  # 확정은 이미 처리됐으니 재시도엔 확정 도구를 빼고 결과만 알려 준다
                    runner = agent.clone(tools=[get_forecast, get_accuracy, calculate])
                    note = " 확정 요청은 이미 처리됐다: " + " / ".join(ledger.published)
                prompt = (f"{question}\n\n(직전 답은 도구 결과에 없는 숫자 {', '.join(str(b[0]) for b in bad)} 때문에 막혔다. "
                          "도구가 준 숫자를 단위까지 표기 그대로만 써서 다시 답하라. 합이나 차가 필요하면 calculate를 부르고, "
                          f"그 밖의 계산값은 쓰지 마라.{note})")
    except Exception as e:
        entry["error"] = type(e).__name__
        raise
    finally:
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


if __name__ == "__main__":
    e = asyncio.run(ask(" ".join(sys.argv[1:]) or input("질문: ")))
    for b in e["blocked"]:
        print(f"\n[가드레일 차단] 도구에 없는 숫자 {b['unverified']}")
    print("\n" + (e["answer"] or "답을 내지 못했다(가드레일 두 번 차단)."))
