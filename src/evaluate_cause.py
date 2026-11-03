"""경보 원인 판단 평가(가상 데이터 data/synthetic, 실제 롯데케미칼 데이터가 아니다). 저장소 루트에서 실행한다.
plan: eval 경보(2021-01~2025-12)에서 고정 seed 층화 표본을 뽑고 정답을 붙여 out/synthetic/cause_sample.csv에 쓴다.
      API를 쓰지 않는다. 비교 기준 점수도 여기서 낸다. 이 파일을 커밋한 뒤 run을 한 번만 돌린다.
run : 표본 경보마다 에이전트(agent.judge)를 한 번씩 돌려 out/synthetic/cause_eval.csv와 요약을 낸다.
      실행 기록은 out/synthetic/agent_log.jsonl. OpenAI API 비용이 든다.
score: cause_eval.csv에서 요약만 다시 낸다(API 안 씀).

정답 규칙(실행 전에 정함, LOG.md 2026-11-03):
- 경보와 사건 대조는 inventory_alarm.match_event: 같은 고객, 사건 시작 <= 경보 달 <= 사건 끝 + 2개월, 여럿이면 시작이 이른 것.
  방향은 보지 않는다(재고를 줄였다가 되돌리며 발주가 느는 경보도 그 사건이다).
- 맞는 사건이 없으면 잡음이고 정답은 판단 보류.
- 사건이 있어도 메모가 없거나 원인을 말하지 않는 메모(memo_vague)뿐이면 정답은 판단 보류(근거가 없으니까).
- 그 밖엔 destock·destock_q4·restock -> 재고조정, demand_up·demand_down -> 실수요 변화.
- 가드레일에 두 번 막혀 답이 없으면 판단 보류로 채점하고 따로 센다(담당자에게 단정이 나가지 않았다).
  API 오류로 답이 없으면 틀린 것으로 채점하고(단정은 아님) 따로 센다.
- 틀린 단정 = 재고조정이나 실수요 변화로 답했는데 정답과 다른 것(정답이 판단 보류인데 단정 + 반대 원인).
비교 기준(LLM 없음): 같은 달 같은 방향 경보 고객 수(same_month)가 3 이상이면 재고조정, 아니면 판단 보류.
"""
import asyncio
import sys

import numpy as np
import pandas as pd

import agent
from inventory_alarm import EVAL, load_events, match_event

OUT = agent.OUT / "synthetic"
SAMPLE, RESULT, LOG = OUT / "cause_sample.csv", OUT / "cause_eval.csv", OUT / "agent_log.jsonl"
SEED, MAX_N, PARALLEL = 2026, 60, 4
CAUSE = {"destock": "재고조정", "destock_q4": "재고조정", "restock": "재고조정",
         "demand_up": "실수요 변화", "demand_down": "실수요 변화"}
HOLD = "판단 보류"


def plan():
    a = pd.read_csv(agent.ALERTS, dtype={"same_month": int})
    a = a[(a["period"] == "eval") & a["month"].between(*EVAL)].copy()
    ev = load_events()
    hits = [match_event(r.customer, r.month, ev) for r in a.itertuples()]
    a["event_id"] = [h["event_id"] if h is not None else "" for h in hits]
    a["event_type"] = [h["type"] if h is not None else "없음" for h in hits]
    a["memo"] = ["-" if h is None else "없음" if pd.isna(h["memo_id"]) else "애매" if h["memo_vague"] else "있음" for h in hits]
    a["truth"] = [CAUSE[t] if m == "있음" else HOLD for t, m in zip(a["event_type"], a["memo"])]
    # 유형별(사건 유형 5개 + 사건 없음) 같은 상한으로 뽑는다. 상한은 합이 MAX_N을 넘지 않는 가장 큰 값
    counts = a["event_type"].value_counts()
    cap = max(c for c in range(1, MAX_N + 1) if counts.clip(upper=c).sum() <= MAX_N)
    s = pd.concat(g.sample(min(len(g), cap), random_state=SEED) for _, g in a.groupby("event_type")).sort_values("alert_id")
    s["baseline"] = np.where(s["same_month"] >= 3, "재고조정", HOLD)
    s.to_csv(SAMPLE, index=False)
    print(f"eval 경보 {len(a)}개 중 표본 {len(s)}개(유형별 상한 {cap}), 정답 {s['truth'].value_counts().to_dict()}")
    print(pd.crosstab(s["event_type"], s["memo"]).to_string())
    print("비교 기준:", summary(s, "baseline"))


def summary(d, col):
    p, t = d[col], d["truth"]
    shown = t != HOLD  # 메모로 원인이 드러난 경보
    asserted = p.isin(["재고조정", "실수요 변화"])
    return {"n": len(d), "정답률": f"{(p == t).sum()}/{len(d)}",
            "정답률(원인 드러난 경보)": f"{(p == t)[shown].sum()}/{shown.sum()}",
            "틀린 단정": int((asserted & (p != t)).sum()), "  보류가 정답인데 단정": int((asserted & ~shown).sum()),
            "  반대 원인": int((asserted & shown & (p != t)).sum()), "보류율": f"{(p == HOLD).sum()}/{len(d)}"}


async def run():
    s = pd.read_csv(SAMPLE, dtype=str)
    gate = asyncio.Semaphore(PARALLEL)

    async def one(aid):
        async with gate:
            try:
                e = await agent.judge(aid, log=LOG)
            except Exception as x:  # judge가 기록은 남겼다
                return dict(alert_id=aid, pred="오류", error=type(x).__name__)
            v = e["answer"] or {"cause": HOLD, "memo_ids": [], "estimate": "없음", "reason": ""}
            return dict(alert_id=aid, pred=v["cause"], memo_ids=" ".join(v["memo_ids"]), estimate=v["estimate"],
                        reason=v["reason"], attempts=e.get("attempts"), blocked=len(e["blocked"]),
                        blocked_twice=e["answer"] is None, fake_id_blocks=sum(bool(b["fake_memo_ids"]) for b in e["blocked"]),
                        tools=" | ".join(e.get("tools", [])))

    rows = await asyncio.gather(*(one(a) for a in s["alert_id"]))
    d = s.merge(pd.DataFrame(rows), on="alert_id")
    d.to_csv(RESULT, index=False)
    report(d)


def report(d):
    print("에이전트:", summary(d, "pred"))
    print("비교 기준:", summary(d, "baseline"))
    num = {c: pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int) for c in ("blocked", "fake_id_blocks")}
    print(f"가드레일 차단 {num['blocked'].sum()}번(가짜 memo_id {num['fake_id_blocks'].sum()}번), "
          f"두 번 막혀 보류 처리 {d['blocked_twice'].astype(str).eq('True').sum()}건, API 오류 {(d['pred'] == '오류').sum()}건")
    g = d[(d["pred"] == HOLD) & (d["estimate"] != "없음") & (d["event_type"] != "없음")]
    print(f"근거 없는 추정을 낸 보류 {((d['pred'] == HOLD) & (d['estimate'] != '없음')).sum()}건, "
          f"그중 사건이 있는 {len(g)}건에서 추정이 사건 유형과 맞음 {(g['estimate'] == g['event_type'].map(CAUSE)).sum()}건")
    print(pd.crosstab(d["truth"], d["pred"]).to_string())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == "plan":
        plan()
    elif mode == "run":
        asyncio.run(run())
    elif mode == "score":
        report(pd.read_csv(RESULT, keep_default_na=False))
