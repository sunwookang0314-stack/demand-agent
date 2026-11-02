"""가상 고객사 발주 데이터를 만든다. 실제 롯데케미칼 데이터가 아니다.

최종 수요는 실제 관세청 ABS 15개국 월별 수출(톤)을 쓰고, 그 위에 가상 고객사 20곳의 재고 정책과
재고조정·실수요 변화 사건을 심는다. 사건 정답(events.csv)은 경보와 에이전트가 읽지 않는 파일로 따로 둔다.
실행: python src/synthetic.py  ->  data/synthetic/{market,orders,events,memos}.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED, N, COVER, ALPHA = 2026, 20, 1.5, 0.5  # 고객 수, 기본 재고(몇 달 치), 재고 차이를 한 달에 메우는 비율
OUT = Path("data/synthetic")
rng = np.random.default_rng(SEED)

c = pd.read_csv("data/customs.csv", parse_dates=["month"])
market = c[c["product"] == "ABS"].groupby("month")["exp_kg"].sum() / 1000
months = market.index
T = len(months)
cust = [f"C{i + 1:02d}" for i in range(N)]

# 고객별 실수요 = 시장 x 비중 x 천천히 움직이는 비중 변화 x 고객 고유 잡음
share = rng.dirichlet(np.ones(N) * 3)
drift = np.exp(np.cumsum(rng.normal(0, 0.01, (T, N)), axis=0))
eps = np.zeros((T, N))
for t in range(1, T):
    eps[t] = 0.5 * eps[t - 1] + rng.normal(0, 0.07, N)
demand = market.to_numpy()[:, None] * share[None, :] * drift * np.exp(eps)
cover = np.full((T, N), COVER)

events, busy = [], np.zeros((T, N), bool)


def add(i, s, e, kind, size):
    """고객 i의 s~e(포함) 달에 사건을 심는다. 같은 고객의 사건끼리 앞뒤 2달 안에 겹치면 건너뛴다."""
    if s < 15 or e >= T or busy[max(0, s - 2):e + 3, i].any():
        return
    busy[s:e + 1, i] = True
    if kind in ("demand_up", "demand_down"):
        demand[s:e + 1, i] *= 1 + size
    else:
        cover[s:e + 1, i] = COVER * (1 + size)
    events.append(dict(event_id=f"E{len(events) + 1:03d}", customer=cust[i], type=kind,
                       start=months[s].strftime("%Y-%m"), end=months[e].strftime("%Y-%m"), size=round(size, 3)))


# 4분기 재고조정: 해마다 업계 강도가 다르고, 그해 고르는 고객 비율이 바뀐다(10~12월 재고 목표를 낮춤)
for y in range(2016, 2026):
    frac = rng.uniform(0.15, 0.6)
    for i in np.flatnonzero(rng.random(N) < frac):
        s = months.get_loc(pd.Timestamp(f"{y}-10-01"))
        add(i, s, s + 2, "destock_q4", -rng.uniform(0.25, 0.5))
# 연중 사건: 고객마다 평균 4~5년에 한 번씩
for i in range(N):
    for kind, rate in (("demand_up", 0.25), ("demand_down", 0.2), ("destock", 0.2), ("restock", 0.15)):
        for _ in range(rng.poisson(rate * T / 12)):
            s = int(rng.integers(15, T - 14))
            if kind.startswith("demand"):
                dur, size = int(rng.integers(6, 13)), rng.uniform(0.15, 0.35)
            else:
                dur, size = int(rng.integers(2, 5)), rng.uniform(0.25, 0.5)
            if kind == "destock" and months[s].month >= 10:
                continue  # 4분기 재고조정과 구분하려고 연중 재고조정은 1~9월에만 심는다
            add(i, s, s + dur - 1, kind, size if kind in ("demand_up", "restock") else -size)

# 발주 = 지난달 수요로 본 이번 달 수요 + 재고 목표와의 차이의 절반 (음수면 0)
orders, inv = np.zeros((T, N)), np.zeros((T, N))
inv_prev = COVER * demand[0]
for t in range(T):
    expect = demand[t - 1] if t else demand[0]
    target = cover[t] * expect
    orders[t] = np.maximum(0, expect + ALPHA * (target - inv_prev))
    inv[t] = inv_prev + orders[t] - demand[t]
    inv_prev = inv[t]

# 영업 메모: 사건의 70%에 메모가 붙는다. 그중 일부는 원인을 밝히지 않는다. 사건과 무관한 메모도 섞는다
TEMPLATES = {
    "destock_q4": ["{c} 구매팀, 연말 결산 전에 재고를 줄이겠다며 4분기 발주를 낮춘다고 함",
                   "{c}: 연말 재고 일수 목표를 낮췄다고. 내년 1월부터는 평소대로 받겠다고 함",
                   "{c} 담당 통화. 최종 제품 판매는 비슷한데 연말이라 창고 재고를 덜어내는 중이라고 함"],
    "destock": ["{c} 자금 사정 때문에 원료 재고를 줄이는 중. 생산량은 그대로라고 함",
                "{c}: 창고 이전 기간이라 두세 달 재고를 적게 들고 간다고 함",
                "{c} 구매 담당, 재고 회전율 목표가 올라가서 발주를 당분간 줄인다고 함"],
    "restock": ["{c} 원료 가격이 오를 것 같다며 미리 재고를 쌓는다고 함",
                "{c}: 공장 정기보수 전에 재고를 더 확보해 두겠다고 함",
                "{c} 구매팀, 해상 운송 지연 대비로 안전재고를 늘린다고 함"],
    "demand_up": ["{c} 신모델 양산 시작. 완제품 판매 목표가 작년보다 높다고 함",
                  "{c}: 새 수출 거래처를 따서 생산 라인을 하나 더 돌린다고 함",
                  "{c} 담당, 신규 차종에 들어가는 부품 수주로 생산량이 늘어난다고 함"],
    "demand_down": ["{c} 주력 모델 단종. 후속 모델 전까지 생산량이 줄어든다고 함",
                    "{c}: 생산 일부를 해외 공장으로 옮겨서 국내 생산이 줄어든다고 함",
                    "{c} 완제품 판매 부진으로 생산 계획을 낮췄다고 함"],
}
VAGUE = ["{c} 담당 통화. 당분간 물량이 달라질 수 있다고만 함", "{c}: 다음 달 발주 계획은 아직 확정 전이라고 함"]
NOISE = ["{c} 단가 협상 요청. 물량 얘기는 없었음", "{c} 신규 grade 샘플 요청", "{c} 품질 클레임 1건 처리 완료",
         "{c} 납기 문의. 기존 일정대로 진행", "{c} 담당자 교체 안내"]
memos = []


def memo(t_idx, text, c, eid=""):
    day = int(rng.integers(1, 28))
    memos.append(dict(date=f"{months[t_idx]:%Y-%m}-{day:02d}", customer=c, text=text, _event=eid))


for ev in events:
    if rng.random() < 0.7:
        s = months.get_loc(pd.Timestamp(ev["start"] + "-01")) + int(rng.integers(-1, 2))
        s = min(max(s, 0), T - 1)
        pool = VAGUE if rng.random() < 0.2 else TEMPLATES[ev["type"]]
        memo(s, pool[int(rng.integers(len(pool)))].format(c=ev["customer"]), ev["customer"], ev["event_id"])
for _ in range(int(3 * T / 12)):
    i = int(rng.integers(N))
    memo(int(rng.integers(T)), NOISE[int(rng.integers(len(NOISE)))].format(c=cust[i]), cust[i])

OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"month": months, "market_t": market.round(1).to_numpy()}).to_csv(OUT / "market.csv", index=False)
o = pd.DataFrame([dict(month=months[t], customer=cust[i], order_t=round(orders[t, i], 1)) for t in range(T) for i in range(N)])
o.to_csv(OUT / "orders.csv", index=False)
truth = pd.DataFrame([dict(month=months[t], customer=cust[i], demand_t=round(demand[t, i], 1), inv_t=round(inv[t, i], 1))
                      for t in range(T) for i in range(N)])
m = pd.DataFrame(memos).sort_values("date").reset_index(drop=True)
m.insert(0, "memo_id", [f"M{i + 1:03d}" for i in range(len(m))])
ev = pd.DataFrame(events)
ev["memo_id"] = ev["event_id"].map(m[m["_event"] != ""].set_index("_event")["memo_id"]).fillna("")
vague = set(m.loc[m["text"].isin([v.format(c=x) for v in VAGUE for x in cust]), "memo_id"])
ev["memo_vague"] = ev["memo_id"].isin(vague)
# 정답 쪽 파일: 경보·에이전트는 읽지 않는다
(OUT / "truth").mkdir(exist_ok=True)
ev.to_csv(OUT / "truth" / "events.csv", index=False)
truth.to_csv(OUT / "truth" / "demand_inventory.csv", index=False)
m.drop(columns="_event").to_csv(OUT / "memos.csv", index=False)
print(f"고객 {N}곳, {T}개월, 사건 {len(ev)}건 {ev['type'].value_counts().to_dict()}, 메모 {len(m)}건")
