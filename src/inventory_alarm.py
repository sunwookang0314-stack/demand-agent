"""가상 발주(data/synthetic)로 재고조정 경보를 시험한다. 실제 롯데케미칼 데이터가 아니다.

경보: 고객 발주가 '시장 x 그 고객의 평소 비중'으로 본 기대치보다 θ 넘게 낮거나(높거나) k달 이어지면 띄운다.
비교 기준: 시장을 안 보고 발주만 보는 경보(작년 같은 달 대비 θ 넘게 줄거나 늘어 k달).
θ·k는 2016~2020년(tune)에서만 고르고, 2021~2025년(eval)은 고른 값을 커밋한 뒤 한 번만 채점한다.
실행: python src/inventory_alarm.py tune  ->  out/synthetic/alarm_params.json, alerts.csv
      python src/inventory_alarm.py eval  ->  out/synthetic/alarm_eval.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D, OUT = Path("data/synthetic"), Path("out/synthetic")
TUNE, EVAL = ("2016-01", "2020-12"), ("2021-01", "2025-12")
GRID = [(th, k) for th in (0.10, 0.15, 0.20, 0.25, 0.30) for k in (1, 2, 3)]
SIGN = {"destock_q4": "down", "destock": "down", "demand_down": "down", "restock": "up", "demand_up": "up"}

orders = pd.read_csv(D / "orders.csv", parse_dates=["month"]).pivot(index="month", columns="customer", values="order_t")
market = pd.read_csv(D / "market.csv", parse_dates=["month"]).set_index("month")["market_t"]


def gaps():
    """기대치 = 시장 x (4~15달 전 12달의 발주/시장 비율 중앙값). 사건이 비중 추정에 바로 섞이지 않게 3달 띄운다."""
    ratio = orders.div(market, axis=0)
    share = ratio.shift(4).rolling(12, min_periods=8).median()
    return orders / share.mul(market, axis=0) - 1


def yoy():
    return orders / orders.shift(12) - 1


def alerts_from(score, th, k):
    rows = []
    for c in score.columns:
        s = score[c]
        for direction, hit in (("down", s <= -th), ("up", s >= th)):
            run = hit.astype(int).groupby((~hit).cumsum()).cumsum()  # 같은 방향으로 이어진 달 수
            for m in run.index[(run == k)]:  # 이어진 구간마다 k번째 달에 한 번만 띄운다
                rows.append(dict(customer=c, month=m, direction=direction, gap_pct=round(float(s[m]), 3), run_months=k))
    a = pd.DataFrame(rows, columns=["customer", "month", "direction", "gap_pct", "run_months"])
    a["same_month"] = a.groupby(["month", "direction"])["customer"].transform("size")
    return a.sort_values(["month", "customer"]).reset_index(drop=True)


def match_event(customer, month, events, direction=None):
    """같은 고객, event.start <= month <= event.end + 2개월인 사건. 여럿이면 시작이 이른 것. 없으면 None."""
    m = pd.Timestamp(month)
    e = events[(events["customer"] == customer) & (events["s"] <= m) & (m <= events["e"] + pd.DateOffset(months=2))]
    if direction is not None:
        e = e[e["type"].map(SIGN) == direction]
    return None if e.empty else e.sort_values("s").iloc[0]


def load_events():
    ev = pd.read_csv(D / "truth" / "events.csv")
    ev["s"] = pd.to_datetime(ev["start"] + "-01")
    ev["e"] = pd.to_datetime(ev["end"] + "-01")
    return ev


def score(a, ev, period):
    lo, hi = (pd.Timestamp(p + "-01") for p in period)
    a = a[(a["month"] >= lo) & (a["month"] <= hi)].copy()
    evp = ev[(ev["s"] >= lo) & (ev["s"] <= hi)]
    hit = [match_event(r.customer, r.month, ev, r.direction) for r in a.itertuples()]
    a["event_id"] = [h["event_id"] if h is not None else "" for h in hit]
    a["delay"] = [(r.month.year - h["s"].year) * 12 + r.month.month - h["s"].month if h is not None else np.nan
                  for r, h in zip(a.itertuples(), hit)]
    found = set(a["event_id"]) - {""}
    years = hi.year - lo.year + 1
    out = dict(alerts=len(a), matched=int((a["event_id"] != "").sum()), events=len(evp),
               recall=round(evp["event_id"].isin(found).mean(), 3),
               precision=round((a["event_id"] != "").mean(), 3) if len(a) else 0.0,
               false_per_customer_year=round((a["event_id"] == "").sum() / (orders.shape[1] * years), 3),
               median_delay=float(np.nanmedian(a["delay"])) if found else np.nan)
    out["f1"] = round(2 * out["recall"] * out["precision"] / (out["recall"] + out["precision"]), 3) if out["recall"] + out["precision"] else 0.0
    by_type = {t: round(g["event_id"].isin(found).mean(), 3) for t, g in evp.groupby("type")}
    return out, by_type, a


METHODS = {"시장 기준 차이(제안)": gaps, "발주만 작년 대비(비교 기준)": yoy}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "tune"
    OUT.mkdir(parents=True, exist_ok=True)
    ev = load_events()
    if mode == "tune":
        params = {}
        for name, f in METHODS.items():
            s = f()
            best = max(GRID, key=lambda g: (score(alerts_from(s, *g), ev, TUNE)[0]["f1"], -g[1]))
            params[name] = dict(theta=best[0], k=best[1], tune=score(alerts_from(s, *best), ev, TUNE)[0])
            print(name, params[name])
        (OUT / "alarm_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")
        p = params["시장 기준 차이(제안)"]
        a = alerts_from(gaps(), p["theta"], p["k"])
        a["period"] = np.where(a["month"] <= pd.Timestamp(TUNE[1] + "-01"), "tune", "eval")
        a = a[a["month"] >= pd.Timestamp(TUNE[0] + "-01")]
        a.insert(0, "alert_id", [f"A{i + 1:03d}" for i in range(len(a))])
        a["month"] = a["month"].dt.strftime("%Y-%m")
        a.to_csv(OUT / "alerts.csv", index=False)
        print("경보", len(a), a["period"].value_counts().to_dict())
    elif mode == "eval":
        params = json.loads((OUT / "alarm_params.json").read_text(encoding="utf-8"))
        rows = []
        for name, f in METHODS.items():
            p = params[name]
            res, by_type, _ = score(alerts_from(f(), p["theta"], p["k"]), ev, EVAL)
            rows.append(dict(method=name, theta=p["theta"], k=p["k"], **res, **{f"recall_{t}": v for t, v in by_type.items()}))
        # 재고조정 물량 분리: 기대치와의 차이(발주 - 기대 수요) vs 정답(발주 - 실제 수요)
        tr = pd.read_csv(D / "truth" / "demand_inventory.csv", parse_dates=["month"]).pivot(index="month", columns="customer", values="demand_t")
        est = orders - orders / (gaps() + 1)
        true = orders - tr
        lo, hi = (pd.Timestamp(x + "-01") for x in EVAL)
        e, t = est.loc[lo:hi].stack(), true.loc[lo:hi].stack()
        corr = round(float(np.corrcoef(e.loc[t.index.intersection(e.index)], t.loc[t.index.intersection(e.index)])[0, 1]), 3)
        r = pd.DataFrame(rows)
        r["inventory_gap_corr"] = corr
        r.to_csv(OUT / "alarm_eval.csv", index=False)
        print(r.T.to_string())
