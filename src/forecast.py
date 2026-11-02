"""예측 단계별 백테스트 + 외부 신호 선별 -> out/

대상: 관세청 ABS·PC 국가별 월 수출량(톤). 롯데 고객사 발주 대신 쓰는 공개 데이터.
백테스트: 2021-12 ~ 2026-06 매달을 기준 시점으로 잡고 1~3개월 뒤를 예측.
기준 시점은 둘로 나눈다. 2021-12~2024-06(31개)은 타깃 형식과 외부 신호를 고르는 데만 쓰고,
2024-07~2026-06(24개)은 고른 결과를 평가하는 데만 쓴다(고른 구간에서 점수를 내면 낙관치가 된다).
단계: 나이브(직전 달, 작년 같은 달) -> AutoETS -> LightGBM(과거 판매만) -> LightGBM + 외부 신호.
LightGBM은 제품별 모델이고, 타깃은 로그 수준과 작년 같은 달 대비 로그 변화 중 선별 구간에서 나은 쪽을 쓴다.
외부 신호는 제품과 연결 근거가 있는 것만 하나씩 넣어 보고, 상대 WAPE가 1% 이상 줄면서
정보가 없는 랜덤 시계열 20개 중 가장 좋은 것보다도 나은 것만 남긴다.
실제로 쓸 모델(chosen)은 고르지 않고 AutoETS와 LightGBM의 평균으로 고정한다. 선별 구간 1위를 고르는 방식은
모델 순위가 기간마다 뒤집혀서 실패했다. 이 규칙은 평가 구간 결과를 한 번 본 뒤에 정했다(LOG.md 2026-11-01).
PC는 선별에서 신호 1개가 남았지만 Combo에는 신호 없는 LightGBM을 쓴다. 이것도 평가 구간을 본 뒤 정했다.
신호는 발표 시차를 감안해 기준 시점보다 한 달 앞 값까지만 쓴다.
"""
from pathlib import Path

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, Naive, SeasonalNaive

from model import H, PLACEBOS, SIGNALS, lgbm_cv, metrics, w, y

FIRST_CUT, SPLIT_CUT, LAST_CUT = pd.Timestamp("2021-12-01"), pd.Timestamp("2024-06-01"), pd.Timestamp("2026-06-01")
SELECT = pd.date_range(FIRST_CUT, SPLIT_CUT, freq="MS")
REPORT = pd.date_range(SPLIT_CUT + pd.DateOffset(months=1), LAST_CUT, freq="MS")
OUT = Path("out")
OUT.mkdir(exist_ok=True)

# ---- 1~2단계: 통계 모델 ----
n_windows = len(SELECT) + len(REPORT)
# n_jobs=1: Windows에선 병렬 자식 프로세스가 이 스크립트를 처음부터 다시 실행해 무한히 멈춘다(10/31 2시간 40분 정지)
sf = StatsForecast(models=[Naive(), SeasonalNaive(12), AutoETS(season_length=12)], freq="MS", n_jobs=1)
cv = sf.cross_validation(h=H, df=y, n_windows=n_windows, step_size=1)
assert cv["cutoff"].min() == FIRST_CUT and cv["cutoff"].max() == LAST_CUT
cv["AutoETS"] = cv["AutoETS"].clip(lower=0)  # 수출량은 음수가 없다(백테스트에서 8건 음수)
print("stats done", flush=True)


# ---- 3~4단계: LightGBM (정의는 model.py) ----
def wape(p):
    return metrics(p, "pred")["wape_total"]


# ---- 선별 구간: 타깃 형식 -> 신호 하나씩 -> 남은 신호 조합 ----
gate, pick, sel_preds = [], {}, []
for product in ["ABS", "PC"]:
    tp = {t: lgbm_cv(product, [], t, SELECT) for t in ["level", "yoy"]}
    tw = {t: wape(p) for t, p in tp.items()}
    target = min(tw, key=tw.get)
    base = tw[target]
    sel_preds.append(tp[target].rename(columns={"pred": "LGBM"}).drop(columns="y"))
    res = {s: wape(lgbm_cv(product, [s], target, SELECT)) for s in SIGNALS[product] + PLACEBOS}
    bar = min(res[p] for p in PLACEBOS)
    for s, v in res.items():
        gate.append({"product": product, "target": target, "signal": s, "wape_base": base, "wape_with": v,
                     "rel_change": v / base - 1, "placebo_best": bar,
                     "kept": s in SIGNALS[product] and v / base - 1 <= -0.01 and v < bar})
    kept = [g["signal"] for g in gate if g["product"] == product and g["kept"]]
    score = {"LGBM": base}
    if kept:
        combo = wape(lgbm_cv(product, kept, target, SELECT)) if len(kept) > 1 else res[kept[0]]
        if combo > min(res[k] for k in kept):
            kept = [min(kept, key=res.get)]  # 조합이 단독보다 못하면 가장 좋은 하나만
        score["LGBM+signals"] = min(combo, res[kept[0]])
    pick[product] = (target, kept, score)
    print(product, "target", tw, "kept", kept, flush=True)
gate = pd.DataFrame(gate)
gate.to_csv(OUT / "signal_gate.csv", index=False)

# ---- 쓸 모델은 고르지 않고 AutoETS와 LightGBM의 평균(Combo)으로 고정한다 ----
# 선별 구간 1위를 고르게 했더니 ABS는 직전 달 나이브가 뽑혔고 평가 구간에서 꼴찌였다(11/1). 모델 순위가 기간마다 뒤집힌다.
# 선별 구간 점수는 이 사실을 보여 주는 참고용으로만 남긴다.
sel = cv[cv["cutoff"].isin(SELECT)].merge(pd.concat(sel_preds), on=["unique_id", "ds", "cutoff"])
assert len(sel) == w.shape[1] * len(SELECT) * H
sel["Combo"] = (sel["AutoETS"] + sel["LGBM"]) / 2
select_scores = []
for product, (target, kept, score) in pick.items():
    s = sel[sel["unique_id"].str.startswith(product + "|")]
    score = {m: metrics(s, m)["wape_total"] for m in ["Naive", "SeasonalNaive", "AutoETS", "Combo"]} | score
    select_scores += [{"product": product, "model": m, "wape_total_select": v} for m, v in score.items()]
    print(product, "select wape", {k: round(v, 4) for k, v in score.items()}, flush=True)
pd.DataFrame(select_scores).to_csv(OUT / "select_scores.csv", index=False)

# ---- 평가 구간: 고른 그대로 한 번만 ----
final = []
for product, (target, kept, _) in pick.items():
    b = lgbm_cv(product, [], target, REPORT).rename(columns={"pred": "LGBM"})
    s = lgbm_cv(product, kept, target, REPORT) if kept else b.rename(columns={"LGBM": "pred"})
    final.append(b.merge(s.drop(columns="y").rename(columns={"pred": "LGBM+signals"}), on=["unique_id", "ds", "cutoff"]))
cv = cv[cv["cutoff"].isin(REPORT)].merge(pd.concat(final).drop(columns="y"), on=["unique_id", "ds", "cutoff"])
assert len(cv) == w.shape[1] * len(REPORT) * H
cv["Combo"] = (cv["AutoETS"] + cv["LGBM"]) / 2
cv.to_csv(OUT / "backtest_preds.csv", index=False)

# ---- 단계별 비교표 (가장 좋은 나이브 대비 개선 = FVA), horizon별로도 ----
MODELS = ["Naive", "SeasonalNaive", "AutoETS", "LGBM", "LGBM+signals", "Combo"]
cv["h"] = (cv["ds"].dt.year - cv["cutoff"].dt.year) * 12 + cv["ds"].dt.month - cv["cutoff"].dt.month
ladder = []
for product in ["ABS", "PC"]:
    p = cv[cv["unique_id"].str.startswith(product + "|")]
    for h, sub in [("1-3", p)] + list(p.groupby("h")):
        m = {model: metrics(sub, model) for model in MODELS}
        naive = min(m["Naive"]["wape_total"], m["SeasonalNaive"]["wape_total"])
        for model in MODELS:
            ladder.append({"product": product, "h": h, "model": model, **m[model],
                           "fva_vs_best_naive": naive - m[model]["wape_total"], "chosen": model == "Combo"})
ladder = pd.DataFrame(ladder)[["product", "h", "model", "chosen", "wape_total", "wape_country", "bias",
                               "fva_vs_best_naive"]]
ladder.to_csv(OUT / "ladder.csv", index=False)
pd.set_option("display.width", 200)
print(gate[~gate["signal"].str.startswith("placebo")].round(4).to_string(index=False))
print(ladder.round(4).to_string(index=False))
