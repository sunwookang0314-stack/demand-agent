"""고정한 규칙을 손대지 않고 두 번 더 채점한다 -> out/

규칙(LOG.md 2026-11-02에 실행 전에 적음): Combo = (AutoETS + LightGBM) / 2. LightGBM 타깃은 ABS 로그 수준, PC 작년 같은 달 대비,
외부 신호 없음, PARAMS는 model.py 그대로.
1. 과거 구간: 기준 시점 2018-12~2021-09(34개), 목표월 2019-01~2021-12. 선별·평가 구간 어디에서도 채점한 적 없는 달이다.
   코로나 급락 달(목표월 2020-03~08)을 뺀 점수도 같이 낸다. 결과가 나빠도 규칙은 바꾸지 않는다.
2. 2026-10~12 예측: 마지막 자료(2026-09)로 낸 예측을 지금 파일로 고정한다. 관세청 10월 자료가 나오면 채점한다.
"""
import hashlib
from pathlib import Path

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, Naive, SeasonalNaive

from model import H, lgbm_cv, lgbm_pred, metrics, w, y

RULE = {"ABS": "level", "PC": "yoy"}
CUTS = pd.date_range("2018-12-01", "2021-09-01", freq="MS")
COVID = pd.date_range("2020-03-01", "2020-08-01", freq="MS")
MODELS = ["Naive", "SeasonalNaive", "AutoETS", "LGBM", "Combo"]
OUT = Path("out")
OUT.mkdir(exist_ok=True)
sf = StatsForecast(models=[Naive(), SeasonalNaive(12), AutoETS(season_length=12)], freq="MS", n_jobs=1)

# ---- 1. 과거 구간 ----
# statsforecast는 창을 자료 끝에서부터 세므로 마지막 목표월(2021-12)에서 자른다
cv = sf.cross_validation(h=H, df=y[y["ds"] <= CUTS[-1] + pd.DateOffset(months=H)], n_windows=len(CUTS), step_size=1)
assert cv["cutoff"].min() == CUTS[0] and cv["cutoff"].max() == CUTS[-1]
cv["AutoETS"] = cv["AutoETS"].clip(lower=0)
lg = pd.concat([lgbm_cv(p, [], t, CUTS) for p, t in RULE.items()]).drop(columns="y")
cv = cv.merge(lg.rename(columns={"pred": "LGBM"}), on=["unique_id", "ds", "cutoff"])
assert len(cv) == w.shape[1] * len(CUTS) * H
cv["Combo"] = (cv["AutoETS"] + cv["LGBM"]) / 2
cv.to_csv(OUT / "holdout_past_preds.csv", index=False)

rows = []
for scope, d in [("all", cv), ("excl_2020-03~08", cv[~cv["ds"].isin(COVID)])]:
    for product in RULE:
        m = {model: metrics(d[d["unique_id"].str.startswith(product + "|")], model) for model in MODELS}
        best = min(v["wape_total"] for v in m.values())
        naive = min(m["Naive"]["wape_total"], m["SeasonalNaive"]["wape_total"])
        rows += [{"scope": scope, "product": product, "model": model, **v, "gap_to_best": v["wape_total"] - best,
                  "fva_vs_best_naive": naive - v["wape_total"]} for model, v in m.items()]
res = pd.DataFrame(rows)
res.to_csv(OUT / "holdout_past.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(4).to_string(index=False))

# ---- 2. 2026-10~12 예측 고정 ----
last = y["ds"].max()
fc = sf.forecast(h=H, df=y)
fc["AutoETS"] = fc["AutoETS"].clip(lower=0)
lg = pd.concat([lgbm_pred(p, [], t, [last]) for p, t in RULE.items()]).drop(columns="cutoff")
fc = fc.merge(lg.rename(columns={"pred": "LGBM"}), on=["unique_id", "ds"])
assert len(fc) == w.shape[1] * H
fc["Combo"] = (fc["AutoETS"] + fc["LGBM"]) / 2
fc.insert(2, "cutoff", last)
path = OUT / "forecast_2026Q4.csv"
fc.to_csv(path, index=False)
print(fc.assign(product=fc["unique_id"].str.split("|").str[0]).groupby(["product", "ds"])[MODELS].sum().round(0))
print(path, "sha256", hashlib.sha256(path.read_bytes()).hexdigest())
