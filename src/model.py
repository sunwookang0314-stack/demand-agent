"""데이터 적재와 LightGBM 예측 함수. forecast.py(백테스트)와 holdout.py(과거 구간 시험, 2026-10~12 예측 고정)가 같이 쓴다."""
import lightgbm as lgb
import numpy as np
import pandas as pd

H = 3

# ---- 데이터 ----
c = pd.read_csv("data/customs.csv", parse_dates=["month"])
y = c.assign(unique_id=c["product"] + "|" + c["country"], ds=c["month"], y=c["exp_kg"] / 1000)[
    ["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])
ex = pd.read_csv("data/ecos.csv", index_col=0, parse_dates=True).join(
    pd.read_csv("data/fred.csv", index_col=0, parse_dates=True))
SIGNALS = {  # 원료·자기 제품 가격, 자기 품목 재고, 수요 산업 생산, 환율
    "ABS": ["ppi_abs", "ppi_sm", "ppi_bd", "abs_inv_t", "kr_durable_ip", "kr_auto_engine",
            "us_ip_motor_vehicles", "us_ip_appliances", "usdkrw", "cnykrw"],
    "PC": ["ppi_bpa", "ep_inv_t", "kr_durable_ip", "kr_auto_engine",
           "us_ip_motor_vehicles", "us_ip_appliances", "usdkrw", "cnykrw"],
}
yoy = np.log(ex).diff(12)  # 신호는 전년 동월 대비 변화율(로그)로 쓴다
# 대조군: 정보가 없는 AR(1) 랜덤 시계열. 실제 신호가 이보다 못하면 효과가 아니라 과적합이다
rng = np.random.default_rng(0)
PLACEBOS = [f"placebo{i}" for i in range(20)]
for p in PLACEBOS:
    yoy[p] = pd.Series(rng.normal(size=len(yoy)), index=yoy.index).ewm(alpha=0.2, adjust=False).mean()
    yoy.loc[yoy.index[:12], p] = np.nan  # 실제 신호처럼 첫 12개월은 비운다


def metrics(df, col):
    """국가별 WAPE, 제품 합계 WAPE, 편향(+면 과대예측)"""
    df = df.assign(product=df["unique_id"].str.split("|").str[0])
    tot = df.groupby(["product", "cutoff", "ds"])[["y", col]].sum()
    return {"wape_country": (df[col] - df["y"]).abs().sum() / df["y"].sum(),
            "wape_total": (tot[col] - tot["y"]).abs().sum() / tot["y"].sum(),
            "bias": (df[col] - df["y"]).sum() / df["y"].sum()}


# ---- LightGBM (제품별, horizon별 direct 모델, 국가 공통) ----
# 마지막 자료 뒤 H개월을 빈 칸으로 늘려 둬야 실제 미래 예측 행이 만들어진다. 학습엔 기준 시점까지만 쓰므로 백테스트엔 영향 없다
w = y.pivot(index="ds", columns="unique_id", values="y")
w = w.reindex(pd.date_range(w.index[0], w.index[-1] + pd.DateOffset(months=H), freq="MS"))
logw = np.log1p(w)


def frame(product, h, signals, target):
    """타깃 월 t, 기준 시점 a=t-h. 특징은 전부 a 이전 정보.
    target="yoy"면 작년 같은 달 대비 로그 변화를 맞히고, 예측 때 작년 값을 다시 더한다."""
    rows = []
    for uid in [u for u in w.columns if u.startswith(product + "|")]:
        s = logw[uid]
        f = pd.DataFrame({"y": s, "lag0": s.shift(h), "lag1": s.shift(h + 1), "lag2": s.shift(h + 2),
                          "lag12": s.shift(12), "ma3": s.shift(h).rolling(3).mean()})
        f["base"] = f["lag12"] if target == "yoy" else 0.0
        f["month"] = f.index.month
        f["uid"] = uid
        for sig in signals:
            f[sig] = yoy[sig].shift(h + 1).reindex(f.index)  # 발표 시차 1개월
        rows.append(f)
    f = pd.concat(rows).dropna(subset=["lag12", "lag2"])
    f["uid"] = f["uid"].astype("category")
    return f


PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 20,
          "verbose": -1, "seed": 0, "deterministic": True}


def lgbm_pred(product, signals, target, cuts):
    assert max(cuts) <= y["ds"].max()  # 늘린 빈 칸이 학습에 들어가지 않게
    preds = []
    for h in range(1, H + 1):
        f = frame(product, h, signals, target)
        feats = [col for col in f.columns if col not in ("y", "base")]
        for cut in cuts:
            tr, te = f[f.index <= cut], f[f.index == cut + pd.DateOffset(months=h)]
            m = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr["y"] - tr["base"]), num_boost_round=200)
            preds.append(pd.DataFrame({"unique_id": te["uid"].astype(str).values, "ds": te.index, "cutoff": cut,
                                       "pred": np.expm1(m.predict(te[feats]) + te["base"].values)}))
    return pd.concat(preds, ignore_index=True)


def lgbm_cv(product, signals, target, cuts):
    return lgbm_pred(product, signals, target, cuts).merge(y, on=["unique_id", "ds"])
