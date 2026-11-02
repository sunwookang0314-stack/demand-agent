"""한국은행 ECOS 월별 지표 -> data/ecos.csv

코드는 ECOS StatisticItemList로 확인한 값이다(2026-10-31).
"""
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.environ["ECOS_KEY"]

SERIES = {  # 이름: (통계표, 항목1, 항목2)
    # 생산자물가지수(품목별)
    "ppi_abs": ("404Y016", "30521103AA", None),
    "ppi_sm": ("404Y016", "30511307AA", None),
    "ppi_an": ("404Y016", "30511402AA", None),
    "ppi_bd": ("404Y016", "30511103AA", None),
    "ppi_bpa": ("404Y016", "30511408AA", None),
    # 품목별 광공업 생산·출하·재고·내수·수출량 (2020-01~)
    "abs_prod_t": ("901Y039", "20220205", "T40"),
    "abs_ship_t": ("901Y039", "20220205", "T41"),
    "abs_inv_t": ("901Y039", "20220205", "T42"),
    "abs_dom_t": ("901Y039", "20220205", "T43"),
    "abs_exp_t": ("901Y039", "20220205", "T44"),
    "ep_prod_t": ("901Y039", "20220214", "T40"),
    "ep_inv_t": ("901Y039", "20220214", "T42"),
    "kr_auto_engine": ("901Y039", "30130101", "T40"),
    # 재별 제조업생산지수: 내구소비재(자동차·가전 포함)
    "kr_durable_ip": ("901Y034", "I31ACA", "I10B"),
    "kr_durable_inv": ("901Y034", "I31ACA", "I10F"),
    # 환율 월평균 (항목2 0000100=평균자료, 0000200=말일자료. 비우면 둘 다 와서 말일 값만 남는다)
    "usdkrw": ("731Y004", "0000001", "0000100"),
    "cnykrw": ("731Y004", "0000053", "0000100"),
}


def fetch(stat, item1, item2):
    path = f"{stat}/M/201501/202612/{item1}" + (f"/{item2}" if item2 else "")
    try:
        j = requests.get(f"https://ecos.bok.or.kr/api/StatisticSearch/{KEY}/json/kr/1/1000/{path}", timeout=60).json()
    except requests.RequestException as e:
        raise RuntimeError(f"{path}: {type(e).__name__}") from None  # 예외 원문엔 키가 든 URL이 있다
    rows = j.get("StatisticSearch", {}).get("row")
    if not rows:
        raise RuntimeError(f"{path}: {j}")
    if len({r["TIME"] for r in rows}) != len(rows):
        raise RuntimeError(f"{path}: 한 달에 여러 행이 왔다. 항목2를 지정해야 한다")
    return pd.Series({pd.Timestamp(r["TIME"][:4] + "-" + r["TIME"][4:]): float(r["DATA_VALUE"]) for r in rows})


df = pd.DataFrame({name: fetch(*spec) for name, spec in SERIES.items()}).sort_index()
df.index.name = "month"
Path("data").mkdir(exist_ok=True)
df.to_csv("data/ecos.csv")
print(df.notna().sum().to_string())
print(len(df), "rows,", df.index.min().date(), "~", df.index.max().date())
