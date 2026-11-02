"""FRED 월별 지표 -> data/fred.csv (키 불필요)"""
import io
from pathlib import Path

import pandas as pd
import requests

SERIES = {
    "us_ip_motor_vehicles": "IPG3361T3S",  # 미국 자동차·부품 산업생산지수
    "us_ip_appliances": "IPG3352S",  # 미국 가전 산업생산지수
}

frames = []
for name, sid in SERIES.items():
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=30)
    r.raise_for_status()
    s = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True).iloc[:, 0].rename(name)
    frames.append(pd.to_numeric(s, errors="coerce"))

df = pd.concat(frames, axis=1).loc["2015":]
Path("data").mkdir(exist_ok=True)
df.to_csv("data/fred.csv")
print(df.tail(3), "\n", len(df), "rows")
