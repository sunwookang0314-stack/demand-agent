"""관세청 품목별 국가별 수출입실적 -> data/customs.csv (월 x 국가 x 품목)

롯데케미칼 고객사 발주 데이터 대신 한국 전체 ABS·PC 수출을 쓴다.
국가 코드가 필수라서 주요 수출국을 하나씩 부른다. 조회 기간은 1년 이내 제한.
"""
import os
import time
import defusedxml.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.environ["DATA_GO_KR_KEY"]
URL = "https://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList"
HS = {"390330": "ABS", "390740": "PC"}
COUNTRIES = ["CN", "VN", "IN", "US", "TR", "ID", "TH", "MY", "MX", "PL", "DE", "JP", "HK", "TW", "BR"]
LAST = pd.Timestamp.today().strftime("%Y%m")


def fetch(hs, cnty, year):
    end = min(f"{year}12", LAST)
    params = {"serviceKey": KEY, "strtYymm": f"{year}01", "endYymm": end, "hsSgn": hs, "cntyCd": cnty}
    for attempt in range(4):  # 관세청 서버가 가끔 500을 내거나 연결이 끊긴다
        try:
            r = requests.get(URL, params=params, timeout=30)
            if r.ok:
                break
            err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            err = type(e).__name__  # 예외 원문엔 키가 든 URL이 있어서 이름만 남긴다
        time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"{hs} {cnty} {year}: {err}")
    root = ET.fromstring(r.content)
    code = root.findtext(".//resultCode")
    if code not in (None, "00"):
        raise RuntimeError(f"{hs} {cnty} {year}: {code} {root.findtext('.//resultMsg')}")
    return [{c.tag: c.text for c in item} for item in root.iter("item")]


rows = []
for hs, name in HS.items():
    for cnty in COUNTRIES:
        for year in range(2015, int(LAST[:4]) + 1):
            for it in fetch(hs, cnty, year):
                if it.get("year", "").count(".") == 1:  # '총계' 행 제외, 'YYYY.MM'만
                    rows.append({"month": it["year"].replace(".", "-"), "product": name, "country": cnty,
                                 "exp_kg": float(it.get("expWgt") or 0), "exp_usd": float(it.get("expDlr") or 0)})
            time.sleep(0.1)
        print(name, cnty, "done")

df = pd.DataFrame(rows)
df["month"] = pd.to_datetime(df["month"])
# 자료 없는 기간은 오류 없이 빈 응답으로 와서, 빠진 국가·월이 조용히 지나가지 않게 막는다
g = df.groupby(["product", "country"]).size()  # 국가가 통째로 빠져도 잡히게 기대 개수와 비교
assert len(g) == len(HS) * len(COUNTRIES) and (g == df["month"].nunique()).all(), "빠진 국가·월이 있다"
Path("data").mkdir(exist_ok=True)
df.sort_values(["product", "country", "month"]).to_csv("data/customs.csv", index=False)
print(len(df), "rows,", df["month"].min().date(), "~", df["month"].max().date())
