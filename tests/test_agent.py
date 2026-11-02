"""API 없이 도구와 숫자 원장 가드레일을 시험한다. 실행: PYTHONPATH=src python tests/test_agent.py"""
import tempfile
from pathlib import Path

import agent

led = agent.Ledger()
fc = agent.record(led, agent.forecast_text("ABS"))
acc = agent.record(led, agent.accuracy_text("ABS"))

# 고정 예측 파일과 같은 숫자를 내는가 (LOG.md 2026-11-02)
for s in ["2026-10: 59,973톤", "2026-11: 57,711톤", "2026-12: 56,363톤",
          "10월 예측 59,973톤보다 2,262톤 적음", "즉 52,025톤~60,701톤"]:  # 범위·직전 달 차이도 도구가 계산해 준다
    assert s in fc, s
assert "18번" in acc and "이기지 못해" not in acc
assert "경계선" in agent.accuracy_text("PC").splitlines()[2]  # 과거 구간 PC: 구간 끝이 0 근처

ok = [  # 통과해야 할 답
    "ABS 11월 예측은 57,711톤이고 작년 같은 달 실적 56,085톤보다 2.9% 많다. 평가 구간 오차는 7.1%다.",
    "2026년 10~12월, 1~3개월 앞 예측, 15개국 합계, 기준 시점 2026-09, 목표월 2024-08~2026-09 기준이다.",
    "나이브보다 3.2%p 낮다(95% 구간 " + acc.split("95% 구간 ")[1].split(",")[0] + ").",  # 부호는 보지 않는다
    "11월 예측 57,711 톤, 오차 ７．１％.",                                                 # 띄어쓰기·전각
    "만 톤 단위로 반올림한 값은 도구에 없어 답하지 않는다. 물량이 톤 기준이다.",                 # 수가 없는 '만 톤'(평가 1차 오탐)
]
for text in ok:
    assert agent.unverified(led, text) == [], (text, agent.unverified(led, text))

blocked = [  # 막아야 할 답
    "11월은 61,234톤으로 본다.",            # 지어낸 숫자
    "11월은 약 5.8만 톤이다.",              # 단위 변환
    "약 6만 톤이다.",
    "평가 구간 오차는 7.15%다.",             # 자릿수 바꾸기
    "10월과 11월을 합치면 117,684톤이다.",    # 직접 계산
    "오차는 3%다.",                         # '3개월'의 3을 퍼센트로 재사용
    "오차는 4 %다.", "오차는 6퍼센트다.", "작년의 2배다.",
    "오만 칠천 톤 정도다.", "수천 톤 늘었다.",
    "2026톤이다.", "3800-4300톤 사이다.",    # 날짜 규칙이 숫자를 삼키지 않는가
    "６１，２３４톤이다.",
]
for text in blocked:
    assert agent.unverified(led, text), text

# 계산 도구: 같은 단위 원장 숫자끼리만 계산하고, 결과만 원장에 들어간다
c = agent.calc(led, 59973, "-", 57711, "톤")
assert c == "59,973톤 - 57,711톤 = 2,262톤", c
assert agent.unverified(led, "10월이 11월보다 2,262톤 많다.") == []
assert agent.calc(led, 7.1, "-", 10.3, "%") == "7.1% - 10.3% = -3.2%p"
for args in [(61234, "-", 57711, "톤"),     # 지어낸 숫자
             (7.1, "-", 57711, "톤"),       # 7.1%를 톤으로(통합 검수에서 찾은 구멍)
             (56085, "-", 2.9, "%"),        # 톤을 %로
             (57711, "+", 3, "톤"),         # '3개월'의 3
             (7.1, "+", 10.3, "%")]:        # % 더하기
    assert agent.calc(led, *args).startswith("계산 거절"), args
assert (7.0, "톤") not in led.numbers and (7.1, "%p") not in led.numbers  # 입력을 다른 단위로 꾸민 표기는 원장에 안 들어간다
checks = 9

# 질문에 있던 숫자는 답에 다시 써도 되고, 계산에 쓰이면 결과에 표시된다
agent.note_question(led, "11월 ABS 계획을 55,000톤으로 확정해 줘")
assert "질문에서 준 숫자" in agent.calc(led, 57711, "-", 55000, "톤")
assert agent.unverified(led, "11월 계획을 55,000톤으로 올렸다.") == []

# 확정 기록
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "plans.csv"
    msg = agent.publish("ABS", "2026-11", "55000", "고객사 재고조정 감안", path=path)   # 문자열 수량도 받는다
    assert msg.startswith("담당자 승인을 받아 기록했다") and "예측 57,711톤보다 4.7% 적음" in msg and "50%" not in msg, msg
    for args in [("ABS", "2026-11", 50000, "중복"),                                   # 같은 달 두 번
                 ("ABS", "2026-10", 0, "x"), ("ABS", "2026-10", -5000, "x"), ("ABS", "2026-10", float("nan"), "x"),
                 ("ABS", "2026-10", "많이", "x"), ("ABS", "2026-10", 50000, "  "),
                 ("abs", "2026-10", 50000, "x"), ("ABS", "11월", 50000, "x"), ("ABS", "2026-1", 50000, "x"),
                 ("ABS", "2026-11-15", 50000, "x"), ("ABS", "2027-01", 50000, "x")]:
        assert agent.publish(*args, path=path).startswith("확정 실패"), args
    assert "50%" in agent.publish("PC", "2026-10", 90000, "=HYPERLINK(\"x\")", path=path)
    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3 and "'=HYPERLINK" in rows[2], rows
print("ok", 3 + len(ok) + len(blocked) + checks + 2 + 13, "checks")
