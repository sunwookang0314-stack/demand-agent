# demand-agent

LOTTE AX Challenge 2026 과제 4(롯데케미칼 첨단소재 수요예측 AI 기획)를 준비하면서 공개 데이터로 먼저 만들어 본 수요예측 모델과 수요계획 에이전트입니다. 롯데케미칼이 받는 발주 데이터 대신 관세청 ABS(HS 390330)·PC(HS 390740) 15개국 월별 수출량을 썼습니다. 날짜별로 한 일은 [LOG.md](LOG.md)에 있습니다.

## 결과

1~3개월 앞 예측, 15개국 합계 WAPE(오차 절댓값 합 ÷ 실적 합) 기준입니다. 최종 예측은 AutoETS와 LightGBM의 평균입니다. 코드와 LOG.md의 선별 구간·평가 구간은 제출 서류의 검증 구간·테스트 구간과 같습니다.

| 구간 | ABS 평균 | ABS 나은 나이브 | PC 평균 | PC 나은 나이브 |
|---|---|---|---|---|
| 평가 구간(기준 시점 2024-07~2026-06, 평균 규칙을 정할 때 본 구간) | 7.1% | 10.3% | 9.3% | 10.7% |
| 과거 구간(기준 시점 2018-12~2021-09, 규칙 고정 뒤 처음 채점) | 8.0% | 8.5% | 10.8% | 12.5% |

나은 나이브와의 차이를 기준 시점 블록 부트스트랩으로 보면 ABS 평가 구간만 0을 포함하지 않고, ABS 과거 구간과 PC 평가 구간은 0을 포함해 차이를 확정할 수 없으며, PC 과거 구간은 경계선입니다. 외부 지표 12개(생산자물가, 국내 재고, 한국·미국 산업생산, 환율)는 판매량과 관계없는 랜덤 시계열 20개 중 가장 좋은 것보다 나을 때만 쓰기로 했는데, 18번 시험에서 1번만 통과했고 그것도 평가 구간에서 오차를 키워서 지금 예측엔 들어가 있지 않습니다.

에이전트는 개발에 쓴 14문항을 3번씩 돌려 41/42, 실행 전에 커밋한 새 질문 10개를 한 번만 돌려 7/10(자동 채점)이었습니다.

### 가상 발주 시험 (가상 데이터, 실제 롯데 데이터 아님)

공개 데이터엔 고객사 발주가 없어서, 실제 관세청 ABS 수출을 최종 수요로 두고 가상 고객사 20곳의 재고 정책과 재고조정·실수요 변화 사건 147건, 가상 영업 메모 134건을 심은 데이터(`src/synthetic.py`, seed 2026)로 방법을 시험했습니다. 정답은 `data/synthetic/truth/`에 따로 두고 경보와 에이전트 도구는 읽지 않습니다.

| 2021~2025년, 사건 67건 | 시장 기준 경보 | 발주만 작년 대비 경보 |
|---|---|---|
| F1 | 0.435 | 0.343 |
| 정밀도 / 재현율 | 37% / 52% | 24% / 60% |
| 오경보(고객당 연) | 0.69건 | 1.61건 |
| 재현율: 실수요 변화 / 재고조정·확대 | 83~100% / 13~44% | 92~100% / 38~44% |

경보만으로는 재고조정을 절반도 못 잡아서, 경보가 뜬 뒤 원인을 가르는 일은 별도 에이전트가 영업 메모를 검색해 판단합니다(재고조정 / 실수요 변화 / 판단 보류 + 근거 메모 번호). 표본 경보 58개에서 정답 56개, 메모로 원인이 드러난 경보 27/28, 틀린 단정 1건이었고, 메모 없이 같은 달 경보 수만 보는 규칙은 정답 17개, 틀린 단정 27건이었습니다. 메모가 사건 유형별 템플릿 3개씩이라 실제 영업 메모보다 쉽습니다.

## 실행 전에 남긴 기록

| 무엇 | 확인할 곳 |
|---|---|
| 새 평가 질문 10개 | 커밋 [e92586f](https://github.com/sunwookang0314-stack/demand-agent/commit/e92586fd9108c728d952a7a5fa2071700f064b85)(11/2 14:57 푸시)의 `src/evaluate.py` HOLDOUT. 결과는 그 뒤 커밋 [23389f9](https://github.com/sunwookang0314-stack/demand-agent/commit/23389f95efca4618f5b9eba0798e53ee476fc872)의 `out/eval_holdout.csv` |
| 2026년 10~12월 예측 | 커밋 [da5a93b](https://github.com/sunwookang0314-stack/demand-agent/commit/da5a93b017e8b9c069d08904d70e7e2edfa4b671)(11/2 13:27)의 `out/forecast_2026Q4.csv`, sha256 `50ef94c1d4a00fe2a278c407f1c6ed51cbf091818329c4396ae8f47e16819633`. 관세청 10월 자료가 나오면 이 파일 그대로 채점합니다 |
| GitHub 서버의 푸시 시각 | [저장소 활동 기록](https://github.com/sunwookang0314-stack/demand-agent/activity). 커밋 시각은 작성자가 바꿀 수 있지만 이 시각은 GitHub 서버가 남깁니다. da5a93b 11/2 13:27:34, e92586f 14:57:59, 23389f9 15:01:00. 새 질문 첫 실행은 `out/agent_log.jsonl` 기준 14:58:09로 e92586f 푸시 뒤입니다 |
| 가상 발주 경보 기준 | 커밋 [9c62903](https://github.com/sunwookang0314-stack/demand-agent/commit/9c62903ce1f129a65c6dd9351d0b2b1cf0e8bbe4)(11/3 03:50)의 `out/synthetic/alarm_params.json`(2016~2020에서 고름). 결과는 커밋 [9b34b4e](https://github.com/sunwookang0314-stack/demand-agent/commit/9b34b4ebeae511a3adea9c4357b230a81415bdf2)의 `out/synthetic/alarm_eval.csv` |
| 경보 원인 판단 표본·정답 규칙 | 커밋 [9b34b4e](https://github.com/sunwookang0314-stack/demand-agent/commit/9b34b4ebeae511a3adea9c4357b230a81415bdf2)의 `out/synthetic/cause_sample.csv`와 `src/evaluate_cause.py`. 결과는 커밋 [f3d3d6f](https://github.com/sunwookang0314-stack/demand-agent/commit/f3d3d6fa8429dd693d3ac29f50440aab47852728)의 `out/synthetic/cause_eval.csv` |
| 과거 구간 시험 규칙 | LOG.md 2026-11-02 첫 줄(00:57 기록, 01:22 실행). 커밋은 결과와 함께 da5a93b에 들어갔고, 이 커밋의 `src/holdout.py`를 다시 돌리면 같은 결과 파일이 나옵니다 |

`.gitattributes`에서 CSV를 바이트 그대로 저장하게 해 두어서 어느 OS에서 받아도 sha256이 같습니다.

## 구조

```
src/collect_customs.py   관세청 품목별·국가별 수출입 → data/customs.csv
src/collect_ecos.py      한국은행 ECOS 지표 → data/ecos.csv
src/collect_fred.py      FRED 미국 산업생산 → data/fred.csv (키 불필요)
src/model.py             데이터 적재, 신호·랜덤 대조군, LightGBM
src/forecast.py          백테스트: 신호 시험, 선별·평가 구간, 모델 단계표 → out/ladder.csv 등
src/holdout.py           과거 구간 시험, 2026년 10~12월 예측 고정 → out/holdout_past*.csv, out/forecast_2026Q4.csv
src/agent.py             에이전트(OpenAI Agents SDK): 도구 4개, 숫자 원장 가드레일, 계획 확정 승인
src/evaluate.py          에이전트 평가 세트(개발 14문항, 새 질문 10문항)
tests/test_agent.py      API 없이 도는 시험 46개
src/synthetic.py         가상 발주 데이터(가상 고객사 20곳, 사건·메모·정답) → data/synthetic/
src/inventory_alarm.py   재고조정 경보: tune(2016~2020에서 기준 선택) / eval(2021~2025 채점) → out/synthetic/
src/evaluate_cause.py    경보 원인 판단 에이전트 평가(plan / run / score)
tests/test_cause.py      원인 판단 쪽 API 없는 시험 26개
out/agent_log.jsonl      에이전트 실행 기록(모델, SDK 판 번호, 호출한 도구, 차단, 승인)
```

에이전트 도구는 `get_forecast`(예측 조회), `get_accuracy`(정확도 조회), `calculate`(원장에 같은 단위로 있는 숫자끼리만 더하고 빼기), `publish_plan`(계획 확정, `needs_approval`로 담당자 승인 뒤에만 기록)입니다. 최종 답에 들어간 숫자는 (값, 단위) 쌍으로 이번 실행의 도구 출력·질문과 대조해서 없으면 답을 막고, 걸린 숫자를 알려 준 뒤 한 번만 다시 쓰게 합니다.

## 재현

Python 3.12와 Windows에서 `PYTHONUTF8=1`을 켜고 돌렸습니다.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # macOS·Linux는 .venv/bin/pip
cp .env.example .env                              # 수집과 에이전트를 돌릴 때만 키를 채운다

python src/forecast.py                            # 백테스트 (data/는 저장소에 들어 있어 수집은 건너뛰어도 된다)
python src/holdout.py                             # 과거 구간 시험과 예측 고정 파일
PYTHONPATH=src python tests/test_agent.py         # API 없이 시험 46개
python src/agent.py "ABS 11월 수출 예측 알려줘"     # OPENAI_API_KEY 필요, 확정 요청이면 터미널에서 승인을 묻는다
python src/evaluate.py 3                          # 개발 세트 3회 반복
python src/synthetic.py                           # 가상 발주 데이터 다시 만들기 (seed 고정, 같은 파일이 나온다)
python src/inventory_alarm.py eval                # 가상 발주 경보 채점
PYTHONPATH=src python tests/test_cause.py         # 원인 판단 쪽 시험 26개
python src/evaluate_cause.py score                # 저장된 원인 판단 결과 요약 (run은 OPENAI_API_KEY 필요)
```

## 한계

- 고객사 발주가 아니라 통관 기준 수출량이고, 15개국 합계라 전체 수출의 일부입니다(2024년 ABS 기준 66%).
- ECOS·FRED 지표는 나중에 수정된 최종값이라 당시 실제로 볼 수 있던 값과 다를 수 있습니다.
- 가드레일은 숫자의 출처만 봅니다. 10월 값을 11월 값이라고 쓰거나 '낮다'를 '높다'로 뒤집어도 통과하고, 날짜와 단위 없는 31 이하 정수는 검사하지 않습니다.
- 평가 구간 점수는 평균 규칙을 정할 때 본 구간이라 새 자료 성적이 아닙니다.
- 가상 발주 시험은 방법이 심어 둔 정답을 찾는지만 보여 줍니다. 실제 고객사 발주의 재고 정책·메모와는 다르고, 경보 기준은 내부 발주로 다시 정해야 합니다. `out/synthetic/alerts.csv`의 period=eval에는 2026년 경보 14개도 들어 있지만 채점은 2021~2025년만 했습니다.

## 데이터 출처

- 관세청 품목별 국가별 수출입실적(공공데이터포털)
- 한국은행 경제통계시스템(ECOS) Open API
- FRED, Federal Reserve Bank of St. Louis: IPG3361T3S, IPG3352S (Board of Governors of the Federal Reserve System)

협업 중인 회사의 판매 데이터는 이 저장소에 올리지 않았습니다(`data/private/`는 gitignore).
