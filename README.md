# 환율·시장지수·변동성이 한국·미국 기술주 ETF 수익률에 미치는 영향 비교 분석

**데이터마이닝 (대학원) | 2026년 1학기 | 구자환 교수**  
**안명현 (2025720536)**

---

> ⚠️ 본 README는 **분석 기간**으로 작성되었습니다.
> 보고서·발표자료에 기재된 모든 수치는 이 시점에 고정되어 있으며,
> 대시보드 데이터(`analysis_ready.csv`)는 이후에도 주 단위로 계속 업데이트됩니다.
> - (현재 기준: 2026-06-12까지 업데이트 완료, 대시보드에서 최신 결과 확인 가능)

## 프로젝트 개요

원/달러 환율, S&P500, KOSPI, VIX, WTI, Google Trends 기반 AI 관심도 등 6개 거시변수가
한국·미국 기술주 ETF 수익률에 미치는 영향을 비교 분석합니다.
주간 데이터 기반 OLS 다중회귀를 기반으로, 다변량 Rolling Beta·국면별 Chow Test·
Bootstrap 강건성 검증까지 확장하였습니다.

- **분석 기간**: 2022-01-14 ~ 2026-05-29 (229주, 주간 수익률)
- **대시보드**: http://13.124.53.41:8501/ (AWS EC2 배포)
---

## 분석 대상

| 구분 | 종목 | 티커 |
|------|------|------|
| 한국 ETF | KODEX 반도체 | 091160.KS |
| 한국 ETF | TIGER 200 IT | 157490.KS |
| 미국 ETF | SOXX (원화환산) | SOXX |
| 미국 ETF | QQQ (원화환산) | QQQ |

- **독립변수 6개**: USD/KRW 환율, S&P500, KOSPI, VIX, WTI, AI관심도(Google Trends)
- **원화 환산 공식**: `r_KRW = (1 + r_USD) × (1 + r_FX) − 1`

---

## 시장 국면 구분 (FOMC 기준)

| 국면 | 기간 |
|------|------|
| 긴축기 | 2022-03-17 ~ 2023-07-26 |
| AI랠리기 | 2023-07-27 ~ 2024-09-17 |
| 불확실성기 | 2024-09-18 ~ 2025-12-31 |
| 현재 | 2026-01-01 ~ |

---

## 주요 가설 및 최종 결과

| 가설 | 내용 | 판정 |
|------|------|------|
| H1-1 | 한국 ETF 환율 민감도 > 미국 ETF | ❌ 기각 |
| H1-2 | 유가가 ETF 수익에 영향 | ⚠️ 부분 지지 (KODEX만) |
| H1-3 | 각국 시장지수가 지배적 | ✅ 지지 |
| H1-4 | AI관심도가 직접 영향 | ❌ 기각 |

---

## 기말 심층 분석 (교수님 Q1~Q4 대응)

| Q | 내용 | 핵심 결과 |
|---|------|----------|
| Q1 | 다변량 Rolling Beta | 6변수 통제 후에도 KODEX 현재 국면 β 음수 유지 |
| Q2 | SOXX β 원화/달러 분리 | β_원화=1.044, β_달러=0.031, 차이≈1 → 회계효과 |
| Q3 | AI관심도 Mediator 탐색 | 직접효과 없음, 간접경로 가능성 시사 |
| Q4 | Bootstrap CI + Chow Test | LOO 전구간 음수, Chow p=0.003 구조변화 확인 |

> 모든 결과는 잔차진단(Durbin-Watson·Breusch-Pagan)을 거쳤으며, KODEX 모형은 이분산성 가능성(BP p=0.026)이 있어 보수적으로 해석하였습니다.

---

## 저장소 구조

```text
data-mining/
├── 01_data_pipeline.ipynb       # 데이터 수집·전처리·VIF·상관관계
├── 02_regression.ipynb          # OLS·Rolling Beta·Bootstrap·Chow Test
├── 데이터마이닝_안명현_기말발표자료.pdf  # 기말 발표 슬라이드
├── README.md                # 데이터 설명
├── data/   
│   ├── analysis_ready.csv       # 최종 분석 데이터 (계속 업데이트)
│   └── google_trends_AI_weekly_final.csv  # Google Trends 원본
└── dashboard/
    ├── streamlit_app.py         # Streamlit 대시보드 소스코드
    ├── analysis_ready.csv       # 대시보드용 데이터 (주 단위 갱신)
    └── requirements.txt         # 라이브러리 목록
```
> ⚠️ `data/analysis_ready.csv`(보고서 고정본)와 `dashboard/analysis_ready.csv`(실시간 갱신본)는 서로 다른 시점의 데이터입니다. 보고서 수치를 재현하려면 `data/` 폴더의 파일을 사용하세요.

---

## 분석 파이프라인

```
데이터 수집(yfinance·Google Trends)
→ 전처리·원화환산·국면라벨링
→ VIF 진단·상관관계 히트맵
→ OLS 다중회귀 (전체기간)
→ 잔차진단 (Durbin-Watson·Breusch-Pagan)
→ 다변량 Rolling Beta (26주창)
→ SOXX β 원화/달러 분리
→ AI관심도 Mediator 탐색 (+ Lag 1~2주 보조 분석)
→ Bootstrap CI + LOO
→ 국면별 OLS + Chow Test
```

---

## 실행 방법

### 노트북 재현 (보고서 기준 결과)
1. Google Colab에서 노트북을 엽니다.
2. Google Drive를 마운트합니다.
3. `data/` 폴더의 CSV 파일을 Drive에 업로드합니다.
4. `01_data_pipeline.ipynb` → `02_regression.ipynb` 순서로 실행합니다.

**대시보드 로컬 실행:**
```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/streamlit_app.py
```

### 대시보드 바로 확인 (갱신본)
👉 http://13.124.53.41:8501

---

## 사용 기술

Python (pandas, numpy, yfinance, statsmodels, scipy, plotly, streamlit)
Google Colab, Google Drive, AWS EC2, GitHub

---

## 한계 및 향후 과제

- 현재 국면 표본 소규모 (보고서 기준 n=22주, 대시보드에서는 매주 증가) → 향후 표본 확대
- 본 연구는 설명 모형에 초점 → out-of-sample 예측, 포트폴리오 백테스트 등 향후 검증 필요


