"""
streamlit_app.py
한·미 기술주 ETF 수익률 분석 대시보드
작성자: 안명현 (2025720536)
마감: 2026-06-04
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# ─────────────────────────────────────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ETF 수익률 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────────────────
PHASE_COLORS = {
    "긴축기":     "rgba(255,193,7,0.18)",
    "AI랠리기":   "rgba(40,167,69,0.15)",
    "불확실성기":  "rgba(220,53,69,0.15)",
    "현재":       "rgba(23,162,184,0.15)",
}

PHASE_DATES = {
    "긴축기":     ("2022-03-17", "2023-07-26"),
    "AI랠리기":   ("2023-07-27", "2024-09-17"),
    "불확실성기":  ("2024-09-18", "2025-12-31"),
    "현재":       ("2026-01-01", "2026-12-31"),
}

ETF_DISPLAY = {
    "KODEX 반도체": "KODEX_반도체_ret",
    "TIGER 200IT":  "TIGER_200IT_ret",
    "SOXX (원화)":  "SOXX_KRW_ret",
    "QQQ (원화)":   "QQQ_KRW_ret",
}

ETF_COLORS = {
    "KODEX 반도체": "#1f77b4",
    "TIGER 200IT":  "#ff7f0e",
    "SOXX (원화)":  "#2ca02c",
    "QQQ (원화)":   "#d62728",
}

# OLS 결과 (하드코딩 — 전체 기간, USDKRW_ret 기준)
OLS_RESULTS = {
    "KODEX 반도체": {
        "USDKRW_ret": {"beta": -0.089, "pval": 0.627},
        "SP500_ret":  {"beta":  0.312, "pval": 0.041},
        "KOSPI_ret":  {"beta":  1.124, "pval": 0.000},
        "VIX_ret":    {"beta": -0.018, "pval": 0.312},
        "WTI_ret":    {"beta":  0.087, "pval": 0.214},
        "AI_interest":{"beta":  0.001, "pval": 0.680},
    },
    "TIGER 200IT": {
        "USDKRW_ret": {"beta": -0.201, "pval": 0.156},
        "SP500_ret":  {"beta":  0.289, "pval": 0.062},
        "KOSPI_ret":  {"beta":  1.052, "pval": 0.000},
        "VIX_ret":    {"beta": -0.015, "pval": 0.418},
        "WTI_ret":    {"beta":  0.043, "pval": 0.531},
        "AI_interest":{"beta":  0.002, "pval": 0.441},
    },
    "SOXX (원화)": {
        "USDKRW_ret": {"beta":  1.057, "pval": 0.000},
        "SP500_ret":  {"beta":  1.423, "pval": 0.000},
        "KOSPI_ret":  {"beta":  0.114, "pval": 0.487},
        "VIX_ret":    {"beta": -0.031, "pval": 0.089},
        "WTI_ret":    {"beta": -0.021, "pval": 0.731},
        "AI_interest":{"beta":  0.004, "pval": 0.157},
    },
    "QQQ (원화)": {
        "USDKRW_ret": {"beta":  0.979, "pval": 0.000},
        "SP500_ret":  {"beta":  1.312, "pval": 0.000},
        "KOSPI_ret":  {"beta":  0.098, "pval": 0.531},
        "VIX_ret":    {"beta": -0.028, "pval": 0.112},
        "WTI_ret":    {"beta": -0.018, "pval": 0.778},
        "AI_interest":{"beta":  0.003, "pval": 0.248},
    },
}

INDEP_VARS = ["USDKRW_ret", "SP500_ret", "KOSPI_ret", "VIX_ret", "WTI_ret", "AI_interest"]
INDEP_LABELS = {
    "USDKRW_ret":  "USD/KRW",
    "SP500_ret":   "S&P500",
    "KOSPI_ret":   "KOSPI",
    "VIX_ret":     "VIX",
    "WTI_ret":     "WTI",
    "AI_interest": "AI관심도",
}

WINDOW = 26  # Rolling Beta 창 크기

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("analysis_ready.csv", index_col="date", parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Rolling Beta 계산
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def compute_rolling_betas(_df):
    """다변량 Rolling Beta (26주 창, USDKRW_ret 계수만 추출)"""
    results = {}
    df_clean = _df.dropna(subset=INDEP_VARS)
    X = sm.add_constant(df_clean[INDEP_VARS])

    for label, col in ETF_DISPLAY.items():
        if col not in df_clean.columns:
            continue
        y_all = df_clean[col]
        betas = [np.nan] * len(df_clean)
        for i in range(WINDOW, len(df_clean) + 1):
            X_w = X.iloc[i - WINDOW : i]
            y_w = y_all.iloc[i - WINDOW : i]
            if y_w.isna().any():
                continue
            try:
                res = sm.OLS(y_w, X_w).fit()
                betas[i - 1] = res.params.get("USDKRW_ret", np.nan)
            except Exception:
                pass
        results[label] = pd.Series(betas, index=df_clean.index)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 국면 배경 vrect 추가 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def add_phase_shapes(fig, phases=None):
    """plotly figure에 국면별 배경색 vrect를 추가한다."""
    if phases is None:
        phases = list(PHASE_DATES.keys())
    for phase in phases:
        start, end = PHASE_DATES[phase]
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=PHASE_COLORS[phase],
            opacity=1,
            layer="below",
            line_width=0,
            annotation_text=phase,
            annotation_position="top left",
            annotation_font_size=10,
        )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.title("🎛️ 필터 설정")
    st.sidebar.markdown("---")

    selected_etfs = st.sidebar.multiselect(
        "ETF 선택",
        options=list(ETF_DISPLAY.keys()),
        default=list(ETF_DISPLAY.keys()),
    )
    if not selected_etfs:
        st.sidebar.warning("ETF를 1개 이상 선택하세요.")
        selected_etfs = list(ETF_DISPLAY.keys())

    selected_phases = st.sidebar.multiselect(
        "국면 필터 (Boxplot용)",
        options=list(PHASE_DATES.keys()),
        default=list(PHASE_DATES.keys()),
    )
    if not selected_phases:
        selected_phases = list(PHASE_DATES.keys())

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "📌 분석 기간: 2022-01-14 ~ 2026-05-08\n\n"
        "작성자: 안명현 (2025720536)\n\n"
        "데이터마이닝 기말 프로젝트"
    )

    return selected_etfs, selected_phases


# ─────────────────────────────────────────────────────────────────────────────
# 탭 1: 누적 수익률
# ─────────────────────────────────────────────────────────────────────────────
def tab_cumulative(df, selected_etfs):
    st.subheader("📈 ETF 누적 수익률 비교")
    st.caption("FOMC 기준 4개 핵심 국면만 음영 처리: 🟡 긴축기  🟢 AI랠리기  🔴 불확실성기  🔵 현재 (초반 기타 구간 제외)")

    fig = go.Figure()

    for label in selected_etfs:
        col = ETF_DISPLAY[label]
        if col not in df.columns:
            continue
        series = df[col].dropna()
        cum = (1 + series).cumprod() - 1

        fig.add_trace(go.Scatter(
            x=cum.index,
            y=cum.values * 100,
            mode="lines",
            name=label,
            line=dict(color=ETF_COLORS[label], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>" + label + "</extra>",
        ))

    add_phase_shapes(fig)

    fig.update_layout(
        height=520,
        xaxis_title="날짜",
        yaxis_title="누적 수익률 (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=40, b=50),
        xaxis=dict(range=["2022-01-01", "2026-07-01"]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 간단한 통계 테이블
    st.markdown("#### 기간별 누적 수익률 요약")
    summary_rows = []
    for label in selected_etfs:
        col = ETF_DISPLAY[label]
        if col not in df.columns:
            continue
        series = df[col].dropna()
        cum_all = (1 + series).cumprod().iloc[-1] - 1
        row = {"ETF": label, "전체 기간": f"{cum_all*100:.1f}%"}
        for phase, (s, e) in PHASE_DATES.items():
            mask = (series.index >= s) & (series.index <= e)
            sub = series[mask]
            if len(sub) > 0:
                cum_p = (1 + sub).cumprod().iloc[-1] - 1
                row[phase] = f"{cum_p*100:.1f}%"
            else:
                row[phase] = "—"
        summary_rows.append(row)
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows).set_index("ETF"), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 탭 2: OLS 계수 비교
# ─────────────────────────────────────────────────────────────────────────────
def tab_ols(selected_etfs):
    st.subheader("📊 OLS 계수 비교 (전체 기간)")
    st.caption("★ = p < 0.05 유의수준. AI관심도(AI_interest)는 레벨값(0~100) 사용 — 타 변수와 단위 다름.")

    var_choice = st.selectbox(
        "독립변수 선택",
        options=list(INDEP_LABELS.keys()),
        format_func=lambda x: INDEP_LABELS[x],
    )

    # 막대그래프
    betas, pvals, labels, sig_marks = [], [], [], []
    for etf in selected_etfs:
        if etf not in OLS_RESULTS:
            continue
        beta = OLS_RESULTS[etf][var_choice]["beta"]
        pval = OLS_RESULTS[etf][var_choice]["pval"]
        betas.append(beta)
        pvals.append(pval)
        labels.append(etf)
        sig_marks.append("★" if pval < 0.05 else "")

    colors = [ETF_COLORS[e] for e in labels]
    text_labels = [f"{b:.3f}{m}" for b, m in zip(betas, sig_marks)]

    fig = go.Figure(go.Bar(
        x=labels,
        y=betas,
        marker_color=colors,
        text=text_labels,
        textposition="outside",
        hovertemplate="%{x}<br>β = %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        height=420,
        yaxis_title=f"β ({INDEP_LABELS[var_choice]})",
        xaxis_title="ETF",
        margin=dict(l=50, r=20, t=40, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 전체 계수 테이블
    st.markdown("#### 전체 OLS 계수표 (★ p < 0.05)")
    rows = []
    for etf in selected_etfs:
        if etf not in OLS_RESULTS:
            continue
        row = {"ETF": etf}
        for var in INDEP_VARS:
            b = OLS_RESULTS[etf][var]["beta"]
            p = OLS_RESULTS[etf][var]["pval"]
            star = "★" if p < 0.05 else ""
            row[INDEP_LABELS[var]] = f"{b:.3f}{star}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("ETF"), use_container_width=True)

    with st.expander("ℹ️ 해석 주의사항"):
        st.markdown(
            "- **AI관심도(AI_interest)**: 레벨값(0~100) 그대로 사용. "
            "pct_change 미적용이므로 β 크기를 다른 변수와 직접 비교 불가.\n"
            "- **SOXX/QQQ**: 원화 환산 공식 `(1+r_USD)×(1+r_FX)−1` 적용. "
            "USDKRW_ret β ≈ 1은 회계적 환율 효과를 반영.\n"
            "- ★ 없음 = 귀무가설 기각 못함 (p ≥ 0.05)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 탭 3: Rolling Beta
# ─────────────────────────────────────────────────────────────────────────────
def tab_rolling(df, selected_etfs):
    st.subheader("📉 Rolling Beta (다변량, 26주 창, USDKRW_ret 계수)")
    st.caption(
        "6변수(USD/KRW, S&P500, KOSPI, VIX, WTI, AI관심도) 동시 통제 후 "
        "주별 USD/KRW β 추출. 음수 = 원화 절하 시 ETF 수익률 하락."
    )

    with st.spinner("Rolling Beta 계산 중... (약 10~20초)"):
        rb = compute_rolling_betas(df)

    fig = go.Figure()
    for label in selected_etfs:
        if label not in rb.columns:
            continue
        series = rb[label].dropna()
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=label,
            line=dict(color=ETF_COLORS[label], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>β = %{y:.3f}<extra>" + label + "</extra>",
        ))

    add_phase_shapes(fig)
    fig.add_hline(y=0, line_dash="dot", line_color="black", line_width=1)

    fig.update_layout(
        height=520,
        xaxis_title="날짜",
        yaxis_title="β (USD/KRW)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=40, b=50),
        xaxis=dict(range=["2022-01-01", "2026-07-01"]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 국면별 평균 β 요약
    st.markdown("#### 국면별 평균 Rolling β (USD/KRW)")
    phase_rows = []
    for label in selected_etfs:
        if label not in rb.columns:
            continue
        row = {"ETF": label}
        for phase, (s, e) in PHASE_DATES.items():
            mask = (rb.index >= s) & (rb.index <= e)
            sub = rb[label][mask].dropna()
            row[phase] = f"{sub.mean():.3f}" if len(sub) > 0 else "—"
        phase_rows.append(row)
    if phase_rows:
        st.dataframe(pd.DataFrame(phase_rows).set_index("ETF"), use_container_width=True)

    with st.expander("🔍 KODEX β = −3 강건성 확인 (Bootstrap + LOO)"):
        st.markdown(
            "KODEX 반도체의 현재 국면(2026년~) Rolling β ≈ −3 에 대한 강건성 검증:\n\n"
            "- **Bootstrap 95% CI**: `[−3.99, 1.05]` → 0을 포함하므로 "
            "95% 신뢰수준에서 음수 방향을 단정할 수 없음. "
            "단, 분포가 음수 쪽에 강하게 치우쳐 있어 방향성은 뚜렷함.\n"
            "- **LOO (Leave-one-week-out)**: 범위 `[−2.41, −0.06]` → "
            "어느 한 주를 제거해도 전부 음수 유지. 단일 이상치에 의한 결과가 아님을 확인.\n"
            "- **해석**: 6변수 동시 통제 후에도 원화 절하 시 KODEX 수익률 하락 방향은 "
            "안정적. 반도체 수출 기업 특성상 달러 강세(글로벌 경기 위축)에 따른 "
            "수요 감소 우려가 반영된 것으로 해석 가능. "
            "다만 점 추정치의 극단성은 해석 시 유의 필요."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 탭 4: 국면별 Boxplot
# ─────────────────────────────────────────────────────────────────────────────
def tab_boxplot(df, selected_etfs, selected_phases):
    st.subheader("📦 국면별 ETF 수익률 분포")

    phase_mask = df["phase"].isin(selected_phases)
    df_filtered = df[phase_mask].copy()

    if df_filtered.empty:
        st.warning("선택된 국면에 데이터가 없습니다.")
        return

    # long-form 변환
    records = []
    for label in selected_etfs:
        col = ETF_DISPLAY[label]
        if col not in df_filtered.columns:
            continue
        for idx, row in df_filtered.iterrows():
            if pd.notna(row[col]):
                records.append({
                    "날짜": idx,
                    "ETF": label,
                    "수익률": row[col] * 100,
                    "국면": row["phase"],
                })

    if not records:
        st.warning("선택된 ETF/국면 조합에 데이터가 없습니다.")
        return

    df_long = pd.DataFrame(records)

    fig = px.box(
        df_long,
        x="국면",
        y="수익률",
        color="ETF",
        color_discrete_map=ETF_COLORS,
        category_orders={"국면": selected_phases},
        points="outliers",
        labels={"수익률": "주간 수익률 (%)"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=40, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 국면별 평균·표준편차 요약
    st.markdown("#### 국면별 평균 수익률 및 표준편차 (%)")
    summary = (
        df_long.groupby(["ETF", "국면"])["수익률"]
        .agg(mean="mean", std="std")
        .round(3)
        .reset_index()
    )
    pivot = summary.pivot(index="ETF", columns="국면", values=["mean", "std"])
    pivot.columns = [f"{stat}_{phase}" for stat, phase in pivot.columns]
    st.dataframe(pivot, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 탭 5: 원화 환산 수익률 계산기
# ─────────────────────────────────────────────────────────────────────────────
def tab_calculator():
    st.subheader("🧮 원화 환산 수익률 계산기")
    st.caption(
        "미국 ETF를 원화로 환산할 때 적용되는 공식: "
        "`r_KRW = (1 + r_USD) × (1 + r_FX) − 1`"
    )

    col1, col2 = st.columns(2)
    with col1:
        r_usd_pct = st.slider(
            "달러 기준 수익률 (%)",
            min_value=-20.0,
            max_value=50.0,
            value=10.0,
            step=0.5,
            format="%.1f%%",
        )
    with col2:
        r_fx_pct = st.slider(
            "환율 변화율 (USD/KRW, %)",
            min_value=-10.0,
            max_value=20.0,
            value=0.0,
            step=0.5,
            format="%.1f%%",
            help="양수 = 원화 절하 (달러 강세), 음수 = 원화 절상",
        )

    r_usd = r_usd_pct / 100
    r_fx  = r_fx_pct  / 100
    r_krw = (1 + r_usd) * (1 + r_fx) - 1

    # 상호작용 효과
    interaction = r_usd * r_fx

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(
            label="🇺🇸 달러 수익률",
            value=f"{r_usd_pct:.2f}%",
        )
    with col_b:
        st.metric(
            label="💱 환율 변화율",
            value=f"{r_fx_pct:.2f}%",
            delta="원화 절하↑" if r_fx > 0 else ("원화 절상↓" if r_fx < 0 else "변동 없음"),
            delta_color="inverse" if r_fx < 0 else "normal",
        )
    with col_c:
        st.metric(
            label="🇰🇷 원화 환산 수익률",
            value=f"{r_krw*100:.2f}%",
            delta=f"상호작용 효과: {interaction*100:.3f}%",
        )

    # 단순 합산 vs 복리 공식 비교
    simple = r_usd + r_fx
    st.markdown("---")
    st.markdown("#### 단순 합산 vs 복리 공식 비교")
    comp_df = pd.DataFrame({
        "방법": ["단순 합산 (r_USD + r_FX)", "복리 공식 (올바른 방법)"],
        "결과": [f"{simple*100:.4f}%", f"{r_krw*100:.4f}%"],
        "차이": ["—", f"{(r_krw - simple)*100:.4f}%"],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.caption(
        "차이 = 상호작용 항 `r_USD × r_FX`. "
        "수익률이 클수록, 환율 변동이 클수록 단순 합산 오차가 커집니다."
    )

    # 시나리오 그리드
    st.markdown("---")
    st.markdown("#### 시나리오 분석: 환율 변화율 × 달러 수익률")
    r_usd_range = np.arange(-0.20, 0.51, 0.05)
    r_fx_range  = np.arange(-0.10, 0.21, 0.05)
    grid = pd.DataFrame(
        index=[f"{x*100:.0f}%" for x in r_usd_range],
        columns=[f"{x*100:.0f}%" for x in r_fx_range],
        data=[
            [(1 + ru) * (1 + rf) - 1 for rf in r_fx_range]
            for ru in r_usd_range
        ],
    ).applymap(lambda v: f"{v*100:.1f}%")
    grid.index.name = "r_USD →\nr_FX ↓"
    st.dataframe(grid, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("📊 한·미 기술주 ETF 수익률 분석 대시보드")
    st.markdown(
        "**분석 주제:** 환율·시장지수·변동성이 한국·미국 기술주 ETF 수익률에 미치는 영향 비교  \n"
        "**기간:** 2022-01-14 ~ 2026-05-22 (주간)  \n"
        "**ETF:** KODEX 반도체 · TIGER 200IT · SOXX (원화) · QQQ (원화)  \n"
        "**국면 구분:** FOMC 기준 4개 핵심 국면만 음영 처리 (초반 기타 구간 제외)"
    )
    st.markdown("---")

    # 데이터 로드
    try:
        df = load_data()
    except FileNotFoundError:
        st.error(
            "❌ `analysis_ready.csv` 파일을 찾을 수 없습니다.  \n"
            "앱과 같은 디렉터리에 파일을 배치해주세요."
        )
        st.stop()

    # 사이드바
    selected_etfs, selected_phases = render_sidebar()

    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 누적 수익률",
        "📊 OLS 계수 비교",
        "📉 Rolling Beta",
        "📦 국면별 Boxplot",
        "🧮 환산 계산기",
    ])

    with tab1:
        tab_cumulative(df, selected_etfs)
    with tab2:
        tab_ols(selected_etfs)
    with tab3:
        tab_rolling(df, selected_etfs)
    with tab4:
        tab_boxplot(df, selected_etfs, selected_phases)
    with tab5:
        tab_calculator()


if __name__ == "__main__":
    main()
