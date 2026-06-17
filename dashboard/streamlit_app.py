"""
streamlit_app.py
한·미 기술주 ETF 수익률 분석 대시보드 (기말 최종)
작성자: 안명현 (2025720536)
마감: 2026-06-11
탭 구성: 연구개요 / Q1 Rolling Beta / Q2 SOXX β / Q3 AI Mediator / Q4 Bootstrap+Chow
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

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
    "긴축기":    "rgba(255,193,7,0.18)",
    "AI랠리기":  "rgba(40,167,69,0.15)",
    "불확실성기": "rgba(220,53,69,0.15)",
    "현재":      "rgba(23,162,184,0.15)",
}
PHASE_DATES = {
    "긴축기":    ("2022-03-17", "2023-07-26"),
    "AI랠리기":  ("2023-07-27", "2024-09-17"),
    "불확실성기": ("2024-09-18", "2025-12-31"),
    "현재":      ("2026-01-01", "2026-12-31"),
}
ETF_COLS = {
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
INDEP = ["USDKRW_ret", "SP500_ret", "KOSPI_ret", "VIX_ret", "WTI_ret", "AI_interest"]
INDEP_LABELS = {
    "USDKRW_ret":  "USD/KRW",
    "SP500_ret":   "S&P500",
    "KOSPI_ret":   "KOSPI",
    "VIX_ret":     "VIX",
    "WTI_ret":     "WTI",
    "AI_interest": "AI관심도",
}
WINDOW = 26

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("analysis_ready.csv", index_col="date", parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 계산 함수 (모두 최상위 — 중첩 캐시 없음)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def run_full_ols(_df):
    """전체 기간 OLS — 4 ETF"""
    df_c = _df.dropna(subset=INDEP)
    results = {}
    for label, col in ETF_COLS.items():
        if col not in df_c.columns:
            continue
        X = sm.add_constant(df_c[INDEP])
        results[label] = sm.OLS(df_c[col], X).fit()
    return results

@st.cache_data
def run_diagnostics(_df):
    """잔차진단 — DW + Breusch-Pagan (실시간 계산)"""
    df_c = _df.dropna(subset=INDEP)
    rows = []
    for label, col in ETF_COLS.items():
        if col not in df_c.columns:
            continue
        X = sm.add_constant(df_c[INDEP])
        res = sm.OLS(df_c[col], X).fit()
        dw = durbin_watson(res.resid)
        _, bp_p, _, _ = het_breuschpagan(res.resid, res.model.exog)
        rows.append({
            "ETF": label,
            "Durbin-Watson": round(dw, 4),
            "자기상관": "✅ 정상" if 1.5 <= dw <= 2.5 else "⚠️ 존재",
            "BP p-value": round(bp_p, 4),
            "이분산성": "⚠️ 존재" if bp_p < 0.05 else "✅ 정상",
        })
    return pd.DataFrame(rows)

@st.cache_data
def compute_rolling_betas(_df):
    """다변량 Rolling Beta (26주 창, USDKRW_ret 계수만 추출)"""
    df_c = _df.dropna(subset=INDEP)
    X_all = sm.add_constant(df_c[INDEP])
    results = {}
    for label, col in ETF_COLS.items():
        if col not in df_c.columns:
            continue
        y_all = df_c[col]
        betas = [np.nan] * len(df_c)
        for i in range(WINDOW, len(df_c) + 1):
            Xw = X_all.iloc[i - WINDOW:i]
            yw = y_all.iloc[i - WINDOW:i]
            if yw.isna().any():
                continue
            try:
                res = sm.OLS(yw, Xw).fit()
                betas[i - 1] = res.params.get("USDKRW_ret", np.nan)
            except Exception:
                pass
        results[label] = pd.Series(betas, index=df_c.index)
    return pd.DataFrame(results)

@st.cache_data
def run_phase_ols(_df):
    """국면별 OLS — USDKRW_ret β 추출"""
    out = {}
    for label, col in ETF_COLS.items():
        out[label] = {}
        for phase, (s, e) in PHASE_DATES.items():
            mask = (_df.index >= s) & (_df.index <= e)
            sub = _df[mask].dropna(subset=INDEP + [col])
            if len(sub) < 10:
                out[label][phase] = {"n": len(sub), "beta": np.nan, "pval": np.nan}
                continue
            X = sm.add_constant(sub[INDEP])
            res = sm.OLS(sub[col], X).fit()
            out[label][phase] = {
                "n": len(sub),
                "beta": res.params.get("USDKRW_ret", np.nan),
                "pval": res.pvalues.get("USDKRW_ret", np.nan),
            }
    return out

@st.cache_data
def run_chow_all(_df):
    """전체 ETF × 국면쌍 Chow Test"""
    phase_pairs = [("긴축기", "AI랠리기"), ("AI랠리기", "불확실성기"), ("불확실성기", "현재")]
    rows = []
    for label, col in ETF_COLS.items():
        for p1, p2 in phase_pairs:
            s1, e1 = PHASE_DATES[p1]
            s2, e2 = PHASE_DATES[p2]
            m1 = _df[(_df.index >= s1) & (_df.index <= e1)].dropna(subset=INDEP + [col])
            m2 = _df[(_df.index >= s2) & (_df.index <= e2)].dropna(subset=INDEP + [col])
            if len(m1) < 10 or len(m2) < 10:
                rows.append({"ETF": label, "국면 쌍": f"{p1} → {p2}",
                             "F통계량": "—", "p값": "—", "판정": "표본 부족"})
                continue
            combined = pd.concat([m1, m2])
            k = len(INDEP) + 1
            n = len(combined)
            def sse(sub):
                return sm.OLS(sub[col], sm.add_constant(sub[INDEP])).fit().ssr
            F = ((sse(combined) - sse(m1) - sse(m2)) / k) / ((sse(m1) + sse(m2)) / (n - 2 * k))
            p = 1 - stats.f.cdf(F, k, n - 2 * k)
            rows.append({
                "ETF": label,
                "국면 쌍": f"{p1} → {p2}",
                "F통계량": f"{F:.3f}",
                "p값": f"{p:.4f}",
                "판정": "★ 구조변화" if p < 0.05 else "유지",
            })
    return pd.DataFrame(rows)

@st.cache_data
def run_bootstrap(_idx_tuple, n_iter, _df):
    """Bootstrap CI — KODEX 현재 국면"""
    s, e = PHASE_DATES["현재"]
    df_b = _df[(_df.index >= s) & (_df.index <= e)].dropna(subset=INDEP + ["KODEX_반도체_ret"])
    np.random.seed(42)
    betas = []
    for _ in range(n_iter):
        idx = np.random.choice(len(df_b), size=len(df_b), replace=True)
        sub = df_b.iloc[idx]
        try:
            res = sm.OLS(sub["KODEX_반도체_ret"], sm.add_constant(sub[INDEP])).fit()
            betas.append(res.params.get("USDKRW_ret", np.nan))
        except Exception:
            pass
    return np.array([b for b in betas if not np.isnan(b)])

@st.cache_data
def run_loo(_idx_tuple, _df):
    """Leave-one-week-out — KODEX 현재 국면"""
    s, e = PHASE_DATES["현재"]
    df_b = _df[(_df.index >= s) & (_df.index <= e)].dropna(subset=INDEP + ["KODEX_반도체_ret"])
    loo_betas, loo_dates = [], []
    for i in range(len(df_b)):
        sub = df_b.drop(df_b.index[i])
        try:
            res = sm.OLS(sub["KODEX_반도체_ret"], sm.add_constant(sub[INDEP])).fit()
            loo_betas.append(res.params.get("USDKRW_ret", np.nan))
            loo_dates.append(df_b.index[i])
        except Exception:
            pass
    return loo_dates, loo_betas

# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def add_phase_vrect(fig):
    for phase, (s, e) in PHASE_DATES.items():
        fig.add_vrect(
            x0=s, x1=e,
            fillcolor=PHASE_COLORS[phase],
            opacity=1, layer="below", line_width=0,
            annotation_text=phase, annotation_position="top left",
            annotation_font_size=10,
        )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(df):
    st.sidebar.title("🎛️ 필터 설정")
    st.sidebar.markdown("---")
    selected_etfs = st.sidebar.multiselect(
        "ETF 선택", options=list(ETF_COLS.keys()), default=list(ETF_COLS.keys())
    )
    if not selected_etfs:
        selected_etfs = list(ETF_COLS.keys())
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"📌 분석 기간: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')} ({len(df)}주)\n\n"
        "작성자: 안명현 (2025720536)\n\n데이터마이닝 기말 프로젝트"
    )
    return selected_etfs

# ─────────────────────────────────────────────────────────────────────────────
# 탭 1 — 연구 개요
# ─────────────────────────────────────────────────────────────────────────────
def tab_overview(df, selected_etfs):
    st.subheader("🔍 연구 개요")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ETF", "4개", "KODEX·TIGER·SOXX·QQQ")
    c2.metric("설명변수", "6개", "환율·지수·VIX·WTI·AI관심도")
    c3.metric("표본", f"{len(df)}주",
              f"{df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")
    c4.metric("분석 방법", "OLS + Rolling", "잔차진단·Bootstrap·Chow")

    st.markdown("---")

    # 가설표 — β/p는 run_full_ols에서 자동 추출
    st.markdown("#### 핵심 가설 및 중간 결과")
    ols_res = run_full_ols(df)

    def fmt(label, var):
        if label not in ols_res:
            return "—"
        b = ols_res[label].params.get(var, np.nan)
        p = ols_res[label].pvalues.get(var, np.nan)
        star = "★" if p < 0.05 else ""
        return f"β={b:.3f}{star} (p={p:.3f})"

    hyp_df = pd.DataFrame([
        {
            "가설": "H1-1 환율",
            "예측": "한국 ETF가 더 큰 환율 민감도",
            "판정": "❌ 기각",
            "KODEX": fmt("KODEX 반도체", "USDKRW_ret"),
            "TIGER": fmt("TIGER 200IT",  "USDKRW_ret"),
            "비고": "SOXX β≈+1은 회계효과(Q2 참고) — 실질 민감도 아님",
        },
        {
            "가설": "H1-2 유가",
            "예측": "유가가 ETF 수익에 영향",
            "판정": "⚠️ 부분 지지",
            "KODEX": fmt("KODEX 반도체", "WTI_ret"),
            "TIGER": fmt("TIGER 200IT",  "WTI_ret"),
            "비고": "KODEX만 유의. 나머지 비유의",
        },
        {
            "가설": "H1-3 지수",
            "예측": "각국 시장지수가 지배적",
            "판정": "✅ 지지",
            "KODEX": fmt("KODEX 반도체", "KOSPI_ret"),
            "TIGER": fmt("TIGER 200IT",  "KOSPI_ret"),
            "비고": "KOSPI·S&P500 압도적 설명력",
        },
        {
            "가설": "H1-4 AI관심도",
            "예측": "AI관심도가 직접 영향",
            "판정": "❌ 기각",
            "KODEX": fmt("KODEX 반도체", "AI_interest"),
            "TIGER": fmt("TIGER 200IT",  "AI_interest"),
            "비고": "전 ETF 비유의. 간접경로 가능성(Q3 참고)",
        },
    ])
    st.dataframe(hyp_df.set_index("가설"), use_container_width=True)

    # 잔차진단 — 실시간 계산
    st.markdown("#### 잔차 진단 결과 (실시간 계산)")
    diag_df = run_diagnostics(df)
    st.dataframe(diag_df.set_index("ETF"), use_container_width=True)
    st.caption("기준: DW 1.5~2.5 = 자기상관 없음 / BP p≥0.05 = 이분산성 없음. KODEX만 이분산성 존재 → p-value 해석 시 유의.")

    st.markdown("---")

    # 누적 수익률
    st.markdown("#### ETF 누적 수익률 × 4대 시장 국면")
    fig = go.Figure()
    for label in selected_etfs:
        col = ETF_COLS[label]
        if col not in df.columns:
            continue
        series = df[col].dropna()
        cum = (1 + series).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values * 100,
            mode="lines", name=label,
            line=dict(color=ETF_COLORS[label], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>" + label + "</extra>",
        ))
    add_phase_vrect(fig)
    fig.update_layout(
        height=460, xaxis_title="날짜", yaxis_title="누적 수익률 (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=40, b=50),
        xaxis=dict(range=["2022-01-01", "2026-07-01"]),
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 탭 2 — Q1. Rolling Beta
# ─────────────────────────────────────────────────────────────────────────────
def tab_rolling(df, selected_etfs):
    st.subheader("📉 Q1. Rolling Beta — 단변량 → 다변량 전환")
    st.info(
        "**교수님 Q1:** 단변량 Rolling Beta가 아니라 6변수 동시 통제 후 환율 β만 추출하면 결과가 달라지는가?\n\n"
        "→ 26주 창, 6변수 OLS에서 USDKRW_ret 계수만 추출. '순수 환율 민감도'에 더 가까운 추정."
    )

    with st.spinner("다변량 Rolling Beta 계산 중... (약 10~20초)"):
        rb = compute_rolling_betas(df)

    fig = go.Figure()
    for label in selected_etfs:
        if label not in rb.columns:
            continue
        s = rb[label].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=label,
            line=dict(color=ETF_COLORS[label], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>β = %{y:.3f}<extra>" + label + "</extra>",
        ))
    add_phase_vrect(fig)
    fig.add_hline(y=0, line_dash="dot", line_color="black", line_width=1)
    fig.update_layout(
        height=500, xaxis_title="날짜", yaxis_title="β (USD/KRW)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=40, b=50),
        xaxis=dict(range=["2022-01-01", "2026-07-01"]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 단변량 vs 다변량 비교표 — 실시간 계산
    st.markdown("#### 단변량 vs 다변량 환율 β — 서로 다른 추정 방식의 참고 비교")
    df_c = df.dropna(subset=INDEP)
    rows = []
    for label, col in ETF_COLS.items():
        if label not in selected_etfs or col not in df_c.columns:
            continue
        y = df_c[col]
        b_uni = sm.OLS(y, sm.add_constant(df_c[["USDKRW_ret"]])).fit().params.get("USDKRW_ret", np.nan)
        b_multi = rb[label].mean() if label in rb.columns else np.nan
        rows.append({
            "ETF": label,
            "단변량 β": round(b_uni, 3),
            "다변량 β (Rolling평균)": round(b_multi, 3),
            "차이": round(b_multi - b_uni, 3),
        })
    st.dataframe(pd.DataFrame(rows).set_index("ETF"), use_container_width=True)
    st.caption("다변량 β: 26주 Rolling 평균. KODEX는 현재 국면에서 음(−) 심화 — Q4에서 강건성 검증.")

# ─────────────────────────────────────────────────────────────────────────────
# 탭 3 — Q2. SOXX β 분리
# ─────────────────────────────────────────────────────────────────────────────
def tab_soxx(df):
    st.subheader("💱 Q2. SOXX β — 원화 기준 vs 달러 기준 분리")
    st.info(
        "**교수님 Q2:** SOXX의 큰 환율 β가 ETF 고유 민감도인가, 원화 환산 회계 구조 때문인가?\n\n"
        "→ SOXX를 달러 기준 수익률로도 OLS 적용. 차이 ≈ 1이면 회계효과."
    )

    df_c = df.dropna(subset=INDEP)
    X = sm.add_constant(df_c[INDEP])

    # 원화 기준
    res_krw = sm.OLS(df_c["SOXX_KRW_ret"], X).fit()
    b_krw = res_krw.params["USDKRW_ret"]
    p_krw = res_krw.pvalues["USDKRW_ret"]

    # 달러 기준 역산
    r_usd = (1 + df_c["SOXX_KRW_ret"]) / (1 + df_c["USDKRW_ret"]) - 1
    res_usd = sm.OLS(r_usd, X).fit()
    b_usd = res_usd.params["USDKRW_ret"]
    p_usd = res_usd.pvalues["USDKRW_ret"]
    diff = b_krw - b_usd

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["SOXX β (원화 기준)", "SOXX β (달러 기준)"],
        y=[b_krw, b_usd],
        marker_color=["#e05c23", "#b0bec5"],
        text=[f"{b_krw:.4f}{'★' if p_krw < 0.05 else ''}", f"{b_usd:.4f}{'★' if p_usd < 0.05 else ''}"],
        textposition="outside",
        width=0.4,
    ))
    fig.add_annotation(
        x=0.5, y=max(b_krw, b_usd) * 0.55,
        text=f"차이: {diff:.4f} ≈ 1<br>(회계 효과)",
        showarrow=True, arrowhead=2, ax=80, ay=-40,
        font=dict(size=13, color="#e05c23"),
        bgcolor="white", bordercolor="#e05c23",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        height=420, yaxis_title="β (USDKRW_ret)",
        margin=dict(l=50, r=20, t=40, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("β (원화 기준)", f"{b_krw:.4f}", f"p={p_krw:.4f} {'★유의' if p_krw < 0.05 else '비유의'}")
    col2.metric("β (달러 기준)", f"{b_usd:.4f}", f"p={p_usd:.4f} {'★유의' if p_usd < 0.05 else '비유의'}")
    col3.metric("차이 (원화−달러)", f"{diff:.4f}", "≈ 1 → 회계효과" if abs(diff - 1) < 0.15 else "")

    st.markdown("---")
    st.markdown(
        f"**결론:** SOXX 원화 기준 환율 β={b_krw:.3f}는 미국 반도체 ETF가 환율에 민감해서가 아니라, "
        f"달러 수익률을 원화로 환산하는 회계 구조에서 대부분 설명됨 (차이≈{diff:.3f})."
    )
    st.caption("★ = p < 0.05. 달러 기준: r_USD = (1+r_KRW)/(1+r_FX) − 1 역산 적용.")

# ─────────────────────────────────────────────────────────────────────────────
# 탭 4 — Q3. AI Mediator
# ─────────────────────────────────────────────────────────────────────────────
def tab_mediator(df):
    st.subheader("🤖 Q3. AI관심도 — Mediator 효과 분해")
    st.info(
        "**교수님 Q3:** AI관심도 계수가 약한데, 시장지수를 경유한 간접 채널 가능성은 있는가?\n\n"
        "→ 시장지수 포함(A) vs 제외(B) 모델 β 비교. β_제외 > β_포함이면 시장지수가 매개."
    )

    df_c = df.dropna(subset=INDEP)
    FORMULA_FULL  = "{dep} ~ USDKRW_ret + SP500_ret + KOSPI_ret + VIX_ret + WTI_ret + AI_interest"
    FORMULA_NOMKT = "{dep} ~ USDKRW_ret + VIX_ret + WTI_ret + AI_interest"

    rows = []
    for label, col in ETF_COLS.items():
        if col not in df_c.columns:
            continue
        r_full  = smf.ols(FORMULA_FULL.format(dep=col),  data=df_c).fit()
        r_nomkt = smf.ols(FORMULA_NOMKT.format(dep=col), data=df_c).fit()
        b_full = r_full.params["AI_interest"];  p_full = r_full.pvalues["AI_interest"]
        b_no   = r_nomkt.params["AI_interest"]; p_no   = r_nomkt.pvalues["AI_interest"]
        rows.append({
            "ETF": label,
            "β_포함(A)": round(b_full, 6),
            "p_포함":    round(p_full, 4),
            "β_제외(B)": round(b_no,   6),
            "p_제외":    round(p_no,   4),
            "변화(B−A)": round(b_no - b_full, 6),
            "매개여부":  "↑ 회복 (매개 가능)" if b_no > b_full else "변화 없음",
        })

    result_df = pd.DataFrame(rows).set_index("ETF")
    st.dataframe(result_df, use_container_width=True)
    st.caption(
        "β_포함(A): 시장지수(SP500·KOSPI) 포함. β_제외(B): 시장지수 제외.\n"
        "β_제외 > β_포함 → 시장지수가 AI_interest 효과를 흡수(매개). ★ = p < 0.05."
    )

    # β 비교 차트
    etf_labels = [r["ETF"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="β_포함 (시장지수 통제)", x=etf_labels,
                         y=[r["β_포함(A)"] for r in rows], marker_color="#5c9bd6"))
    fig.add_trace(go.Bar(name="β_제외 (시장지수 없음)", x=etf_labels,
                         y=[r["β_제외(B)"] for r in rows], marker_color="#e07b54"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        barmode="group", height=350, yaxis_title="AI_interest 계수 (β)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=40, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Lag 분석
    st.markdown("#### 보조: AI_interest Lag 1~2주 — 선행지표 가능성")
    df_lag = df_c.copy()
    df_lag["AI_lag1"] = df_lag["AI_interest"].shift(1)
    df_lag["AI_lag2"] = df_lag["AI_interest"].shift(2)
    df_lag = df_lag.dropna()

    lag_rows = []
    for label, col in ETF_COLS.items():
        if col not in df_lag.columns:
            continue
        INDEP_L1 = ["USDKRW_ret","SP500_ret","KOSPI_ret","VIX_ret","WTI_ret","AI_lag1"]
        INDEP_L2 = ["USDKRW_ret","SP500_ret","KOSPI_ret","VIX_ret","WTI_ret","AI_lag2"]
        r1 = sm.OLS(df_lag[col], sm.add_constant(df_lag[INDEP_L1])).fit()
        r2 = sm.OLS(df_lag[col], sm.add_constant(df_lag[INDEP_L2])).fit()
        lag_rows.append({
            "ETF":    label,
            "β_lag1": round(r1.params["AI_lag1"], 6),
            "p_lag1": round(r1.pvalues["AI_lag1"], 4),
            "β_lag2": round(r2.params["AI_lag2"], 6),
            "p_lag2": round(r2.pvalues["AI_lag2"], 4),
            "판정":   "기각 유지" if r1.pvalues["AI_lag1"] >= 0.05 and r2.pvalues["AI_lag2"] >= 0.05 else "★ 선행 가능성",
        })
    st.dataframe(pd.DataFrame(lag_rows).set_index("ETF"), use_container_width=True)
    st.caption("전 ETF lag1·lag2 모두 p≥0.05 → AI관심도는 선행지표로서도 기각. 직접·선행효과 없음, 간접경로만 가능성.")

    st.markdown("---")
    st.markdown(
        "**결론:** AI 트렌드 심리는 ETF 수익률에 직접 작용하지 않고, "
        "시장지수(KOSPI·S&P500)를 경유한 간접경로 가능성을 시사함. "
        "정식 매개효과 검정은 추가 분석이 필요함. "
        "단, β 절댓값이 0.0001 수준으로 매우 작아 실질적 경제 영향은 제한적."
    )

# ─────────────────────────────────────────────────────────────────────────────
# 탭 5 — Q4. Bootstrap + Chow
# ─────────────────────────────────────────────────────────────────────────────
def tab_bootstrap_chow(df):
    st.subheader("🔬 Q4. 강건성 검증 — Bootstrap + LOO + Chow Test")
    st.info(
        "**교수님 Q4:** 현재 국면 표본이 작고 KODEX β 극단값이 우연인지 검증하라.\n\n"
        "→ Bootstrap CI, LOO, 국면별 OLS, Chow Test로 구조적 안정성 확인."
    )

    s, e = PHASE_DATES["현재"]
    df_now = df[(df.index >= s) & (df.index <= e)].dropna(subset=INDEP + ["KODEX_반도체_ret"])
    n_now = len(df_now)
    idx_tuple = tuple(df_now.index)

    st.markdown(f"**현재 국면 표본: n={n_now}주** ({s} ~ {df_now.index.max().strftime('%Y-%m-%d')})")

    # ── Bootstrap ─────────────────────────────────────────────────
    st.markdown("#### Q4-1. Bootstrap CI (KODEX 현재 국면)")
    n_iter = st.slider("Bootstrap 반복 횟수", min_value=100, max_value=2000, value=1000, step=100)

    with st.spinner(f"Bootstrap {n_iter}회 계산 중..."):
        boot_betas = run_bootstrap(idx_tuple, n_iter, df)

    ci_low  = np.percentile(boot_betas, 2.5)
    ci_high = np.percentile(boot_betas, 97.5)
    b_mean  = boot_betas.mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Bootstrap 평균 β", f"{b_mean:.4f}")
    col2.metric("95% CI 하한", f"{ci_low:.4f}")
    col3.metric("95% CI 상한", f"{ci_high:.4f}",
                "0 포함 → 단정 불가" if ci_low < 0 < ci_high else "0 미포함")

    fig_boot = go.Figure()
    fig_boot.add_trace(go.Histogram(x=boot_betas, nbinsx=50,
                                    marker_color="#5c9bd6", opacity=0.8))
    fig_boot.add_vline(x=ci_low,  line_dash="dash",  line_color="red",
                       annotation_text=f"CI 하한 {ci_low:.2f}")
    fig_boot.add_vline(x=ci_high, line_dash="dash",  line_color="red",
                       annotation_text=f"CI 상한 {ci_high:.2f}")
    fig_boot.add_vline(x=b_mean,  line_dash="solid", line_color="black",
                       annotation_text=f"평균 {b_mean:.2f}")
    fig_boot.update_layout(
        height=350, xaxis_title="β (USDKRW_ret)", yaxis_title="Count",
        margin=dict(l=50, r=20, t=40, b=50),
    )
    st.plotly_chart(fig_boot, use_container_width=True)
    st.caption(
        f"Bootstrap {n_iter}회 | 평균 β={b_mean:.4f} | 95% CI [{ci_low:.4f}, {ci_high:.4f}] | n={n_now}\n"
        "CI에 0 포함 → 95% 수준 단정 불가. 단 분포 질량 대부분이 음수 구간에 집중 → 음의 방향성이 반복 관찰되나 95% 수준 단정은 보류."
    )

    # ── LOO ──────────────────────────────────────────────────────
    st.markdown("#### Q4-1. Leave-one-week-out (LOO)")
    with st.spinner("LOO 계산 중..."):
        loo_dates, loo_betas = run_loo(idx_tuple, df)
    loo_arr = np.array(loo_betas)

    col1, col2, col3 = st.columns(3)
    col1.metric("LOO β 최솟값", f"{loo_arr.min():.4f}")
    col2.metric("LOO β 최댓값", f"{loo_arr.max():.4f}")
    col3.metric("전부 음수?",
                "✅ 예" if (loo_arr < 0).all() else "❌ 아니오",
                "이상치 의존 없음" if (loo_arr < 0).all() else "특정 주 의존 가능성")

    fig_loo = go.Figure()
    fig_loo.add_trace(go.Scatter(
        x=loo_dates, y=loo_betas,
        mode="lines+markers", line=dict(color="#5c9bd6", width=2),
        marker=dict(size=6),
        hovertemplate="%{x|%Y-%m-%d}<br>β = %{y:.4f}<extra></extra>",
    ))
    fig_loo.add_hline(y=0, line_dash="dash", line_color="red", line_width=1)
    fig_loo.update_layout(
        height=300, xaxis_title="제거된 주", yaxis_title="β (USDKRW_ret)",
        margin=dict(l=50, r=20, t=30, b=50),
    )
    st.plotly_chart(fig_loo, use_container_width=True)
    st.caption(
        f"LOO 범위 [{loo_arr.min():.4f}, {loo_arr.max():.4f}] (n-1={n_now-1}주 기반) — "
        "어느 한 주를 제거해도 전부 음수. 특정 이상치 1개에 의존한 결과 아님."
    )

    st.markdown(
        f"**Bootstrap + LOO 종합:** β의 부호(음수)는 단순 재표집과 LOO 기준에서 비교적 일관적. "
        f"단 크기는 n={n_now} 소표본 한계로 유동적. "
        "현재 국면에서 달러 강세 시 KODEX 수익률 하락 패턴은 음의 방향성이 반복 관찰되나, 통계적 단정은 보류."
    )

    st.markdown("---")

    # ── 국면별 OLS + Chow ────────────────────────────────────────
    st.markdown("#### Q4-2. 국면별 OLS + Chow Test — KODEX 구조변화 검정")
    phase_res = run_phase_ols(df)
    phases = ["긴축기", "AI랠리기", "불확실성기", "현재"]

    # KODEX β 막대그래프 — nan 안전 처리
    kodex_data = phase_res["KODEX 반도체"]
    betas_ph = [kodex_data[p]["beta"] for p in phases]
    pvals_ph = [kodex_data[p]["pval"] for p in phases]
    ns_ph    = [kodex_data[p]["n"]    for p in phases]
    colors_ph = [
        ("#e07b54" if (not np.isnan(b) and b < 0) else "#5c9bd6")
        for b in betas_ph
    ]
    texts_ph = [
        f"β={b:.2f}{'★' if (p is not None and not np.isnan(p) and p < 0.05) else ''}\n(n={n})"
        if not np.isnan(b) else f"(n={n}, 부족)"
        for b, p, n in zip(betas_ph, pvals_ph, ns_ph)
    ]

    fig_phase = go.Figure(go.Bar(
        x=phases, y=[b if not np.isnan(b) else 0 for b in betas_ph],
        marker_color=colors_ph,
        text=texts_ph, textposition="outside",
        hovertemplate="%{x}<br>β = %{y:.3f}<extra></extra>",
    ))
    fig_phase.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig_phase.update_layout(
        height=380, yaxis_title="β (USDKRW_ret)", xaxis_title="시장 국면",
        title="KODEX 반도체 — 국면별 환율 β",
        margin=dict(l=50, r=20, t=50, b=50),
    )
    st.plotly_chart(fig_phase, use_container_width=True)

    # 전체 ETF 국면별 테이블
    st.markdown("**전체 ETF 국면별 환율 β**")
    table_rows = []
    for label in ETF_COLS.keys():
        row = {"ETF": label}
        for phase in phases:
            d = phase_res[label][phase]
            b, p, n = d["beta"], d["pval"], d["n"]
            if np.isnan(b):
                row[phase] = f"n={n} (부족)"
            else:
                star = "★" if (p is not None and not np.isnan(p) and p < 0.05) else ""
                row[phase] = f"{b:.2f}{star} (n={n})"
        table_rows.append(row)
    st.dataframe(pd.DataFrame(table_rows).set_index("ETF"), use_container_width=True)
    st.caption(f"★ = p < 0.05. 현재 국면 n={n_now}로 소표본 — F값 분산 클 수 있음.")

    # Chow Test
    st.markdown("**Chow Test — 인접 국면 간 구조변화 검정**")
    with st.spinner("Chow Test 계산 중..."):
        chow_df = run_chow_all(df)
    st.dataframe(chow_df.set_index("ETF"), use_container_width=True)
    bf_pass = [
        f"{r['ETF']} {r['국면 쌍']}"
        for _, r in chow_df.iterrows()
        if r["p값"] != "—" and float(r["p값"]) < 0.004
    ]
    st.caption(
        "⚠️ 탐색적 증거 수준 — 4 ETF × 3 국면쌍 = 12개 동시 검정, 다중비교 보정 미적용.\n"
        f"Bonferroni 기준 p < 0.004 적용 시 통과: {', '.join(bf_pass)}."
    )

    # Chow Test 캡션 아래, 구조변화 해석 위에 추가
    chow_p_kodex = chow_df[
        (chow_df["ETF"] == "KODEX 반도체") & (chow_df["국면 쌍"] == "긴축기 → AI랠리기")
    ]["p값"].iloc[0]

    # 구조변화 해석 — 실시간 β값 사용
    b_긴축 = kodex_data["긴축기"]["beta"]
    b_ai   = kodex_data["AI랠리기"]["beta"]
    b_현재 = kodex_data["현재"]["beta"]
    p_현재 = kodex_data["현재"]["pval"]
    st.markdown(
        f"**구조변화 해석:** 긴축기(β={b_긴축:.2f})에서 AI랠리기(β={b_ai:.2f})로 부호까지 전환되며 "
        f"계수가 구조적으로 달라짐 (Chow p={chow_p_kodex}). "
        f"현재 국면에서는 β={b_현재:.2f}(p={p_현재:.3f})로 재차 음전환 — "
        "단순 전체 기간 평균 β로는 포착 불가능한 국면별 이질성 확인."
    )

# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("📊 한·미 기술주 ETF 수익률 분석 대시보드")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌ `analysis_ready.csv` 파일을 찾을 수 없습니다. 앱과 같은 디렉터리에 배치해주세요.")
        st.stop()

    st.markdown(
        f"**주제:** 환율·시장지수·변동성이 한국·미국 기술주 ETF 수익률에 미치는 영향 비교 | "
        f"**기간:** {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')} ({len(df)}주) | "
        "**ETF:** KODEX반도체·TIGER200IT·SOXX·QQQ"
    )
    st.markdown("---")

    selected_etfs = render_sidebar(df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 연구 개요",
        "📉 Q1. Rolling Beta",
        "💱 Q2. SOXX β 분리",
        "🤖 Q3. AI Mediator",
        "🔬 Q4. Bootstrap + Chow",
    ])

    with tab1:
        tab_overview(df, selected_etfs)
    with tab2:
        tab_rolling(df, selected_etfs)
    with tab3:
        tab_soxx(df)
    with tab4:
        tab_mediator(df)
    with tab5:
        tab_bootstrap_chow(df)


if __name__ == "__main__":
    main()