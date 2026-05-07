"""
Page 5 — Medical Trend Monitor
Quarterly inflation tracking, per-DRG cost trajectories,
length-of-stay trends, and seasonal claim patterns (2023–2025).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    BRAND, DRG_COLOR_MAP, DRG_CATEGORIES, PLAN_TYPES, REGIONS,
    fmt_rm, inject_css, data_loader_widget,
)

inject_css()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:0.95rem;font-weight:700;color:#111827;"
        "letter-spacing:-0.01em;'>MHIT Intelligence</div>"
        "<div style='font-size:0.7rem;color:#9CA3AF;margin-top:2px;"
        "text-transform:uppercase;letter-spacing:0.07em;'>UMACT 2026 · BUZHIDAO</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Filters**")
    sel_drg = st.multiselect("DRG Category", DRG_CATEGORIES, default=DRG_CATEGORIES)
    sel_regions = st.multiselect("Region", REGIONS, default=REGIONS)

# ─── Data ────────────────────────────────────────────────────────────────────
df_full = data_loader_widget()
if df_full is None:
    st.stop()

df = df_full[
    df_full["drg_category"].isin(sel_drg)
    & df_full["region"].isin(sel_regions)
].copy()

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("<div class='page-header'>📅 Medical Trend Monitor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='page-sub'>Quarterly claim volume and severity trends (2023–2025), "
    "per-DRG cost trajectories, and seasonal patterns.</div>",
    unsafe_allow_html=True,
)

# ─── Quarterly aggregation ─────────────────────────────────────────────────────
quarterly = (
    df.groupby("admission_quarter")
    .agg(
        claim_count=("claim_id", "count"),
        avg_claim=("total_claim_amount", "mean"),
        total_claim=("total_claim_amount", "sum"),
        avg_los=("length_of_stay", "mean"),
        avg_insurer=("insurance_paid", "mean"),
    )
    .reset_index()
    .sort_values("admission_quarter")
)

# Compute QoQ growth
quarterly["claim_growth"] = quarterly["avg_claim"].pct_change() * 100
quarterly["volume_growth"] = quarterly["claim_count"].pct_change() * 100

# ─── KPI Row ─────────────────────────────────────────────────────────────────
if len(quarterly) >= 2:
    latest = quarterly.iloc[-1]
    prev = quarterly.iloc[-2]
    claim_delta = latest["avg_claim"] - prev["avg_claim"]
    vol_delta = latest["claim_count"] - prev["claim_count"]
    los_delta = latest["avg_los"] - prev["avg_los"]

    c1, c2, c3, c4 = st.columns(4)
    for col, (label, value, delta) in zip([c1, c2, c3, c4], [
        ("Latest Avg Claim", fmt_rm(latest["avg_claim"]),
         f"{'▲' if claim_delta > 0 else '▼'} {fmt_rm(abs(claim_delta))} vs prev quarter"),
        ("Latest Claim Volume", f"{latest['claim_count']:,}",
         f"{'▲' if vol_delta > 0 else '▼'} {abs(int(vol_delta))} vs prev quarter"),
        ("Avg Length of Stay", f"{latest['avg_los']:.1f} days",
         f"{'▲' if los_delta > 0 else '▼'} {abs(los_delta):.1f} vs prev quarter"),
        ("Periods Analysed", f"{len(quarterly)}", "quarters of data"),
    ]):
        with col:
            st.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-delta">{delta}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)

# ─── Row 1: Quarterly Trend ────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Quarterly Claim Volume & Severity (2023–2025)</div>",
            unsafe_allow_html=True)

fig_q = make_subplots(specs=[[{"secondary_y": True}]])
fig_q.add_trace(
    go.Bar(
        x=quarterly["admission_quarter"], y=quarterly["claim_count"],
        name="Claim Count", marker_color=BRAND["primary"], opacity=0.7,
    ),
    secondary_y=False,
)
fig_q.add_trace(
    go.Scatter(
        x=quarterly["admission_quarter"], y=quarterly["avg_claim"],
        name="Avg Claim (RM)", mode="lines+markers",
        line=dict(color=BRAND["accent"], width=2.5),
        marker=dict(size=7, symbol="circle"),
    ),
    secondary_y=True,
)
fig_q.add_trace(
    go.Scatter(
        x=quarterly["admission_quarter"], y=quarterly["avg_insurer"],
        name="Avg Insurer Payment (RM)", mode="lines+markers",
        line=dict(color=BRAND["secondary"], width=2, dash="dot"),
        marker=dict(size=5),
    ),
    secondary_y=True,
)
fig_q.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(orientation="h", y=1.08, x=0),
    margin=dict(t=10, b=0, l=0, r=0),
    height=320,
    hovermode="x unified",
    yaxis=dict(title="Claim Count", gridcolor=BRAND["mid"]),
    yaxis2=dict(title="Amount (RM)", gridcolor=BRAND["mid"]),
    xaxis=dict(title="Quarter", gridcolor=BRAND["mid"]),
    font=dict(family="Inter, sans-serif", size=12),
)
st.plotly_chart(fig_q, use_container_width=True)

# ─── Row 2: Per-DRG Trend + LOS Trend ─────────────────────────────────────────
st.markdown("<div class='section-title'>Per-DRG Average Claim Trend</div>",
            unsafe_allow_html=True)
col_drg, col_los = st.columns([3, 2])

with col_drg:
    drg_quarterly = (
        df.groupby(["admission_quarter", "drg_category"])["total_claim_amount"]
        .mean()
        .reset_index()
        .sort_values("admission_quarter")
    )
    fig_drg = go.Figure()
    for drg_name in sel_drg:
        sub = drg_quarterly[drg_quarterly["drg_category"] == drg_name]
        if sub.empty:
            continue
        fig_drg.add_trace(go.Scatter(
            x=sub["admission_quarter"], y=sub["total_claim_amount"],
            name=drg_name, mode="lines+markers",
            line=dict(color=DRG_COLOR_MAP.get(drg_name, BRAND["secondary"]), width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{drg_name}</b><br>%{{x}}: RM %{{y:,.0f}}<extra></extra>",
        ))
    fig_drg.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Quarter", gridcolor=BRAND["mid"]),
        yaxis=dict(title="Avg Claim (RM)", gridcolor=BRAND["mid"]),
        legend=dict(orientation="h", y=1.05, x=0, font_size=10),
        margin=dict(t=10, b=0, l=0, r=0),
        height=300,
        hovermode="x unified",
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_drg, use_container_width=True)

with col_los:
    st.markdown("<div class='section-title'>Avg LOS by DRG</div>",
                unsafe_allow_html=True)
    los_drg = (
        df.groupby("drg_category")["length_of_stay"]
        .mean()
        .reset_index()
        .sort_values("length_of_stay", ascending=True)
    )
    fig_los = go.Figure(go.Bar(
        x=los_drg["length_of_stay"],
        y=los_drg["drg_category"],
        orientation="h",
        marker_color=[DRG_COLOR_MAP.get(d, BRAND["secondary"]) for d in los_drg["drg_category"]],
        text=[f"{v:.1f}d" for v in los_drg["length_of_stay"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Avg LOS: %{x:.1f} days<extra></extra>",
    ))
    fig_los.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Avg Length of Stay (days)", gridcolor=BRAND["mid"]),
        yaxis=dict(title=""),
        margin=dict(t=10, b=0, l=0, r=10),
        height=300,
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_los, use_container_width=True)

# ─── Row 3: Monthly Seasonality Heatmap ─────────────────────────────────────
st.markdown("<div class='section-title'>Monthly Claim Volume Heatmap (Seasonality)</div>",
            unsafe_allow_html=True)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

season = (
    df.groupby(["admission_year", "admission_month"])["claim_id"]
    .count()
    .reset_index()
    .rename(columns={"claim_id": "claims"})
    .pivot(index="admission_year", columns="admission_month", values="claims")
)
season.columns = [MONTH_NAMES[m - 1] for m in season.columns]
season = season.reindex(columns=MONTH_NAMES, fill_value=None)

fig_season = go.Figure(go.Heatmap(
    z=season.values,
    x=season.columns.tolist(),
    y=season.index.astype(str).tolist(),
    colorscale=[
        [0.0, "#EFF6FF"],
        [0.5, BRAND["secondary"]],
        [1.0, BRAND["primary"]],
    ],
    text=[[str(int(v)) if not np.isnan(v) else "" for v in row] for row in season.values],
    texttemplate="%{text}",
    hovertemplate="Year: %{y}<br>Month: %{x}<br>Claims: %{z}<extra></extra>",
    colorbar=dict(title="Claims"),
))
fig_season.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(title="Month"),
    yaxis=dict(title="Year"),
    margin=dict(t=10, b=0, l=0, r=0),
    height=200,
    font=dict(family="Inter, sans-serif", size=12),
)
st.plotly_chart(fig_season, use_container_width=True)
st.caption("Darker cells = higher claim volume. Seasonal spikes may indicate endemic disease outbreaks (e.g., Dengue) or elective procedure cycles.")

# ─── Row 4: Inflation Rate Table ─────────────────────────────────────────────
st.markdown("<div class='section-title'>Year-on-Year Medical Inflation by DRG</div>",
            unsafe_allow_html=True)

yoy = (
    df.groupby(["admission_year", "drg_category"])["total_claim_amount"]
    .mean()
    .unstack(level=0)
)
inflation_rows = []
years_avail = sorted(df["admission_year"].unique())

for drg_name in yoy.index:
    row = {"DRG Category": drg_name}
    for i, year in enumerate(years_avail):
        if year in yoy.columns:
            row[str(year)] = fmt_rm(yoy.loc[drg_name, year])
        if i > 0:
            prev_yr = years_avail[i - 1]
            if prev_yr in yoy.columns and year in yoy.columns:
                pct = (yoy.loc[drg_name, year] / yoy.loc[drg_name, prev_yr] - 1) * 100
                row[f"{prev_yr}→{year} Growth"] = f"{pct:+.1f}%"
    inflation_rows.append(row)

st.dataframe(pd.DataFrame(inflation_rows), use_container_width=True, hide_index=True)
st.caption(
    "Year-on-year growth in average claim severity per DRG. "
    "Positive values indicate medical cost inflation for that category."
)
