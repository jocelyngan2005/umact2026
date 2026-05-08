"""
Page 1 — Executive Dashboard
Interactive KPI cards, claim trends, distribution charts, and regional breakdown.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    BRAND, DRG_COLOR_MAP, PLAN_COLOR_MAP, REGION_COLOR_MAP,
    PLAN_TYPES, REGIONS, DRG_CATEGORIES,
    fmt_rm, fmt_pct, inject_css, data_loader_widget,
)

inject_css()

# ─── Data ────────────────────────────────────────────────────────────────────
df_full = data_loader_widget()
if df_full is None:
    st.stop()

# ─── Sidebar Filters ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Filters**")

    years = sorted(df_full["admission_year"].unique())
    sel_years = st.multiselect("Year", years, default=years)

    sel_plans = st.multiselect("Plan Type", PLAN_TYPES, default=PLAN_TYPES)
    sel_regions = st.multiselect("Region", REGIONS, default=REGIONS)
    sel_drg = st.multiselect("DRG Category", DRG_CATEGORIES, default=DRG_CATEGORIES)

df = df_full[
    df_full["admission_year"].isin(sel_years)
    & df_full["plan_type"].isin(sel_plans)
    & df_full["region"].isin(sel_regions)
    & df_full["drg_category"].isin(sel_drg)
].copy()

# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom: 1.8rem; padding-bottom: 1.4rem;
                border-bottom: 1px solid #E5E7EB;">
        <div style="font-size:0.68rem; font-weight:700; color:#ABABAB;
                    text-transform:uppercase; letter-spacing:0.1em;
                    margin-bottom:0.5rem;">
            UMACT Hackathon 2026 · Finals Submission
        </div>
        <div style="font-size:2rem; font-weight:700; color:#0A0A0A;
                    line-height:1.2; margin-bottom:0.55rem;
                    letter-spacing:-0.025em;">
            MHIT Insurance Intelligence
        </div>
        <div style="font-size:0.9rem; color:#6B7280; line-height:1.6;">
            An end-to-end actuarial analytics and policy decision tool built on
            20,000 Malaysian health insurance claims — featuring live risk scoring,
            interactive policy simulation, and medical trend analysis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if df.empty:
    st.warning("No data matches the current filter selection.")
    st.stop()

# ─── KPI Row ─────────────────────────────────────────────────────────────────
total_claims = len(df)
avg_claim = df["total_claim_amount"].mean()
cap_hit_rate = df["hits_cap"].mean()
insurer_liability = df["base_mhit_insurer_pays"].sum()
avg_los = df["length_of_stay"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    ("Total Claims", f"{total_claims:,}", "hospitalisation episodes"),
    ("Avg Claim Amount", fmt_rm(avg_claim), "per admission"),
    ("Cap Hit Rate", f"{cap_hit_rate*100:.1f}%", "breach RM 3,000 co-pay cap"),
    ("Insurer Liability", fmt_rm(insurer_liability), "Base MHIT plan"),
    ("Avg Length of Stay", f"{avg_los:.1f} days", "across filtered claims"),
]
for col, (label, value, delta) in zip([c1, c2, c3, c4, c5], kpis):
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

# ─── Row 2: Monthly Trend + DRG Donut ─────────────────────────────────────────
st.markdown("<div class='section-title'>Claim Volume & Severity Trend</div>", unsafe_allow_html=True)
col_trend, col_donut = st.columns([3, 2])

with col_trend:
    monthly = (
        df.groupby(["admission_year", "admission_month"])
        .agg(claim_count=("claim_id", "count"), avg_claim=("total_claim_amount", "mean"))
        .reset_index()
    )
    monthly["period"] = pd.to_datetime(
        monthly["admission_year"].astype(str) + "-" + monthly["admission_month"].astype(str) + "-01"
    )
    monthly = monthly.sort_values("period")

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Bar(
            x=monthly["period"], y=monthly["claim_count"],
            name="Claim Count", marker_color=BRAND["primary"],
            opacity=0.7,
        ),
        secondary_y=False,
    )
    fig_trend.add_trace(
        go.Scatter(
            x=monthly["period"], y=monthly["avg_claim"],
            name="Avg Claim (RM)", mode="lines+markers",
            line=dict(color=BRAND["accent"], width=2.5),
            marker=dict(size=5),
        ),
        secondary_y=True,
    )
    fig_trend.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(t=10, b=0, l=0, r=0),
        height=300,
        hovermode="x unified",
        yaxis=dict(title="Claim Count", gridcolor=BRAND["mid"]),
        yaxis2=dict(title="Avg Claim (RM)", gridcolor=BRAND["mid"]),
        xaxis=dict(gridcolor=BRAND["mid"]),
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col_donut:
    drg_counts = df.groupby("drg_category")["total_claim_amount"].sum().reset_index()
    drg_counts.columns = ["DRG Category", "Total Claims (RM)"]
    fig_donut = px.pie(
        drg_counts, values="Total Claims (RM)", names="DRG Category",
        color="DRG Category", color_discrete_map=DRG_COLOR_MAP,
        hole=0.52,
    )
    fig_donut.update_traces(
        textposition="outside", textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>RM %{value:,.0f}<extra></extra>",
    )
    fig_donut.update_layout(
        showlegend=False, margin=dict(t=10, b=0, l=0, r=0),
        height=300, paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11),
        annotations=[dict(
            text=f"<b>{fmt_rm(df['total_claim_amount'].sum())}</b><br>total",
            x=0.5, y=0.5, font_size=13, showarrow=False,
        )],
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# ─── Row 3: Claim Distribution + Regional Breakdown ──────────────────────────
st.markdown("<div class='section-title'>Claim Severity Distribution & Regional Breakdown</div>",
            unsafe_allow_html=True)
col_dist, col_region = st.columns([3, 2])

with col_dist:
    claims = df["total_claim_amount"].dropna()
    fig_dist = go.Figure()

    # Histogram
    fig_dist.add_trace(go.Histogram(
        x=claims, nbinsx=60,
        marker_color=BRAND["primary"], opacity=0.65,
        name="Claim Distribution",
        hovertemplate="RM %{x:,.0f} — %{y} claims<extra></extra>",
    ))

    # Lognormal fit overlay
    shape, loc, scale = scipy_stats.lognorm.fit(claims, floc=0)
    x_fit = np.linspace(claims.min(), min(claims.quantile(0.995), claims.max()), 300)
    pdf_fit = scipy_stats.lognorm.pdf(x_fit, shape, loc, scale)
    bin_width = (claims.max() - claims.min()) / 60
    fig_dist.add_trace(go.Scatter(
        x=x_fit, y=pdf_fit * len(claims) * bin_width,
        mode="lines", name="Lognormal Fit",
        line=dict(color=BRAND["accent"], width=2.5, dash="dash"),
    ))

    # Cap line
    cap_threshold = 15_000.65
    fig_dist.add_vline(
        x=cap_threshold, line_dash="dot",
        line_color=BRAND["danger"], line_width=1.5,
        annotation_text="RM 3k cap trigger",
        annotation_position="top right",
        annotation_font_size=10,
    )

    fig_dist.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Total Claim Amount (RM)", gridcolor=BRAND["mid"]),
        yaxis=dict(title="Count", gridcolor=BRAND["mid"]),
        legend=dict(orientation="h", y=1.05),
        margin=dict(t=10, b=0, l=0, r=0),
        height=300, hovermode="x",
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col_region:
    region_avg = (
        df.groupby("region")["total_claim_amount"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_claim", "count": "n_claims"})
        .sort_values("avg_claim", ascending=True)
    )
    fig_region = go.Figure(go.Bar(
        x=region_avg["avg_claim"],
        y=region_avg["region"],
        orientation="h",
        marker_color=[REGION_COLOR_MAP.get(r, BRAND["secondary"]) for r in region_avg["region"]],
        text=[fmt_rm(v) for v in region_avg["avg_claim"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Avg: RM %{x:,.0f}<extra></extra>",
    ))
    fig_region.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Avg Claim Amount (RM)", gridcolor=BRAND["mid"]),
        yaxis=dict(title=""),
        margin=dict(t=10, b=0, l=0, r=10),
        height=300,
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_region, use_container_width=True)

# ─── Row 4: Plan Type Box Plot + Top Diagnoses ───────────────────────────────
st.markdown("<div class='section-title'>Claim Severity by Plan Type & Top Diagnoses</div>",
            unsafe_allow_html=True)
col_box, col_diag = st.columns([2, 3])

with col_box:
    fig_box = go.Figure()
    for plan in ["Basic", "Silver", "Gold"]:
        subset = df[df["plan_type"] == plan]["total_claim_amount"]
        if subset.empty:
            continue
        fig_box.add_trace(go.Box(
            y=subset, name=plan,
            marker_color=PLAN_COLOR_MAP.get(plan, BRAND["primary"]),
            boxmean=True,
        ))
    fig_box.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="Claim Amount (RM)", gridcolor=BRAND["mid"]),
        xaxis=dict(title="Plan Type"),
        margin=dict(t=10, b=0, l=0, r=0),
        height=300, showlegend=False,
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_box, use_container_width=True)

with col_diag:
    top_diag = (
        df.groupby("procedure_diagnosis")["total_claim_amount"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"procedure_diagnosis": "Diagnosis", "mean": "Avg Claim", "count": "Count"})
        .sort_values("Avg Claim", ascending=False)
        .head(10)
    )
    top_diag["Avg Claim (RM)"] = top_diag["Avg Claim"].apply(fmt_rm)

    fig_diag = go.Figure(go.Bar(
        x=top_diag["Avg Claim"],
        y=top_diag["Diagnosis"],
        orientation="h",
        marker_color=BRAND["secondary"],
        text=top_diag["Avg Claim (RM)"],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Avg: RM %{x:,.0f}<extra></extra>",
    ))
    fig_diag.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Avg Claim (RM)", gridcolor=BRAND["mid"]),
        yaxis=dict(title="", autorange="reversed"),
        margin=dict(t=10, b=0, l=0, r=10),
        height=300,
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_diag, use_container_width=True)

# ─── Footer caption ──────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Showing **{total_claims:,}** claims | "
    f"Total claim value: **{fmt_rm(df['total_claim_amount'].sum())}** | "
    f"Cap hit rate: **{cap_hit_rate*100:.1f}%**"
)
