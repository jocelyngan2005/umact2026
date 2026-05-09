"""
Page 4 — Portfolio Risk Analytics
Loss ratio heatmaps, hospital efficiency benchmarking, DRG cost breakdown,
and patient risk cluster analysis.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    BRAND, DRG_COLOR_MAP, DRG_CATEGORIES, PLAN_TYPES, REGIONS,
    fmt_rm, inject_css, data_loader_widget, load_models, cluster_label_fallback,
)

inject_css()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Filters**")
    sel_plans = st.multiselect("Plan Type", PLAN_TYPES, default=PLAN_TYPES)
    sel_regions = st.multiselect("Region", REGIONS, default=REGIONS)
    hospital_tier_filter = st.multiselect(
        "Hospital Tier", ["Tier 1", "Tier 2"], default=["Tier 1", "Tier 2"]
    )

# ─── Data ────────────────────────────────────────────────────────────────────
df_full = data_loader_widget()
if df_full is None:
    st.stop()

df = df_full[
    df_full["plan_type"].isin(sel_plans)
    & df_full["region"].isin(sel_regions)
    & df_full["hospital_tier"].isin(hospital_tier_filter)
].copy()

# ─── Load ML Models for Clustering ───────────────────────────────────────────
severity_model, kmeans, scaler = load_models()
models_loaded = kmeans is not None

# ─── Header ──────────────────────────────────────────────────────────────────
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
            Portfolio Risk Analytics
        </div>
        <div style="font-size:0.9rem; color:#6B7280; line-height:1.6;">
            Loss ratio segmentation, hospital efficiency benchmarking,
            DRG cost breakdown, and risk cluster profiling.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── KPI Row ─────────────────────────────────────────────────────────────────
n_hospitals = df["hospital_name"].nunique()
tier2_pct = (df["hospital_tier"] == "Tier 2").mean()
avg_variance = df["cost_variance_pct"].mean()
high_cost_pct = (df["cost_variance_pct"] > 0.20).mean()

c1, c2, c3, c4 = st.columns(4)
for col, (label, value, delta) in zip([c1, c2, c3, c4], [
    ("Unique Hospitals", f"{n_hospitals}", "in filtered portfolio"),
    ("Tier 2 Share", f"{tier2_pct*100:.1f}%", "inefficient hospitals"),
    ("Avg Cost Variance", f"{avg_variance*100:+.1f}%", "vs FMV benchmark"),
    ("High-Cost Claims", f"{high_cost_pct*100:.1f}%", ">20% above FMV"),
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

# ─── Row 1: Loss Ratio Heatmap + DRG Breakdown ───────────────────────────────
col_heat, col_drg = st.columns([3, 2])

with col_heat:
    st.markdown("<div class='section-title'>Avg Claim Amount — Plan Type × Region</div>",
                unsafe_allow_html=True)
    pivot = (
        df.groupby(["plan_type", "region"])["insurance_paid"]
        .mean()
        .reset_index()
        .pivot(index="plan_type", columns="region", values="insurance_paid")
        .reindex(index=["Basic", "Silver", "Gold"])
    )
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0, "#F0FDFA"],
            [0.5, BRAND["secondary"]],
            [1.0, BRAND["primary"]],
        ],
        text=[[fmt_rm(v) for v in row] for row in pivot.values],
        texttemplate="%{text}",
        hovertemplate="Plan: %{y}<br>Region: %{x}<br>Avg Insurer Pays: RM %{z:,.0f}<extra></extra>",
        colorbar=dict(title="Avg Insurer<br>Payment (RM)"),
    ))
    fig_heat.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Region"),
        yaxis=dict(title="Plan Type"),
        margin=dict(t=10, b=0, l=0, r=0),
        height=280,
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col_drg:
    st.markdown("<div class='section-title'>Avg Claim by DRG Category</div>",
                unsafe_allow_html=True)
    drg_avg = (
        df.groupby("drg_category")["total_claim_amount"]
        .mean()
        .reset_index()
        .sort_values("total_claim_amount", ascending=True)
    )
    fig_drg = go.Figure(go.Bar(
        x=drg_avg["total_claim_amount"],
        y=drg_avg["drg_category"],
        orientation="h",
        marker_color=[DRG_COLOR_MAP.get(d, BRAND["secondary"]) for d in drg_avg["drg_category"]],
        text=[fmt_rm(v) for v in drg_avg["total_claim_amount"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Avg: RM %{x:,.0f}<extra></extra>",
    ))
    fig_drg.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Avg Claim (RM)", gridcolor=BRAND["mid"]),
        yaxis=dict(title=""),
        margin=dict(t=10, b=0, l=0, r=10),
        height=280,
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_drg, use_container_width=True)

# ─── Row 2: Hospital Benchmarking Scatter ─────────────────────────────────────
st.markdown("<div class='section-title'>Hospital Efficiency Benchmarking — Actual vs FMV</div>",
            unsafe_allow_html=True)

hosp_summary = (
    df.groupby(["hospital_name", "hospital_tier", "drg_category"])
    .agg(
        actual=("total_claim_amount", "mean"),
        fmv=("expected_fmv_cost", "mean"),
        n_claims=("claim_id", "count"),
    )
    .reset_index()
)
hosp_summary["variance_pct"] = (hosp_summary["actual"] / hosp_summary["fmv"] - 1) * 100

fig_scatter = px.scatter(
    hosp_summary,
    x="fmv", y="actual",
    color="hospital_tier",
    color_discrete_map={"Tier 1": BRAND["success"], "Tier 2": BRAND["danger"]},
    size="n_claims",
    size_max=20,
    hover_name="hospital_name",
    hover_data={
        "fmv": ":,.0f",
        "actual": ":,.0f",
        "variance_pct": ":.1f",
        "n_claims": True,
    },
    labels={
        "fmv": "FMV Benchmark (RM)",
        "actual": "Actual Avg Claim (RM)",
        "hospital_tier": "Hospital Tier",
    },
)
# Reference line (actual = FMV)
max_val = max(hosp_summary["fmv"].max(), hosp_summary["actual"].max())
fig_scatter.add_trace(go.Scatter(
    x=[0, max_val], y=[0, max_val],
    mode="lines",
    name="Actual = FMV (fair pricing)",
    line=dict(color=BRAND["gray"], dash="dash", width=1.5),
))
fig_scatter.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(gridcolor=BRAND["mid"], title="FMV Benchmark (RM)"),
    yaxis=dict(gridcolor=BRAND["mid"], title="Actual Avg Claim (RM)"),
    legend=dict(orientation="h", y=1.05, x=0),
    margin=dict(t=10, b=0, l=0, r=0),
    height=380,
    font=dict(family="Inter, sans-serif", size=12),
)
st.plotly_chart(fig_scatter, use_container_width=True)
st.caption(
    "Points **above** the dashed line indicate hospitals charging more than the FMV benchmark. "
    "**Tier 2** (red) hospitals contribute disproportionately to insurer over-expenditure."
)

# ─── Row 3: Risk Cluster Profiles ─────────────────────────────────────────────
st.markdown("<div class='section-title'>Patient Risk Cluster Profiling</div>",
            unsafe_allow_html=True)

if not models_loaded:
    st.warning(
        "⚠️ K-Means clustering model not loaded. Using rule-based clustering. "
        "Drop model files in `models/` folder for ML-driven assignments.",
        icon="⚠️"
    )

# Compute cluster assignments using trained K-Means if available
if models_loaded and "risk_cluster" not in df.columns:
    try:
        features = df[["patient_age", "bmi", "chronic_conditions", "has_previous_claims"]].fillna(0)
        X_scaled = scaler.transform(features)
        df["risk_cluster"] = kmeans.predict(X_scaled)
    except Exception as e:
        st.warning(f"Could not compute K-Means clusters: {e}")

# Map cluster numbers to meaningful labels
if "risk_cluster" not in df.columns:
    bins = [0, 1.15, 1.40, 1.70, 2.00, 99]
    labels = ["0 – Young & Healthy", "1 – Low-Moderate", "2 – Moderate", "3 – High Risk", "4 – Very High"]
    df["risk_cluster_label"] = pd.cut(df["patient_risk_score"], bins=bins, labels=labels)
else:
    # Convert cluster numbers to descriptive labels based on ex-ante feature profiles
    cluster_names = {
        0: "Cluster 0 – High Risk (Older, Chronic)",
        1: "Cluster 1 – Low-Moderate (Young w/ History)",
        2: "Cluster 2 – Lowest Risk (Young & Healthy)",
        3: "Cluster 3 – High Risk (Older, Chronic & Claims)",
        4: "Cluster 4 – Moderate-High (High BMI)",
    }
    df["risk_cluster_label"] = df["risk_cluster"].map(lambda x: cluster_names.get(x, f"Cluster {x}"))

col_scatter, col_profile = st.columns([3, 2])

with col_scatter:
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig_clust = px.scatter(
        sample, x="patient_age", y="bmi",
        color="risk_cluster_label",
        size="total_claim_amount",
        size_max=15,
        opacity=0.65,
        hover_data={"patient_risk_score": ":.2f", "chronic_conditions": True,
                    "total_claim_amount": ":,.0f"},
        labels={
            "patient_age": "Patient Age",
            "bmi": "BMI",
            "risk_cluster_label": "Risk Cluster",
        },
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_clust.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor=BRAND["mid"]),
        yaxis=dict(gridcolor=BRAND["mid"]),
        legend=dict(orientation="h", y=1.05, x=0, font_size=11),
        margin=dict(t=10, b=0, l=0, r=0),
        height=340,
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_clust, use_container_width=True)

with col_profile:
    cluster_profile = (
        df.groupby("risk_cluster_label")
        .agg(
            Patients=("claim_id", "count"),
            Avg_Age=("patient_age", "mean"),
            Avg_BMI=("bmi", "mean"),
            Avg_Chronic=("chronic_conditions", "mean"),
            Avg_Claim=("total_claim_amount", "mean"),
            Avg_Risk_Score=("patient_risk_score", "mean"),
        )
        .round(2)
        .reset_index()
    )
    cluster_profile.columns = [
        "Cluster", "# Patients", "Avg Age", "Avg BMI",
        "Avg Chronic", "Avg Claim (RM)", "Avg Risk Score"
    ]
    cluster_profile["Avg Claim (RM)"] = cluster_profile["Avg Claim (RM)"].apply(fmt_rm)
    st.dataframe(cluster_profile, use_container_width=True, hide_index=True)

    clustering_method = "**K-Means ML Model**" if models_loaded else "**Rule-Based Binning**"
    st.caption(
        f"Clustering method: {clustering_method} • "
        "Bubble size in scatter = claim amount. "
        "Cluster separation is driven by age, BMI, chronic conditions, and prior claims."
    )

# ─── Row 4: Admission Type & Hospital Type Breakdown ─────────────────────────
st.markdown("<div class='section-title'>Admission Type & Hospital Type Cost Analysis</div>",
            unsafe_allow_html=True)

col_adm, col_htype = st.columns(2)

with col_adm:
    adm_avg = (
        df.groupby("admission_type")["total_claim_amount"].mean()
        .reset_index()
        .sort_values("total_claim_amount", ascending=False)
    )
    fig_adm = go.Figure(go.Bar(
        x=adm_avg["admission_type"],
        y=adm_avg["total_claim_amount"],
        marker_color=BRAND["secondary"],
        text=[fmt_rm(v) for v in adm_avg["total_claim_amount"]],
        textposition="outside",
    ))
    fig_adm.update_layout(
        title_text="Avg Claim by Admission Type",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="Avg Claim (RM)", gridcolor=BRAND["mid"]),
        margin=dict(t=40, b=0, l=0, r=0),
        height=280,
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_adm, use_container_width=True)

with col_htype:
    htype_avg = (
        df.groupby("hospital_type")["total_claim_amount"].mean()
        .reset_index()
        .sort_values("total_claim_amount", ascending=False)
    )
    fig_htype = go.Figure(go.Bar(
        x=htype_avg["hospital_type"],
        y=htype_avg["total_claim_amount"],
        marker_color=BRAND["accent"],
        text=[fmt_rm(v) for v in htype_avg["total_claim_amount"]],
        textposition="outside",
    ))
    fig_htype.update_layout(
        title_text="Avg Claim by Hospital Type",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="Avg Claim (RM)", gridcolor=BRAND["mid"]),
        margin=dict(t=40, b=0, l=0, r=0),
        height=280,
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_htype, use_container_width=True)
