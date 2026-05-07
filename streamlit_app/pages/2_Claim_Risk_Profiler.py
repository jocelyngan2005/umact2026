"""
Page 2 — Claim Risk Profiler
Real-time ML-powered patient risk scoring, cluster assignment, and
expected claim prediction using the trained Lognormal OLS + K-Means models.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    BRAND, DRG_CATEGORIES, DRG_COLOR_MAP, HOSPITAL_TYPES, PLAN_TYPES, REGIONS,
    PROCEDURES, DRG_MAPPING,
    calculate_risk_score_single, cluster_label_fallback,
    fmt_rm, inject_css, load_models, data_loader_widget, predict_claim, risk_label,
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
    st.markdown("**About the models**")
    st.caption(
        "**Severity model:** Lognormal OLS with DRG, region, hospital type, "
        "risk cluster, and patient risk score as predictors. R²=0.981.\n\n"
        "**Risk clustering:** K-Means (k=5) on age, BMI, chronic conditions, "
        "and prior claims history. Zero data leakage — trained on ex-ante features only."
    )

# ─── Load data + models ───────────────────────────────────────────────────────
df = data_loader_widget()
if df is None:
    st.stop()

severity_model, kmeans, scaler = load_models()
models_loaded = severity_model is not None

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("<div class='page-header'>🎯 Claim Risk Profiler</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='page-sub'>Enter a patient profile below to receive an instant "
    "risk assessment, cluster assignment, and expected claim prediction.</div>",
    unsafe_allow_html=True,
)

if models_loaded:
    st.markdown(
        "<div class='model-banner ok'>✅ ML models are active — "
        "predictions use the trained Lognormal OLS + K-Means pipeline.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='model-banner warn'>⚠️ ML model files not found in <code>models/</code>. "
        "Showing benchmark-based estimates. See the sidebar for export instructions.</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ─── Input Form + Results ─────────────────────────────────────────────────────
col_form, col_results = st.columns([2, 3], gap="large")

with col_form:
    st.markdown("<div class='section-title'>Patient Profile</div>", unsafe_allow_html=True)

    with st.form("risk_form"):
        st.markdown("**Demographics**")
        age = st.slider("Age", min_value=0, max_value=100, value=45, step=1)
        gender = st.selectbox("Gender", ["M", "F"])
        smoker = st.selectbox("Smoker Status", ["Non-Smoker", "Smoker"])
        bmi = st.slider("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.5)

        st.markdown("**Clinical History**")
        chronic = st.slider("Number of Chronic Conditions", 0, 5, 0)
        prev_claims = st.selectbox("Prior Claims?", [0, 1], format_func=lambda x: "Yes" if x else "No")

        st.markdown("**Admission Details**")
        procedure = st.selectbox("Procedure / Diagnosis", PROCEDURES)
        hospital_type = st.selectbox("Hospital Type", HOSPITAL_TYPES)
        region = st.selectbox("Region", REGIONS)
        plan_type = st.selectbox("Plan Type", PLAN_TYPES)

        submitted = st.form_submit_button(
            "Score This Patient →",
            use_container_width=True,
            type="primary",
        )

with col_results:
    if not submitted:
        st.markdown(
            f"""<div style="
                height: 420px; display:flex; flex-direction:column;
                align-items:center; justify-content:center;
                border: 2px dashed {BRAND['mid']}; border-radius: 14px;
                color:{BRAND['gray']}; text-align:center; padding: 2rem;
            ">
                <div style="font-size:3rem;margin-bottom:1rem;">🎯</div>
                <div style="font-size:1rem;font-weight:600;">Fill in the patient profile</div>
                <div style="font-size:0.85rem;margin-top:0.5rem;">
                    and click <b>Score This Patient</b> to see<br>
                    risk tier, cluster assignment, and predicted claim.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        drg = DRG_MAPPING.get(procedure, "Respiratory")
        risk_score = calculate_risk_score_single(age, bmi, chronic, smoker)
        label, label_color = risk_label(risk_score)

        if models_loaded:
            pred_amount, cluster_id, _ = predict_claim(
                severity_model, kmeans, scaler,
                age, bmi, chronic, prev_claims, smoker,
                drg, region, hospital_type,
            )
            cluster_name = f"Cluster {cluster_id}"
            model_note = "Lognormal OLS + Duan's smearing correction"
        else:
            # Fallback: benchmark from historical data
            mask = (df["drg_category"] == drg) & (df["hospital_type"] == hospital_type)
            base_amount = df.loc[mask, "total_claim_amount"].median()
            if pd.isna(base_amount):
                base_amount = df["total_claim_amount"].median()
            pred_amount = base_amount * risk_score
            cluster_name = cluster_label_fallback(risk_score)
            model_note = "Group median × risk score (ML model not loaded)"

        # ── Gauge chart ──────────────────────────────────────────────────
        gauge_max = 3.0
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            delta={"reference": 1.0, "increasing": {"color": BRAND["danger"]}},
            number={"font": {"size": 40, "color": BRAND["primary"]}},
            gauge={
                "axis": {"range": [1, gauge_max], "tickwidth": 1},
                "bar": {"color": label_color, "thickness": 0.25},
                "steps": [
                    {"range": [1.0, 1.5], "color": "#D1FAE5"},
                    {"range": [1.5, 2.0], "color": "#FEF3C7"},
                    {"range": [2.0, gauge_max], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": BRAND["danger"], "width": 3},
                    "thickness": 0.75,
                    "value": risk_score,
                },
            },
            title={"text": "Patient Risk Score", "font": {"size": 14, "color": BRAND["gray"]}},
        ))
        fig_gauge.update_layout(
            height=220, margin=dict(t=30, b=0, l=30, r=30),
            paper_bgcolor="white",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Result cards ─────────────────────────────────────────────────
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-label">Risk Tier</div>
                    <div class="kpi-value" style="font-size:1.2rem;">{label}</div>
                    <div class="kpi-delta">Risk score: {risk_score:.2f}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-label">Expected Claim</div>
                    <div class="kpi-value" style="font-size:1.35rem;">{fmt_rm(pred_amount)}</div>
                    <div class="kpi-delta">{model_note[:28]}…</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-label">Risk Cluster</div>
                    <div class="kpi-value" style="font-size:1rem;">{cluster_name}</div>
                    <div class="kpi-delta">K-Means segmentation</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # ── Premium loading table ─────────────────────────────────────────
        st.markdown("<div class='section-title'>Premium Loading Recommendation</div>",
                    unsafe_allow_html=True)

        base_premiums = {"Basic": 150, "Silver": 250, "Gold": 350}
        base = base_premiums[plan_type]
        loading_pct = (risk_score - 1.0) * 100
        loaded_premium = base * risk_score
        insurer_exposure = max(0, pred_amount - min(pred_amount * 0.20, 3000))

        loading_df = pd.DataFrame({
            "Item": [
                "Base Annual Premium (RM)",
                "Risk Loading (%)",
                "Risk-Adjusted Premium (RM)",
                "Expected Insurer Exposure (RM)",
                "DRG Category",
                "Hospital Tier",
                "Region",
            ],
            "Value": [
                f"RM {base:,.0f}",
                f"+{loading_pct:.0f}%",
                f"RM {loaded_premium:,.0f}",
                fmt_rm(insurer_exposure),
                drg,
                hospital_type,
                region,
            ],
        })
        st.dataframe(loading_df, use_container_width=True, hide_index=True)

        # ── How is this calculated? ────────────────────────────────────────
        with st.expander("How is this calculated?"):
            st.markdown(f"""
**Risk Score Formula** (from actuarial feature engineering):
```
score = 1.0
+ 0.50 if age > 65  (or +0.25 if age > 50)
+ 0.15 if BMI ≥ 30
+ 0.25 if Smoker
+ 0.20 × chronic_conditions
```
Current score: **{risk_score:.2f}**

**Severity Prediction** ({'OLS model' if models_loaded else 'Fallback'}):
{"Uses the trained Lognormal OLS model: `log(claim) ~ DRG + Region + Hospital + Cluster + RiskScore`, then applies Duan's smearing factor to correct re-transformation bias." if models_loaded else f"Uses historical group median for {drg} × {hospital_type}, scaled by the risk score."}

**Premium Loading**: Base room-and-board rate × risk score.
            """)

        # ── Historical comparison ──────────────────────────────────────────
        st.markdown("<div class='section-title'>vs. Historical Claims — Same DRG & Hospital Type</div>",
                    unsafe_allow_html=True)
        comp = df[(df["drg_category"] == drg) & (df["hospital_type"] == hospital_type)][
            "total_claim_amount"
        ]
        if not comp.empty:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=comp, nbinsx=40,
                marker_color=BRAND["primary"], opacity=0.6,
                name="Historical claims",
            ))
            fig_hist.add_vline(
                x=pred_amount, line_dash="dash",
                line_color=BRAND["accent"], line_width=2.5,
                annotation_text=f"Your patient: {fmt_rm(pred_amount)}",
                annotation_font_color=BRAND["accent"],
            )
            fig_hist.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Claim Amount (RM)"),
                yaxis=dict(title="Count"),
                margin=dict(t=10, b=0, l=0, r=0),
                height=200, showlegend=False,
                font=dict(family="Inter, sans-serif", size=11),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            pct_rank = (comp < pred_amount).mean() * 100
            st.caption(
                f"This patient's expected claim is at the "
                f"**{pct_rank:.0f}th percentile** of historical "
                f"{drg} claims at {hospital_type} hospitals "
                f"(n={len(comp):,})."
            )
