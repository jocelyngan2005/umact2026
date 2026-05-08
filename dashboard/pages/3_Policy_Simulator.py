"""
Page 3 — Policy Co-Payment Simulator
Live comparison of Base MHIT vs Alt A (Tiered Cap) vs Alt B (Dynamic DRG Cap)
vs a fully custom policy — all recomputed in real time.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    BRAND, DRG_CATEGORIES, PLAN_TYPES, REGIONS,
    POLICY_COLOR_MAP,
    fmt_rm, fmt_pct, inject_css, data_loader_widget, simulate_custom_policy,
)

inject_css()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Scenario Definitions**")
    st.caption(
        "**Base MHIT:** 20% co-pay, RM 3,000 cap.\n\n"
        "**Alt A – Tiered Cap:** RM 3k cap for Tier 1 (efficient) hospitals, "
        "RM 10k cap for Tier 2 (inefficient) hospitals.\n\n"
        "**Alt B – Dynamic DRG:** Patient pays 10% of FMV benchmark + "
        "100% of any amount the hospital charges above the FMV."
    )
    st.divider()
    st.markdown("**Portfolio Filter**")
    sel_plans = st.multiselect("Plan Type", PLAN_TYPES, default=PLAN_TYPES)
    sel_regions = st.multiselect("Region", REGIONS, default=REGIONS)

# ─── Data ────────────────────────────────────────────────────────────────────
df_full = data_loader_widget()
if df_full is None:
    st.stop()

df = df_full[
    df_full["plan_type"].isin(sel_plans)
    & df_full["region"].isin(sel_regions)
].copy()

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
            Policy Co-Payment Simulator
        </div>
        <div style="font-size:0.9rem; color:#6B7280; line-height:1.6;">
            Interactively compare co-payment structures and their financial
            impact on both patients and the insurer — in real time.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Custom Policy Controls ────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Custom Policy Designer</div>", unsafe_allow_html=True)
ctrl1, ctrl2, ctrl3 = st.columns(3)
with ctrl1:
    custom_pct = st.slider(
        "Co-Pay Percentage", min_value=5, max_value=50, value=20, step=5,
        format="%d%%",
    ) / 100
with ctrl2:
    custom_cap = st.slider(
        "Co-Pay Cap (RM)", min_value=500, max_value=30_000, value=3_000, step=500,
        format="RM %d",
    )
with ctrl3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        f"**Custom Policy:**  \n"
        f"{custom_pct*100:.0f}% co-pay, capped at {fmt_rm(custom_cap)}"
    )

# ─── Compute Custom Policy ─────────────────────────────────────────────────
custom_patient, custom_insurer = simulate_custom_policy(df, custom_pct, custom_cap)
df["custom_patient_pays"] = custom_patient
df["custom_insurer_pays"] = custom_insurer

# ─── Aggregate Metrics ─────────────────────────────────────────────────────
scenarios = {
    "Base MHIT": {
        "patient_col": "base_mhit_patient_pays",
        "insurer_col": "base_mhit_insurer_pays",
        "description": "20% co-pay, RM 3k cap",
        "cap": 3000,
        "recommended": False,
    },
    "Alt A – Tiered Cap": {
        "patient_col": "alt_a_patient_pays",
        "insurer_col": "alt_a_insurer_pays",
        "description": "Tier 1: RM 3k cap / Tier 2: RM 10k cap",
        "cap": None,
        "recommended": False,
    },
    "Alt B – Dynamic DRG": {
        "patient_col": "alt_b_patient_pays",
        "insurer_col": "alt_b_insurer_pays",
        "description": "10% of FMV + 100% of overage",
        "cap": None,
        "recommended": True,
    },
    "Custom Policy": {
        "patient_col": "custom_patient_pays",
        "insurer_col": "custom_insurer_pays",
        "description": f"{custom_pct*100:.0f}% co-pay, {fmt_rm(custom_cap)} cap",
        "cap": custom_cap,
        "recommended": False,
    },
}

base_liability = df["base_mhit_insurer_pays"].sum()

# ─── Policy Card Row ─────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Insurer Liability — All Scenarios</div>",
            unsafe_allow_html=True)
cols = st.columns(4)
for col, (name, cfg) in zip(cols, scenarios.items()):
    insurer_total = df[cfg["insurer_col"]].sum()
    savings = base_liability - insurer_total
    savings_pct = savings / base_liability * 100
    cap_hit = (
        (df[cfg["patient_col"]] >= (cfg["cap"] - 1)).mean() * 100
        if cfg["cap"] else
        (df[cfg["patient_col"]] == df[cfg["patient_col"]].max()).mean() * 100
    )
    rec_class = "recommended" if cfg["recommended"] else ""
    savings_str = (
        f"+{fmt_rm(abs(savings))} ({abs(savings_pct):.1f}% more)"
        if savings < 0 else
        f"Saves {fmt_rm(savings)} ({savings_pct:.1f}%)"
    )
    with col:
        st.markdown(
            f"""<div class="policy-card {rec_class}">
                <div class="pc-name">{name}{' (Recommended)' if cfg['recommended'] else ''}</div>
                <div class="pc-amount">{fmt_rm(insurer_total)}</div>
                <div class="pc-savings">{cfg['description']}</div>
                <div class="pc-savings" style="color:{'#10B981' if savings>0 else '#EF4444'}">
                    {savings_str if name != 'Base MHIT' else 'Baseline'}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ─── Comparison Charts Row ─────────────────────────────────────────────────
col_bar, col_cap = st.columns([3, 2])

with col_bar:
    st.markdown("<div class='section-title'>Insurer vs Patient Liability Breakdown</div>",
                unsafe_allow_html=True)
    names = list(scenarios.keys())
    insurer_vals = [df[cfg["insurer_col"]].sum() / 1e6 for cfg in scenarios.values()]
    patient_vals = [df[cfg["patient_col"]].sum() / 1e6 for cfg in scenarios.values()]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="Insurer Pays", x=names, y=insurer_vals,
        marker_color=BRAND["primary"],
        text=[f"RM {v:.1f}M" for v in insurer_vals], textposition="inside",
    ))
    fig_bar.add_trace(go.Bar(
        name="Patient Pays", x=names, y=patient_vals,
        marker_color=BRAND["accent"],
        text=[f"RM {v:.1f}M" for v in patient_vals], textposition="inside",
    ))
    fig_bar.update_layout(
        barmode="stack",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="Total Amount (RM Millions)", gridcolor=BRAND["mid"]),
        xaxis=dict(title=""),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(t=10, b=0, l=0, r=0),
        height=320,
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_cap:
    st.markdown("<div class='section-title'>Cap Hit Rate & Avg Patient Liability</div>",
                unsafe_allow_html=True)

    cap_data = []
    for name, cfg in scenarios.items():
        cap_data.append({
            "Scenario": name,
            "Avg Patient Pays (RM)": df[cfg["patient_col"]].mean(),
            "Avg Insurer Pays (RM)": df[cfg["insurer_col"]].mean(),
        })
    cap_df = pd.DataFrame(cap_data)

    fig_avg = go.Figure(go.Bar(
        x=cap_df["Avg Patient Pays (RM)"],
        y=cap_df["Scenario"],
        orientation="h",
        marker_color=[
            POLICY_COLOR_MAP.get(n, BRAND["secondary"]) for n in cap_df["Scenario"]
        ],
        text=[fmt_rm(v) for v in cap_df["Avg Patient Pays (RM)"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Avg patient: RM %{x:,.0f}<extra></extra>",
    ))
    fig_avg.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Avg Patient Liability (RM)", gridcolor=BRAND["mid"]),
        yaxis=dict(title="", autorange="reversed"),
        margin=dict(t=10, b=0, l=0, r=10),
        height=320,
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_avg, use_container_width=True)

# ─── Patient Payment Distribution Comparison ──────────────────────────────────
st.markdown("<div class='section-title'>Patient Payment Distribution — All Scenarios</div>",
            unsafe_allow_html=True)

fig_dist = go.Figure()
for name, cfg in scenarios.items():
    fig_dist.add_trace(go.Histogram(
        x=df[cfg["patient_col"]],
        name=name,
        opacity=0.6,
        nbinsx=50,
        marker_color=POLICY_COLOR_MAP.get(name, BRAND["secondary"]),
        hovertemplate=f"<b>{name}</b><br>RM %{{x:,.0f}} — %{{y}} claims<extra></extra>",
    ))

fig_dist.update_layout(
    barmode="overlay",
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(
        title="Patient Co-Payment (RM)",
        gridcolor=BRAND["mid"],
        range=[0, min(df["total_claim_amount"].quantile(0.99), 30_000)],
    ),
    yaxis=dict(title="Number of Claims", gridcolor=BRAND["mid"]),
    legend=dict(orientation="h", y=1.05, x=0),
    margin=dict(t=10, b=0, l=0, r=0),
    height=300,
    hovermode="x",
    font=dict(family="Inter, sans-serif", size=12),
)
st.plotly_chart(fig_dist, use_container_width=True)

# ─── Detailed Stats Table ─────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Full Financial Comparison Table</div>",
            unsafe_allow_html=True)

rows = []
for name, cfg in scenarios.items():
    pp = df[cfg["patient_col"]]
    ip = df[cfg["insurer_col"]]
    savings = base_liability - ip.sum()
    rows.append({
        "Scenario": f"{name}{' (Recommended)' if cfg['recommended'] else ''}",
        "Total Insurer Liability": fmt_rm(ip.sum()),
        "Savings vs Base MHIT": fmt_rm(savings) if name != "Base MHIT" else "—",
        "Savings %": f"{savings/base_liability*100:.2f}%" if name != "Base MHIT" else "—",
        "Avg Patient Pays": fmt_rm(pp.mean()),
        "Median Patient Pays": fmt_rm(pp.median()),
        "Max Patient Pays": fmt_rm(pp.max()),
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─── Insight Callout ────────────────────────────────────────────────────────
with st.expander(":material/lightbulb: Key Findings from the Policy Analysis"):
    b_saves = df["base_mhit_insurer_pays"].sum() - df["alt_b_insurer_pays"].sum()
    a_saves = df["base_mhit_insurer_pays"].sum() - df["alt_a_insurer_pays"].sum()
    st.markdown(f"""
**Critical flaw identified:** {df['hits_cap'].mean()*100:.0f}% of claims breach the RM 3,000 co-pay cap under the Base MHIT Plan,
meaning the insurer absorbs virtually all excess costs for the vast majority of cases.

**Alternative A (Tiered Cap)** saves **{fmt_rm(a_saves)}** — a modest improvement because it still protects
patients from the true cost of over-servicing at inefficient hospitals.

**Alternative B (Dynamic DRG Cap)** saves **{fmt_rm(b_saves)}** by aligning patient exposure with
Fair Market Value benchmarks. It creates a direct financial incentive for patients to choose
efficient hospitals and for hospitals to stay close to the FMV benchmark.

**Recommendation:** Adopt the **Dynamic DRG Cap (Alt B)** to support the MHIT plan.
    """)
