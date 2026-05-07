"""
Shared utilities: constants, data loading, feature engineering,
model loading, policy simulation helpers, and formatting.
"""

import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ─── Constants ────────────────────────────────────────────────────────────────

DRG_MAPPING = {
    "Dengue Fever": "Infectious Disease",
    "Dengue Haemorrhagic Fever": "Infectious Disease",
    "Asthma Exacerbation": "Respiratory",
    "COPD Exacerbation": "Respiratory",
    "Bronchitis": "Respiratory",
    "Pneumonia": "Respiratory",
    "Normal Delivery": "Obstetrics & Gynecology",
    "C-Section": "Obstetrics & Gynecology",
    "CABG": "Cardiovascular",
    "Heart Valve Replacement": "Cardiovascular",
    "Angioplasty with Stent": "Cardiovascular",
    "Knee Replacement": "Orthopedics",
    "Hip Replacement": "Orthopedics",
    "Spinal Surgery": "Orthopedics",
    "Arthroscopy": "Orthopedics",
}

PROCEDURES = list(DRG_MAPPING.keys())
DRG_CATEGORIES = sorted(set(DRG_MAPPING.values()))
REGIONS = ["Central", "East Coast", "East Malaysia", "Northern", "Southern"]
HOSPITAL_TYPES = ["Government", "Private"]
PLAN_TYPES = ["Basic", "Silver", "Gold"]

BRAND = {
    "primary": "#0D3B66",
    "secondary": "#1A7A8A",
    "accent": "#E87722",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "gray": "#6B7280",
    "light": "#F8FAFC",
    "mid": "#E5E7EB",
}

DRG_COLOR_MAP = {
    "Cardiovascular": "#EF4444",
    "Infectious Disease": "#F97316",
    "Obstetrics & Gynecology": "#EC4899",
    "Orthopedics": "#8B5CF6",
    "Respiratory": "#3B82F6",
}

PLAN_COLOR_MAP = {
    "Basic": "#6B7280",
    "Silver": "#94A3B8",
    "Gold": "#F59E0B",
}

REGION_COLOR_MAP = {
    "Central": "#0D3B66",
    "Northern": "#1A7A8A",
    "Southern": "#E87722",
    "East Coast": "#10B981",
    "East Malaysia": "#8B5CF6",
}

POLICY_COLOR_MAP = {
    "Base MHIT": "#6B7280",
    "Alt A – Tiered Cap": "#3B82F6",
    "Alt B – Dynamic DRG": "#10B981",
    "Custom Policy": "#E87722",
}

# ─── Risk Score (mirrors notebook Cell 39) ────────────────────────────────────


def calculate_risk_score(row):
    score = 1.0
    age = row["patient_age"]
    if age > 65:
        score += 0.50
    elif age > 50:
        score += 0.25
    if row["bmi"] >= 30.0:
        score += 0.15
    if str(row.get("smoker_status", "")).lower() == "smoker":
        score += 0.25
    score += row["chronic_conditions"] * 0.20
    return round(score, 2)


def calculate_risk_score_single(age: int, bmi: float, chronic: int, smoker: str) -> float:
    score = 1.0
    if age > 65:
        score += 0.50
    elif age > 50:
        score += 0.25
    if bmi >= 30.0:
        score += 0.15
    if smoker.lower() == "smoker":
        score += 0.25
    score += chronic * 0.20
    return round(score, 2)


def risk_label(score: float):
    if score >= 2.0:
        return "High Risk", BRAND["danger"]
    elif score >= 1.5:
        return "Moderate Risk", BRAND["warning"]
    else:
        return "Low Risk", BRAND["success"]


def cluster_label_fallback(risk_score: float) -> str:
    """Rule-based cluster label used when K-Means model is not loaded."""
    if risk_score < 1.15:
        return "Cluster 0 — Young & Healthy"
    elif risk_score < 1.40:
        return "Cluster 1 — Low-Moderate Risk"
    elif risk_score < 1.70:
        return "Cluster 2 — Moderate Risk"
    elif risk_score < 2.00:
        return "Cluster 3 — High Risk (Chronic)"
    else:
        return "Cluster 4 — Very High Risk"


# ─── ML Model Loading ─────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """
    ═══════════════════════════════════════════════════════════════════════
    TODO: Export your trained models from the notebook and drop them into
          the  models/  folder before launching the app.

    Required files
    ──────────────
    models/severity_model_fmv.pkl   ← statsmodels OLS result (Cell 49)
    models/kmeans_model.pkl         ← sklearn KMeans, k=5   (Cell 41)
    models/kmeans_scaler.pkl        ← sklearn StandardScaler (Cell 41)

    Add these lines at the end of Cell 41 in your notebook:
    ─────────────────────────────────────────────────────────
    import pickle, os
    save_dir = '/content/drive/MyDrive/UMACT 2026/'
    with open(os.path.join(save_dir, 'kmeans_model.pkl'), 'wb') as f:
        pickle.dump(km_final, f)
    with open(os.path.join(save_dir, 'kmeans_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    ═══════════════════════════════════════════════════════════════════════
    """
    model_dir = Path("models")
    paths = {
        "severity": model_dir / "severity_model_fmv.pkl",
        "kmeans": model_dir / "kmeans_model.pkl",
        "scaler": model_dir / "kmeans_scaler.pkl",
    }
    if all(p.exists() for p in paths.values()):
        with open(paths["severity"], "rb") as f:
            severity_model = pickle.load(f)
        with open(paths["kmeans"], "rb") as f:
            kmeans = pickle.load(f)
        with open(paths["scaler"], "rb") as f:
            scaler = pickle.load(f)
        return severity_model, kmeans, scaler
    return None, None, None


def predict_claim(severity_model, kmeans, scaler,
                  age, bmi, chronic, prev_claims, smoker, drg, region, hospital_type):
    """Run a full prediction pipeline for a single patient."""
    risk_score = calculate_risk_score_single(age, bmi, chronic, smoker)
    X_scaled = scaler.transform([[age, bmi, chronic, prev_claims]])
    cluster = int(kmeans.predict(X_scaled)[0])

    new_data = pd.DataFrame({
        "drg_category": [drg],
        "region": [region],
        "hospital_type": [hospital_type],
        "risk_cluster": [cluster],
        "patient_risk_score": [risk_score],
    })

    log_pred = severity_model.predict(new_data)[0]
    smearing_factor = float(np.exp(severity_model.resid).mean())
    predicted_amount = np.exp(log_pred) * smearing_factor
    return predicted_amount, cluster, risk_score


# ─── Data Loading & Feature Engineering ──────────────────────────────────────

@st.cache_data(show_spinner="Loading and engineering features…")
def load_and_engineer_data(uploaded_bytes: bytes | None = None) -> pd.DataFrame | None:
    """
    Load the cleaned CSV, apply all deterministic feature engineering from
    the notebook, and compute pre-calculated policy simulations.

    Accepts either:
      - uploaded_bytes: raw CSV bytes from st.file_uploader
      - auto-loads from  data/UMACT_HACKATHON_2026_CLEANED.csv  if present
    """
    if uploaded_bytes is not None:
        df = pd.read_csv(
            io.BytesIO(uploaded_bytes),
            parse_dates=["admission_date", "discharge_date"],
        )
    else:
        data_path = Path("data/UMACT_HACKATHON_2026_CLEANED.csv")
        if not data_path.exists():
            return None
        df = pd.read_csv(data_path, parse_dates=["admission_date", "discharge_date"])

    # ── Deterministic feature engineering (Cell 39) ──────────────────────
    df["drg_category"] = df["procedure_diagnosis"].map(DRG_MAPPING)
    df["patient_risk_score"] = df.apply(calculate_risk_score, axis=1)
    df["admission_year"] = df["admission_date"].dt.year
    df["admission_month"] = df["admission_date"].dt.month
    df["admission_quarter"] = df["admission_date"].dt.to_period("Q").astype(str)
    df["cost_per_day"] = df["total_claim_amount"] / df["length_of_stay"].replace(0, 1)

    # ── FMV Benchmark (fallback: group median — replaced when model loaded) ─
    fmv = (
        df.groupby(["drg_category", "hospital_type"])["total_claim_amount"]
        .median()
        .reset_index()
        .rename(columns={"total_claim_amount": "expected_fmv_cost"})
    )
    df = df.merge(fmv, on=["drg_category", "hospital_type"], how="left")
    df["cost_variance_pct"] = (df["total_claim_amount"] / df["expected_fmv_cost"]) - 1
    df["hospital_tier"] = np.where(df["cost_variance_pct"] <= 0.15, "Tier 1", "Tier 2")

    # ── Policy simulations (Cell 51) ──────────────────────────────────────
    df["base_mhit_patient_pays"] = np.minimum(df["total_claim_amount"] * 0.20, 3000)
    df["base_mhit_insurer_pays"] = df["total_claim_amount"] - df["base_mhit_patient_pays"]

    def _tiered_copay(row):
        base = row["total_claim_amount"] * 0.20
        cap = 3000 if row["hospital_tier"] == "Tier 1" else 10_000
        return min(base, cap)

    df["alt_a_patient_pays"] = df.apply(_tiered_copay, axis=1)
    df["alt_a_insurer_pays"] = df["total_claim_amount"] - df["alt_a_patient_pays"]

    def _dynamic_drg_copay(row):
        fmv = row["expected_fmv_cost"]
        actual = row["total_claim_amount"]
        base = 0.10 * fmv
        overage = max(0, actual - fmv)
        return min(actual, base + overage)

    df["alt_b_patient_pays"] = df.apply(_dynamic_drg_copay, axis=1)
    df["alt_b_insurer_pays"] = df["total_claim_amount"] - df["alt_b_patient_pays"]

    return df


def simulate_custom_policy(df: pd.DataFrame, copay_pct: float, cap_amount: float):
    patient_pays = np.minimum(df["total_claim_amount"] * copay_pct, cap_amount)
    insurer_pays = df["total_claim_amount"] - patient_pays
    return patient_pays, insurer_pays


# ─── Formatting helpers ───────────────────────────────────────────────────────

def fmt_rm(val: float) -> str:
    if abs(val) >= 1_000_000_000:
        return f"RM {val / 1_000_000_000:.2f}B"
    if abs(val) >= 1_000_000:
        return f"RM {val / 1_000_000:.2f}M"
    return f"RM {val:,.0f}"


def fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


# ─── Shared CSS injected into every page ─────────────────────────────────────

SHARED_CSS = """
<style>
    /* ── Hide Streamlit chrome ── */
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    .stDeployButton { display: none !important; }
    footer { visibility: hidden !important; }

    /* ── Global surface ── */
    .stApp { background-color: #FAFAFA !important; }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1280px;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E8E8E8 !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
    }

    /* ── Typography ── */
    .page-header {
        font-size: 1.5rem; font-weight: 700;
        color: #0A0A0A; margin-bottom: 0.15rem;
        letter-spacing: -0.02em; line-height: 1.25;
    }
    .page-sub {
        font-size: 0.85rem; color: #8C8C8C;
        margin-bottom: 1.5rem; font-weight: 400;
    }
    .section-title {
        font-size: 0.68rem; font-weight: 700; color: #8C8C8C;
        text-transform: uppercase; letter-spacing: 0.1em;
        border-left: 3px solid #0A0A0A;
        padding-left: 0.65rem; margin: 1.6rem 0 0.8rem 0;
    }

    /* ── KPI cards ── */
    .kpi-card {
        background: #FFFFFF;
        padding: 1.2rem 1.1rem; border-radius: 10px;
        border: 1px solid #E8E8E8;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        text-align: center;
    }
    .kpi-card .kpi-label {
        font-size: 0.67rem; color: #8C8C8C;
        text-transform: uppercase; letter-spacing: 0.09em;
        margin-bottom: 0.4rem; font-weight: 600;
    }
    .kpi-card .kpi-value {
        font-size: 1.55rem; font-weight: 700;
        color: #0A0A0A; line-height: 1.1;
    }
    .kpi-card .kpi-delta {
        font-size: 0.67rem; color: #ABABAB;
        margin-top: 0.3rem;
    }

    /* ── Policy cards ── */
    .policy-card {
        background: #FFFFFF;
        border: 1px solid #E8E8E8; border-radius: 10px;
        padding: 1rem; text-align: center; height: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .policy-card.recommended {
        border-color: #0A0A0A; background: #F5F5F5;
    }
    .policy-card .pc-name {
        font-weight: 600; font-size: 0.85rem; color: #3D3D3D;
    }
    .policy-card .pc-amount {
        font-size: 1.4rem; font-weight: 700;
        color: #0A0A0A; margin: 0.3rem 0;
    }
    .policy-card .pc-savings { font-size: 0.75rem; color: #8C8C8C; }

    /* ── Model banners ── */
    .model-banner {
        padding: 0.6rem 1rem; border-radius: 8px;
        font-size: 0.81rem; margin-bottom: 1rem;
    }
    .model-banner.warn {
        background: #FAFAFA; border: 1px solid #D4D4D4; color: #3D3D3D;
    }
    .model-banner.ok {
        background: #F5F5F5; border: 1px solid #D4D4D4; color: #1A1A1A;
    }

    /* ── Risk result card ── */
    .risk-result-card {
        background: #FFFFFF;
        border-radius: 10px; padding: 1.2rem;
        border: 1px solid #E8E8E8;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        text-align: center;
    }

    /* ── Streamlit element overrides ── */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E8E8E8;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetricValue"] { color: #0A0A0A !important; }
    div[data-testid="stMetricLabel"] { color: #8C8C8C !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: #0A0A0A !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background: #3D3D3D !important;
    }
</style>
"""


def inject_css():
    st.markdown(SHARED_CSS, unsafe_allow_html=True)


def data_loader_widget() -> pd.DataFrame | None:
    """
    Try to load data from disk; if absent, show an upload widget.
    Returns the engineered DataFrame or None.
    """
    df = load_and_engineer_data()
    if df is not None:
        return df

    st.warning(
        "**Data file not found.**  \n"
        "Place `UMACT_HACKATHON_2026_CLEANED.csv` inside the `data/` folder, "
        "or upload it below."
    )
    uploaded = st.file_uploader(
        "Upload UMACT_HACKATHON_2026_CLEANED.csv", type="csv", key="csv_upload"
    )
    if uploaded:
        return load_and_engineer_data(uploaded.read())
    return None
