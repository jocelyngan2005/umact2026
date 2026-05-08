# MHIT Insurance Intelligence Platform
**UMACT Hackathon 2026 — Finals Submission**

An end-to-end actuarial analytics and policy decision tool built on 20,000 Malaysian health insurance claims. Features live ML-powered risk scoring, interactive policy simulation, hospital efficiency benchmarking, and medical trend analysis — all served as a multi-page Streamlit dashboard.

---

## Project Structure

```
umact2026/
├── dashboard/
│   ├── app.py                          # Entry point — Streamlit navigation
│   ├── utils.py                        # Shared constants, data loading, ML helpers, formatting
│   ├── requirements.txt
│   ├── .streamlit/
│   │   └── config.toml                 # Theme & server config
│   ├── data/
│   │   └── UMACT_HACKATHON_2026_CLEANED.csv   # 20,000 health insurance claims
│   ├── models/
│   │   ├── severity_model_fmv.pkl      # Lognormal OLS severity model (statsmodels)
│   │   ├── kmeans_model.pkl            # K-Means risk clustering (k=5, sklearn)
│   │   └── kmeans_scaler.pkl           # StandardScaler for K-Means features
│   └── pages/
│       ├── 1_Executive_Dashboard.py
│       ├── 2_Claim_Risk_Profiler.py
│       ├── 3_Policy_Simulator.py
│       ├── 4_Portfolio_Analytics.py
│       └── 5_Medical_Trend_Monitor.py
└── UMACTHackathon_PreRound_BUZHIDAO.ipynb   # Analysis & model training notebook
```

---

## Pages

### 1. Executive Dashboard
High-level KPI overview with interactive sidebar filters (year, plan type, region, DRG category).

- **KPIs:** total claims, average claim amount, cap hit rate, insurer liability, average length of stay
- Monthly claim volume & severity trend (bar + line dual-axis)
- DRG category spend breakdown (donut chart)
- Claim severity distribution with lognormal fit overlay and RM 3k cap marker
- Regional average claim comparison
- Claim severity by plan type (box plots) and top diagnoses by average cost

### 2. Claim Risk Profiler
Real-time, single-patient risk scoring and claim prediction using the trained ML pipeline.

- Input form: age, gender, BMI, smoker status, chronic conditions, prior claims, procedure, hospital type, region, plan type
- **Risk score gauge** (Low / Moderate / High)
- Expected claim prediction via Lognormal OLS + Duan's smearing correction
- K-Means cluster assignment (5 risk segments)
- Premium loading recommendation table
- Percentile rank vs. historical claims for the same DRG and hospital type

### 3. Policy Co-Payment Simulator
Side-by-side comparison of four co-payment structures applied to the full claim portfolio.

| Scenario | Structure |
|---|---|
| Base MHIT | 20% co-pay, RM 3,000 cap |
| Alt A – Tiered Cap | RM 3k cap (Tier 1 hospitals) / RM 10k cap (Tier 2 hospitals) |
| Alt B – Dynamic DRG *(recommended)* | 10% of FMV benchmark + 100% of any overage above FMV |
| Custom Policy | User-defined co-pay % and cap via sliders |

- Insurer liability cards per scenario with savings vs. baseline
- Stacked bar: insurer vs. patient split (RM millions)
- Patient payment distribution overlay (all scenarios)
- Full financial comparison table

### 4. Portfolio Risk Analytics
Portfolio-level risk segmentation and hospital efficiency analysis.

- Loss ratio heatmap: plan type × region (average insurer payment)
- DRG average claim breakdown
- Hospital efficiency scatter: actual vs. FMV benchmark, coloured by tier (Tier 1 = efficient, Tier 2 = inefficient)
- Patient risk cluster profiling: age × BMI scatter, cluster summary table
- Admission type and hospital type cost analysis

### 5. Medical Trend Monitor
Time-series inflation and seasonality analysis across the 2023–2025 claim period.

- Quarterly claim volume & severity trend (dual-axis)
- Per-DRG average claim trajectory
- Average length of stay by DRG
- Monthly seasonality heatmap (year × month claim volume)
- Year-on-year medical inflation table by DRG category

---

## ML Models

### Severity Model (`severity_model_fmv.pkl`)
- **Type:** Lognormal OLS (statsmodels)
- **Formula:** `log(claim_amount) ~ DRG_category + region + hospital_type + risk_cluster + patient_risk_score`
- **Performance:** R² = 0.981
- **Correction:** Duan's smearing factor applied on re-transformation to correct for log-space bias

### Risk Clustering (`kmeans_model.pkl` + `kmeans_scaler.pkl`)
- **Type:** K-Means, k = 5 (sklearn)
- **Features:** age, BMI, chronic conditions, prior claims history
- **Zero data leakage** — trained on ex-ante patient features only

### Risk Score Formula
```
score = 1.0
      + 0.50  if age > 65  (else +0.25 if age > 50)
      + 0.15  if BMI ≥ 30
      + 0.25  if Smoker
      + 0.20 × number_of_chronic_conditions
```

| Score range | Risk tier |
|---|---|
| < 1.5 | Low Risk |
| 1.5 – 2.0 | Moderate Risk |
| ≥ 2.0 | High Risk |

---

## Setup & Running

### Prerequisites
- Python 3.10+
- Recommended: create and activate a virtual environment

### Install dependencies
```bash
cd dashboard
pip install -r requirements.txt
```

### Run the dashboard
```bash
cd dashboard
streamlit run app.py
```

The app auto-loads `data/UMACT_HACKATHON_2026_CLEANED.csv` on startup. If the file is not present, a file-uploader widget is shown on each page so you can supply the CSV manually.

### Model files
The three `.pkl` files in `models/` are required for ML-powered predictions on the Claim Risk Profiler page. If they are absent, the profiler falls back to a group-median × risk score benchmark estimate.

---

## Data

| Field | Description |
|---|---|
| `claim_id` | Unique claim identifier |
| `admission_date` / `discharge_date` | Hospital stay dates |
| `patient_age`, `bmi`, `smoker_status` | Patient demographics |
| `chronic_conditions` | Number of chronic conditions |
| `procedure_diagnosis` | Clinical procedure (15 types across 5 DRG categories) |
| `hospital_name`, `hospital_type` | Hospital identifier and type (Government / Private) |
| `region` | Central, Northern, Southern, East Coast, East Malaysia |
| `plan_type` | Basic, Silver, Gold |
| `total_claim_amount`, `insurance_paid`, `length_of_stay` | Claim financials |

**DRG categories:** Cardiovascular · Infectious Disease · Obstetrics & Gynecology · Orthopedics · Respiratory

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| streamlit | ≥ 1.35 | Dashboard framework |
| pandas | ≥ 2.1 | Data manipulation |
| numpy | ≥ 1.26 | Numerical operations |
| plotly | ≥ 5.20 | Interactive charts |
| scipy | ≥ 1.12 | Statistical distributions & fitting |
| scikit-learn | ≥ 1.4 | K-Means clustering |
| statsmodels | ≥ 0.14 | OLS severity model |
