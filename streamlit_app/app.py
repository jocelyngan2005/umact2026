"""
MHIT Insurance Intelligence Platform
Entry point — defines navigation (app itself is excluded from the nav).
"""

import streamlit as st

st.set_page_config(
    page_title="MHIT Intelligence Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation(
    [
        st.Page("pages/1_Executive_Dashboard.py",  title="Executive Dashboard",  icon="📊"),
        st.Page("pages/2_Claim_Risk_Profiler.py",  title="Claim Risk Profiler",  icon="🎯"),
        st.Page("pages/3_Policy_Simulator.py",     title="Policy Simulator",     icon="⚖️"),
        st.Page("pages/4_Portfolio_Analytics.py",  title="Portfolio Analytics",  icon="📈"),
        st.Page("pages/5_Medical_Trend_Monitor.py",title="Medical Trend Monitor",icon="📅"),
    ],
    position="sidebar",
)
pg.run()
