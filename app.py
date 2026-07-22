"""
RetailPulse -- AI-Powered Demand & Inventory Intelligence Platform
Router / entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from config import settings as cfg
from src.ui_components import inject_global_css, sidebar_brand

st.set_page_config(
    page_title=cfg.APP_TITLE,
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()

VIEWS_DIR = Path(__file__).resolve().parent / "views"

home_page = st.Page(str(VIEWS_DIR / "home.py"), title="Home",
                     icon=":material/space_dashboard:", default=True)
dataset_upload_page = st.Page(str(VIEWS_DIR / "dataset_upload.py"), title="Dataset Upload",
                               icon=":material/upload_file:")
sales_page = st.Page(str(VIEWS_DIR / "sales_analytics.py"), title="Sales Analytics",
                      icon=":material/monitoring:")
customer_page = st.Page(str(VIEWS_DIR / "customer_analytics.py"), title="Customer Analytics",
                         icon=":material/groups:")
segmentation_page = st.Page(str(VIEWS_DIR / "customer_segmentation.py"), title="Customer Segmentation",
                             icon=":material/donut_small:")
forecasting_page = st.Page(str(VIEWS_DIR / "demand_forecasting.py"), title="Demand Forecasting",
                            icon=":material/trending_up:")
churn_page = st.Page(str(VIEWS_DIR / "churn_analysis.py"), title="Churn Analysis",
                      icon=":material/report_problem:")
inventory_page = st.Page(str(VIEWS_DIR / "inventory_optimization.py"), title="Inventory Optimization",
                          icon=":material/inventory_2:")
insights_page = st.Page(str(VIEWS_DIR / "business_insights.py"), title="Business Insights",
                         icon=":material/lightbulb:")
performance_page = st.Page(str(VIEWS_DIR / "model_performance.py"), title="Model Performance",
                            icon=":material/speed:")
about_page = st.Page(str(VIEWS_DIR / "about_project.py"), title="About Project",
                      icon=":material/info:")

with st.sidebar:
    sidebar_brand()

pg = st.navigation(
    {
        "Overview": [home_page, dataset_upload_page],
        "Analytics": [sales_page, customer_page, segmentation_page],
        "AI Models": [forecasting_page, churn_page, inventory_page],
        "Reporting": [insights_page, performance_page],
        "Platform": [about_page],
    },
    position="sidebar",
)

with st.sidebar:
    st.divider()
    if "uploaded_file_bytes" in st.session_state:
        st.success(f"Using: {st.session_state.get('uploaded_file_name')}", icon=":material/check_circle:")
    else:
        st.info("Using bundled Online Retail II dataset", icon=":material/database:")
    st.caption("RetailPulse v1.0 -- AI Retail Intelligence")

pg.run()
