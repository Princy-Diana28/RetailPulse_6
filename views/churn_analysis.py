"""RetailPulse — Churn Analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings as cfg
from src.pipeline import ensure_pipeline_ready
from src.ui_components import kpi_row, page_header, style_fig
from src.utils import format_number, format_percent

page_header(
    "Churn Analysis",
    f"Customers are flagged Churned after {cfg.CHURN_INACTIVITY_DAYS} days of inactivity. "
    "A Random Forest model predicts forward-looking churn probability from behavioral features.",
)

with st.spinner("Loading data..."):
    data = ensure_pipeline_ready()

churn_df = data["churn_df"]

active_n = (churn_df["ChurnRiskLabel"] == "Active").sum()
at_risk_n = (churn_df["ChurnRiskLabel"] == "At Risk").sum()
churned_n = (churn_df["ChurnRiskLabel"] == "Churned").sum()

kpi_row([
    {"label": "Active Customers", "value": format_number(active_n)},
    {"label": "At Risk", "value": format_number(at_risk_n)},
    {"label": "Churned", "value": format_number(churned_n)},
    {"label": "Churn Rate", "value": format_percent(churned_n / len(churn_df) if len(churn_df) else 0)},
])
st.divider()

color_map = {"Active": cfg.SUCCESS_COLOR, "At Risk": cfg.WARNING_COLOR, "Churned": cfg.DANGER_COLOR}

c1, c2 = st.columns(2)
with c1:
    status_counts = churn_df["ChurnRiskLabel"].value_counts().reset_index()
    status_counts.columns = ["Status", "Customers"]
    fig = px.pie(status_counts, names="Status", values="Customers", hole=0.55,
                 color="Status", color_discrete_map=color_map)
    st.plotly_chart(style_fig(fig, title="Churn Status Breakdown"), width="stretch")

with c2:
    fig = px.histogram(churn_df, x="Recency", color="ChurnRiskLabel", nbins=40, barmode="overlay",
                        color_discrete_map=color_map, opacity=0.65)
    st.plotly_chart(style_fig(fig, title="Recency Distribution by Status",
                               x_title="Days Since Last Purchase", y_title="Customers"), width="stretch")

st.subheader("Highest Churn-Risk Customers")
at_risk_table = churn_df[churn_df["ChurnRiskLabel"] != "Churned"].sort_values(
    "ChurnProbability", ascending=False
).head(20)[["CustomerID", "Recency", "Frequency", "Monetary", "ChurnProbability", "ChurnRiskLabel"]].copy()
at_risk_table["ChurnProbability"] = (at_risk_table["ChurnProbability"] * 100).round(1).astype(str) + "%"
st.dataframe(at_risk_table, width="stretch", hide_index=True)

st.divider()
st.download_button(
    "Download churn analysis results (CSV)",
    churn_df.to_csv(index=False).encode("utf-8"),
    file_name="retailpulse_churn_analysis.csv",
    mime="text/csv",
    icon=":material/download:",
)
