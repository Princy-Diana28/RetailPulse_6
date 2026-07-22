"""RetailPulse — Home / Overview."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings as cfg
from src.pipeline import ensure_pipeline_ready
from src.ui_components import brand_hero, kpi_row, page_header, render_insights, style_fig
from src.utils import format_currency, format_number, format_percent

brand_hero("AI-Powered Demand & Inventory Intelligence Platform")

page_header(
    "Overview",
    "A single view of revenue, customers, forecasted demand, and churn risk across the business.",
)

with st.spinner("Running the analytics pipeline..."):
    data = ensure_pipeline_ready()

featured = data["featured_df"]
churn_df = data["churn_df"]
fc_summary = data["forecast_summary"]

total_revenue = featured["TotalPrice"].sum()
total_orders = featured["Invoice"].nunique()
total_customers = featured["CustomerID"].nunique()
avg_order_value = total_revenue / total_orders if total_orders else 0
top_country = featured.groupby("Country", observed=True)["TotalPrice"].sum().idxmax()
churn_rate = (churn_df["ChurnRiskLabel"] == "Churned").mean() if len(churn_df) else 0

kpi_row([
    {"label": "Total Revenue", "value": format_currency(total_revenue, cfg.CURRENCY_SYMBOL)},
    {"label": "Total Orders", "value": format_number(total_orders)},
    {"label": "Total Customers", "value": format_number(total_customers)},
    {"label": "Avg Order Value", "value": format_currency(avg_order_value, cfg.CURRENCY_SYMBOL)},
])
st.write("")
kpi_row([
    {"label": "Top Market", "value": top_country},
    {"label": "Countries Served", "value": format_number(featured["Country"].nunique())},
    {
        "label": f"{cfg.FORECAST_HORIZON_DAYS}-Day Forecasted Revenue",
        "value": format_currency(fc_summary.get("forecast_total", 0), cfg.CURRENCY_SYMBOL),
        "delta": f"{fc_summary.get('pct_change_vs_trailing', 0)*100:+.1f}% vs trailing avg",
        "positive": fc_summary.get("pct_change_vs_trailing", 0) >= 0,
    },
    {"label": "Customer Churn Rate", "value": format_percent(churn_rate)},
])

st.write("")
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Monthly Revenue Trend")
    monthly = featured.groupby("YearMonth", observed=True)["TotalPrice"].sum().reset_index()
    fig = px.area(monthly, x="YearMonth", y="TotalPrice")
    fig.update_traces(line_color=cfg.PRIMARY_COLOR, fillcolor="rgba(99,102,241,0.22)")
    st.plotly_chart(
        style_fig(fig, height=360, x_title="Month", y_title=f"Revenue ({cfg.CURRENCY_SYMBOL})"),
        width="stretch",
    )

with col2:
    st.subheader("Revenue by Segment")
    seg_rev = data["rfm"].groupby("Segment")["Monetary"].sum().reset_index().sort_values("Monetary")
    fig2 = px.bar(seg_rev, x="Monetary", y="Segment", orientation="h")
    st.plotly_chart(
        style_fig(fig2, height=360, x_title=f"Revenue ({cfg.CURRENCY_SYMBOL})", y_title="Segment"),
        width="stretch",
    )

st.divider()
st.subheader("Top Business Insights")
render_insights(data["insights"][:4])

st.caption(
    "Use the sidebar to explore the full breakdown — sales analytics, customer segmentation, "
    "demand forecasting, churn risk, inventory recommendations, and model performance."
)
