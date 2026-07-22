"""RetailPulse — Customer Analytics."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings as cfg
from src.pipeline import ensure_pipeline_ready
from src.ui_components import kpi_row, page_header, style_fig
from src.utils import format_currency, format_number

page_header("Customer Analytics", "Spend, frequency, and value distribution across the customer base.")

with st.spinner("Loading data..."):
    data = ensure_pipeline_ready()
df = data["featured_df"]

cust = df.groupby("CustomerID").agg(
    TotalSpend=("TotalPrice", "sum"),
    TotalOrders=("Invoice", "nunique"),
    FirstPurchase=("InvoiceDate", "min"),
    LastPurchase=("InvoiceDate", "max"),
    Country=("Country", "first"),
).reset_index()
cust["AvgOrderValue"] = cust["TotalSpend"] / cust["TotalOrders"]

kpi_row([
    {"label": "Total Customers", "value": format_number(cust.shape[0])},
    {"label": "Avg Customer Spend", "value": format_currency(cust["TotalSpend"].mean(), cfg.CURRENCY_SYMBOL)},
    {"label": "Avg Orders / Customer", "value": f"{cust['TotalOrders'].mean():.1f}"},
    {"label": "Repeat Customers", "value": format_number((cust["TotalOrders"] > 1).sum())},
])
st.divider()

c1, c2 = st.columns(2)
with c1:
    freq_capped = cust[cust["TotalOrders"] <= cust["TotalOrders"].quantile(0.99)]
    fig = px.histogram(freq_capped, x="TotalOrders", nbins=30)
    st.plotly_chart(style_fig(fig, title="Customer Purchase Frequency",
                               x_title="Orders per Customer", y_title="Customers"), width="stretch")

with c2:
    spend_capped = cust[cust["TotalSpend"] <= cust["TotalSpend"].quantile(0.97)]
    fig = px.histogram(spend_capped, x="TotalSpend", nbins=40, color_discrete_sequence=[cfg.ACCENT_COLOR])
    st.plotly_chart(style_fig(fig, title="Customer Spend Distribution",
                               x_title=f"Total Spend ({cfg.CURRENCY_SYMBOL})", y_title="Customers"), width="stretch")

c3, c4 = st.columns(2)
with c3:
    top_cust = cust.sort_values("TotalSpend", ascending=False).head(15)
    fig = px.bar(top_cust, x="TotalSpend", y="CustomerID", orientation="h")
    fig.update_yaxes(categoryorder="total ascending", type="category")
    st.plotly_chart(style_fig(fig, title="Top 15 Customers by Spend",
                               x_title=f"Total Spend ({cfg.CURRENCY_SYMBOL})", y_title="Customer ID"), width="stretch")

with c4:
    by_country = cust.groupby("Country", observed=True)["CustomerID"].nunique().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(by_country, x="CustomerID", y="Country", orientation="h", color_discrete_sequence=[cfg.SUCCESS_COLOR])
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(style_fig(fig, title="Customers by Country (Top 10)",
                               x_title="Customers", y_title=""), width="stretch")

clv = df.drop_duplicates("CustomerID")[["CustomerID", "CustomerLifetimeValue", "PurchaseFrequency", "CustomerTotalSpend"]]
clv = clv[clv["CustomerLifetimeValue"] <= clv["CustomerLifetimeValue"].quantile(0.97)]
fig = px.scatter(clv, x="PurchaseFrequency", y="CustomerLifetimeValue",
                  size="CustomerTotalSpend", opacity=0.6, color_discrete_sequence=[cfg.PRIMARY_COLOR])
st.plotly_chart(
    style_fig(fig, title="Customer Lifetime Value vs Purchase Frequency", height=460,
              x_title="Purchase Frequency (orders)", y_title=f"Lifetime Value ({cfg.CURRENCY_SYMBOL})"),
    width="stretch",
)

st.divider()
st.download_button(
    "Download customer-level summary (CSV)",
    cust.to_csv(index=False).encode("utf-8"),
    file_name="retailpulse_customer_summary.csv",
    mime="text/csv",
    icon=":material/download:",
)
