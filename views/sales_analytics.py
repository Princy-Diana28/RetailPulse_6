"""RetailPulse — Sales Analytics."""

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

page_header("Sales Analytics", "Revenue trends, product performance, and market breakdown.")

with st.spinner("Loading data..."):
    data = ensure_pipeline_ready()
df = data["featured_df"]

with st.expander("Filters", icon=":material/tune:"):
    countries = st.multiselect("Country", sorted(df["Country"].unique()), default=[])
    date_min, date_max = df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date()
    date_range = st.date_input("Date range", value=(date_min, date_max),
                                min_value=date_min, max_value=date_max)

filtered = df.copy()
if countries:
    filtered = filtered[filtered["Country"].isin(countries)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        (filtered["InvoiceDate"].dt.date >= start) & (filtered["InvoiceDate"].dt.date <= end)
    ]

if filtered.empty:
    st.warning("No transactions match the selected filters.")
    st.stop()

revenue = filtered["TotalPrice"].sum()
orders = filtered["Invoice"].nunique()
aov = revenue / orders if orders else 0
units = filtered["Quantity"].sum()

kpi_row([
    {"label": "Revenue (filtered)", "value": format_currency(revenue, cfg.CURRENCY_SYMBOL)},
    {"label": "Orders", "value": format_number(orders)},
    {"label": "Avg Order Value", "value": format_currency(aov, cfg.CURRENCY_SYMBOL)},
    {"label": "Units Sold", "value": format_number(units)},
])
st.divider()

c1, c2 = st.columns(2)

with c1:
    monthly = filtered.groupby("YearMonth", observed=True)["TotalPrice"].sum().reset_index()
    fig = px.line(monthly, x="YearMonth", y="TotalPrice", markers=True)
    st.plotly_chart(style_fig(fig, title="Monthly Revenue Trend", x_title="Month",
                               y_title=f"Revenue ({cfg.CURRENCY_SYMBOL})"), width="stretch")

with c2:
    by_country = filtered.groupby("Country", observed=True)["TotalPrice"].sum().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(by_country, x="TotalPrice", y="Country", orientation="h")
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(style_fig(fig, title="Sales by Country (Top 10)",
                               x_title=f"Revenue ({cfg.CURRENCY_SYMBOL})", y_title=""), width="stretch")

c3, c4 = st.columns(2)

with c3:
    top_rev = filtered.groupby("Description", observed=True)["TotalPrice"].sum().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(top_rev, x="TotalPrice", y="Description", orientation="h")
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(style_fig(fig, title="Top 10 Products by Revenue",
                               x_title=f"Revenue ({cfg.CURRENCY_SYMBOL})", y_title=""), width="stretch")

with c4:
    top_qty = filtered.groupby("Description", observed=True)["Quantity"].sum().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(top_qty, x="Quantity", y="Description", orientation="h", color_discrete_sequence=[cfg.ACCENT_COLOR])
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(style_fig(fig, title="Top 10 Products by Quantity",
                               x_title="Units Sold", y_title=""), width="stretch")

c5, c6 = st.columns(2)

with c5:
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    # fill_value=0 avoids NaN bars/tooltips if a filtered slice has zero
    # orders on a given weekday (e.g. a single-country filter).
    dow = filtered.groupby("DayOfWeek", observed=True)["Invoice"].nunique().reindex(dow_order, fill_value=0).reset_index()
    dow.columns = ["DayOfWeek", "Orders"]
    fig = px.bar(dow, x="DayOfWeek", y="Orders")
    st.plotly_chart(style_fig(fig, title="Orders by Day of Week", x_title="", y_title="Orders"), width="stretch")

with c6:
    season = filtered.groupby("Season", observed=True)["TotalPrice"].sum().reset_index()
    fig = px.pie(season, names="Season", values="TotalPrice", hole=0.55)
    st.plotly_chart(style_fig(fig, title="Seasonal Sales Distribution"), width="stretch")

c7, c8 = st.columns(2)

with c7:
    rev_capped = filtered[filtered["TotalPrice"] < filtered["TotalPrice"].quantile(0.99)]
    fig = px.histogram(rev_capped, x="TotalPrice", nbins=50)
    st.plotly_chart(style_fig(fig, title="Revenue Distribution (per line item)",
                               x_title=f"Revenue ({cfg.CURRENCY_SYMBOL})", y_title="Line Items"), width="stretch")

with c8:
    qty_capped = filtered[filtered["Quantity"] < filtered["Quantity"].quantile(0.99)]
    fig = px.histogram(qty_capped, x="Quantity", nbins=50, color_discrete_sequence=[cfg.WARNING_COLOR])
    st.plotly_chart(style_fig(fig, title="Quantity Distribution (per line item)",
                               x_title="Quantity", y_title="Line Items"), width="stretch")

numeric_cols = ["Quantity", "UnitPrice", "TotalPrice", "InvoiceTotalValue",
                 "CustomerLifetimeValue", "PurchaseFrequency", "AverageOrderValue"]
corr = filtered[numeric_cols].corr()
fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Purples", aspect="auto")
st.plotly_chart(style_fig(fig, title="Correlation Heatmap", height=460), width="stretch")

st.divider()
st.download_button(
    "Download filtered transactions (CSV)",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name="retailpulse_sales_filtered.csv",
    mime="text/csv",
    icon=":material/download:",
)
