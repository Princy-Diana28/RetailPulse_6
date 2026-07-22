"""RetailPulse — Inventory Optimization."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings as cfg
from src.pipeline import ensure_pipeline_ready
from src.ui_components import kpi_row, page_header, style_fig
from src.utils import format_number

page_header(
    "Inventory Optimization",
    "Stock recommendations for top-selling SKUs, comparing recent vs. prior 30-day demand.",
)

with st.spinner("Loading data..."):
    data = ensure_pipeline_ready()

inv_df = data["inventory_df"]
summary = data["inventory_summary"]

kpi_row([
    {"label": "Increase Stock", "value": format_number(summary["increase_count"]), "positive": True},
    {"label": "Maintain Stock", "value": format_number(summary["maintain_count"])},
    {"label": "Reduce Stock", "value": format_number(summary["reduce_count"]), "positive": False},
    {"label": "Total Suggested Reorder Units", "value": format_number(summary["total_reorder_units"])},
])
st.divider()

action_filter = st.multiselect(
    "Filter by recommended action",
    ["Increase Stock", "Maintain Stock", "Reduce Stock"],
    default=["Increase Stock", "Maintain Stock", "Reduce Stock"],
)
view = inv_df[inv_df["RecommendedAction"].isin(action_filter)]

if view.empty:
    st.warning("No SKUs match the selected filter.")
    st.stop()

color_map = {"Increase Stock": cfg.SUCCESS_COLOR, "Maintain Stock": cfg.WARNING_COLOR, "Reduce Stock": cfg.DANGER_COLOR}

c1, c2 = st.columns([1, 1.4])
with c1:
    counts = view["RecommendedAction"].value_counts().reset_index()
    counts.columns = ["Action", "SKUs"]
    fig = px.pie(counts, names="Action", values="SKUs", hole=0.55, color="Action", color_discrete_map=color_map)
    st.plotly_chart(style_fig(fig, title="Action Distribution"), width="stretch")

with c2:
    top20 = view.reindex(view["PctChange"].abs().sort_values(ascending=False).index).head(20)
    top20 = top20.sort_values("PctChange")
    fig = px.bar(top20, x="PctChange", y="Description", orientation="h", color="RecommendedAction",
                 color_discrete_map=color_map)
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(
        style_fig(fig, title="Demand Change: Recent vs Prior 30 Days", height=460,
                  x_title="% Change", y_title=""),
        width="stretch",
    )

st.subheader("Recommendations Table")
table = view[["StockCode", "Description", "RecentAvgDailyQty", "PriorAvgDailyQty",
              "PctChange", "RecommendedAction", "Confidence", "SuggestedReorderQty"]].copy()
table["PctChange"] = (table["PctChange"] * 100).round(1).astype(str) + "%"
table["Confidence"] = (table["Confidence"] * 100).round(0).astype(int).astype(str) + "%"
st.dataframe(table, width="stretch", hide_index=True)

st.divider()
st.download_button(
    "Download inventory recommendations (CSV)",
    inv_df.to_csv(index=False).encode("utf-8"),
    file_name="retailpulse_inventory_recommendations.csv",
    mime="text/csv",
    icon=":material/download:",
)
