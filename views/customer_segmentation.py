"""RetailPulse — Customer Segmentation (RFM + KMeans)."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings as cfg
from src.pipeline import ensure_pipeline_ready, run_segmentation
from src.ui_components import kpi_row, page_header, style_fig
from src.utils import format_currency, format_number

page_header(
    "Customer Segmentation",
    "RFM analysis clustered with KMeans, labeled automatically by relative segment value.",
)

with st.spinner("Loading data..."):
    data = ensure_pipeline_ready()

n_clusters = st.slider("Number of segments (KMeans clusters)", min_value=3, max_value=6,
                        value=cfg.RFM_N_CLUSTERS)

if n_clusters == cfg.RFM_N_CLUSTERS:
    rfm, silhouette, summary = data["rfm"], data["silhouette"], data["segment_summary"]
else:
    with st.spinner("Re-clustering..."):
        rfm, silhouette, summary = run_segmentation(data["featured_df"], n_clusters=n_clusters)

kpi_row([
    {"label": "Segmented Customers", "value": format_number(len(rfm))},
    {"label": "Segments", "value": str(n_clusters)},
    {"label": "Silhouette Score", "value": f"{silhouette:.3f}" if silhouette else "n/a"},
    {"label": "Top Segment Revenue Share", "value": f"{summary['PctOfRevenue'].max():.1f}%"},
])
st.divider()

c1, c2 = st.columns([1.3, 1])
with c1:
    fig = px.scatter(rfm, x="Recency", y="Monetary", color="Segment", opacity=0.65,
                      hover_data=["Frequency"])
    st.plotly_chart(
        style_fig(fig, title="Segment Map — Recency vs Monetary", height=440,
                  x_title="Recency (days since last purchase)", y_title=f"Monetary ({cfg.CURRENCY_SYMBOL})"),
        width="stretch",
    )

with c2:
    fig = px.pie(summary, names="Segment", values="Customers", hole=0.55)
    st.plotly_chart(style_fig(fig, title="Customers per Segment", height=440), width="stretch")

st.subheader("Segment Summary")
display_summary = summary.copy()
display_summary["AvgRecencyDays"] = display_summary["AvgRecencyDays"].round(1)
display_summary["AvgFrequency"] = display_summary["AvgFrequency"].round(1)
display_summary["AvgMonetary"] = display_summary["AvgMonetary"].apply(lambda v: format_currency(v, cfg.CURRENCY_SYMBOL))
display_summary["TotalMonetary"] = display_summary["TotalMonetary"].apply(lambda v: format_currency(v, cfg.CURRENCY_SYMBOL))
display_summary["PctOfCustomers"] = display_summary["PctOfCustomers"].round(1).astype(str) + "%"
display_summary["PctOfRevenue"] = display_summary["PctOfRevenue"].round(1).astype(str) + "%"
st.dataframe(display_summary, width="stretch", hide_index=True)

with st.expander("What do these segments mean?", icon=":material/help:"):
    st.markdown("""
- **Champions** — bought recently, buy often, spend the most. Reward and retain them.
- **Loyal Customers** — consistent buyers with solid spend; good upsell targets.
- **Potential Loyalists** — recent, moderate activity; nurture toward loyalty.
- **Needs Attention** — below-average recency/frequency/value; re-engage with targeted offers.
- **At Risk** — were valuable, now going quiet; win-back campaigns recommended.
- **Hibernating / Lost** — long inactive, low historical value; low-cost reactivation only.
    """)

st.divider()
st.download_button(
    "Download customer segmentation results (CSV)",
    rfm.to_csv(index=False).encode("utf-8"),
    file_name="retailpulse_customer_segments.csv",
    mime="text/csv",
    icon=":material/download:",
)
