"""RetailPulse — Business Insights."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pipeline import ensure_pipeline_ready
from src.ui_components import page_header, render_insights

page_header("Business Insights", "Automatically generated from the current dataset — no manual curation.")

with st.spinner("Generating insights..."):
    data = ensure_pipeline_ready()

insights = data["insights"]

categories = sorted(set(i["category"] for i in insights))
tabs = st.tabs(["All"] + categories)

with tabs[0]:
    render_insights(insights, show_category=True)

for tab, cat in zip(tabs[1:], categories):
    with tab:
        render_insights([i for i in insights if i["category"] == cat])

st.divider()
insights_df = pd.DataFrame(insights)
st.download_button(
    "Download insights (CSV)",
    insights_df.to_csv(index=False).encode("utf-8"),
    file_name="retailpulse_business_insights.csv",
    mime="text/csv",
    icon=":material/download:",
)
