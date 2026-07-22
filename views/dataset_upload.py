"""RetailPulse — Dataset Upload."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pipeline import load_uploaded_dataset
from src.ui_components import page_header, section_label

page_header(
    "Dataset Upload",
    "Swap in your own retail transaction export and the entire pipeline re-runs against it.",
)

st.markdown(
    "RetailPulse ships with the **Online Retail II** dataset pre-loaded, so every page works "
    "out of the box. Upload a file below to re-run cleaning, feature engineering, segmentation, "
    "forecasting, churn, and inventory optimization against your own data."
)

st.info(
    "Expected columns (Online Retail / Online Retail II schema): "
    "**Invoice, StockCode, Description, Quantity, InvoiceDate, Price (or UnitPrice), "
    "Customer ID (or CustomerID), Country**",
    icon=":material/info:",
)

uploaded_file = st.file_uploader("Upload a retail transactions file", type=["csv", "xlsx", "xls"])

col_a, col_b = st.columns(2)
with col_a:
    if uploaded_file is not None:
        if st.button("Use this file for the whole app", type="primary", icon=":material/check:"):
            st.session_state["uploaded_file_bytes"] = uploaded_file.getvalue()
            st.session_state["uploaded_file_name"] = uploaded_file.name
            st.cache_data.clear()
            st.success(f"Now using **{uploaded_file.name}** across all pages.")
            st.rerun()
with col_b:
    if "uploaded_file_bytes" in st.session_state:
        if st.button("Revert to bundled dataset", icon=":material/undo:"):
            del st.session_state["uploaded_file_bytes"]
            del st.session_state["uploaded_file_name"]
            st.cache_data.clear()
            st.rerun()

st.divider()

if uploaded_file is not None:
    try:
        preview_df = load_uploaded_dataset(uploaded_file.getvalue(), uploaded_file.name)

        section_label("Preview")
        st.dataframe(preview_df.head(20), width="stretch")

        section_label("Summary Statistics")
        numeric_df = preview_df.select_dtypes(include="number")
        categorical_df = preview_df.select_dtypes(exclude="number")

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Numeric columns")
            if not numeric_df.empty:
                st.dataframe(
                    numeric_df.describe().transpose().round(2).fillna("--"),
                    width="stretch",
                )
            else:
                st.caption("No numeric columns detected.")
        with c2:
            st.caption("Categorical columns")
            if not categorical_df.empty:
                cat_summary = categorical_df.describe().transpose().fillna("--")
                st.dataframe(cat_summary, width="stretch")
            else:
                st.caption("No categorical columns detected.")

        st.caption(f"{len(preview_df):,} rows x {len(preview_df.columns)} columns detected.")
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
else:
    section_label("Currently Active Dataset")
    if "uploaded_file_bytes" in st.session_state:
        st.write(f"**{st.session_state['uploaded_file_name']}** (uploaded)")
    else:
        st.write(
            "**Online Retail II** (bundled) — UK-based online retailer, Dec 2009 - Dec 2011, "
            "~1.07M raw transaction lines across 43 countries."
        )
