"""
RetailPulse — Pipeline orchestrator.

Wraps the individual src/ modules into cached, Streamlit-friendly functions.
Every stage is memoized with @st.cache_data / @st.cache_resource keyed on
the underlying data, so navigating between pages never re-runs cleaning,
clustering, or model training unless the source data actually changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings as cfg
from src.churn import analyze_churn
from src.feature_engineering import engineer_features
from src.forecasting import forecast_demand, forecast_summary
from src.insights import generate_insights
from src.inventory import inventory_summary, optimize_inventory
from src.preprocessing import clean_dataset
from src.segmentation import segment_customers, segment_summary
from src.utils import get_logger, load_raw_dataset

logger = get_logger(__name__)


_DEFAULT_PARQUET_CACHE = cfg.RAW_DATA_DIR / "online_retail_ii_combined.parquet"


@st.cache_data(show_spinner=False)
def load_default_dataset() -> pd.DataFrame:
    """
    Loads the bundled Online Retail II dataset. Prefers a cached Parquet
    snapshot (~85% smaller, loads in a fraction of the time) over
    re-parsing the raw multi-sheet .xlsx export on every cold start. The
    cache is created automatically the first time the raw file is parsed.
    """
    if _DEFAULT_PARQUET_CACHE.exists():
        return pd.read_parquet(_DEFAULT_PARQUET_CACHE)

    path = cfg.RAW_DATA_DIR / cfg.DEFAULT_RAW_FILENAME
    df = load_raw_dataset(path)
    try:
        df.to_parquet(_DEFAULT_PARQUET_CACHE, index=False)
    except Exception:
        pass  # caching is a best-effort optimization, not required for correctness
    return df


@st.cache_data(show_spinner=False)
def load_uploaded_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    import io
    buffer = io.BytesIO(file_bytes)
    buffer.name = filename  # load_raw_dataset inspects .name for extension routing
    return load_raw_dataset(buffer)


@st.cache_data(show_spinner=False)
def run_preprocessing(raw_df: pd.DataFrame):
    """
    Cleaning + feature engineering combined into a single cached stage.
    The intermediate cleaned-but-not-yet-featured dataframe is never
    surfaced in the UI, so keeping it as its own long-lived cache entry
    only doubled memory usage for no benefit — this merges the two steps
    so only one large dataframe (featured_df) is retained per dataset.
    """
    clean_df, report = clean_dataset(raw_df)
    featured_df = engineer_features(clean_df)
    del clean_df  # drop the intermediate copy explicitly once no longer needed
    return featured_df, report


@st.cache_data(show_spinner=False)
def run_segmentation(featured_df: pd.DataFrame, n_clusters: int = cfg.RFM_N_CLUSTERS):
    rfm, silhouette = segment_customers(featured_df, n_clusters=n_clusters)
    summary = segment_summary(rfm)
    return rfm, silhouette, summary


@st.cache_data(show_spinner=False)
def run_forecasting(featured_df: pd.DataFrame, horizon_days: int = cfg.FORECAST_HORIZON_DAYS,
                     use_prophet: bool = False):
    forecast, engine = forecast_demand(featured_df, horizon_days=horizon_days, use_prophet=use_prophet)
    summary = forecast_summary(forecast)
    return forecast, engine, summary


@st.cache_data(show_spinner=False)
def run_churn_analysis(featured_df: pd.DataFrame):
    churn_df, metrics = analyze_churn(featured_df)
    return churn_df, metrics


@st.cache_data(show_spinner=False)
def run_inventory_optimization(featured_df: pd.DataFrame, top_n: int = 50):
    inv_df = optimize_inventory(featured_df, top_n_products=top_n)
    summary = inventory_summary(inv_df)
    return inv_df, summary


@st.cache_data(show_spinner=False)
def run_insights(featured_df: pd.DataFrame, rfm: pd.DataFrame, inv_df: pd.DataFrame):
    return generate_insights(featured_df, rfm, inv_df)


def get_active_raw_dataset() -> tuple[pd.DataFrame, str]:
    """
    Returns the raw DataFrame currently active for the session: an
    uploaded file if the user provided one, otherwise the bundled
    Online Retail II dataset.
    """
    uploaded = st.session_state.get("uploaded_file_bytes")
    if uploaded is not None:
        df = load_uploaded_dataset(uploaded, st.session_state.get("uploaded_file_name", "upload.csv"))
        return df, st.session_state.get("uploaded_file_name", "Uploaded file")
    return load_default_dataset(), cfg.DEFAULT_RAW_FILENAME


def ensure_pipeline_ready():
    """
    Runs (or fetches from cache) every pipeline stage for the currently
    active dataset and returns a single dict bag of results, so pages can
    do: `data = ensure_pipeline_ready()` and read what they need.
    """
    raw_df, source_name = get_active_raw_dataset()
    featured_df, clean_report = run_preprocessing(raw_df)
    rfm, silhouette, segment_summary_df = run_segmentation(featured_df)
    forecast_df, forecast_engine, forecast_summary_dict = run_forecasting(featured_df)
    churn_df, churn_metrics = run_churn_analysis(featured_df)
    inv_df, inv_summary = run_inventory_optimization(featured_df)
    insights = run_insights(featured_df, rfm, inv_df)

    return {
        "source_name": source_name,
        "clean_report": clean_report,
        "featured_df": featured_df,
        "rfm": rfm,
        "silhouette": silhouette,
        "segment_summary": segment_summary_df,
        "forecast_df": forecast_df,
        "forecast_engine": forecast_engine,
        "forecast_summary": forecast_summary_dict,
        "churn_df": churn_df,
        "churn_metrics": churn_metrics,
        "inventory_df": inv_df,
        "inventory_summary": inv_summary,
        "insights": insights,
    }
