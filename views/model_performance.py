"""RetailPulse — Model Performance."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings as cfg
from src.pipeline import ensure_pipeline_ready
from src.ui_components import kpi_row, page_header, section_label, style_fig

page_header("Model Performance", "Evaluation metrics for every model behind the platform.")

with st.spinner("Loading data..."):
    data = ensure_pipeline_ready()

section_label("Customer Segmentation — KMeans")
sil = data["silhouette"]
c1, c2 = st.columns(2)
with c1:
    st.metric("Silhouette Score", f"{sil:.3f}" if sil else "n/a",
               help="Ranges -1 to 1. Above ~0.25 indicates reasonably separated clusters "
                    "for real-world customer behavior data.")
with c2:
    st.metric("Clusters", cfg.RFM_N_CLUSTERS)
st.caption("Silhouette score measures how well-separated the KMeans clusters are — "
           "higher is better (1.0 = perfectly separated, 0 = overlapping, negative = likely mislabeled).")

st.divider()

section_label("Churn Prediction — Random Forest Classifier")
metrics = data["churn_metrics"]

if not metrics:
    st.warning("Churn model was not trained (insufficient history in the current dataset).")
else:
    kpi_row([
        {"label": "Accuracy", "value": f"{metrics['accuracy']*100:.1f}%"},
        {"label": "Precision", "value": f"{metrics['precision']*100:.1f}%"},
        {"label": "Recall", "value": f"{metrics['recall']*100:.1f}%"},
        {"label": "F1 Score", "value": f"{metrics['f1_score']*100:.1f}%"},
    ])
    roc_auc_text = f"ROC-AUC: {metrics['roc_auc']*100:.1f}%" if metrics.get("roc_auc") else ""
    st.caption(
        f"Trained on {metrics['n_train']:,} customers, evaluated on a held-out "
        f"{metrics['n_test']:,}-customer test set. {roc_auc_text}"
    )

    c1, c2 = st.columns(2)
    with c1:
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["Actual: Retained", "Actual: Churned"],
                              columns=["Predicted: Retained", "Predicted: Churned"])
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Purples", aspect="auto")
        st.plotly_chart(style_fig(fig, title="Confusion Matrix", height=380), width="stretch")

    with c2:
        fi = pd.DataFrame(list(metrics["feature_importance"].items()), columns=["Feature", "Importance"])
        fi = fi.sort_values("Importance")
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[cfg.ACCENT_COLOR])
        st.plotly_chart(style_fig(fig, title="Feature Importance", height=380,
                                   x_title="Importance", y_title=""), width="stretch")

    st.info(
        "The model is trained on a time-based holdout: features are computed as of a snapshot "
        f"{cfg.CHURN_INACTIVITY_DAYS} days before the dataset's true end date, and the label is "
        "whether the customer purchased again afterward — preventing lookahead leakage.",
        icon=":material/verified_user:",
    )

st.divider()

section_label("Demand Forecasting")
forecast = data["forecast_df"]
engine = data["forecast_engine"]
st.write(f"**Engine used:** {engine}")

history = forecast[~forecast["IsForecast"]].dropna(subset=["y"])
if len(history) > 5:
    err = (history["y"] - history["yhat"]).abs()
    mape = (err / history["y"].replace(0, pd.NA)).dropna().mean() * 100
    mae = err.mean()
    c1, c2 = st.columns(2)
    c1.metric("In-Sample MAE", f"{cfg.CURRENCY_SYMBOL}{mae:,.0f}")
    c2.metric("In-Sample MAPE", f"{mape:.1f}%")
    st.caption("Mean Absolute Error / Mean Absolute Percentage Error of the model's fit against "
               "known historical daily revenue — a sanity check, not a guarantee of future accuracy.")

st.divider()
section_label("Inventory Optimization")
inv_df = data["inventory_df"]
st.write(
    f"Recommendations generated for **{len(inv_df)}** top SKUs using a 30-day trailing vs. "
    f"prior-30-day demand comparison, with a 95% service-level safety-stock buffer "
    f"(z = {cfg.REORDER_SERVICE_LEVEL_Z}) applied to the reorder quantity."
)
st.caption("Confidence score is heuristic — scaled by the magnitude of the demand shift and "
           "the amount of historical data available for that SKU.")
