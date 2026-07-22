"""
RetailPulse — Churn analysis.

Two layers, per the project brief:
1. Analytical churn label — derived directly from customer inactivity
   (no purchase within CHURN_INACTIVITY_DAYS of the dataset snapshot).
2. Predictive model — a Random Forest classifier trained on RFM + behavioral
   features to predict churn risk, so the platform can flag customers who
   are *trending* toward churn even before they cross the inactivity line
   (evaluated with a snapshot held one cycle back, so the label isn't
   trivially derived from features computed at the same snapshot).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings as cfg
from src.utils import get_logger

logger = get_logger(__name__)

FEATURE_COLS = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "TenureDays"]


def _customer_features_at(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Build RFM-style features for every customer using only transactions
    up to (and including) snapshot_date — used both for the live analytical
    view and for building a leakage-free training set."""
    hist = df[df["InvoiceDate"] <= snapshot_date]
    if hist.empty:
        return pd.DataFrame(columns=["CustomerID"] + FEATURE_COLS)

    grp = hist.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        FirstPurchase=("InvoiceDate", "min"),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    grp["Recency"] = (snapshot_date - grp["LastPurchase"]).dt.days
    grp["TenureDays"] = (snapshot_date - grp["FirstPurchase"]).dt.days.clip(lower=1)
    grp["AvgOrderValue"] = grp["Monetary"] / grp["Frequency"].replace(0, np.nan)
    grp = grp.fillna(0).reset_index()
    return grp[["CustomerID"] + FEATURE_COLS]


class ChurnAnalyzer:
    """Analytical inactivity-based churn flag + a predictive RF classifier."""

    def __init__(self, inactivity_days: int = cfg.CHURN_INACTIVITY_DAYS):
        self.inactivity_days = inactivity_days
        self.model: RandomForestClassifier | None = None
        self.metrics: dict = {}

    def analytical_churn(self, df: pd.DataFrame) -> pd.DataFrame:
        """Current churn status for every customer, as of the latest date
        in the dataset — the 'ground truth' operational view."""
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        feats = _customer_features_at(df, snapshot_date)
        feats["IsChurned"] = feats["Recency"] > self.inactivity_days
        feats["ChurnRiskLabel"] = np.where(
            feats["Recency"] > self.inactivity_days, "Churned",
            np.where(feats["Recency"] > self.inactivity_days * 0.6, "At Risk", "Active"),
        )
        return feats

    def train_predictive_model(self, df: pd.DataFrame) -> dict:
        """
        Train a classifier to predict churn using a train snapshot taken
        ~inactivity_days before the true end of the data, so the label
        (churn observed by the final date) reflects genuinely future
        behavior relative to the features (avoids leakage).
        """
        max_date = df["InvoiceDate"].max()
        train_snapshot = max_date - pd.Timedelta(days=self.inactivity_days)

        if (train_snapshot - df["InvoiceDate"].min()).days < 30:
            logger.warning("History too short for a leakage-free churn model; skipping training.")
            return {}

        X_feats = _customer_features_at(df, train_snapshot)
        # Label: did the customer make ANY purchase in the window after the
        # train snapshot up to the true max date? If not -> churned.
        future_customers = set(
            df[df["InvoiceDate"] > train_snapshot]["CustomerID"].unique()
        )
        X_feats["Churned"] = (~X_feats["CustomerID"].isin(future_customers)).astype(int)

        X = X_feats[FEATURE_COLS]
        y = X_feats["Churned"]
        if y.nunique() < 2 or len(X_feats) < 50:
            logger.warning("Insufficient class diversity/rows for churn model training.")
            return {}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=cfg.RANDOM_STATE, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=120, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=cfg.RANDOM_STATE, n_jobs=2,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)) if y.nunique() == 2 else None,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "feature_importance": dict(zip(FEATURE_COLS, self.model.feature_importances_.round(4).tolist())),
        }
        joblib.dump(self.model, cfg.CHURN_MODEL_PATH)
        logger.info("Churn model trained: %s", self.metrics)
        return self.metrics

    def predict_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the trained model to current customers for a forward-looking
        churn probability, blended with the analytical label."""
        analytical = self.analytical_churn(df)
        if self.model is None:
            analytical["ChurnProbability"] = np.where(analytical["IsChurned"], 0.9, 0.1)
            return analytical

        X = analytical[FEATURE_COLS]
        analytical["ChurnProbability"] = self.model.predict_proba(X)[:, 1]
        return analytical


def analyze_churn(featured_df: pd.DataFrame):
    analyzer = ChurnAnalyzer()
    metrics = analyzer.train_predictive_model(featured_df)
    result = analyzer.predict_risk_scores(featured_df)
    return result, metrics


if __name__ == "__main__":
    from src.utils import load_parquet, save_parquet

    featured = load_parquet(cfg.FEATURED_DATA_PATH)
    churn_df, metrics = analyze_churn(featured)
    save_parquet(churn_df, cfg.CHURN_PATH)
    logger.info("Saved churn analysis to %s | metrics=%s", cfg.CHURN_PATH, metrics)
