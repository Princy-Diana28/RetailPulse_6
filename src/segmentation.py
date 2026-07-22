"""
RetailPulse — Customer segmentation.

Builds an RFM (Recency, Frequency, Monetary) profile per customer, scales
it, and clusters customers with KMeans. Clusters are then translated into
business-readable labels (Champions, Loyal, At Risk, ...) based on their
relative RFM statistics rather than hardcoded cluster indices — KMeans
cluster IDs are not stable across runs, so labeling must be data-driven.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings as cfg
from src.utils import get_logger

logger = get_logger(__name__)


class CustomerSegmenter:
    """RFM feature construction + KMeans clustering with business labeling."""

    def __init__(self, df: pd.DataFrame, n_clusters: int = cfg.RFM_N_CLUSTERS):
        self.df = df.copy()
        self.n_clusters = n_clusters
        self.scaler: StandardScaler | None = None
        self.kmeans: KMeans | None = None
        self.silhouette: float | None = None

    def build_rfm(self) -> pd.DataFrame:
        df = self.df
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

        rfm = df.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        rfm = rfm[rfm["Monetary"] > 0]
        return rfm

    def fit_predict(self) -> pd.DataFrame:
        rfm = self.build_rfm()

        # Log-transform monetary/frequency to tame skew before scaling.
        rfm_log = rfm.copy()
        rfm_log["Frequency"] = np.log1p(rfm_log["Frequency"])
        rfm_log["Monetary"] = np.log1p(rfm_log["Monetary"])

        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(rfm_log[["Recency", "Frequency", "Monetary"]])

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=cfg.RANDOM_STATE, n_init=10)
        rfm["Cluster"] = self.kmeans.fit_predict(scaled)

        if len(rfm) > self.n_clusters:
            self.silhouette = float(silhouette_score(scaled, rfm["Cluster"]))
        else:
            self.silhouette = None

        rfm["Segment"] = self._label_clusters(rfm)
        logger.info(
            "Segmentation complete: %s customers, %s clusters, silhouette=%.3f",
            len(rfm), self.n_clusters, self.silhouette or -1,
        )
        return rfm.reset_index()

    def _label_clusters(self, rfm: pd.DataFrame) -> pd.Series:
        """
        Rank clusters by an RFM composite score and assign business-readable
        labels. Lower recency, higher frequency, higher monetary = better.
        """
        stats = rfm.groupby("Cluster").agg(
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean"),
        )
        # Rank each dimension (1 = best) then average the ranks into a score.
        stats["RecencyRank"] = stats["Recency"].rank(ascending=True)
        stats["FrequencyRank"] = stats["Frequency"].rank(ascending=False)
        stats["MonetaryRank"] = stats["Monetary"].rank(ascending=False)
        stats["Score"] = stats[["RecencyRank", "FrequencyRank", "MonetaryRank"]].mean(axis=1)
        ordered_clusters = stats.sort_values("Score").index.tolist()

        n = len(ordered_clusters)
        label_pool = ["Champions", "Loyal Customers", "Potential Loyalists",
                       "Needs Attention", "At Risk", "Hibernating", "Lost"]
        if n <= len(label_pool):
            # Evenly sample the label pool so labels stay meaningful for any n_clusters.
            idxs = np.linspace(0, len(label_pool) - 1, n).round().astype(int)
            labels = [label_pool[i] for i in idxs]
        else:
            labels = [f"Segment {i+1}" for i in range(n)]

        cluster_to_label = dict(zip(ordered_clusters, labels))
        return rfm["Cluster"].map(cluster_to_label)

    def save(self, path_prefix: Path | None = None) -> None:
        path_prefix = path_prefix or cfg.MODELS_DIR
        joblib.dump(self.scaler, cfg.SCALER_MODEL_PATH)
        joblib.dump(self.kmeans, cfg.KMEANS_MODEL_PATH)


def segment_customers(featured_df: pd.DataFrame, n_clusters: int = cfg.RFM_N_CLUSTERS):
    segmenter = CustomerSegmenter(featured_df, n_clusters=n_clusters)
    rfm = segmenter.fit_predict()
    segmenter.save()
    return rfm, segmenter.silhouette


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Business-facing summary table: size and average RFM per segment."""
    summary = rfm.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        AvgRecencyDays=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
        TotalMonetary=("Monetary", "sum"),
    ).sort_values("TotalMonetary", ascending=False).reset_index()
    summary["PctOfCustomers"] = 100 * summary["Customers"] / summary["Customers"].sum()
    summary["PctOfRevenue"] = 100 * summary["TotalMonetary"] / summary["TotalMonetary"].sum()
    return summary


if __name__ == "__main__":
    from src.utils import load_parquet, save_parquet

    featured = load_parquet(cfg.FEATURED_DATA_PATH)
    rfm, sil = segment_customers(featured)
    save_parquet(rfm, cfg.RFM_SEGMENTS_PATH)
    logger.info("Saved segmentation to %s | silhouette=%.3f", cfg.RFM_SEGMENTS_PATH, sil or -1)
