"""
RetailPulse — Inventory optimization.

Translates per-product demand trends into stock action recommendations
(Increase / Maintain / Reduce) with a reasoned confidence score, plus a
reorder quantity suggestion using a classic safety-stock formula. This
runs on unit *quantity* (not revenue) since that's what a warehouse
actually reorders against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings as cfg
from src.utils import get_logger

logger = get_logger(__name__)


def _trend_window_stats(product_daily: pd.DataFrame, window: int = 30) -> dict:
    """Compare the most recent window vs. the prior window of the same length."""
    if len(product_daily) < window:
        recent = product_daily["Quantity"].mean()
        prior = recent
    else:
        recent = product_daily["Quantity"].tail(window).mean()
        prior_slice = product_daily["Quantity"].iloc[-2 * window:-window]
        prior = prior_slice.mean() if len(prior_slice) else recent
    pct_change = (recent - prior) / prior if prior else 0.0
    return {"recent_avg_daily": recent, "prior_avg_daily": prior, "pct_change": pct_change}


class InventoryOptimizer:
    """Generates per-product stock recommendations from historical + forecast demand."""

    def __init__(self, top_n_products: int = 50):
        self.top_n_products = top_n_products

    def recommend(self, df: pd.DataFrame) -> pd.DataFrame:
        # A handful of StockCodes carry more than one Description variant in
        # the raw data (relabeled products over the 2-year window). Resolve
        # each StockCode to its single most frequent description so a SKU
        # doesn't appear twice under near-identical names.
        desc_lookup = (
            df.groupby("StockCode", observed=True)["Description"]
            .agg(lambda s: s.value_counts().idxmax())
        )

        top_codes = (
            df.groupby("StockCode", observed=True)["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(self.top_n_products)
            .index
        )

        rows = []
        for stock_code in top_codes:
            description = desc_lookup.loc[stock_code]
            sub = df[df["StockCode"] == stock_code]
            daily = sub.groupby(sub["InvoiceDate"].dt.date)["Quantity"].sum()
            daily.index = pd.to_datetime(daily.index)
            daily = daily.asfreq("D", fill_value=0).reset_index()
            daily.columns = ["InvoiceDate", "Quantity"]

            stats = _trend_window_stats(daily)
            std_daily = daily["Quantity"].tail(60).std() if len(daily) >= 2 else 0.0
            std_daily = 0.0 if pd.isna(std_daily) else std_daily

            forecast_daily = max(stats["recent_avg_daily"], 0)
            lead_time_days = 7  # assumed supplier lead time
            safety_stock = cfg.REORDER_SERVICE_LEVEL_Z * std_daily * np.sqrt(lead_time_days)
            reorder_qty = forecast_daily * lead_time_days + safety_stock

            if stats["pct_change"] >= cfg.INVENTORY_INCREASE_THRESHOLD:
                action = "Increase Stock"
            elif stats["pct_change"] <= cfg.INVENTORY_REDUCE_THRESHOLD:
                action = "Reduce Stock"
            else:
                action = "Maintain Stock"

            confidence = min(0.95, 0.5 + min(abs(stats["pct_change"]), 1.0) * 0.4 + (0.05 if len(daily) >= 60 else 0))

            rows.append({
                "StockCode": stock_code,
                "Description": description,
                "RecentAvgDailyQty": round(stats["recent_avg_daily"], 2),
                "PriorAvgDailyQty": round(stats["prior_avg_daily"], 2),
                "PctChange": round(stats["pct_change"], 4),
                "RecommendedAction": action,
                "Confidence": round(confidence, 2),
                "SuggestedReorderQty": int(round(max(reorder_qty, 0))),
                "TotalHistoricalQty": int(sub["Quantity"].sum()),
            })

        result = pd.DataFrame(rows).sort_values("TotalHistoricalQty", ascending=False)
        logger.info("Inventory recommendations generated for %s SKUs", len(result))
        return result.reset_index(drop=True)


def optimize_inventory(df: pd.DataFrame, top_n_products: int = 50) -> pd.DataFrame:
    return InventoryOptimizer(top_n_products=top_n_products).recommend(df)


def inventory_summary(inv_df: pd.DataFrame) -> dict:
    counts = inv_df["RecommendedAction"].value_counts().to_dict()
    return {
        "increase_count": counts.get("Increase Stock", 0),
        "maintain_count": counts.get("Maintain Stock", 0),
        "reduce_count": counts.get("Reduce Stock", 0),
        "total_reorder_units": int(inv_df["SuggestedReorderQty"].sum()),
    }


if __name__ == "__main__":
    from src.utils import load_parquet, save_parquet

    featured = load_parquet(cfg.FEATURED_DATA_PATH)
    inv = optimize_inventory(featured)
    save_parquet(inv, cfg.INVENTORY_PATH)
    logger.info("Saved inventory recommendations to %s | summary=%s",
                cfg.INVENTORY_PATH, inventory_summary(inv))
