"""
RetailPulse — Feature engineering.

Derives business-meaningful features from the cleaned transaction table:
calendar features, demand-level buckets, customer-level RFM/CLV/frequency
aggregates. All features are engineered directly from the existing columns
— nothing is fabricated.
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

_SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}


class FeatureEngineer:
    """Adds calendar, demand, and customer-value features to clean transactions."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def run(self) -> pd.DataFrame:
        df = self.df
        df = self._add_calendar_features(df)
        df = self._add_demand_level(df)
        df = self._add_order_value_features(df)
        df = self._add_customer_value_features(df)
        df = self._optimize_dtypes(df)
        logger.info("Feature engineering complete: %s columns", len(df.columns))
        return df

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts repetitive string columns to pandas' `category` dtype.
        These columns store a small/moderate number of distinct values
        repeated across hundreds of thousands of rows (e.g. ~5K product
        descriptions repeated 790K times) — as plain strings, pandas keeps
        a separate Python string object per row; as categories, it stores
        one integer code per row plus a small lookup table, typically
        cutting memory for these columns by 70-90% with no behavior change
        (categorical columns still support the same groupby/filter/string
        operations used elsewhere in the pipeline).
        """
        categorical_cols = ["DayOfWeek", "Season", "DemandLevel", "MonthName", "YearMonth"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    # ------------------------------------------------------------------
    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Year"] = df["InvoiceDate"].dt.year
        df["Month"] = df["InvoiceDate"].dt.month
        df["MonthName"] = df["InvoiceDate"].dt.strftime("%b")
        df["Quarter"] = df["InvoiceDate"].dt.quarter
        df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
        df["DayOfWeekNum"] = df["InvoiceDate"].dt.dayofweek
        df["Hour"] = df["InvoiceDate"].dt.hour
        df["Season"] = df["Month"].map(_SEASON_MAP)
        df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
        return df

    def _add_demand_level(self, df: pd.DataFrame) -> pd.DataFrame:
        low_q = df["Quantity"].quantile(cfg.DEMAND_LOW_QUANTILE)
        high_q = df["Quantity"].quantile(cfg.DEMAND_HIGH_QUANTILE)

        def bucket(qty: float) -> str:
            if qty <= low_q:
                return "Low"
            if qty <= high_q:
                return "Medium"
            return "High"

        df["DemandLevel"] = df["Quantity"].apply(bucket)
        return df

    def _add_order_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        order_value = df.groupby("Invoice")["TotalPrice"].transform("sum")
        order_size = df.groupby("Invoice")["Quantity"].transform("sum")
        df["InvoiceTotalValue"] = order_value
        df["InvoiceTotalQuantity"] = order_size
        return df

    def _add_customer_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

        cust = df.groupby("CustomerID").agg(
            FirstPurchaseDate=("InvoiceDate", "min"),
            LastPurchaseDate=("InvoiceDate", "max"),
            TotalOrders=("Invoice", "nunique"),
            TotalSpend=("TotalPrice", "sum"),
            TotalItemsPurchased=("Quantity", "sum"),
        )
        cust["CustomerTenureDays"] = (
            snapshot_date - cust["FirstPurchaseDate"]
        ).dt.days.clip(lower=1)
        cust["RecencyDays"] = (snapshot_date - cust["LastPurchaseDate"]).dt.days
        cust["AverageOrderValue"] = cust["TotalSpend"] / cust["TotalOrders"].replace(0, np.nan)
        cust["PurchaseFrequency"] = cust["TotalOrders"] / (cust["CustomerTenureDays"] / 30.0)
        # Simple historical CLV proxy: avg order value * purchase frequency (monthly) * tenure in months
        cust["CustomerLifetimeValue"] = (
            cust["AverageOrderValue"] * cust["PurchaseFrequency"] * (cust["CustomerTenureDays"] / 30.0)
        ).fillna(0)
        cust = cust.fillna(0)

        df = df.merge(
            cust[["AverageOrderValue", "PurchaseFrequency", "CustomerLifetimeValue",
                  "RecencyDays", "TotalOrders", "TotalSpend"]].rename(columns={
                "TotalOrders": "CustomerTotalOrders",
                "TotalSpend": "CustomerTotalSpend",
            }),
            left_on="CustomerID", right_index=True, how="left",
        )
        return df


def engineer_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    return FeatureEngineer(clean_df).run()


if __name__ == "__main__":
    from src.utils import load_parquet, save_parquet

    clean = load_parquet(cfg.CLEANED_DATA_PATH)
    featured = engineer_features(clean)
    save_parquet(featured, cfg.FEATURED_DATA_PATH)
    logger.info("Saved feature-engineered dataset to %s (%s rows, %s cols)",
                cfg.FEATURED_DATA_PATH, len(featured), len(featured.columns))
