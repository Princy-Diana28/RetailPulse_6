"""
RetailPulse — Data cleaning & preprocessing.

Transforms the raw Online Retail II export into an analysis-ready
transaction table. Every cleaning decision is encapsulated in its own
method so the pipeline is auditable and independently testable.
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


class DataCleaner:
    """Encapsulates the full cleaning pipeline for raw retail transactions."""

    def __init__(self, df: pd.DataFrame):
        self._input_df = df
        self.report: dict = {}

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        df = self._input_df.copy()
        start_rows = len(df)

        df = self._standardize_columns(df)
        df = self._drop_duplicates(df)
        df = self._handle_missing_values(df)
        df = self._parse_datetime(df)
        df = self._flag_cancellations(df)
        df = self._filter_invalid_quantity_price(df)
        df = self._cap_outliers(df)
        df = self._finalize_types(df)

        self.report["start_rows"] = start_rows
        self.report["end_rows"] = len(df)
        self.report["rows_removed"] = start_rows - len(df)
        self.report["pct_removed"] = round(
            100 * self.report["rows_removed"] / start_rows, 2
        ) if start_rows else 0.0

        logger.info(
            "Cleaning complete: %s -> %s rows (%.2f%% removed)",
            start_rows, len(df), self.report["pct_removed"],
        )
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Individual cleaning steps
    # ------------------------------------------------------------------
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "Customer ID": "CustomerID",
            "Price": "UnitPrice",
        }
        df = df.rename(columns=rename_map)
        expected = ["Invoice", "StockCode", "Description", "Quantity",
                    "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing expected columns: {missing}")
        return df

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        logger.info("Dropped %s duplicate rows", before - len(df))
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # CustomerID is essential for segmentation/churn/CLV — drop if missing.
        before = len(df)
        df = df.dropna(subset=["CustomerID"])
        logger.info("Dropped %s rows with missing CustomerID", before - len(df))

        # Description is non-critical for numeric analysis; fill for display.
        df["Description"] = df["Description"].fillna("UNKNOWN ITEM").str.strip()
        df = df[df["Description"] != ""]

        df["CustomerID"] = df["CustomerID"].astype("int64").astype(str)
        return df

    def _parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["InvoiceDate"])
        logger.info("Dropped %s rows with unparseable InvoiceDate", before - len(df))
        return df

    def _flag_cancellations(self, df: pd.DataFrame) -> pd.DataFrame:
        # Invoices prefixed with 'C' are cancellations in this dataset.
        df["IsCancelled"] = df["Invoice"].astype(str).str.startswith("C")
        return df

    def _filter_invalid_quantity_price(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        # Keep only genuine sales transactions: positive quantity & price.
        # Cancellations (negative quantity) are flagged separately via
        # IsCancelled for potential returns analysis, but excluded here
        # from the core "sales" table used for forecasting/segmentation.
        df = df[(~df["IsCancelled"]) & (df["Quantity"] > 0) & (df["UnitPrice"] >= cfg.MIN_VALID_PRICE)]
        logger.info("Dropped %s rows with invalid Quantity/UnitPrice or cancellations",
                     before - len(df))
        # Drop known non-product stock codes (postage, bank charges, etc.)
        non_product_codes = {"POST", "D", "DOT", "M", "S", "BANK CHARGES",
                              "CRUK", "PADS", "AMAZONFEE", "TEST001", "TEST002"}
        before = len(df)
        df = df[~df["StockCode"].astype(str).str.upper().isin(non_product_codes)]
        logger.info("Dropped %s non-product stock code rows", before - len(df))
        return df

    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        # Winsorize extreme bulk-order quantities and mispriced items rather
        # than deleting them outright — preserves volume while limiting the
        # influence of data-entry errors on aggregates and models.
        q_cap = df["Quantity"].quantile(cfg.OUTLIER_QUANTITY_CAP_PERCENTILE)
        p_cap = df["UnitPrice"].quantile(cfg.OUTLIER_PRICE_CAP_PERCENTILE)
        df["Quantity"] = np.minimum(df["Quantity"], q_cap)
        df["UnitPrice"] = np.minimum(df["UnitPrice"], p_cap)
        return df

    def _finalize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        # category dtype for repetitive low/medium-cardinality string columns:
        # a handful of distinct StockCodes/Descriptions/Countries repeat across
        # hundreds of thousands of rows, so storing one integer code per row
        # (plus a small lookup table) instead of a full string object per row
        # cuts memory for these columns by roughly 70-90% with no behavior
        # change for the groupby/filter operations used throughout the app.
        df["Invoice"] = df["Invoice"].astype(str).astype("category")
        df["StockCode"] = df["StockCode"].astype(str).astype("category")
        df["Description"] = df["Description"].astype("category")
        df["Country"] = df["Country"].astype(str).str.strip().astype("category")
        df["CustomerID"] = df["CustomerID"].astype("category")
        df["Quantity"] = df["Quantity"].astype("float32")
        df["UnitPrice"] = df["UnitPrice"].astype("float32")
        df["TotalPrice"] = (df["Quantity"] * df["UnitPrice"]).astype("float32")
        cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
                "UnitPrice", "CustomerID", "Country", "TotalPrice", "IsCancelled"]
        return df[cols]


def clean_dataset(raw_df: pd.DataFrame):
    """Convenience wrapper: run the cleaner and return (clean_df, report)."""
    cleaner = DataCleaner(raw_df)
    clean_df = cleaner.run()
    return clean_df, cleaner.report


if __name__ == "__main__":
    raw_path = cfg.RAW_DATA_DIR / cfg.DEFAULT_RAW_FILENAME
    logger.info("Loading raw dataset from %s", raw_path)
    from src.utils import load_raw_dataset, save_parquet

    raw = load_raw_dataset(raw_path)
    clean, rpt = clean_dataset(raw)
    save_parquet(clean, cfg.CLEANED_DATA_PATH)
    logger.info("Saved cleaned dataset to %s | report=%s", cfg.CLEANED_DATA_PATH, rpt)
