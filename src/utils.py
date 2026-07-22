"""
RetailPulse — Shared utility helpers.

Small, dependency-light helpers reused across the preprocessing, modeling,
and presentation layers. Keeping these centralized avoids duplicate logic
scattered through the Streamlit pages.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Union

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def load_raw_dataset(source) -> pd.DataFrame:
    """
    Load the Online Retail II dataset from an .xlsx (multi-sheet, one per
    year) or .csv file, returning a single concatenated DataFrame.

    Accepts a path, an uploaded Streamlit file-like object, or raw bytes.
    """
    if hasattr(source, "name"):
        filename = source.name.lower()
    else:
        filename = str(source).lower()

    if filename.endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(source, sheet_name=None, engine="openpyxl")
        frames = []
        for sheet_name, frame in sheets.items():
            frame = frame.copy()
            frame["SourceSheet"] = sheet_name
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
    elif filename.endswith(".csv"):
        df = pd.read_csv(source, encoding="ISO-8859-1")
    else:
        raise ValueError(f"Unsupported file type for: {filename}")

    return df


def save_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


# ---------------------------------------------------------------------------
# Formatting helpers (used heavily in the Streamlit KPI cards)
# ---------------------------------------------------------------------------
def format_currency(value: float, symbol: str = "£") -> str:
    if value is None or pd.isna(value):
        return f"{symbol}0"
    if abs(value) >= 1_000_000:
        return f"{symbol}{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{symbol}{value / 1_000:,.1f}K"
    return f"{symbol}{value:,.0f}"


def format_number(value: float) -> str:
    if value is None or pd.isna(value):
        return "0"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.1f}K"
    return f"{value:,.0f}"


def format_percent(value: float, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "0%"
    return f"{value * 100:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not denominator:
        return default
    return numerator / denominator
