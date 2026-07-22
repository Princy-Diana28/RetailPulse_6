"""
RetailPulse — Automated business insights.

Scans the feature-engineered dataset (plus segmentation/inventory outputs
when available) and produces a list of plain-English insight statements —
the kind an analyst would pull out for an exec summary slide.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings as cfg
from src.utils import format_currency, format_number, get_logger

logger = get_logger(__name__)


def generate_insights(df: pd.DataFrame, rfm: pd.DataFrame | None = None,
                       inventory: pd.DataFrame | None = None) -> list[dict]:
    insights: list[dict] = []
    sym = cfg.CURRENCY_SYMBOL

    # --- Revenue by month --------------------------------------------------
    monthly = df.groupby("YearMonth", observed=True)["TotalPrice"].sum().sort_values(ascending=False)
    if len(monthly):
        insights.append({
            "category": "Revenue",
            "title": "Highest Revenue Month",
            "detail": f"{monthly.index[0]} generated {format_currency(monthly.iloc[0], sym)} in revenue, "
                      f"the strongest month in the dataset.",
        })
        insights.append({
            "category": "Revenue",
            "title": "Lowest Revenue Month",
            "detail": f"{monthly.index[-1]} was the weakest month, with only "
                      f"{format_currency(monthly.iloc[-1], sym)} in revenue.",
        })

    # --- Country ------------------------------------------------------------
    country_rev = df.groupby("Country", observed=True)["TotalPrice"].sum().sort_values(ascending=False)
    if len(country_rev):
        top_country = country_rev.index[0]
        share = 100 * country_rev.iloc[0] / country_rev.sum()
        insights.append({
            "category": "Geography",
            "title": "Top Market",
            "detail": f"{top_country} is the top market, contributing {format_currency(country_rev.iloc[0], sym)} "
                      f"({share:.1f}% of total revenue).",
        })

    # --- Products -------------------------------------------------------------
    product_rev = df.groupby("Description", observed=True)["TotalPrice"].sum().sort_values(ascending=False)
    product_qty = df.groupby("Description", observed=True)["Quantity"].sum().sort_values(ascending=False)
    if len(product_rev):
        insights.append({
            "category": "Product",
            "title": "Top Revenue Product",
            "detail": f"'{product_rev.index[0]}' is the highest-grossing product at "
                      f"{format_currency(product_rev.iloc[0], sym)} in total sales.",
        })
    if len(product_qty):
        insights.append({
            "category": "Product",
            "title": "Best-Selling Product by Volume",
            "detail": f"'{product_qty.index[0]}' is the top seller by units, with "
                      f"{format_number(product_qty.iloc[0])} units sold.",
        })

    # --- Fastest growing / declining product (recent vs prior period) -----
    if "InvoiceDate" in df.columns:
        recent_cutoff = df["InvoiceDate"].max() - pd.Timedelta(days=30)
        prior_cutoff = recent_cutoff - pd.Timedelta(days=30)
        recent = df[df["InvoiceDate"] > recent_cutoff].groupby("Description", observed=True)["Quantity"].sum()
        prior = df[(df["InvoiceDate"] <= recent_cutoff) & (df["InvoiceDate"] > prior_cutoff)].groupby("Description", observed=True)["Quantity"].sum()
        both = pd.concat([recent.rename("recent"), prior.rename("prior")], axis=1).dropna()
        both = both[both["prior"] >= 30]
        if len(both):
            both["growth"] = (both["recent"] - both["prior"]) / both["prior"]
            fastest = both["growth"].idxmax()
            declining = both["growth"].idxmin()
            insights.append({
                "category": "Trend",
                "title": "Fastest Growing Product",
                "detail": f"'{fastest}' grew {both.loc[fastest, 'growth']*100:.0f}% in the most recent 30-day "
                          f"period versus the prior 30 days.",
            })
            insights.append({
                "category": "Trend",
                "title": "Declining Demand",
                "detail": f"'{declining}' demand fell {abs(both.loc[declining, 'growth'])*100:.0f}% over the "
                          f"same comparison window and may need markdown or review.",
            })

    # --- Customer segments ----------------------------------------------------
    if rfm is not None and len(rfm):
        seg_value = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=False)
        if len(seg_value):
            top_seg = seg_value.index[0]
            share = 100 * seg_value.iloc[0] / seg_value.sum()
            insights.append({
                "category": "Customers",
                "title": "Most Valuable Customer Segment",
                "detail": f"'{top_seg}' customers drive {share:.1f}% of total customer revenue despite being "
                          f"{100 * (rfm['Segment'] == top_seg).mean():.1f}% of the customer base.",
            })

    # --- Inventory --------------------------------------------------------
    if inventory is not None and len(inventory):
        increase_n = (inventory["RecommendedAction"] == "Increase Stock").sum()
        reduce_n = (inventory["RecommendedAction"] == "Reduce Stock").sum()
        insights.append({
            "category": "Inventory",
            "title": "Stock Action Needed",
            "detail": f"{increase_n} top products show rising demand and should have stock increased; "
                      f"{reduce_n} show declining demand and are candidates for stock reduction.",
        })

    return insights


if __name__ == "__main__":
    from src.utils import load_parquet

    featured = load_parquet(cfg.FEATURED_DATA_PATH)
    rfm = load_parquet(cfg.RFM_SEGMENTS_PATH) if cfg.RFM_SEGMENTS_PATH.exists() else None
    inv = load_parquet(cfg.INVENTORY_PATH) if cfg.INVENTORY_PATH.exists() else None
    for ins in generate_insights(featured, rfm, inv):
        print(f"[{ins['category']}] {ins['title']}: {ins['detail']}")
