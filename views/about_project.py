"""RetailPulse — About Project."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings as cfg
from src.ui_components import page_header

page_header("About Project", "What RetailPulse does, how it's built, and how to deploy it.")

st.markdown("""
**RetailPulse** is an end-to-end AI-powered retail analytics platform built on the
**Online Retail II** dataset (a UK-based online gift retailer, Dec 2009 - Dec 2011,
~1.07M raw transaction lines across 43 countries).

### What it does
- **Data Cleaning** — deduplication, missing-value handling, cancellation/return
  flagging, non-product code filtering, outlier winsorization.
- **Feature Engineering** — calendar features, demand-level buckets, order value,
  customer lifetime value, purchase frequency, recency.
- **Exploratory Data Analysis** — revenue trends, top products, country breakdowns,
  seasonality, distributions, correlations.
- **Customer Segmentation** — RFM analysis + KMeans clustering with automatic,
  data-driven business labeling (Champions, Loyal, At Risk, etc.).
- **Demand Forecasting** — Holt-Winters (statsmodels) is the default, lightweight
  forecasting engine; Prophet is available as an optional, higher-fidelity engine
  users can opt into on the Demand Forecasting page.
- **Churn Analysis** — an inactivity-based analytical label plus a Random Forest
  classifier trained on a leakage-free, time-based holdout.
- **Inventory Optimization** — per-SKU stock action recommendations (Increase /
  Maintain / Reduce) with a safety-stock reorder quantity.
- **Business Insights** — automatically generated, plain-English findings.

### Technology Stack
| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| Data | Pandas, NumPy |
| ML | Scikit-learn (KMeans, RandomForest), Prophet, Statsmodels |
| Visualization | Plotly |
| App | Streamlit |
| Persistence | Joblib (models), Parquet (data) |

### Project Structure
```
RetailPulse/
├── app.py                   # Router / entry point (st.navigation)
├── requirements.txt
├── README.md
├── config/settings.py        # Central configuration
├── data/{raw,processed}/
├── src/
│   ├── preprocessing.py      # Data cleaning
│   ├── feature_engineering.py
│   ├── segmentation.py       # RFM + KMeans
│   ├── forecasting.py        # Prophet / Holt-Winters
│   ├── churn.py              # Analytical + predictive churn
│   ├── inventory.py          # Stock recommendations
│   ├── insights.py           # Automated insights
│   ├── pipeline.py           # Cached orchestration for Streamlit
│   ├── ui_components.py      # Shared theme / KPI cards
│   └── utils.py
├── views/                    # Application pages (routed via st.navigation)
├── saved_models/             # Persisted KMeans, scaler, churn model
└── outputs/                  # Exported reports
```

### Deployment
Runnable locally with:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Compatible with Streamlit Community Cloud and Render. Holt-Winters is the
default forecasting engine specifically because it's lightweight enough for
free-tier memory limits — Prophet remains available as an opt-in toggle on
the Demand Forecasting page for anyone who wants its richer seasonality
modeling and has the memory headroom to spare.

### Notes on the Data
The dataset ends mid-December 2011, so the final calendar month is partial —
month-over-month comparisons involving December 2011 should be read with that
in mind. Customer-level analyses (segmentation, churn, CLV) exclude the ~23%
of raw rows with no Customer ID, since those can't be attributed to a person.
""")

st.info(
    f"Currency shown throughout the app is {cfg.CURRENCY_SYMBOL} (GBP), matching the "
    "source dataset's home market.",
    icon=":material/payments:",
)
