"""
RetailPulse — Central configuration.

All tunable constants, paths, and thresholds live here so the rest of the
codebase never hardcodes a magic number. Keeping this isolated also makes
the project trivially configurable for a different retailer / dataset.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "saved_models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
ASSETS_DIR = ROOT_DIR / "assets"

for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR, ASSETS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

DEFAULT_RAW_FILENAME = "online_retail_II.xlsx"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_transactions.parquet"
FEATURED_DATA_PATH = PROCESSED_DATA_DIR / "feature_engineered.parquet"
RFM_SEGMENTS_PATH = PROCESSED_DATA_DIR / "customer_segments.parquet"
FORECAST_PATH = PROCESSED_DATA_DIR / "demand_forecast.parquet"
CHURN_PATH = PROCESSED_DATA_DIR / "churn_predictions.parquet"
INVENTORY_PATH = PROCESSED_DATA_DIR / "inventory_recommendations.parquet"

KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_segmentation.joblib"
SCALER_MODEL_PATH = MODELS_DIR / "rfm_scaler.joblib"
CHURN_MODEL_PATH = MODELS_DIR / "churn_classifier.joblib"

# ---------------------------------------------------------------------------
# Raw column names (as they appear in the Online Retail II dataset)
# ---------------------------------------------------------------------------
COL_INVOICE = "Invoice"
COL_STOCKCODE = "StockCode"
COL_DESCRIPTION = "Description"
COL_QUANTITY = "Quantity"
COL_INVOICE_DATE = "InvoiceDate"
COL_PRICE = "Price"
COL_CUSTOMER_ID = "Customer ID"
COL_COUNTRY = "Country"

# ---------------------------------------------------------------------------
# Business logic thresholds
# ---------------------------------------------------------------------------
CHURN_INACTIVITY_DAYS = 90          # No purchase in N days -> considered churned
FORECAST_HORIZON_DAYS = 30          # Default forecast window
RFM_N_CLUSTERS = 4                  # Customer segments via KMeans
OUTLIER_QUANTITY_CAP_PERCENTILE = 0.995   # Winsorize extreme quantities
OUTLIER_PRICE_CAP_PERCENTILE = 0.995      # Winsorize extreme prices
MIN_VALID_PRICE = 0.01
DEMAND_LOW_QUANTILE = 0.33          # DemandLevel bucket boundaries
DEMAND_HIGH_QUANTILE = 0.67

# Inventory decision bands, expressed as % change vs. trailing demand
INVENTORY_INCREASE_THRESHOLD = 0.10     # forecast > trailing * 1.10 -> Increase Stock
INVENTORY_REDUCE_THRESHOLD = -0.10      # forecast < trailing * 0.90 -> Reduce Stock
REORDER_SERVICE_LEVEL_Z = 1.65          # ~95% service level z-score for safety stock
SAFETY_STOCK_FACTOR = 0.15              # fallback % buffer on forecasted demand

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Streamlit theme
# ---------------------------------------------------------------------------
APP_TITLE = "RetailPulse — AI-Powered Demand & Inventory Intelligence"
APP_ICON = "📊"
PRIMARY_COLOR = "#6366F1"      # indigo
ACCENT_COLOR = "#22D3EE"       # cyan
SUCCESS_COLOR = "#34D399"
WARNING_COLOR = "#FBBF24"
DANGER_COLOR = "#F87171"
BG_DARK = "#0B0F19"
SURFACE_DARK = "#141A2A"
BORDER_DARK = "#232B3E"
TEXT_PRIMARY = "#E5E7EB"
TEXT_MUTED = "#9CA3AF"

CURRENCY_SYMBOL = "£"  # Online Retail II is a UK-based retailer
