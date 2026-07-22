"""
RetailPulse — Demand forecasting.

Forecasts daily revenue (and, optionally, per-product quantity) using
Prophet when available. Prophet's build (cmdstan) can be slow or fail on
constrained deployment targets (e.g. some free-tier PaaS), so this module
transparently falls back to a Holt-Winters exponential smoothing model
(statsmodels) if Prophet cannot be imported or fails to fit — the rest of
the app never needs to know which engine produced the forecast.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings as cfg
from src.utils import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from prophet import Prophet
    _PROPHET_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _PROPHET_AVAILABLE = False


class DemandForecaster:
    """Forecasts a daily time series with Prophet, falling back to Holt-Winters."""

    def __init__(self, horizon_days: int = cfg.FORECAST_HORIZON_DAYS):
        self.horizon_days = horizon_days
        self.engine_used: str = "none"

    def _prepare_series(self, df: pd.DataFrame, value_col: str = "TotalPrice") -> pd.DataFrame:
        daily = (
            df.groupby(df["InvoiceDate"].dt.date)[value_col]
            .sum()
            .reset_index()
        )
        daily.columns = ["ds", "y"]
        daily["ds"] = pd.to_datetime(daily["ds"])
        daily = daily.sort_values("ds").reset_index(drop=True)
        return daily

    def fit_predict(self, df: pd.DataFrame, value_col: str = "TotalPrice",
                     use_prophet: bool = False) -> pd.DataFrame:
        """
        Holt-Winters (statsmodels) is the default engine: it's fast and
        has a negligible memory footprint, which matters a lot on
        resource-constrained deployments (Streamlit Cloud / Render free
        tiers). Prophet is measurably heavier (~400MB peak, mostly its
        compiled Stan subprocess) and is only used when explicitly
        requested via use_prophet=True — e.g. a user opting in on the
        Demand Forecasting page for a higher-fidelity forecast.
        """
        daily = self._prepare_series(df, value_col)

        if len(daily) < 14:
            raise ValueError("Not enough daily history to forecast (need >= 14 days).")

        if use_prophet and _PROPHET_AVAILABLE:
            try:
                return self._forecast_prophet(daily)
            except Exception as exc:  # pragma: no cover
                logger.warning("Prophet failed (%s); falling back to Holt-Winters.", exc)

        return self._forecast_holt_winters(daily)

    # ------------------------------------------------------------------
    def _forecast_prophet(self, daily: pd.DataFrame) -> pd.DataFrame:
        model = Prophet(
            yearly_seasonality=len(daily) >= 365,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.90,
        )
        model.fit(daily)
        future = model.make_future_dataframe(periods=self.horizon_days)
        forecast = model.predict(future)
        self.engine_used = "Prophet"

        out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        out = out.merge(daily, on="ds", how="left")  # attach actuals where available
        out["IsForecast"] = out["y"].isna()
        out["yhat"] = out["yhat"].clip(lower=0)
        out["yhat_lower"] = out["yhat_lower"].clip(lower=0)
        return out

    def _forecast_holt_winters(self, daily: pd.DataFrame) -> pd.DataFrame:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        series = daily.set_index("ds")["y"].asfreq("D").interpolate()
        seasonal_periods = 7 if len(series) >= 14 else None
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add" if seasonal_periods else None,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True)

        fitted = model.fittedvalues
        forecast_vals = model.forecast(self.horizon_days)
        self.engine_used = "Holt-Winters (statsmodels)"

        resid_std = float((series - fitted).std())
        future_idx = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=self.horizon_days, freq="D")

        hist = pd.DataFrame({"ds": series.index, "y": series.values, "yhat": fitted.values})
        fut = pd.DataFrame({
            "ds": future_idx,
            "y": np.nan,
            "yhat": np.clip(forecast_vals.values, 0, None),
        })
        out = pd.concat([hist, fut], ignore_index=True)
        out["yhat_lower"] = np.clip(out["yhat"] - 1.645 * resid_std, 0, None)
        out["yhat_upper"] = out["yhat"] + 1.645 * resid_std
        out["IsForecast"] = out["y"].isna()
        return out


def forecast_demand(df: pd.DataFrame, horizon_days: int = cfg.FORECAST_HORIZON_DAYS,
                     value_col: str = "TotalPrice", use_prophet: bool = False):
    forecaster = DemandForecaster(horizon_days=horizon_days)
    forecast = forecaster.fit_predict(df, value_col=value_col, use_prophet=use_prophet)
    return forecast, forecaster.engine_used


def forecast_summary(forecast: pd.DataFrame) -> dict:
    future = forecast[forecast["IsForecast"]]
    history = forecast[~forecast["IsForecast"]]
    trailing_avg_daily = history["y"].tail(30).mean() if len(history) else np.nan
    forecast_avg_daily = future["yhat"].mean() if len(future) else np.nan
    pct_change = (
        (forecast_avg_daily - trailing_avg_daily) / trailing_avg_daily
        if trailing_avg_daily else np.nan
    )
    return {
        "forecast_total": float(future["yhat"].sum()) if len(future) else 0.0,
        "trailing_avg_daily": float(trailing_avg_daily) if pd.notna(trailing_avg_daily) else 0.0,
        "forecast_avg_daily": float(forecast_avg_daily) if pd.notna(forecast_avg_daily) else 0.0,
        "pct_change_vs_trailing": float(pct_change) if pd.notna(pct_change) else 0.0,
    }


if __name__ == "__main__":
    from src.utils import load_parquet, save_parquet

    featured = load_parquet(cfg.FEATURED_DATA_PATH)
    forecast, engine = forecast_demand(featured)
    save_parquet(forecast, cfg.FORECAST_PATH)
    logger.info("Saved forecast (%s) to %s | summary=%s", engine, cfg.FORECAST_PATH,
                forecast_summary(forecast))
