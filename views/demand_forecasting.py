"""RetailPulse — Demand Forecasting."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings as cfg
from src.pipeline import ensure_pipeline_ready, run_forecasting
from src.ui_components import kpi_row, page_header, style_fig
from src.utils import format_currency

page_header(
    "Demand Forecasting",
    "Time-series forecast of daily revenue. Holt-Winters is the default engine "
    "(fast, lightweight); Prophet is available as an optional, higher-fidelity "
    "engine for a deeper look.",
)

with st.spinner("Loading data..."):
    data = ensure_pipeline_ready()

col_a, col_b = st.columns([2, 1])
with col_a:
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=90,
                         value=cfg.FORECAST_HORIZON_DAYS, step=1)
with col_b:
    use_prophet = st.toggle(
        "Use Prophet engine",
        value=False,
        help="Prophet can capture richer seasonality but uses substantially more "
             "memory and compute than the default Holt-Winters engine.",
    )

needs_recompute = horizon != cfg.FORECAST_HORIZON_DAYS or use_prophet
if not needs_recompute:
    forecast, engine, summary = data["forecast_df"], data["forecast_engine"], data["forecast_summary"]
else:
    with st.spinner(f"Forecasting with {'Prophet' if use_prophet else 'Holt-Winters'}..."):
        forecast, engine, summary = run_forecasting(
            data["featured_df"], horizon_days=horizon, use_prophet=use_prophet
        )

st.caption(f"Forecast engine in use: **{engine}**")

kpi_row([
    {"label": f"Forecasted Revenue (next {horizon}d)", "value": format_currency(summary["forecast_total"], cfg.CURRENCY_SYMBOL)},
    {"label": "Trailing Avg Daily Revenue", "value": format_currency(summary["trailing_avg_daily"], cfg.CURRENCY_SYMBOL)},
    {"label": "Forecast Avg Daily Revenue", "value": format_currency(summary["forecast_avg_daily"], cfg.CURRENCY_SYMBOL)},
    {
        "label": "Change vs Trailing Avg",
        "value": f"{summary['pct_change_vs_trailing']*100:+.1f}%",
        "positive": summary["pct_change_vs_trailing"] >= 0,
    },
])
st.divider()

st.subheader("Forecast vs Actuals")
fig = go.Figure()
history = forecast[~forecast["IsForecast"]]
future = forecast[forecast["IsForecast"]]

fig.add_trace(go.Scatter(x=history["ds"], y=history["y"], mode="lines", name="Actual Revenue",
                          line=dict(color=cfg.TEXT_MUTED, width=1.5)))
fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Model Fit / Forecast",
                          line=dict(color=cfg.PRIMARY_COLOR, width=2)))
fig.add_trace(go.Scatter(
    x=list(future["ds"]) + list(future["ds"][::-1]),
    y=list(future["yhat_upper"]) + list(future["yhat_lower"][::-1]),
    fill="toself", fillcolor="rgba(99,102,241,0.15)", line=dict(width=0),
    name="Confidence Interval", showlegend=True,
))
st.plotly_chart(
    style_fig(fig, height=460, x_title="Date", y_title=f"Revenue ({cfg.CURRENCY_SYMBOL})"),
    width="stretch",
)

st.subheader(f"Next {horizon} Days — Forecast Table")
table = future[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
table.columns = ["Date", "Forecast", "Lower Bound", "Upper Bound"]
table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
for col in ["Forecast", "Lower Bound", "Upper Bound"]:
    table[col] = table[col].map(lambda v: f"{cfg.CURRENCY_SYMBOL}{v:,.0f}")
st.dataframe(table, width="stretch", hide_index=True)

st.divider()
st.download_button(
    "Download forecast results (CSV)",
    forecast.to_csv(index=False).encode("utf-8"),
    file_name="retailpulse_demand_forecast.csv",
    mime="text/csv",
    icon=":material/download:",
)
