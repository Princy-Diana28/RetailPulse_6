"""
RetailPulse — Shared Streamlit UI components.

Centralizes the theme CSS, KPI card renderer, page header, and Plotly
styling so every page shares one consistent, professional visual language.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings as cfg

PLOTLY_TEMPLATE = "plotly_dark"
CHART_COLORWAY = [cfg.PRIMARY_COLOR, cfg.ACCENT_COLOR, cfg.SUCCESS_COLOR,
                  cfg.WARNING_COLOR, cfg.DANGER_COLOR, "#A78BFA", "#F472B6"]


def inject_global_css() -> None:
    st.html(textwrap.dedent(f"""\
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        .stApp {{
            background: radial-gradient(circle at 15% 0%, #12172A 0%, {cfg.BG_DARK} 45%);
        }}
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}
        [data-testid="stSidebar"] {{
            background-color: {cfg.SURFACE_DARK};
            border-right: 1px solid {cfg.BORDER_DARK};
        }}
        [data-testid="stSidebarNav"] {{
            padding-top: 0.25rem;
        }}
        .block-container {{
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1360px;
            animation: rp-fade-in 0.35s ease-out;
        }}
        @keyframes rp-fade-in {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}

        .rp-brand {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 4px 2px 14px 2px;
        }}
        .rp-brand-mark {{
            width: 34px; height: 34px;
            border-radius: 9px;
            background: linear-gradient(135deg, {cfg.PRIMARY_COLOR}, {cfg.ACCENT_COLOR});
            display: flex; align-items: center; justify-content: center;
            font-weight: 800; color: white; font-size: 1.05rem;
            box-shadow: 0 4px 14px rgba(99,102,241,0.35);
        }}
        .rp-brand-text {{ line-height: 1.15; }}
        .rp-brand-title {{ font-weight: 800; font-size: 1.02rem; color: {cfg.TEXT_PRIMARY}; letter-spacing: -0.01em; }}
        .rp-brand-subtitle {{ font-size: 0.68rem; color: {cfg.TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.08em; }}

        .rp-page-header {{
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 0.35rem;
        }}
        .rp-page-icon {{
            width: 46px; height: 46px;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(34,211,238,0.12));
            border: 1px solid {cfg.BORDER_DARK};
            display: flex; align-items: center; justify-content: center;
            font-size: 1.35rem;
        }}
        .rp-page-title {{
            font-size: 1.65rem;
            font-weight: 800;
            color: {cfg.TEXT_PRIMARY};
            letter-spacing: -0.01em;
            margin: 0;
        }}
        .rp-page-subtitle {{
            color: {cfg.TEXT_MUTED};
            font-size: 0.92rem;
            margin-top: 2px;
        }}
        .rp-divider {{
            height: 1px;
            background: linear-gradient(90deg, {cfg.BORDER_DARK}, transparent);
            margin: 1.1rem 0 1.3rem 0;
            border: none;
        }}

        .rp-hero {{
            display: flex;
            align-items: center;
            gap: 18px;
            padding: 6px 0 18px 0;
        }}
        .rp-hero-mark {{
            width: 56px; height: 56px;
            border-radius: 16px;
            background: linear-gradient(135deg, {cfg.PRIMARY_COLOR}, {cfg.ACCENT_COLOR});
            display: flex; align-items: center; justify-content: center;
            font-weight: 800; color: white; font-size: 1.6rem;
            box-shadow: 0 8px 22px rgba(99,102,241,0.35);
            flex-shrink: 0;
        }}
        .rp-hero-title {{
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.1;
            margin: 0;
            background: linear-gradient(135deg, {cfg.TEXT_PRIMARY}, {cfg.ACCENT_COLOR});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .rp-hero-tagline {{
            color: {cfg.TEXT_MUTED};
            font-size: 1rem;
            margin-top: 4px;
        }}

        .rp-card {{
            background: linear-gradient(160deg, {cfg.SURFACE_DARK} 0%, #10162A 100%);
            border: 1px solid {cfg.BORDER_DARK};
            border-radius: 16px;
            padding: 20px 22px;
            height: 100%;
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }}
        .rp-card:hover {{
            transform: translateY(-3px);
            border-color: rgba(99,102,241,0.45);
            box-shadow: 0 12px 28px rgba(99,102,241,0.16);
        }}
        .rp-card .rp-label {{
            color: {cfg.TEXT_MUTED};
            font-size: 0.76rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 8px;
        }}
        .rp-card .rp-value {{
            color: {cfg.TEXT_PRIMARY};
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1.2;
            letter-spacing: -0.01em;
        }}
        .rp-card .rp-delta {{
            font-size: 0.83rem;
            font-weight: 600;
            margin-top: 6px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }}
        .rp-delta-up {{ color: {cfg.SUCCESS_COLOR}; }}
        .rp-delta-down {{ color: {cfg.DANGER_COLOR}; }}

        .rp-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }}
        .rp-badge-increase {{ background: rgba(52,211,153,0.14); color: {cfg.SUCCESS_COLOR}; border: 1px solid rgba(52,211,153,0.28); }}
        .rp-badge-reduce   {{ background: rgba(248,113,113,0.14); color: {cfg.DANGER_COLOR}; border: 1px solid rgba(248,113,113,0.28); }}
        .rp-badge-maintain {{ background: rgba(251,191,36,0.14); color: {cfg.WARNING_COLOR}; border: 1px solid rgba(251,191,36,0.28); }}

        .rp-insight {{
            border-left: 3px solid {cfg.PRIMARY_COLOR};
            background: linear-gradient(90deg, {cfg.SURFACE_DARK}, rgba(22,26,35,0.4));
            padding: 12px 18px;
            border-radius: 0 10px 10px 0;
            margin-bottom: 12px;
            transition: border-color 0.18s ease, transform 0.18s ease;
        }}
        .rp-insight:hover {{
            border-left-color: {cfg.ACCENT_COLOR};
            transform: translateX(2px);
        }}
        .rp-insight .rp-insight-title {{
            font-weight: 700;
            color: {cfg.TEXT_PRIMARY};
            font-size: 0.94rem;
            margin-bottom: 2px;
        }}
        .rp-insight .rp-insight-detail {{
            color: {cfg.TEXT_MUTED};
            font-size: 0.87rem;
            line-height: 1.5;
        }}
        .rp-insight-category {{
            display: inline-block;
            font-size: 0.66rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {cfg.ACCENT_COLOR};
            margin-bottom: 4px;
        }}

        .rp-section-label {{
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: {cfg.TEXT_MUTED};
            margin: 0.4rem 0 0.6rem 0;
        }}

        .stButton > button {{
            border-radius: 10px !important;
            font-weight: 600 !important;
            border: 1px solid {cfg.BORDER_DARK} !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(99,102,241,0.25);
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {cfg.PRIMARY_COLOR}, #4F46E5) !important;
            border: none !important;
        }}
        [data-testid="stMetric"] {{
            background: {cfg.SURFACE_DARK};
            border: 1px solid {cfg.BORDER_DARK};
            border-radius: 14px;
            padding: 14px 16px;
        }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid {cfg.BORDER_DARK};
        }}
        [data-testid="stExpander"] {{
            border: 1px solid {cfg.BORDER_DARK};
            border-radius: 12px;
            background: {cfg.SURFACE_DARK};
        }}
        [data-testid="stTabs"] button {{
            border-radius: 8px 8px 0 0;
        }}
        div[data-baseweb="tab-list"] {{
            gap: 4px;
        }}
        .stAlert {{
            border-radius: 12px;
        }}
        h1, h2, h3, h4 {{ color: {cfg.TEXT_PRIMARY}; letter-spacing: -0.01em; }}
        hr {{ border-color: {cfg.BORDER_DARK}; }}
        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-thumb {{ background: {cfg.BORDER_DARK}; border-radius: 8px; }}
        </style>
        """))


def sidebar_brand() -> None:
    st.html(textwrap.dedent("""\
        <div class="rp-brand">
            <div class="rp-brand-mark">R</div>
            <div class="rp-brand-text">
                <div class="rp-brand-title">RetailPulse</div>
                <div class="rp-brand-subtitle">Retail Intelligence Platform</div>
            </div>
        </div>
        """))


def brand_hero(tagline: str = "") -> None:
    """Large product-name title, meant for the very top of the Home page —
    distinct from page_header(), which every other page uses."""
    st.html(textwrap.dedent(f"""\
        <div class="rp-hero">
            <div class="rp-hero-mark">R</div>
            <div>
                <p class="rp-hero-title">RetailPulse</p>
                {f'<div class="rp-hero-tagline">{tagline}</div>' if tagline else ''}
            </div>
        </div>
        """))


def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    icon_html = f'<div class="rp-page-icon">{icon}</div>' if icon else ""
    st.html(textwrap.dedent(f"""\
        <div class="rp-page-header">
            {icon_html}
            <div>
                <p class="rp-page-title">{title}</p>
                {f'<div class="rp-page-subtitle">{subtitle}</div>' if subtitle else ''}
            </div>
        </div>
        <hr class="rp-divider" />
        """))


def section_label(text: str) -> None:
    st.html(f'<div class="rp-section-label">{text}</div>')


def kpi_card(label: str, value: str, delta: str | None = None, positive: bool | None = None) -> str:
    delta_html = ""
    if delta:
        cls = "rp-delta-up" if positive else "rp-delta-down" if positive is False else ""
        arrow = "\u25b2" if positive else "\u25bc" if positive is False else ""
        delta_html = f'<div class="rp-delta {cls}">{arrow} {delta}</div>'
    return textwrap.dedent(f"""\
        <div class="rp-card">
            <div class="rp-label">{label}</div>
            <div class="rp-value">{value}</div>
            {delta_html}
        </div>
        """)


def kpi_row(cards: list[dict]) -> None:
    """cards: list of {label, value, delta, positive}"""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.html(
                kpi_card(card["label"], card["value"], card.get("delta"), card.get("positive")),
            )


def style_fig(fig: go.Figure, title: str = "", height: int = 420,
              x_title: str | None = None, y_title: str | None = None) -> go.Figure:
    """Apply the RetailPulse dark theme to a Plotly figure with explicit,
    never-blank titles/axis labels — avoids relying on implicit defaults
    that can render as empty or 'undefined' in some environments."""
    layout_kwargs = dict(
        template=PLOTLY_TEMPLATE,
        colorway=CHART_COLORWAY,
        paper_bgcolor=cfg.SURFACE_DARK,
        plot_bgcolor=cfg.SURFACE_DARK,
        font=dict(color=cfg.TEXT_PRIMARY, family="Inter, sans-serif", size=13),
        margin=dict(l=30, r=20, t=50 if title else 20, b=30),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", title_text=""),
        hoverlabel=dict(bgcolor=cfg.SURFACE_DARK, font_family="Inter, sans-serif"),
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(size=15))
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(gridcolor=cfg.BORDER_DARK, zerolinecolor=cfg.BORDER_DARK,
                      title=dict(text=x_title) if x_title else None)
    fig.update_yaxes(gridcolor=cfg.BORDER_DARK, zerolinecolor=cfg.BORDER_DARK,
                      title=dict(text=y_title) if y_title else None)
    return fig


def action_badge(action: str) -> str:
    cls = {"Increase Stock": "rp-badge-increase", "Reduce Stock": "rp-badge-reduce",
           "Maintain Stock": "rp-badge-maintain"}.get(action, "rp-badge-maintain")
    return f'<span class="rp-badge {cls}">{action}</span>'


def insight_block(title: str, detail: str, category: str = "") -> str:
    cat_html = f'<div class="rp-insight-category">{category}</div>' if category else ""
    return textwrap.dedent(f"""\
        <div class="rp-insight">
            {cat_html}
            <div class="rp-insight-title">{title}</div>
            <div class="rp-insight-detail">{detail}</div>
        </div>
        """)


def render_insights(insights: list[dict], show_category: bool = False) -> None:
    for ins in insights:
        st.html(
            insight_block(ins["title"], ins["detail"], ins["category"] if show_category else ""),
        )
