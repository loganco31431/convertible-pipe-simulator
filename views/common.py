"""Formatting, layout and chart helpers shared by the dashboard pages."""
import numpy as np
import streamlit as st

# Reference palette, slots 1 and 2 (validated colorblind-safe as an adjacent pair in both modes).
DARK = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
NO_IMPACT = "#3987e5" if DARK else "#2a78d6"     # blue: frictionless / no investor selling
IMPACT = "#d95926" if DARK else "#eb6834"        # orange: with volume limits and price impact
MONTH = 21                                       # trading days per month


def pct(x, d=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:,.{d}f}%"


def usd(x, d=1):
    if x is None or not np.isfinite(x):
        return "n/a"
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x / 1e9:,.{d}f}B" if x >= 1e9 else f"{sign}${x / 1e6:,.{d}f}M"


def tiles(n, per_row=3):
    """Metric slots laid out in rows of `per_row`, so values never truncate on a narrow window."""
    cols = []
    for i in range(0, n, per_row):
        cols += st.columns(per_row)
    return cols


def style(fig, title, x_title, y_title, y_fmt=None, height=340):
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=height, margin=dict(l=10, r=10, t=40, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=-0.22, x=0, xanchor="left", yanchor="top"),
                      xaxis_title=x_title, yaxis_title=y_title)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1, zeroline=False, tickformat=y_fmt)
    return fig
