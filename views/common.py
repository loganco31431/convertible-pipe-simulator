"""Look and feel shared by the dashboard pages: desk styling, ticker strip, KPI cards, tables, charts.

Colors come from the dataviz reference palette (dark mode): blue is the market (what actually
traded, or the model with no impact), orange is the desk (its own selling and its effect). Status
green and red are reserved for gains and losses and always come with an arrow, never color alone.
"""
import html

import numpy as np
import pandas as pd
import streamlit as st

MARKET = "#3987e5"       # series 1, dark step: the market / no investor selling
DESK = "#d95926"         # series 2, dark step: the desk's own selling and its effect
NO_IMPACT, IMPACT = MARKET, DESK
GOOD, BAD = "#0ca30c", "#d03b3b"
TEXT, TEXT2, MUTED = "#ecebe6", "#c3c2b7", "#898781"
SURFACE, GRID = "#1a1a19", "#2c2c2a"
MONTH = 21

_CSS = f"""
<style>
.block-container {{ padding-top: 4.2rem; padding-bottom: 2rem; max-width: 1500px; }}
[data-testid="stMetricValue"], [data-testid="stDataFrame"], .kpi-value, .strip-value, .num {{
  font-variant-numeric: tabular-nums; }}
[data-testid="stExpander"] details summary p {{ font-size: 0.85rem; font-weight: 600; color: {TEXT2}; }}
[data-testid="stTabs"] button p {{ font-size: 0.9rem; font-weight: 600; }}
[data-testid="stForm"] {{ border: 1px solid {GRID}; background: {SURFACE}; padding: 0.9rem 1rem 0.4rem; }}
div[data-testid="stHorizontalBlock"] {{ gap: 0.75rem; }}
.strip {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.35rem 1.6rem; padding: 0.55rem 0 0.7rem;
  border-bottom: 1px solid {GRID}; margin-bottom: 0.9rem; }}
.strip-ticker {{ font-size: 1.55rem; font-weight: 700; letter-spacing: 0.02em; color: {TEXT}; margin-right: 0.4rem; }}
.strip-item {{ display: flex; flex-direction: column; }}
.strip-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; color: {MUTED}; }}
.strip-value {{ font-size: 1.0rem; font-weight: 600; color: {TEXT}; }}
.strip-sub {{ font-size: 0.78rem; font-weight: 600; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(122px, 1fr)); gap: 0.6rem; margin: 0.2rem 0 0.9rem; }}
.kpi {{ background: {SURFACE}; border: 1px solid {GRID}; border-radius: 0.3rem; padding: 0.6rem 0.8rem 0.55rem; }}
.kpi-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; color: {MUTED}; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }}
.kpi-value {{ font-size: 1.3rem; font-weight: 650; color: {TEXT}; line-height: 1.35; white-space: nowrap; }}
.kpi-sub {{ font-size: 0.76rem; color: {TEXT2}; line-height: 1.25; }}
.kpi[title] {{ cursor: help; }}
.good {{ color: {GOOD}; }} .bad {{ color: {BAD}; }} .muted {{ color: {MUTED}; }}
.section {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: {MUTED}; font-weight: 600;
  border-bottom: 1px solid {GRID}; padding-bottom: 0.3rem; margin: 1.1rem 0 0.6rem; }}
.footnote {{ font-size: 0.75rem; color: {MUTED}; margin-top: 0.2rem; }}
.pill {{ display: inline-block; font-size: 0.72rem; color: {TEXT2}; border: 1px solid {GRID}; border-radius: 1rem;
  padding: 0.05rem 0.55rem; margin-left: 0.4rem; }}
</style>
"""


def inject_css():
    st.html(_CSS)


# ---------------------------------------------------------------- number formatting
def pct(x, d=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:,.{d}f}%"


def usd(x, d=1):
    """$1.23B, $4.56M, $78.9K, $512."""
    if x is None or not np.isfinite(x):
        return "n/a"
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e9:
        return f"{sign}${a / 1e9:,.{d}f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:,.{d}f}M"
    if a >= 1e3:
        return f"{sign}${a / 1e3:,.{d}f}K"
    return f"{sign}${a:,.0f}"


def shares(x, d=2):
    if x is None or not np.isfinite(x):
        return "n/a"
    a = abs(x)
    return f"{x / 1e6:,.{d}f}M sh" if a >= 1e6 else (f"{x / 1e3:,.{d - 1}f}K sh" if a >= 1e3 else f"{x:,.0f} sh")


def bps(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+,.0f} bps"


def price(x):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


def signed(text, x, good_when_positive=True):
    """Wrap `text` in the gain or loss color, with an arrow so color is never the only cue."""
    if x is None or not np.isfinite(x) or abs(x) < 1e-12:
        return html.escape(text)
    up = x > 0
    cls = "good" if up == good_when_positive else "bad"
    text = text.lstrip("-+")                      # the arrow carries the sign
    return f'<span class="{cls}">{"▲" if up else "▼"} {html.escape(text)}</span>'


# ---------------------------------------------------------------- layout pieces
def strip(ticker: str, items: list):
    """Ticker strip. `items` are (label, value) or (label, value, sub_html)."""
    parts = [f'<span class="strip-ticker">{html.escape(ticker)}</span>']
    for it in items:
        label, value = it[0], it[1]
        sub = f' <span class="strip-sub">{it[2]}</span>' if len(it) > 2 and it[2] else ""
        parts.append(f'<div class="strip-item"><span class="strip-label">{html.escape(label)}</span>'
                     f'<span class="strip-value">{html.escape(value)}{sub}</span></div>')
    st.html(f'<div class="strip">{"".join(parts)}</div>')


def kpis(cards: list):
    """KPI cards. Each card is a dict: label, value, sub (html, optional), tip (hover text, optional)."""
    out = []
    for c in cards:
        tip = f' title="{html.escape(c["tip"])}"' if c.get("tip") else ""
        sub = f'<div class="kpi-sub">{c["sub"]}</div>' if c.get("sub") else '<div class="kpi-sub">&nbsp;</div>'
        out.append(f'<div class="kpi"{tip}><div class="kpi-label">{html.escape(c["label"])}</div>'
                   f'<div class="kpi-value">{html.escape(str(c["value"]))}</div>{sub}</div>')
    st.html(f'<div class="kpis">{"".join(out)}</div>')


def section(title: str, note: str = ""):
    n = f' <span class="pill">{html.escape(note)}</span>' if note else ""
    st.html(f'<div class="section">{html.escape(title)}{n}</div>')


def footnote(text: str):
    st.html(f'<div class="footnote">{html.escape(text)}</div>')


def table(df: pd.DataFrame, fmt: dict, signed_cols=(), highlight_first=False, height=None):
    """Formatted, index-free table; signed columns in gain / loss colors."""
    sty = df.style.format(fmt, na_rep="n/a")
    for c in signed_cols:
        if c in df:
            sty = sty.map(lambda v: "" if not isinstance(v, (int, float)) or not np.isfinite(v) or v == 0
                          else f"color: {GOOD if v > 0 else BAD}", subset=[c])
    if highlight_first and len(df):
        sty = sty.apply(lambda r: ["font-weight: 700" if r.name == df.index[0] else "" for _ in r], axis=1)
    st.dataframe(sty, width="stretch", hide_index=True, height=height if height is not None else "content")


def style(fig, title, x_title, y_title, y_fmt=None, height=320, legend=True):
    fig.update_layout(
        title=dict(text=title.upper() if title else None, font=dict(size=11, color=MUTED), x=0, xanchor="left", y=0.98),
        height=height, margin=dict(l=8, r=8, t=34, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT2, size=12), hovermode="x unified", showlegend=legend,
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID, font=dict(color=TEXT)),
        legend=dict(orientation="h", y=1.0, x=1, xanchor="right", yanchor="bottom", font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis_title=x_title, yaxis_title=y_title)
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED), title_font=dict(size=11, color=MUTED))
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, tickformat=y_fmt, tickfont=dict(color=MUTED),
                     title_font=dict(size=11, color=MUTED))
    return fig


def chart(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


@st.cache_data(ttl=3600, show_spinner=False)
def quote(ticker: str):
    """Header numbers from two years of daily data: last, 1-day and 1-month change, vol, ADV, size."""
    from pipesim.market import calibrate_market

    m = calibrate_market(ticker)
    c = m.history["Close"] if m.history is not None else pd.Series([m.s0])
    return {"mkt": m, "last": m.s0,
            "d1": float(c.iloc[-1] / c.iloc[-2] - 1) if len(c) > 1 else np.nan,
            "m1": float(c.iloc[-1] / c.iloc[-22] - 1) if len(c) > 22 else np.nan,
            "asof": c.index[-1] if hasattr(c.index, "strftime") else None}


def ticker_strip(ticker: str, q: dict, extra: list | None = None):
    m = q["mkt"]
    med = float(m.history["Volume"].tail(60).median()) if m.history is not None else m.adv_shares
    items = [("Last", price(q["last"]), signed(pct(abs(q["d1"])), q["d1"])),
             ("1M", "", signed(pct(abs(q["m1"])), q["m1"]) if np.isfinite(q["m1"]) else "n/a"),
             ("Vol 1Y", pct(m.sigma, 0)),
             ("ADV 60D", f"{m.adv_shares / 1e6:,.2f}M sh", f'<span class="muted">{usd(m.adv_dollars)}</span>'),
             ("Med vol 60D", f"{med / 1e6:,.2f}M sh", f'<span class="muted">{usd(med * m.s0)}</span>'),
             ("Mkt cap", usd(m.s0 * m.shares_out)),
             ("Shares out", f"{m.shares_out / 1e6:,.1f}M")]
    if q.get("asof") is not None:
        items.append(("As of", f"{q['asof']:%d %b %Y}"))
    strip(ticker, items + (extra or []))


def ticker_input(key: str) -> str:
    """Ticker box shared by both pages; the last ticker carries over when switching pages."""
    t = st.text_input("Ticker", value=st.session_state.get("ticker", "OTLK"), key=key,
                      help="Any Yahoo Finance ticker. Non-US listings need the suffix, e.g. BHP.AX").strip().upper()
    st.session_state["ticker"] = t or "OTLK"
    return st.session_state["ticker"]
