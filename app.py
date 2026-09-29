"""Dashboard: enter a ticker, see how a convertible note or equity line trades out.

    python -m streamlit run app.py        (or double-click Simulator.bat)

Everything here is a thin layer over `pipesim`: market data from `market.py`, the frictionless
engine from `engine.py`, the volume- and impact-aware engine from `execution.py`, and the grid
search from `optimize.py`. The dashboard adds a deal-size capacity sweep and an Excel export.
"""
import dataclasses
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipesim import ConvertibleNote, StandbyEquityFacility, simulate, summarize
from pipesim.execution import ExecutionModel, execution_stats, run_execution
from pipesim.market import calibrate_market
from pipesim.optimize import search

st.set_page_config(page_title="PIPE Simulator", layout="wide")

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


@st.cache_data(ttl=3600, show_spinner=False)
def load_market(ticker):
    return calibrate_market(ticker)


@st.cache_data(show_spinner=False, max_entries=20)
def run_models(ticker, instr, ex, n_paths):
    mkt = load_market(ticker)
    free = simulate(instr, mkt.s0, mkt.sigma, mkt.shares_out, n_paths=n_paths)
    res = run_execution(instr, mkt, ex, n_paths=n_paths)
    gap = 1.0 - res.paths / res.base_paths
    return {"free": summarize(free), "real": summarize(res), "x": execution_stats(res),
            "gap_q": np.nanpercentile(gap, [10, 50, 90], axis=0)}


@st.cache_data(show_spinner=False, max_entries=20)
def run_replay(ticker, instr, ex):
    mkt = load_market(ticker)
    if mkt.history is None:
        return None
    h = mkt.history.tail(instr.maturity_days + ex.tail_days + 1)
    if len(h) - 1 < instr.maturity_days + ex.tail_days:
        return None
    r = run_execution(instr, mkt, ex, base_paths=h["Close"].to_numpy()[None, :],
                      volumes=h["Volume"].to_numpy(dtype=float)[None, :])
    daily = pd.DataFrame({"actual_close": r.base_paths[0], "close_with_our_selling": r.paths[0],
                          "shares_sold": r.sold[0], "market_volume": r.volumes[0],
                          "investor_cash": r.cashflows[0]}, index=h.index)
    stock_ret = float(h["Close"].iloc[instr.maturity_days] / h["Close"].iloc[0] - 1)
    return {"s": summarize(r), "x": execution_stats(r), "daily": daily, "stock_ret": stock_ret}


@st.cache_data(show_spinner=False, max_entries=10)
def run_search(ticker, instr, ex, n_paths):
    return search(instr, load_market(ticker), ex, n_paths=n_paths)


@st.cache_data(show_spinner=False, max_entries=10)
def run_capacity(ticker, instr, ex, n_paths, multiples=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Same terms at several deal sizes; tranche scales with size so the schedule is unchanged."""
    mkt = load_market(ticker)
    is_note = isinstance(instr, ConvertibleNote)
    size = instr.principal if is_note else instr.commitment
    step = instr.tranche if is_note else instr.advance
    rows = []
    for m in multiples:
        kw = {"principal": size * m, "tranche": step * m} if is_note else {"commitment": size * m, "advance": step * m}
        r = run_execution(dataclasses.replace(instr, **kw), mkt, ex, n_paths=n_paths)
        s, x = summarize(r), execution_stats(r)
        rows.append({"deal_size": size * m, "days_of_volume": size * m / mkt.adv_dollars,
                     "irr_p50": s["irr_p50"], "irr_p10": s["irr_p10"], "pnl_p50": s["pnl_p50"],
                     "prob_loss": s["prob_loss"], "dilution_p50": s["dilution_p50"],
                     "converted_p50": s["converted_p50"], "impact_bps_p50": x["impact_bps_p50"],
                     "price_drag_peak_p50": x["price_drag_peak_p50"], "exit_day_p50": x["exit_day_p50"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- inputs
with st.sidebar.form("inputs"):
    st.header("Deal")
    ticker = st.text_input("Ticker", value="OTLK", help="Any Yahoo Finance ticker. Non-US listings need the suffix, e.g. BHP.AX").strip().upper()
    oid = coupon = 0.0
    kind = st.radio("Instrument", ["Convertible note", "Equity line (SEPA)"], horizontal=True)
    if kind == "Convertible note":
        size_m = st.number_input("Principal ($M)", 0.5, 500.0, 10.0, 0.5)
        discount = st.slider("Discount to VWAP", 0.0, 0.30, 0.10, 0.01, format="%.2f")
        oid = st.slider("Original issue discount", 0.0, 0.20, 0.05, 0.01, format="%.2f")
        coupon = st.slider("Coupon (annual)", 0.0, 0.15, 0.06, 0.01, format="%.2f")
        lookback = st.number_input("VWAP lookback (trading days)", 1, 30, 10)
        cadence = st.number_input("Days between conversions", 1, 63, 10)
        step_pct = st.slider("Tranche (% of principal)", 0.01, 0.50, 0.10, 0.01, format="%.2f")
        floor_pct = st.slider("Floor price (% of today's price, 0 = none)", 0.0, 1.0, 0.0, 0.05, format="%.2f")
        months = st.number_input("Term (months)", 3, 36, 12)
    else:
        size_m = st.number_input("Commitment ($M)", 1.0, 1000.0, 50.0, 1.0)
        discount = st.slider("Discount to VWAP", 0.0, 0.20, 0.05, 0.01, format="%.2f")
        lookback = st.number_input("Pricing period (trading days)", 1, 30, 3)
        cadence = st.number_input("Days between draws", 1, 63, 5)
        step_pct = st.slider("Advance (% of commitment)", 0.01, 0.30, 0.04, 0.01, format="%.2f")
        floor_pct = st.slider("Issuer won't draw below (% of today's price)", 0.0, 1.0, 0.0, 0.05, format="%.2f")
        months = st.number_input("Term (months)", 3, 36, 12)

    st.header("Execution")
    participation = st.slider("Max share of daily volume sold", 0.01, 0.50, 0.10, 0.01, format="%.2f")
    eta = st.slider("Price impact strength", 0.0, 2.0, 0.5, 0.1,
                    help="Square-root law coefficient. 0 = selling never moves the price. Placeholder, not fitted.")
    carried = st.slider("Share of each day's move that carries over", 0.0, 1.0, 0.5, 0.05)
    no_recovery = st.checkbox("Price never recovers from our selling")
    half_life = st.number_input("Recovery half-life (trading days)", 1.0, 250.0, 10.0, 1.0, disabled=no_recovery)
    residual = st.slider("Share of carried move that never fades", 0.0, 1.0, 0.3, 0.05, disabled=no_recovery)
    n_paths = st.select_slider("Simulated paths", [500, 1000, 2000, 3000, 5000], value=2000)
    go_btn = st.form_submit_button("Run", type="primary", width="stretch")

if go_btn:
    st.session_state["ran"] = True
if not st.session_state.get("ran"):
    st.title("Convertible Note & PIPE Simulator")
    st.write("Enter a ticker and deal terms on the left, then press **Run**.")
    st.stop()

try:
    with st.spinner(f"Pulling {ticker} from Yahoo Finance..."):
        mkt = load_market(ticker)
except Exception as e:
    st.error(f"Couldn't load {ticker}: {e}")
    st.stop()

size = size_m * 1e6
days = int(months * MONTH)
if kind == "Convertible note":
    instr = ConvertibleNote(principal=size, tranche=step_pct * size, discount=discount, oid=oid, coupon=coupon,
                            vwap_lookback=int(lookback), cadence=int(cadence), floor_price=floor_pct * mkt.s0,
                            maturity_days=days)
else:
    instr = StandbyEquityFacility(commitment=size, advance=step_pct * size, discount=discount,
                                  vwap_lookback=int(lookback), cadence=int(cadence),
                                  min_draw_price=floor_pct * mkt.s0, maturity_days=days)
ex = ExecutionModel(participation=participation, eta=eta, permanent=carried,
                    half_life=float("inf") if no_recovery else half_life, residual=residual)

# ---------------------------------------------------------------- market snapshot
st.title(f"{ticker}: {kind.lower()}, {usd(size, 0)} at a {pct(discount, 0)} discount")
c = tiles(6)
c[0].metric("Price", f"${mkt.s0:,.2f}")
c[1].metric("Volatility (1y)", pct(mkt.sigma, 0))
c[2].metric("Market cap", usd(mkt.s0 * mkt.shares_out))
c[3].metric("Traded per day", usd(mkt.adv_dollars), help="Average over the last 60 trading days")
c[4].metric("Deal size", f"{size / mkt.adv_dollars:,.1f} days", help="Deal size divided by the stock's average daily dollar volume")
c[5].metric("Time to sell at cap", f"{size / (participation * mkt.adv_dollars):,.0f} days",
            help="Trading days to sell shares worth the full deal at the volume cap, ignoring the discount")

with st.spinner("Simulating..."):
    out = run_models(ticker, instr, ex, n_paths)
f, r, x = out["free"], out["real"], out["x"]

st.subheader("What the deal earns once selling is realistic")
c = tiles(6)
c[0].metric("Median profit", usd(r["pnl_p50"], 2), delta=usd(r["pnl_p50"] - f["pnl_p50"], 2) + " vs no impact")
c[1].metric("Median IRR", pct(r["irr_p50"], 0), delta=f"{100 * (r['irr_p50'] - f['irr_p50']):+,.0f} pts")
c[2].metric("Bad case IRR (p10)", pct(r["irr_p10"], 0), delta=f"{100 * (r['irr_p10'] - f['irr_p10']):+,.0f} pts")
c[3].metric("Chance of a loss", pct(r["prob_loss"]))
c[4].metric("Issuer dilution (median)", pct(r["dilution_p50"]), delta=f"{100 * (r['dilution_p50'] - f['dilution_p50']):+,.1f} pts",
            delta_color="off")
c[5].metric("Cost of our own selling", f"{x['impact_bps_p50']:,.0f} bps",
            help="Median price impact paid on every share sold, in basis points of the sale")

cmp = pd.DataFrame({
    "No impact (old model)": [pct(f["irr_p50"]), pct(f["irr_p10"]), pct(f["irr_p90"]), usd(f["pnl_p50"], 2),
                              pct(f["prob_loss"]), pct(f["dilution_p50"]), pct(f["dilution_p90"]), usd(f["converted_p50"]),
                              "none", "none", "same day"],
    "With volume limits and impact": [pct(r["irr_p50"]), pct(r["irr_p10"]), pct(r["irr_p90"]), usd(r["pnl_p50"], 2),
                                      pct(r["prob_loss"]), pct(r["dilution_p50"]), pct(r["dilution_p90"]),
                                      usd(r["converted_p50"]), pct(x["price_drag_peak_p50"]), pct(x["price_drag_p50"]),
                                      f"day {x['exit_day_p50']:,.0f}"],
}, index=["Median IRR", "Bad case IRR (p10)", "Good case IRR (p90)", "Median profit", "Chance of a loss",
          "Median dilution", "Bad case dilution (p90)", "Converted (median)", "Price drag from our selling, worst",
          "Price drag from our selling, at end", "Fully sold by"])

tabs = st.tabs(["Overview", "Replay on real prices", "Best setup", "How big a deal can it take", "Assumptions"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    left, right = st.columns([1, 1])
    with left:
        lo = np.nanpercentile(np.concatenate([f["irr"], r["irr"]]), 1)
        hi = np.nanpercentile(np.concatenate([f["irr"], r["irr"]]), 99)
        fig = go.Figure()
        for name, arr, col in (("No impact", f["irr"], NO_IMPACT), ("With impact", r["irr"], IMPACT)):
            a = arr[np.isfinite(arr) & (arr >= lo) & (arr <= hi)]   # drop the outer 1% tails
            fig.add_histogram(x=a, name=name, marker_color=col, opacity=0.6, nbinsx=50,
                              hovertemplate="IRR %{x:.0%}<br>%{y} paths<extra>" + name + "</extra>")
        fig.update_layout(barmode="overlay", bargap=0.05)
        fig = style(fig, "IRR across simulated paths", "Annualized IRR", "Paths", height=360)
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch")
    with right:
        q = out["gap_q"]
        t = np.arange(q.shape[1])
        fig = go.Figure()
        fig.add_scatter(x=np.r_[t, t[::-1]], y=np.r_[q[2], q[0][::-1]], fill="toself", fillcolor=IMPACT, opacity=0.18,
                        line=dict(width=0), hoverinfo="skip", name="10th to 90th percentile")
        fig.add_scatter(x=t, y=q[1], line=dict(color=IMPACT, width=2), name="Median",
                        hovertemplate="Day %{x}: %{y:.1%} below<extra></extra>")
        fig.add_vline(x=instr.maturity_days, line=dict(width=1, dash="dot"),
                      annotation_text="maturity", annotation_position="top")
        fig = style(fig, "Price gap caused by our selling", "Trading day",
                    "Price gap", y_fmt=".0%", height=360)
        st.plotly_chart(fig, width="stretch")
    st.dataframe(cmp, width="stretch", height=35 * (len(cmp) + 1) + 3)

# ---------------------------------------------------------------- replay
with tabs[1]:
    rep = run_replay(ticker, instr, ex)
    if rep is None:
        st.info(f"Not enough history to replay a {months}-month term plus the {ex.tail_days}-day selling tail. "
                "Try a shorter term.")
    else:
        d, sr, xr = rep["daily"], rep["s"], rep["x"]
        st.caption(f"The same deal run on {ticker}'s actual prices and volume from {d.index[0]:%d %b %Y} to "
                   f"{d.index[-1]:%d %b %Y}, with our own selling layered on top of what really traded.")
        c = tiles(6)
        c[0].metric("Profit", usd(sr["pnl_p50"], 2))
        c[1].metric("IRR", pct(sr["irr_p50"], 0))
        c[2].metric("Stock over the term", pct(rep["stock_ret"], 0))
        c[3].metric("Dilution", pct(sr["dilution_p50"]))
        c[4].metric("Worst price drag", pct(xr["price_drag_peak_p50"]))
        fig = go.Figure()
        fig.add_scatter(x=d.index, y=d["actual_close"], name="What actually traded", line=dict(color=NO_IMPACT, width=2))
        fig.add_scatter(x=d.index, y=d["close_with_our_selling"], name="With our selling", line=dict(color=IMPACT, width=2))
        st.plotly_chart(style(fig, "Price: actual vs with our selling", None, "Close ($)", y_fmt="$,.2f"),
                        width="stretch")
        share = np.where(d["market_volume"] > 0, d["shares_sold"] / d["market_volume"], 0.0)
        fig = go.Figure(go.Bar(x=d.index, y=share, marker_color=IMPACT, name="Our share of volume",
                               hovertemplate="%{x|%d %b %Y}: %{y:.1%} of volume<extra></extra>"))
        st.plotly_chart(style(fig, "Our selling as a share of each day's volume", None, "Share of volume",
                              y_fmt=".0%", height=260), width="stretch")
        with st.expander("Lookahead check"):
            st.write("Each conversion and sale uses only closes and volume up to that day. The leak risk is in the "
                     "inputs: volatility and volume are calibrated on this same window, so the replay is not a clean "
                     "out-of-sample test.")

# ---------------------------------------------------------------- best setup
with tabs[2]:
    st.write("Tries every combination of conversion frequency (5, 10, 21 days), tranche size (5%, 10%, 20%) and "
             "selling speed (5%, 10%, 20% of volume), all on the same simulated prices. Ranked by median profit "
             "among setups with a 10% or lower chance of loss.")
    if st.button("Find the best setup (about 30 seconds)"):
        st.session_state["search_key"] = (ticker, instr, ex)
    if st.session_state.get("search_key") == (ticker, instr, ex):
        with st.spinner("Searching 27 setups..."):
            df = run_search(ticker, instr, ex, max(1_000, n_paths // 2))
        b = df.iloc[0]
        msg = (f"convert {pct(b.tranche_pct, 0)} every {b.cadence_days:.0f} days and sell at {pct(b.participation, 0)} "
               f"of volume. Median profit {usd(b.pnl_p50, 2)}, IRR {pct(b.irr_p50, 0)}, chance of loss {pct(b.prob_loss)}.")
        if b.eligible:
            st.success("Best: " + msg)
        else:
            st.warning("No setup keeps the chance of loss at 10% or lower. Least bad: " + msg)
        show = df.drop(columns="eligible").rename(columns={
            "cadence_days": "Every (days)", "tranche_pct": "Tranche", "participation": "Share of volume",
            "irr_p10": "IRR p10", "irr_p50": "IRR median", "prob_loss": "P(loss)", "pnl_p50": "Profit median",
            "moic_p50": "MOIC", "dilution_p50": "Dilution", "converted_p50": "Converted",
            "impact_bps_p50": "Impact (bps)", "exit_day_p50": "Sold by day"})
        st.dataframe(show.style.format({"Tranche": "{:.0%}", "Share of volume": "{:.0%}", "IRR p10": "{:.0%}",
                                        "IRR median": "{:.0%}", "P(loss)": "{:.1%}", "Profit median": "${:,.0f}",
                                        "MOIC": "{:.2f}x", "Dilution": "{:.1%}", "Converted": "${:,.0f}",
                                        "Impact (bps)": "{:,.0f}", "Sold by day": "{:,.0f}"}),
                     width="stretch", hide_index=True)
        st.session_state["search_df"] = df

# ---------------------------------------------------------------- capacity
with tabs[3]:
    st.write("The same terms at a quarter, half, 1x, 2x and 4x the deal size. Past a point the stock can't absorb "
             "the shares: impact rises, the exit takes longer and the return falls.")
    with st.spinner("Running five deal sizes..."):
        cap = run_capacity(ticker, instr, ex, min(n_paths, 1_500))
    left, right = st.columns(2)
    labels = [usd(v, 1) for v in cap["deal_size"]]
    with left:
        fig = go.Figure(go.Scatter(x=labels, y=cap["irr_p50"], mode="lines+markers", line=dict(color=IMPACT, width=2),
                                   marker=dict(size=9), name="Median IRR",
                                   hovertemplate="%{x}: %{y:.0%}<extra></extra>"))
        st.plotly_chart(style(fig, "Median IRR by deal size", "Deal size", "IRR", y_fmt=".0%", height=300),
                        width="stretch")
    with right:
        fig = go.Figure(go.Scatter(x=labels, y=cap["dilution_p50"], mode="lines+markers", line=dict(color=IMPACT, width=2),
                                   marker=dict(size=9), name="Median dilution",
                                   hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
        st.plotly_chart(style(fig, "Issuer dilution by deal size", "Deal size", "Dilution", y_fmt=".0%", height=300),
                        width="stretch")
    st.dataframe(cap.rename(columns={
        "deal_size": "Deal size", "days_of_volume": "Days of volume", "irr_p50": "IRR median", "irr_p10": "IRR p10",
        "pnl_p50": "Profit median", "prob_loss": "P(loss)", "dilution_p50": "Dilution", "converted_p50": "Converted",
        "impact_bps_p50": "Impact (bps)", "price_drag_peak_p50": "Worst price drag", "exit_day_p50": "Sold by day"
    }).style.format({"Deal size": "${:,.0f}", "Days of volume": "{:.1f}", "IRR median": "{:.0%}", "IRR p10": "{:.0%}",
                     "Profit median": "${:,.0f}", "P(loss)": "{:.1%}", "Dilution": "{:.1%}", "Converted": "${:,.0f}",
                     "Impact (bps)": "{:,.0f}", "Worst price drag": "{:.1%}", "Sold by day": "{:,.0f}"}),
        width="stretch", hide_index=True)

# ---------------------------------------------------------------- assumptions
with tabs[4]:
    st.markdown(f"""
- **Prices** are simulated with zero drift and the stock's realized 1-year volatility ({pct(mkt.sigma, 0)}). No jumps,
  no default, no delisting, so the real loss case (the issuer failing before conversion) is not in here.
- **Volume** is lognormal around the 60-day average ({mkt.adv_shares / 1e6:,.2f}M shares a day). It does not rise on
  down days or react to our selling.
- **Price impact** follows the square-root law: selling q shares into daily volume V costs
  impact strength x daily vol x sqrt(q / V). Part of that move carries over, and part of the carried move fades with a
  half-life. **These settings are placeholders, not fitted to real PIPE selling.** A desk's own record of how much its
  selling moves a stock is the right calibration.
- **Not modeled:** contractual daily conversion caps, the 4.99% ownership blocker, registration delays, borrow for
  hedging, and any signaling effect from the market knowing a PIPE investor is selling.
- **No lookahead:** every decision on day t uses data up to day t only.
""")

# ---------------------------------------------------------------- export
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as w:
    pd.DataFrame({"input": ["ticker", "instrument", "size", "discount", "cadence_days", "step_pct", "lookback_days",
                            "floor_pct_of_spot", "term_months", "participation", "impact_eta", "carried",
                            "half_life_days", "residual", "paths", "spot", "vol", "shares_out", "adv_dollars"],
                  "value": [ticker, kind, size, discount, cadence, step_pct, lookback, floor_pct, months, participation,
                            eta, carried, "never" if no_recovery else half_life, residual, n_paths, mkt.s0, mkt.sigma,
                            mkt.shares_out, mkt.adv_dollars]}).to_excel(w, sheet_name="Inputs", index=False)
    cmp.to_excel(w, sheet_name="Summary")
    cap.to_excel(w, sheet_name="Deal size", index=False)
    rep = run_replay(ticker, instr, ex)
    if rep is not None:
        daily = rep["daily"].copy()
        if daily.index.tz is not None:
            daily.index = daily.index.tz_localize(None)
        daily.to_excel(w, sheet_name="Replay daily")
    if st.session_state.get("search_key") == (ticker, instr, ex) and "search_df" in st.session_state:
        st.session_state["search_df"].to_excel(w, sheet_name="Best setup", index=False)
st.sidebar.download_button("Download results (Excel)", buf.getvalue(), file_name=f"{ticker}_pipe_sim.xlsx",
                           width="stretch")
