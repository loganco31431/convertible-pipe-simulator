"""Deal economics page: Monte Carlo on daily prices, frictionless vs volume- and impact-aware."""
import dataclasses
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipesim import ConvertibleNote, StandbyEquityFacility, Warrant, simulate, summarize
from pipesim.execution import ExecutionModel, execution_stats, run_execution
from pipesim.instruments import TRADING_DAYS
from pipesim.market import calibrate_market
from pipesim.optimize import search
from pipesim.presets import PRESETS
from pipesim.warrants import warrant_vol
from views.common import IMPACT, MONTH, NO_IMPACT, pct, style, tiles, usd

NOTE_PRICING = {"Lowest daily VWAP before the notice": "lowest", "Average, including the notice day": "average"}
SEPA_PRICING = {"Option 1: same-day VWAP": "option1", "Option 2: lowest daily VWAP": "option2",
                "Trailing average (not a filed structure)": "trailing"}
REPAY = {"At maturity only": "none", "Monthly from a start date": "scheduled",
         "Monthly after the stock falls below the floor": "on_trigger"}


def pick(options: dict, value):
    return list(options.values()).index(value)


@st.cache_data(ttl=3600, show_spinner=False)
def load_market(ticker):
    return calibrate_market(ticker)


def scale(instr, warrant, m):
    """The same deal at `m` times the size: every dollar and share amount scales, every rate stays."""
    if isinstance(instr, ConvertibleNote):
        instr = dataclasses.replace(instr, principal=instr.principal * m, tranche=instr.tranche * m,
                                    monthly_payment=instr.monthly_payment * m, structuring_fee=instr.structuring_fee * m)
    else:
        instr = dataclasses.replace(instr, commitment=instr.commitment * m, advance=instr.advance * m,
                                    commitment_shares=instr.commitment_shares * m,
                                    structuring_fee=instr.structuring_fee * m)
    if warrant is not None:
        warrant = dataclasses.replace(warrant, shares=warrant.shares * m)
    return instr, warrant


@st.cache_data(show_spinner=False, max_entries=20)
def run_models(ticker, instr, warrant, ex, n_paths):
    mkt = load_market(ticker)
    free = simulate(instr, mkt.s0, mkt.sigma, mkt.shares_out, n_paths=n_paths, warrant=warrant)
    res = run_execution(instr, mkt, ex, n_paths=n_paths, warrant=warrant)
    gap = 1.0 - res.paths / res.base_paths
    size = instr.principal if isinstance(instr, ConvertibleNote) else instr.commitment
    mix = {"Converted or drawn into shares": float(np.mean(res.converted)) / size,
           "Repaid in cash before maturity": float(np.mean(res.repaid_cash)) / size,
           "Repaid in cash at maturity": float(np.mean(res.repaid_maturity)) / size}
    return {"free": summarize(free), "real": summarize(res), "x": execution_stats(res),
            "gap_q": np.nanpercentile(gap, [10, 50, 90], axis=0), "mix": mix}


@st.cache_data(show_spinner=False, max_entries=20)
def run_replay(ticker, instr, warrant, ex):
    mkt = load_market(ticker)
    if mkt.history is None:
        return None
    h = mkt.history.tail(instr.maturity_days + ex.tail_days + 1)
    if len(h) - 1 < instr.maturity_days + ex.tail_days:
        return None
    r = run_execution(instr, mkt, ex, base_paths=h["Close"].to_numpy()[None, :],
                      volumes=h["Volume"].to_numpy(dtype=float)[None, :], warrant=warrant)
    daily = pd.DataFrame({"actual_close": r.base_paths[0], "close_with_our_selling": r.paths[0],
                          "shares_sold": r.sold[0], "market_volume": r.volumes[0],
                          "investor_cash": r.cashflows[0]}, index=h.index)
    stock_ret = float(h["Close"].iloc[instr.maturity_days] / h["Close"].iloc[0] - 1)
    return {"s": summarize(r), "x": execution_stats(r), "daily": daily, "stock_ret": stock_ret}


@st.cache_data(show_spinner=False, max_entries=10)
def run_search(ticker, instr, ex, n_paths):
    return search(instr, load_market(ticker), ex, n_paths=n_paths)


@st.cache_data(show_spinner=False, max_entries=10)
def run_capacity(ticker, instr, warrant, ex, n_paths, multiples=(0.25, 0.5, 1.0, 2.0, 4.0)):
    mkt = load_market(ticker)
    size = instr.principal if isinstance(instr, ConvertibleNote) else instr.commitment
    rows = []
    for m in multiples:
        i_m, w_m = scale(instr, warrant, m)
        r = run_execution(i_m, mkt, ex, n_paths=n_paths, warrant=w_m)
        s, x = summarize(r), execution_stats(r)
        ret = "irr" if isinstance(instr, ConvertibleNote) else "margin"
        rows.append({"deal_size": size * m, "days_of_volume": size * m / mkt.adv_dollars,
                     "irr_p50": s.get(f"pkg_{ret}_p50", s[f"{ret}_p50"]),
                     "irr_p10": s.get(f"pkg_{ret}_p10", s[f"{ret}_p10"]),
                     "pnl_p50": s.get("pkg_pnl_p50", s["pnl_p50"]),
                     "prob_loss": s.get("pkg_prob_loss", s["prob_loss"]), "dilution_p50": s["dilution_p50"],
                     "converted_p50": s["converted_p50"], "impact_bps_p50": x["impact_bps_p50"],
                     "price_drag_peak_p50": x["price_drag_peak_p50"], "exit_day_p50": x["exit_day_p50"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- inputs
st.sidebar.header("Deal")
ticker = st.sidebar.text_input("Ticker", value="OTLK",
                               help="Any Yahoo Finance ticker. Non-US listings need the suffix, e.g. BHP.AX").strip().upper()
preset_name = st.sidebar.selectbox("Start from the terms of", list(PRESETS), index=0,
                                   help="Terms read from filed Yorkville agreements. Every term can be changed below.")
try:
    with st.spinner(f"Pulling {ticker} from Yahoo Finance..."):
        mkt = load_market(ticker)
except Exception as e:
    st.error(f"Couldn't load {ticker}: {e}")
    st.stop()

preset = PRESETS[preset_name](mkt.s0)
base = preset.note or preset.sepa
is_note = preset.note is not None
base_size = base.principal if is_note else base.commitment
# a deal sized for the filing's company is usually too big for this one: start at 7% of market cap
start_size = float(np.clip(round(min(base_size, 0.07 * mkt.s0 * mkt.shares_out) / 5e5) * 5e5, 5e5, 5e8))
k = f"{ticker}|{preset_name}|"          # widget keys: new ticker or preset resets every default

with st.sidebar.form("inputs"):
    st.caption("Prices below are entered as a share of today's price, the way the contracts set them at signing.")
    if is_note:
        size_m = st.number_input("Principal ($M)", 0.5, 500.0, start_size / 1e6, 0.5, key=k + "size")
        oid = st.slider("Funding discount (paid less than face)", 0.0, 0.20, base.oid, 0.01, key=k + "oid")
        coupon = st.slider("Interest (annual)", 0.0, 0.20, base.coupon, 0.01, key=k + "cpn")
        months = st.number_input("Term (months)", 3, 36, int(round(base.maturity_days / MONTH)), key=k + "term")
        with st.expander("Conversion", expanded=base.tranche > 0):
            converts = st.checkbox("Investor converts into shares", value=base.tranche > 0, key=k + "conv")
            discount = st.slider("Discount to VWAP", 0.0, 0.30, base.discount, 0.01, key=k + "disc")
            pricing = st.selectbox("VWAP measure", list(NOTE_PRICING), index=pick(NOTE_PRICING, base.pricing), key=k + "pr")
            lookback = st.number_input("VWAP days", 1, 30, base.vwap_lookback, key=k + "lb")
            fixed_pct = st.number_input("Fixed price (x today's price, 0 = none)", 0.0, 5.0,
                                        round(base.fixed_price / mkt.s0, 2), 0.05, key=k + "fix")
            floor_pct = st.number_input("Floor price (x today's price, 0 = none)", 0.0, 1.0,
                                        round(base.floor_price / mkt.s0, 2), 0.05, key=k + "flr")
            cadence = st.number_input("Days between conversions", 1, 63, base.cadence, key=k + "cad")
            step_pct = st.slider("Converted each time (share of principal)", 0.01, 0.50,
                                 float(np.clip(base.tranche / base.principal if base.tranche else 0.10, 0.01, 0.5)),
                                 0.01, key=k + "step")
        with st.expander("Repayment in cash", expanded=base.payment_mode != "none"):
            repay = st.selectbox("Issuer repays", list(REPAY), index=pick(REPAY, base.payment_mode), key=k + "rep")
            pay_pct = st.slider("Monthly payment (share of principal)", 0.0, 0.50,
                                round(base.monthly_payment / base.principal, 3), 0.005, format="%.3f", key=k + "pay")
            premium = st.slider("Premium on each payment", 0.0, 0.20, base.payment_premium, 0.01, key=k + "prem")
            pay_start = st.number_input("First payment (trading days after closing)", 1, 252, base.payment_start_day,
                                        key=k + "ps", help="Used for payments from a start date. 60 calendar days is about 42.")
            in_shares = st.checkbox("Payments settled by equity-line advances", value=base.pay_in_shares, key=k + "pis",
                                    help="The pre-paid advance loop: instead of cash, the issuer delivers shares at "
                                         "the equity-line price and the proceeds offset the note.")
        w0 = preset.warrant
        with st.expander("Warrants", expanded=w0 is not None):
            cover = st.slider("Coverage (share of principal, 0 = none)", 0.0, 1.0,
                              round(w0.shares * w0.strike / base.principal, 3) if w0 else 0.0, 0.005, format="%.3f",
                              key=k + "wc", help="Warrant shares x strike, divided by principal")
            strike_pct = st.number_input("Strike (x today's price)", 0.1, 5.0,
                                         round(w0.strike / mkt.s0, 2) if w0 else 1.0, 0.05, key=k + "wk")
            w_years = st.number_input("Expires after (years)", 0.5, 10.0, w0.term_days / TRADING_DAYS if w0 else 5.0,
                                      0.5, key=k + "wt")
            w_wait = st.number_input("Exercisable after (months)", 0, 24, int(round(w0.exercisable_after / MONTH)) if w0 else 0,
                                     key=k + "ww")
            vol_cap = st.slider("Volatility used to value them, capped at", 0.2, 2.0, 1.0, 0.05, key=k + "wv",
                                help=f"{ticker}'s realized volatility is {pct(mkt.sigma, 0)}. Black-Scholes at very "
                                     "high volatility values a warrant close to the stock itself, so desks haircut it.")
    else:
        size_m = st.number_input("Commitment ($M)", 1.0, 1000.0, max(start_size / 1e6, 1.0), 1.0, key=k + "size")
        adv_pct = st.slider("Each advance (share of commitment)", 0.01, 0.30,
                            float(np.clip(base.advance / base.commitment, 0.01, 0.30)), 0.01, key=k + "step")
        pricing = st.selectbox("Pricing", list(SEPA_PRICING), index=pick(SEPA_PRICING, base.pricing), key=k + "pr")
        discount = st.slider("Discount to VWAP", 0.0, 0.20, base.discount, 0.005, format="%.3f", key=k + "disc")
        lookback = st.number_input("Pricing days (Option 2 and trailing)", 1, 30, max(base.vwap_lookback, 1), key=k + "lb")
        cadence = st.number_input("Days between advances", 1, 63, base.cadence, key=k + "cad")
        months = st.number_input("Term (months)", 3, 36, int(round(base.maturity_days / MONTH)), key=k + "term")
        with st.expander("Limits on each advance", expanded=True):
            vol_thr = st.number_input("Option 1 volume threshold (0 = none)", 0.0, 1.0, base.volume_threshold, 0.05,
                                      key=k + "vt", help="The advance is cut if volume that day is below advance / threshold")
            adv_cap = st.number_input("Max advance (x prior 5-day avg volume, 0 = none)", 0.0, 10.0,
                                      base.max_advance_adv, 0.25, key=k + "ac")
            floor_pct = st.number_input("Minimum acceptable price (x today's price, 0 = none)", 0.0, 1.0,
                                        round(base.min_draw_price / mkt.s0, 2), 0.05, key=k + "flr")
    with st.expander("Caps and fees"):
        own_cap = st.number_input("Ownership cap (0 = none)", 0.0, 0.25, base.ownership_cap, 0.0001, format="%.4f",
                                  key=k + "own")
        exch_cap = st.number_input("Exchange cap (0 = none, or approved by shareholders)", 0.0, 1.0, base.exchange_cap,
                                   0.0001, format="%.4f", key=k + "exc")
        fee = st.number_input("Structuring fee paid to the investor ($)", 0.0, 5e6, float(base.structuring_fee), 5e3,
                              key=k + "fee")
        commit_sh = 0.0
        if not is_note:
            commit_sh = st.number_input("Commitment shares (thousands)", 0.0, 1e5,
                                        round(base.commitment_shares * (start_size / base_size) / 1e3, 1), 5.0,
                                        key=k + "cs")
    with st.expander("Execution", expanded=True):
        participation = st.slider("Max share of daily volume sold", 0.01, 0.50, 0.10, 0.01, format="%.2f")
        slip = st.slider("Spread and fees on each sale", 0.0, 0.05, 0.02, 0.0025, format="%.4f",
                         help="Paid on every share sold, on top of price impact. Sub-dollar stocks trade in one-cent "
                              "ticks, so the spread alone can be over 1%.")
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
    st.title("Deal economics")
    st.write("Pick a ticker and a deal to start from on the left, change any term, then press **Run**.")
    st.caption(f"{preset.name}. Source: {preset.source}.")
    st.stop()

size = size_m * 1e6
days = int(months * MONTH)
warrant = None
if is_note:
    instr = ConvertibleNote(
        principal=size, oid=oid, coupon=coupon, discount=discount, vwap_lookback=int(lookback), cadence=int(cadence),
        tranche=step_pct * size if converts else 0.0, floor_price=floor_pct * mkt.s0, maturity_days=days,
        pricing=NOTE_PRICING[pricing], fixed_price=fixed_pct * mkt.s0, payment_mode=REPAY[repay],
        monthly_payment=pay_pct * size, payment_premium=premium, payment_start_day=int(pay_start),
        trigger_days=base.trigger_days, trigger_window=base.trigger_window, trigger_lag=base.trigger_lag,
        cure_days=base.cure_days, pay_in_shares=in_shares, sepa_discount=base.sepa_discount,
        ownership_cap=own_cap, exchange_cap=exch_cap, structuring_fee=fee, sell_slippage=slip)
    if cover > 0:
        warrant = Warrant.from_coverage(size, cover, mkt.s0, strike_pct=strike_pct, term_days=int(w_years * TRADING_DAYS),
                                        exercisable_after=int(w_wait * MONTH), vol_cap=vol_cap)
else:
    instr = StandbyEquityFacility(
        commitment=size, advance=adv_pct * size, discount=discount, vwap_lookback=int(lookback), cadence=int(cadence),
        min_draw_price=floor_pct * mkt.s0, maturity_days=days, pricing=SEPA_PRICING[pricing],
        volume_threshold=vol_thr, max_advance_adv=adv_cap, ownership_cap=own_cap, exchange_cap=exch_cap,
        commitment_shares=commit_sh * 1e3, structuring_fee=fee, sell_slippage=slip)
ex = ExecutionModel(participation=participation, eta=eta, permanent=carried,
                    half_life=float("inf") if no_recovery else half_life, residual=residual)
converting = (instr.tranche > 0) if is_note else True
kind = ("note" if converting else "loan") if is_note else "equity line"

# ---------------------------------------------------------------- market snapshot
st.title(f"{ticker}: {usd(size, 1)} {kind}" + (" with warrants" if warrant else ""))
st.caption(f"Started from: {preset.name}. Source: {preset.source}. Filled in by the model: {preset.assumed}"
           .replace("$", r"\$"))
c = tiles(6)
c[0].metric("Price", f"${mkt.s0:,.2f}")
c[1].metric("Volatility (1y)", pct(mkt.sigma, 0))
c[2].metric("Market cap", usd(mkt.s0 * mkt.shares_out))
c[3].metric("Traded per day", usd(mkt.adv_dollars), help="Average over the last 60 trading days")
c[4].metric("Deal size", f"{size / mkt.adv_dollars:,.1f} days", help="Deal size divided by the stock's average daily dollar volume")
c[5].metric("Time to sell at cap", f"{size / (participation * mkt.adv_dollars):,.0f} days",
            help="Trading days to sell shares worth the full deal at the volume cap, ignoring the discount")

with st.spinner("Simulating..."):
    out = run_models(ticker, instr, warrant, ex, n_paths)
f, r, x = out["free"], out["real"], out["x"]
P = "pkg_" if warrant else ""           # headline numbers include the warrants when there are any
RET = "irr" if is_note else "margin"    # an equity line ties up no capital, so it has a margin, not an IRR
RET_NAME = "IRR" if is_note else "margin"
RET_HELP = ("Annualized. A note that converts within a few months shows a very high IRR on a modest profit, so "
            "read it next to the profit and the margin in the table below." if is_note else
            "Profit over dollars drawn. The desk sells the shares before it pays for them, so no capital is tied up "
            "and an IRR would be meaningless.")
DP = 0 if is_note else 1

st.subheader("What the deal earns once selling is realistic" + (" (cash plus warrants)" if warrant else ""))
c = tiles(6)
c[0].metric("Median profit", usd(r[P + "pnl_p50"], 2), delta=usd(r[P + "pnl_p50"] - f[P + "pnl_p50"], 2) + " vs no impact")
c[1].metric(f"Median {RET_NAME}", pct(r[P + RET + "_p50"], DP), help=RET_HELP,
            delta=f"{100 * (r[P + RET + '_p50'] - f[P + RET + '_p50']):+,.{DP}f} pts")
c[2].metric(f"Bad case {RET_NAME} (p10)", pct(r[P + RET + "_p10"], DP),
            delta=f"{100 * (r[P + RET + '_p10'] - f[P + RET + '_p10']):+,.{DP}f} pts")
c[3].metric("Chance of a loss", pct(r[P + "prob_loss"]))
c[4].metric("Issuer dilution (median)", pct(r["dilution_p50"]), delta=f"{100 * (r['dilution_p50'] - f['dilution_p50']):+,.1f} pts",
            delta_color="off")
c[5].metric("Cost of our own selling", "none" if not np.isfinite(x["impact_bps_p50"]) else f"{x['impact_bps_p50']:,.0f} bps",
            help="Median price impact paid on every share sold, in basis points of the sale")

if warrant:
    st.subheader("Warrants")
    c = tiles(6)
    c[0].metric("Warrant shares", f"{warrant.shares / 1e6:,.2f}M")
    c[0].caption(f"struck at \\${warrant.strike:,.4f}, {pct(warrant.shares / mkt.shares_out)} of shares outstanding")
    c[1].metric("Value at closing", usd(r["warrant_value0"], 2), help="Black-Scholes at today's price")
    c[1].caption(f"{pct(r['warrant_value0'] / size)} of principal, valued at {pct(warrant_vol(warrant, mkt.sigma), 0)} volatility")
    c[2].metric("Average value at the end", usd(r["warrant_value_mean"], 2),
                help="Exercise payoff if they expire inside the run, otherwise Black-Scholes on the last day")
    c[3].metric("Lost to our own selling", usd(r["warrant_lost_to_selling_mean"], 2),
                help="Average warrant value at prices with no investor selling, minus the value at prices with it")
    c[4].metric("Profit without the warrants", usd(r["pnl_p50"], 2), help="Median, cash only")
    c[5].metric("Dilution with warrants", pct(r["dilution_with_warrants_p50"]),
                help="Median shares issued plus every warrant share, over shares outstanding")

rows = [("Median IRR", pct(f[P + "irr_p50"]), pct(r[P + "irr_p50"])),
        ("Bad case IRR (p10)", pct(f[P + "irr_p10"]), pct(r[P + "irr_p10"]))] if is_note else []
rows += [
        ("Median profit", usd(f[P + "pnl_p50"], 2), usd(r[P + "pnl_p50"], 2)),
        ("Average profit", usd(f[P + "pnl_mean"], 2), usd(r[P + "pnl_mean"], 2)),
        ("Median margin on " + ("cash funded" if is_note else "dollars drawn"),
         pct(f[P + "margin_p50"]), pct(r[P + "margin_p50"])),
        ("Bad case margin (p10)", pct(f[P + "margin_p10"]), pct(r[P + "margin_p10"])),
        ("Bad case profit (p10)", usd(f[P + "pnl_p10"], 2), usd(r[P + "pnl_p10"], 2)),
        ("Chance of a loss", pct(f[P + "prob_loss"]), pct(r[P + "prob_loss"])),
        ("Median dilution", pct(f["dilution_p50"]), pct(r["dilution_p50"])),
        ("Bad case dilution (p90)", pct(f["dilution_p90"]), pct(r["dilution_p90"])),
        ("Converted or drawn (median)", usd(f["converted_p50"]), usd(r["converted_p50"])),
        ("Price drag from our selling, worst", "none", pct(x["price_drag_peak_p50"])),
        ("Price drag from our selling, at end", "none", pct(x["price_drag_p50"])),
        ("Fully sold by", "same day", f"day {x['exit_day_p50']:,.0f}")]
if is_note:
    rows += [("Repaid in cash before maturity (median)", usd(f["repaid_cash_p50"]), usd(r["repaid_cash_p50"])),
             ("Repaid at maturity (median)", usd(f["repaid_maturity_p50"]), usd(r["repaid_maturity_p50"])),
             ("Chance the stock falls through the floor", pct(f["prob_trigger"]), pct(r["prob_trigger"]))]
cmp = pd.DataFrame(rows, columns=["", "No impact", "With volume limits and impact"]).set_index("")

tabs = st.tabs(["Overview", "How the investor gets paid back", "Replay on real prices", "Best setup",
                "How big a deal can it take", "Assumptions"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    left, right = st.columns([1, 1])
    with left:
        a0, a1 = f[P + RET], r[P + RET]
        both = np.concatenate([a0[np.isfinite(a0)], a1[np.isfinite(a1)]])
        if both.size:
            lo, hi = np.percentile(both, [1, 99])
            fig = go.Figure()
            for name, arr, col in (("No impact", a0, NO_IMPACT), ("With impact", a1, IMPACT)):
                a = arr[np.isfinite(arr) & (arr >= lo) & (arr <= hi)]   # drop the outer 1% tails
                fig.add_histogram(x=a, name=name, marker_color=col, opacity=0.6, nbinsx=50,
                                  hovertemplate=RET_NAME + " %{x:.1%}<br>%{y} paths<extra>" + name + "</extra>")
            fig.update_layout(barmode="overlay", bargap=0.05)
            fig = style(fig, ("IRR" if is_note else "Margin") + " across simulated paths",
                        "Annualized IRR" if is_note else "Profit over dollars drawn", "Paths", height=360)
            fig.update_xaxes(tickformat=".0%")
            fig.update_layout(hovermode="closest")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Nothing to show: no shares were issued on any path.")
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
        fig = style(fig, "Price gap caused by our selling", "Trading day", "Price gap", y_fmt=".0%", height=360)
        st.plotly_chart(fig, width="stretch")
    st.dataframe(cmp, width="stretch", height=35 * (len(cmp) + 1) + 3)

# ---------------------------------------------------------------- payback mix
with tabs[1]:
    mix = out["mix"]
    st.write("Average across paths, as a share of the deal size. Shares mean the investor has to sell into the "
             "market; cash means the issuer has to have it.")
    fig = go.Figure(go.Bar(x=list(mix.values()), y=list(mix), orientation="h", marker_color=IMPACT,
                           text=[pct(v, 0) for v in mix.values()], textposition="outside",
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig = style(fig, "Where the principal goes", "Share of deal size", None, height=260)
    fig.update_xaxes(tickformat=".0%", range=[0, max(1.0, max(mix.values()) * 1.15)])
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, width="stretch")
    if is_note and instr.payment_mode == "on_trigger":
        st.write(f"The stock fell through the floor (VWAP below \\${instr.floor_price:,.4f} on {instr.trigger_days} of "
                 f"{instr.trigger_window} days) on **{pct(r['prob_trigger'])}** of paths. From then on the issuer owes "
                 f"{usd(instr.monthly_payment, 2).replace('$', chr(92) + '$')} a month"
                 + (", settled in shares at the equity-line price." if instr.pay_in_shares
                    else f" in cash, plus a {pct(instr.payment_premium, 0)} premium."))
    st.warning("Every cash payment is assumed to be made. The model has no issuer default, so the part repaid in cash "
               "is a ceiling: for a company that needs this kind of financing, not getting paid is the main risk.")

# ---------------------------------------------------------------- replay
with tabs[2]:
    rep = run_replay(ticker, instr, warrant, ex)
    if rep is None:
        st.info(f"Not enough history to replay a {months}-month term plus the {ex.tail_days}-day selling tail. "
                "Try a shorter term.")
    else:
        d, sr, xr = rep["daily"], rep["s"], rep["x"]
        st.caption(f"The same deal run on {ticker}'s actual prices and volume from {d.index[0]:%d %b %Y} to "
                   f"{d.index[-1]:%d %b %Y}, with our own selling layered on top of what really traded.")
        c = tiles(6)
        c[0].metric("Profit", usd(sr[P + "pnl_p50"], 2))
        c[1].metric("IRR" if is_note else "Margin", pct(sr[P + RET + "_p50"], DP))
        c[2].metric("Stock over the term", pct(rep["stock_ret"], 0))
        c[3].metric("Dilution", pct(sr["dilution_p50"]))
        c[4].metric("Worst price drag", pct(xr["price_drag_peak_p50"]))
        c[5].metric("Repaid in cash", usd(sr["repaid_cash_p50"] + sr["repaid_maturity_p50"]))
        fig = go.Figure()
        fig.add_scatter(x=d.index, y=d["actual_close"], name="What actually traded", line=dict(color=NO_IMPACT, width=2))
        fig.add_scatter(x=d.index, y=d["close_with_our_selling"], name="With our selling", line=dict(color=IMPACT, width=2))
        if is_note and instr.floor_price > 0:
            fig.add_hline(y=instr.floor_price, line=dict(width=1, dash="dot"), annotation_text="floor",
                          annotation_position="bottom right")
        st.plotly_chart(style(fig, "Price: actual vs with our selling", None, "Close ($)", y_fmt="$,.2f"),
                        width="stretch")
        share = np.where(d["market_volume"] > 0, d["shares_sold"] / d["market_volume"], 0.0)
        fig = go.Figure(go.Bar(x=d.index, y=share, marker_color=IMPACT, name="Our share of volume",
                               hovertemplate="%{x|%d %b %Y}: %{y:.1%} of volume<extra></extra>"))
        st.plotly_chart(style(fig, "Our selling as a share of each day's volume", None, "Share of volume",
                              y_fmt=".0%", height=260), width="stretch")
        with st.expander("Lookahead check"):
            st.write("Each conversion and sale uses only closes and volume up to that day. The fixed price, floor and "
                     "warrant strike are set from today's price and then applied to last year's prices, so the replay "
                     "shows how these terms behave on a real path, not what a deal signed a year ago would have "
                     "earned. Volatility and volume are calibrated on this same window, so it is not a clean "
                     "out-of-sample test.")

# ---------------------------------------------------------------- best setup
with tabs[3]:
    if not converting:
        st.info("This deal is repaid in cash, so there is no conversion schedule to tune.")
    else:
        st.write("Tries every combination of how often (every 5, 10, 21 days), size each time (5%, 10%, 20%) and "
                 "selling speed (5%, 10%, 20% of volume), all on the same simulated prices. Ranked by median cash "
                 "profit among setups with a 10% or lower chance of loss. Warrants are left out of the ranking.")
        if st.button("Find the best setup (about 30 seconds)"):
            st.session_state["search_key"] = (ticker, instr, ex)
        if st.session_state.get("search_key") == (ticker, instr, ex):
            with st.spinner("Searching 27 setups..."):
                df = run_search(ticker, instr, ex, max(1_000, n_paths // 2))
            b = df.iloc[0]
            verb = "convert" if is_note else "draw"
            ret = f"IRR {pct(b.irr_p50, 0)}" if is_note else f"margin {pct(b.margin_p50)}"
            msg = (f"{verb} {pct(b.tranche_pct, 0)} every {b.cadence_days:.0f} days and sell at {pct(b.participation, 0)} "
                   f"of volume. Median profit {usd(b.pnl_p50, 2)}, {ret}, chance of loss {pct(b.prob_loss)}."
                   ).replace("$", r"\$")
            if b.eligible:
                st.success("Best: " + msg)
            else:
                st.warning("No setup keeps the chance of loss at 10% or lower. Least bad: " + msg)
            show = df.drop(columns=["eligible", "margin_p50" if is_note else "irr_p50"]
                           + ([] if is_note else ["irr_p10", "moic_p50"])).rename(columns={
                "cadence_days": "Every (days)", "tranche_pct": "Size each time", "participation": "Share of volume",
                "irr_p10": "IRR p10", "irr_p50": "IRR median", "margin_p50": "Margin median",
                "prob_loss": "P(loss)", "pnl_p50": "Profit median",
                "moic_p50": "MOIC", "dilution_p50": "Dilution", "converted_p50": "Converted",
                "impact_bps_p50": "Impact (bps)", "exit_day_p50": "Sold by day"})
            st.dataframe(show.style.format({"Size each time": "{:.0%}", "Share of volume": "{:.0%}", "IRR p10": "{:.0%}",
                                            "IRR median": "{:.0%}", "Margin median": "{:.1%}", "P(loss)": "{:.1%}",
                                            "Profit median": "${:,.0f}",
                                            "MOIC": "{:.2f}x", "Dilution": "{:.1%}", "Converted": "${:,.0f}",
                                            "Impact (bps)": "{:,.0f}", "Sold by day": "{:,.0f}"}),
                         width="stretch", hide_index=True)
            st.session_state["search_df"] = df

# ---------------------------------------------------------------- capacity
with tabs[4]:
    st.write("The same terms at a quarter, half, 1x, 2x and 4x the deal size. Past a point the stock can't absorb "
             "the shares: impact rises, the caps bind, the exit takes longer and the return falls. "
             + ("Return is the annualized IRR." if is_note else "Return is profit over dollars drawn."))
    with st.spinner("Running five deal sizes..."):
        cap = run_capacity(ticker, instr, warrant, ex, min(n_paths, 1_500))
    left, right = st.columns(2)
    labels = [usd(v, 1) for v in cap["deal_size"]]
    with left:
        fig = go.Figure(go.Scatter(x=labels, y=cap["irr_p50"], mode="lines+markers", line=dict(color=IMPACT, width=2),
                                   marker=dict(size=9), name=f"Median {RET_NAME}",
                                   hovertemplate="%{x}: %{y:.0%}<extra></extra>"))
        st.plotly_chart(style(fig, f"Median {RET_NAME} by deal size", "Deal size", RET_NAME, y_fmt=".0%", height=300),
                        width="stretch")
    with right:
        fig = go.Figure(go.Scatter(x=labels, y=cap["dilution_p50"], mode="lines+markers", line=dict(color=IMPACT, width=2),
                                   marker=dict(size=9), name="Median dilution",
                                   hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
        st.plotly_chart(style(fig, "Issuer dilution by deal size", "Deal size", "Dilution", y_fmt=".0%", height=300),
                        width="stretch")
    st.dataframe(cap.rename(columns={
        "deal_size": "Deal size", "days_of_volume": "Days of volume", "irr_p50": "Return median", "irr_p10": "Return p10",
        "pnl_p50": "Profit median", "prob_loss": "P(loss)", "dilution_p50": "Dilution", "converted_p50": "Converted",
        "impact_bps_p50": "Impact (bps)", "price_drag_peak_p50": "Worst price drag", "exit_day_p50": "Sold by day"
    }).style.format({"Deal size": "${:,.0f}", "Days of volume": "{:.1f}", "Return median": "{:.1%}", "Return p10": "{:.1%}",
                     "Profit median": "${:,.0f}", "P(loss)": "{:.1%}", "Dilution": "{:.1%}", "Converted": "${:,.0f}",
                     "Impact (bps)": "{:,.0f}", "Worst price drag": "{:.1%}", "Sold by day": "{:,.0f}"}, na_rep="none"),
        width="stretch", hide_index=True)

# ---------------------------------------------------------------- assumptions
with tabs[5]:
    st.markdown(f"""
- **Terms** start from {preset.name} ({preset.source}). Filled in by the model: {preset.assumed}
- **No issuer default.** Every cash payment and the repayment at maturity are assumed to be made. That is the largest
  gap for a loan or for any deal where the stock falls through the floor.
- **Prices** are simulated with zero drift and the stock's realized 1-year volatility ({pct(mkt.sigma, 0)}). No jumps
  and no delisting.
- **Volume** is lognormal around the 60-day average ({mkt.adv_shares / 1e6:,.2f}M shares a day). It does not rise on
  down days or react to our selling.
- **Daily VWAP** is the day's price less half of our own impact that day. Every VWAP-based term uses it, so our
  selling lowers our own conversion price and can push the stock through the floor.
- **Price impact** follows the square-root law: selling q shares into daily volume V costs
  impact strength x daily vol x sqrt(q / V). Part of that move carries over, and part of the carried move fades with a
  half-life. **These settings are placeholders, not fitted to real selling.**
- **Warrants** are valued by Black-Scholes with volatility capped at the level set in the sidebar and a 4% risk-free
  rate. The valuation ignores the ownership blocker, the cost of selling the exercised shares, and anti-dilution
  adjustments.
- **Not modeled:** registration delays, events of default and default interest, the investor's right to accelerate
  payments, mandatory redemption on a new financing, and any signaling effect from the market knowing the investor
  is selling.
- **No lookahead:** every decision on day t uses data up to day t only. Option 1 and Option 2 purchase prices are set
  after the fact by contract, and the cash is booked the day after the pricing period ends.
""".replace("$", r"\$"))

# ---------------------------------------------------------------- export
terms = {**{"ticker": ticker, "started_from": preset.name, "source": preset.source},
         **dataclasses.asdict(instr), **({f"warrant_{a}": b for a, b in dataclasses.asdict(warrant).items()} if warrant else {}),
         **{f"execution_{a}": b for a, b in dataclasses.asdict(ex).items()},
         "paths": n_paths, "spot": mkt.s0, "vol": mkt.sigma, "shares_out": mkt.shares_out, "adv_dollars": mkt.adv_dollars}
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as w:
    pd.DataFrame({"input": list(terms), "value": [str(v) if isinstance(v, float) and not np.isfinite(v) else v
                                                  for v in terms.values()]}).to_excel(w, sheet_name="Inputs", index=False)
    cmp.to_excel(w, sheet_name="Summary")
    pd.DataFrame({"where_the_principal_goes": list(out["mix"]), "share": list(out["mix"].values())}).to_excel(
        w, sheet_name="Payback", index=False)
    cap.to_excel(w, sheet_name="Deal size", index=False)
    rep = run_replay(ticker, instr, warrant, ex)
    if rep is not None:
        daily = rep["daily"].copy()
        if daily.index.tz is not None:
            daily.index = daily.index.tz_localize(None)
        daily.to_excel(w, sheet_name="Replay daily")
    if st.session_state.get("search_key") == (ticker, instr, ex) and "search_df" in st.session_state:
        st.session_state["search_df"].to_excel(w, sheet_name="Best setup", index=False)
st.sidebar.download_button("Download results (Excel)", buf.getvalue(), file_name=f"{ticker}_pipe_sim.xlsx",
                           width="stretch")
