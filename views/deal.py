"""Deal economics page: Monte Carlo on daily prices, frictionless vs volume- and impact-aware."""
import dataclasses
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pipesim import ConvertibleNote, StandbyEquityFacility, Warrant, simulate, summarize
from pipesim.execution import ExecutionModel, execution_stats, run_execution
from pipesim.instruments import TRADING_DAYS
from pipesim.optimize import search
from pipesim.presets import PRESETS
from pipesim.warrants import warrant_vol
from views.common import (DESK, MARKET, MONTH, MUTED, chart, footnote, inject_css, kpis, pct, price, quote, section,
                          shares, signed, style, table, ticker_input, ticker_strip, usd)

NOTE_PRICING = {"Lowest daily VWAP before the notice": "lowest", "Average, including the notice day": "average"}
SEPA_PRICING = {"Option 1: same-day VWAP": "option1", "Option 2: lowest daily VWAP": "option2",
                "Trailing average (not a filed structure)": "trailing"}
REPAY = {"At maturity only": "none", "Monthly from a start date": "scheduled",
         "Monthly after the stock falls below the floor": "on_trigger"}


def pick(options: dict, value):
    return list(options.values()).index(value)


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
    mkt = quote(ticker)["mkt"]
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
    mkt = quote(ticker)["mkt"]
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
    return search(instr, quote(ticker)["mkt"], ex, n_paths=n_paths)


@st.cache_data(show_spinner=False, max_entries=10)
def run_capacity(ticker, instr, warrant, ex, n_paths, multiples=(0.25, 0.5, 1.0, 2.0, 4.0)):
    mkt = quote(ticker)["mkt"]
    size = instr.principal if isinstance(instr, ConvertibleNote) else instr.commitment
    ret = "irr" if isinstance(instr, ConvertibleNote) else "margin"
    rows = []
    for m in multiples:
        i_m, w_m = scale(instr, warrant, m)
        r = run_execution(i_m, mkt, ex, n_paths=n_paths, warrant=w_m)
        s, x = summarize(r), execution_stats(r)
        rows.append({"deal_size": size * m, "days_of_volume": size * m / mkt.adv_dollars,
                     "ret_p50": s.get(f"pkg_{ret}_p50", s[f"{ret}_p50"]),
                     "ret_p10": s.get(f"pkg_{ret}_p10", s[f"{ret}_p10"]),
                     "pnl_p50": s.get("pkg_pnl_p50", s["pnl_p50"]),
                     "prob_loss": s.get("pkg_prob_loss", s["prob_loss"]), "dilution_p50": s["dilution_p50"],
                     "impact_bps_p50": x["impact_bps_p50"], "price_drag_peak_p50": x["price_drag_peak_p50"],
                     "exit_day_p50": x["exit_day_p50"]})
    return pd.DataFrame(rows)


inject_css()

# ---------------------------------------------------------------- controls
top = st.columns([1.1, 3.2, 1.2], vertical_alignment="bottom")
with top[0]:
    ticker = ticker_input("deal_ticker")
preset_name = top[1].selectbox("Structure", list(PRESETS), key="deal_preset",
                               help="Terms read from filed Yorkville (YA II PN) agreements. Change any term below.")
export = top[2].popover("Export", width="stretch")
try:
    q = quote(ticker)
except Exception as e:
    st.error(f"No data for {ticker}: {e}")
    st.stop()
mkt = q["mkt"]

preset = PRESETS[preset_name](mkt.s0)
base = preset.note or preset.sepa
is_note = preset.note is not None
base_size = base.principal if is_note else base.commitment
# a deal sized for the filing's company is usually too big for this one: start at 7% of market cap
start_size = float(np.clip(round(min(base_size, 0.07 * mkt.s0 * mkt.shares_out) / 5e5) * 5e5, 5e5, 5e8))
k = f"{ticker}|{preset_name}|"          # widget keys: a new ticker or structure resets every default
w0 = preset.warrant

with st.form("deal"):
    row = st.columns([1, 1, 1, 1, 0.8], vertical_alignment="bottom")
    if is_note:
        size_m = row[0].number_input("Principal ($M)", 0.5, 500.0, start_size / 1e6, 0.5, key=k + "size")
    else:
        size_m = row[0].number_input("Commitment ($M)", 1.0, 1000.0, max(start_size / 1e6, 1.0), 1.0, key=k + "size")
    months = row[1].number_input("Term (months)", 3, 36, int(round(base.maturity_days / MONTH)), key=k + "term")
    discount = row[2].number_input("Discount to VWAP (%)", 0.0, 30.0, round(base.discount * 100, 2), 0.5, key=k + "disc") / 100
    n_paths = row[3].select_slider("Paths", [500, 1000, 1500, 2000, 3000, 5000], value=1500, key=k + "paths")
    row[4].form_submit_button("Run", type="primary", width="stretch")

    with st.expander("Full terms, caps and execution"):
        tab_names = (["Conversion", "Repayment", "Warrants"] if is_note else ["Advances"]) + ["Caps and fees", "Execution"]
        tt = dict(zip(tab_names, st.tabs(tab_names)))
        if is_note:
            with tt["Conversion"]:
                a, b, c, d = st.columns(4)
                converts = a.toggle("Investor converts", value=base.tranche > 0, key=k + "conv")
                pricing = b.selectbox("VWAP measure", list(NOTE_PRICING), index=pick(NOTE_PRICING, base.pricing), key=k + "pr")
                lookback = c.number_input("VWAP days", 1, 30, base.vwap_lookback, key=k + "lb")
                cadence = d.number_input("Days between conversions", 1, 63, base.cadence, key=k + "cad")
                a, b, c, d = st.columns(4)
                step_pct = a.number_input("Converted each time (% of principal)", 1.0, 50.0,
                                          float(np.clip(100 * base.tranche / base.principal if base.tranche else 10.0, 1, 50)),
                                          1.0, key=k + "step") / 100
                fixed_pct = b.number_input("Fixed price (x today, 0 = none)", 0.0, 5.0, round(base.fixed_price / mkt.s0, 2),
                                           0.05, key=k + "fix")
                floor_pct = c.number_input("Floor price (x today, 0 = none)", 0.0, 1.0, round(base.floor_price / mkt.s0, 2),
                                           0.05, key=k + "flr")
                oid = d.number_input("Funding discount (%)", 0.0, 20.0, round(base.oid * 100, 2), 0.5, key=k + "oid") / 100
            with tt["Repayment"]:
                a, b, c, d = st.columns(4)
                repay = a.selectbox("Issuer repays", list(REPAY), index=pick(REPAY, base.payment_mode), key=k + "rep")
                pay_pct = b.number_input("Monthly payment (% of principal)", 0.0, 50.0,
                                         round(100 * base.monthly_payment / base.principal, 2), 0.5, key=k + "pay") / 100
                premium = c.number_input("Premium on each payment (%)", 0.0, 20.0, round(base.payment_premium * 100, 2), 0.5,
                                         key=k + "prem") / 100
                coupon = d.number_input("Interest (% a year)", 0.0, 20.0, round(base.coupon * 100, 2), 0.5, key=k + "cpn") / 100
                a, b = st.columns([1, 3])
                pay_start = a.number_input("First payment (trading day)", 1, 252, base.payment_start_day, key=k + "ps",
                                           help="For payments from a start date. 60 calendar days is about 42.")
                in_shares = b.toggle("Settle payments with equity-line advances (pre-paid advance loop)",
                                     value=base.pay_in_shares, key=k + "pis")
            with tt["Warrants"]:
                a, b, c, d, e = st.columns(5)
                cover = a.number_input("Coverage (% of principal)", 0.0, 100.0,
                                       round(100 * w0.shares * w0.strike / base.principal, 2) if w0 else 0.0, 1.0,
                                       key=k + "wc", help="Warrant shares x strike / principal. 0 = no warrants") / 100
                strike_pct = b.number_input("Strike (x today)", 0.1, 5.0, round(w0.strike / mkt.s0, 2) if w0 else 1.0, 0.05,
                                            key=k + "wk")
                w_years = c.number_input("Term (years)", 0.5, 10.0, w0.term_days / TRADING_DAYS if w0 else 5.0, 0.5, key=k + "wt")
                w_wait = d.number_input("Exercisable after (months)", 0, 24,
                                        int(round(w0.exercisable_after / MONTH)) if w0 else 0, key=k + "ww")
                vol_cap = e.number_input("Valuation vol cap (%)", 20.0, 200.0, 100.0, 5.0, key=k + "wv",
                                         help=f"Realized vol is {pct(mkt.sigma, 0)}. Uncapped Black-Scholes values a "
                                              "long warrant near the stock itself at microcap vol.") / 100
        else:
            with tt["Advances"]:
                a, b, c, d = st.columns(4)
                pricing = a.selectbox("Pricing", list(SEPA_PRICING), index=pick(SEPA_PRICING, base.pricing), key=k + "pr")
                adv_pct = b.number_input("Each advance (% of commitment)", 1.0, 30.0,
                                         float(np.clip(100 * base.advance / base.commitment, 1, 30)), 1.0, key=k + "step") / 100
                cadence = c.number_input("Days between advances", 1, 63, base.cadence, key=k + "cad")
                lookback = d.number_input("Pricing days (Option 2)", 1, 30, max(base.vwap_lookback, 1), key=k + "lb")
                a, b, c, _ = st.columns(4)
                vol_thr = a.number_input("Option 1 volume threshold", 0.0, 1.0, base.volume_threshold, 0.05, key=k + "vt")
                adv_cap = b.number_input("Max advance (x 5-day ADV)", 0.0, 10.0, base.max_advance_adv, 0.25, key=k + "ac")
                floor_pct = c.number_input("Min acceptable price (x today)", 0.0, 1.0, round(base.min_draw_price / mkt.s0, 2),
                                           0.05, key=k + "flr")
        with tt["Caps and fees"]:
            a, b, c, d = st.columns(4)
            own_cap = a.number_input("Ownership cap", 0.0, 0.25, base.ownership_cap, 0.0001, format="%.4f", key=k + "own")
            exch_cap = b.number_input("Exchange cap", 0.0, 1.0, base.exchange_cap, 0.0001, format="%.4f", key=k + "exc")
            fee = c.number_input("Structuring fee ($)", 0.0, 5e6, float(base.structuring_fee), 5e3, key=k + "fee")
            commit_sh = 0.0
            if not is_note:
                commit_sh = d.number_input("Commitment shares (K)", 0.0, 1e5,
                                           round(base.commitment_shares * (start_size / base_size) / 1e3, 1), 5.0, key=k + "cs")
        with tt["Execution"]:
            a, b, c, d = st.columns(4)
            participation = a.number_input("Max % of daily volume", 1, 50, 10, 1, key=k + "part") / 100
            slip = b.number_input("Spread and fees per sale (%)", 0.0, 5.0, 2.0, 0.25, key=k + "slip") / 100
            eta = c.number_input("Impact strength", 0.0, 2.0, 0.5, 0.1, key=k + "eta", help="Square-root coefficient. Placeholder.")
            carried = d.number_input("Share of move carried over", 0.0, 1.0, 0.5, 0.05, key=k + "car")
            a, b, c, _ = st.columns(4)
            half_life = a.number_input("Recovery half-life (days)", 1.0, 250.0, 10.0, 1.0, key=k + "hl")
            residual = b.number_input("Share that never fades", 0.0, 1.0, 0.3, 0.05, key=k + "res")
            no_recovery = c.toggle("Price never recovers", value=False, key=k + "norec")

# ---------------------------------------------------------------- build the deal
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
P = "pkg_" if warrant else ""           # headline numbers include the warrants when there are any
RET = "irr" if is_note else "margin"    # an equity line ties up no capital, so it has a margin, not an IRR
RET_NAME = "IRR" if is_note else "Margin"

ticker_strip(ticker, q, [("Deal", f"{usd(size)} {kind}" + (" + warrants" if warrant else ""),
                          f'<span class="muted">{size / mkt.adv_dollars:.1f}x ADV</span>')])

with st.spinner("Simulating"):
    out = run_models(ticker, instr, warrant, ex, n_paths)
f, r, x = out["free"], out["real"], out["x"]

d_pnl = r[P + "pnl_p50"] - f[P + "pnl_p50"]
kpis([
    {"label": "Median P&L", "value": usd(r[P + "pnl_p50"], 2),
     "sub": f"{signed(usd(d_pnl, 2), d_pnl)} vs no impact",
     "tip": "Median profit with volume limits and price impact" + (", warrants included" if warrant else "")},
    {"label": f"Median {RET_NAME}", "value": pct(r[P + RET + "_p50"], 0 if is_note else 1),
     "sub": f"bad case {pct(r[P + RET + '_p10'], 0 if is_note else 1)}",
     "tip": ("Annualized. A note that converts in a few months shows a high IRR on a modest profit." if is_note else
             "Profit over dollars drawn. The desk sells before it pays, so no capital is tied up and IRR is meaningless.")},
    {"label": "Chance of a loss", "value": pct(r[P + "prob_loss"]), "sub": "no issuer default modeled",
     "tip": "Every cash payment is assumed to be made. Default is the main risk these deals carry."},
    {"label": "Issuer dilution", "value": pct(r["dilution_p50"]),
     "sub": f"bad case {pct(r['dilution_p90'])}" + (f", {pct(r['dilution_with_warrants_p50'])} w/ warrants" if warrant else ""),
     "tip": "Median shares issued over shares outstanding; bad case is the 90th percentile"},
    {"label": "Impact cost", "value": "none" if not np.isfinite(x["impact_bps_p50"]) else f"{x['impact_bps_p50']:,.0f} bps",
     "sub": f"worst price drag {pct(x['price_drag_peak_p50'])}", "tip": "Median impact paid on each share sold"},
    {"label": "Fully sold by", "value": f"day {x['exit_day_p50']:,.0f}" if np.isfinite(x["exit_day_p50"]) else "n/a",
     "sub": f"at {pct(participation, 0)} of daily volume"},
])
if warrant:
    section("Warrants", f"{shares(warrant.shares)} at {price(warrant.strike)}")
    kpis([
        {"label": "Value at closing", "value": usd(r["warrant_value0"], 2),
         "sub": f"{pct(r['warrant_value0'] / size)} of principal at {pct(warrant_vol(warrant, mkt.sigma), 0)} vol"},
        {"label": "Average value at end", "value": usd(r["warrant_value_mean"], 2),
         "sub": "payoff if expired, else Black-Scholes"},
        {"label": "Lost to own selling", "value": usd(r["warrant_lost_to_selling_mean"], 2),
         "sub": "vs prices with no selling", "tip": "Converting deals only: selling pushes the stock down"},
        {"label": "P&L without warrants", "value": usd(r["pnl_p50"], 2), "sub": "median, cash only"},
        {"label": "Warrant shares", "value": pct(warrant.shares / mkt.shares_out), "sub": "of shares outstanding"},
    ])
footnote(f"{preset.name}. {preset.source}. Every term is editable; assumptions in Notes.")

tabs = st.tabs(["Summary", "Payback", "Replay", "Optimizer", "Capacity", "Notes"])

# ---------------------------------------------------------------- summary
with tabs[0]:
    rows = [("IRR", pct(f[P + "irr_p50"]), pct(r[P + "irr_p50"])),
            ("IRR, bad case (p10)", pct(f[P + "irr_p10"]), pct(r[P + "irr_p10"]))] if is_note else []
    rows += [("P&L", usd(f[P + "pnl_p50"], 2), usd(r[P + "pnl_p50"], 2)),
             ("P&L, average", usd(f[P + "pnl_mean"], 2), usd(r[P + "pnl_mean"], 2)),
             ("P&L, bad case (p10)", usd(f[P + "pnl_p10"], 2), usd(r[P + "pnl_p10"], 2)),
             ("Margin on " + ("cash funded" if is_note else "dollars drawn"), pct(f[P + "margin_p50"]), pct(r[P + "margin_p50"])),
             ("Chance of a loss", pct(f[P + "prob_loss"]), pct(r[P + "prob_loss"])),
             ("Dilution", pct(f["dilution_p50"]), pct(r["dilution_p50"])),
             ("Dilution, bad case (p90)", pct(f["dilution_p90"]), pct(r["dilution_p90"])),
             ("Converted or drawn", usd(f["converted_p50"]), usd(r["converted_p50"])),
             ("Price drag, worst / end", "none", f"{pct(x['price_drag_peak_p50'])} / {pct(x['price_drag_p50'])}")]
    if is_note:
        rows += [("Repaid in cash early", usd(f["repaid_cash_p50"]), usd(r["repaid_cash_p50"])),
                 ("Repaid at maturity", usd(f["repaid_maturity_p50"]), usd(r["repaid_maturity_p50"])),
                 ("Stock falls through floor", pct(f["prob_trigger"]), pct(r["prob_trigger"]))]
    cmp = pd.DataFrame(rows, columns=["Median unless noted", "No impact", "With impact"])
    left, right = st.columns([1, 1.25])
    with left:
        table(cmp, {})
    with right:
        a0, a1 = f[P + RET], r[P + RET]
        both = np.concatenate([a0[np.isfinite(a0)], a1[np.isfinite(a1)]])
        if both.size:
            lo, hi = np.percentile(both, [1, 99])
            fig = go.Figure()
            for name, arr, col in (("No impact", a0, MARKET), ("With impact", a1, DESK)):
                a = arr[np.isfinite(arr) & (arr >= lo) & (arr <= hi)]
                fig.add_histogram(x=a, name=name, marker_color=col, opacity=0.65, nbinsx=50,
                                  hovertemplate=RET_NAME + " %{x:.1%}: %{y} paths<extra>" + name + "</extra>")
            fig.update_layout(barmode="overlay", bargap=0.04, hovermode="closest")
            fig = style(fig, f"{RET_NAME} across simulated paths", None, "Paths", height=250)
            fig.update_xaxes(tickformat=".0%")
            chart(fig)
        qg = out["gap_q"]
        t = np.arange(qg.shape[1])
        fig = go.Figure()
        fig.add_scatter(x=np.r_[t, t[::-1]], y=np.r_[qg[2], qg[0][::-1]], fill="toself", fillcolor=DESK, opacity=0.18,
                        line=dict(width=0), hoverinfo="skip", name="10th to 90th pct")
        fig.add_scatter(x=t, y=qg[1], line=dict(color=DESK, width=2), name="Median",
                        hovertemplate="Day %{x}: %{y:.1%} below<extra></extra>")
        fig.add_vline(x=instr.maturity_days, line=dict(width=1, dash="dot", color=MUTED))
        chart(style(fig, "Price gap from own selling, by trading day (dotted = maturity)", None, None, y_fmt=".0%", height=230))

# ---------------------------------------------------------------- payback
with tabs[1]:
    mix = out["mix"]
    fig = go.Figure(go.Bar(x=list(mix.values()), y=list(mix), orientation="h", marker_color=DESK,
                           text=[pct(v, 0) for v in mix.values()], textposition="outside",
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig = style(fig, "Where the principal goes, average across paths", None, None, height=230, legend=False)
    fig.update_xaxes(tickformat=".0%", range=[0, max(1.0, max(mix.values()) * 1.15)])
    fig.update_layout(hovermode="closest")
    chart(fig)
    if is_note and instr.payment_mode == "on_trigger":
        kpis([
            {"label": "Falls through floor", "value": pct(r["prob_trigger"]), "sub": f"VWAP < {price(instr.floor_price)}, "
             f"{instr.trigger_days} of {instr.trigger_window} days"},
            {"label": "Monthly payment", "value": usd(instr.monthly_payment, 2),
             "sub": "in shares via advances" if instr.pay_in_shares else f"cash + {pct(instr.payment_premium, 0)} premium"},
            {"label": "Repaid in cash early", "value": usd(r["repaid_cash_p50"]), "sub": "median"},
            {"label": "Repaid at maturity", "value": usd(r["repaid_maturity_p50"]), "sub": "median"},
        ])
    footnote("Shares mean the desk has to sell into the market; cash means the issuer has to have it. "
             "No issuer default is modeled, so the cash legs are a ceiling.")

# ---------------------------------------------------------------- replay
with tabs[2]:
    rep = run_replay(ticker, instr, warrant, ex)
    if rep is None:
        footnote(f"Not enough history to replay a {months}-month term plus the {ex.tail_days}-day selling tail.")
    else:
        d, sr, xr = rep["daily"], rep["s"], rep["x"]
        kpis([
            {"label": "P&L", "value": usd(sr[P + "pnl_p50"], 2)},
            {"label": RET_NAME, "value": pct(sr[P + RET + "_p50"], 0 if is_note else 1)},
            {"label": "Stock over the term", "value": pct(rep["stock_ret"], 0)},
            {"label": "Dilution", "value": pct(sr["dilution_p50"])},
            {"label": "Worst price drag", "value": pct(xr["price_drag_peak_p50"])},
            {"label": "Repaid in cash", "value": usd(sr["repaid_cash_p50"] + sr["repaid_maturity_p50"])},
        ])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
        fig.add_scatter(x=d.index, y=d["actual_close"], name="Traded", line=dict(color=MARKET, width=1.6), row=1, col=1)
        fig.add_scatter(x=d.index, y=d["close_with_our_selling"], name="With desk selling", line=dict(color=DESK, width=1.6),
                        row=1, col=1)
        if is_note and instr.floor_price > 0:
            fig.add_hline(y=instr.floor_price, line=dict(width=1, dash="dash", color=MUTED), row=1, col=1,
                          annotation_text="floor", annotation_position="bottom right", annotation_font=dict(size=10, color=MUTED))
        share = np.where(d["market_volume"] > 0, d["shares_sold"] / d["market_volume"], 0.0)
        fig.add_bar(x=d.index, y=share, name="Desk % of volume", marker_color=DESK, row=2, col=1,
                    hovertemplate="%{x|%d %b %Y}: %{y:.1%} of volume<extra></extra>")
        fig = style(fig, f"Last {len(d) - 1} trading days, actual prices and volume, desk selling layered on top",
                    None, None, height=440)
        fig.update_yaxes(tickformat="$,.2f", row=1, col=1)
        fig.update_yaxes(tickformat=".0%", row=2, col=1)
        chart(fig)
        footnote("Fixed price, floor and strike are set from today's price and applied to last year's path, so this "
                 "shows how the terms behave on a real path, not what a deal signed a year ago earned.")

# ---------------------------------------------------------------- optimizer
with tabs[3]:
    if not converting:
        footnote("This deal is repaid in cash, so there is no conversion schedule to tune.")
    else:
        a, b = st.columns([1, 3], vertical_alignment="center")
        if a.button("Run optimizer", width="stretch"):
            st.session_state["search_key"] = (ticker, instr, ex)
        with b:
            footnote("27 setups: every 5, 10 or 21 days x 5%, 10% or 20% each time x selling at 5%, 10% or 20% of volume, "
                     "on the same simulated prices. Ranked by median cash P&L with a 10% or lower chance of loss.")
        if st.session_state.get("search_key") == (ticker, instr, ex):
            with st.spinner("Searching 27 setups"):
                df = run_search(ticker, instr, ex, max(1_000, n_paths // 2))
            b0 = df.iloc[0]
            kpis([
                {"label": "Best setup", "value": f"{pct(b0.tranche_pct, 0)} every {b0.cadence_days:.0f}d",
                 "sub": f"sell at {pct(b0.participation, 0)} of volume" + ("" if b0.eligible else ", loss cap not met")},
                {"label": "Median P&L", "value": usd(b0.pnl_p50, 2)},
                {"label": "IRR" if is_note else "Margin", "value": pct(b0.irr_p50 if is_note else b0.margin_p50, 0 if is_note else 1)},
                {"label": "Chance of a loss", "value": pct(b0.prob_loss)},
            ])
            show = df.rename(columns={"cadence_days": "Every (d)", "tranche_pct": "Size", "participation": "% of vol",
                                      "pnl_p50": "P&L med", "irr_p50": "IRR med", "irr_p10": "IRR p10",
                                      "margin_p50": "Margin med", "prob_loss": "P(loss)", "dilution_p50": "Dilution",
                                      "impact_bps_p50": "Impact (bps)", "exit_day_p50": "Sold by day"})
            keep = ["Every (d)", "Size", "% of vol", "P&L med"] + (["IRR med", "IRR p10"] if is_note else ["Margin med"]) + \
                ["P(loss)", "Dilution", "Impact (bps)", "Sold by day"]
            table(show[keep], {"Size": "{:.0%}", "% of vol": "{:.0%}", "P&L med": "${:,.0f}", "IRR med": "{:.0%}",
                               "IRR p10": "{:.0%}", "Margin med": "{:.1%}", "P(loss)": "{:.1%}", "Dilution": "{:.1%}",
                               "Impact (bps)": "{:,.0f}", "Sold by day": "{:,.0f}"},
                  signed_cols=["P&L med"], highlight_first=True, height=420)
            st.session_state["search_df"] = df

# ---------------------------------------------------------------- capacity
with tabs[4]:
    a, b = st.columns([1, 3], vertical_alignment="center")
    if a.button("Run size sweep", width="stretch"):
        st.session_state["cap_key"] = (ticker, instr, warrant, ex)
    with b:
        footnote("Same terms at 0.25x, 0.5x, 1x, 2x and 4x the size: where the stock stops absorbing the shares.")
    if st.session_state.get("cap_key") == (ticker, instr, warrant, ex):
        with st.spinner("Running five sizes"):
            cap = run_capacity(ticker, instr, warrant, ex, min(n_paths, 1_500))
        st.session_state["cap_df"] = cap
        labels = [usd(v, 1) for v in cap["deal_size"]]
        left, right = st.columns(2)
        with left:
            fig = go.Figure(go.Scatter(x=labels, y=cap["ret_p50"], mode="lines+markers", line=dict(color=DESK, width=2),
                                       marker=dict(size=8), name=f"Median {RET_NAME}", hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
            chart(style(fig, f"Median {RET_NAME} by size", None, None, y_fmt=".0%", height=260, legend=False))
        with right:
            fig = go.Figure(go.Scatter(x=labels, y=cap["dilution_p50"], mode="lines+markers", line=dict(color=DESK, width=2),
                                       marker=dict(size=8), name="Median dilution", hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
            chart(style(fig, "Issuer dilution by size", None, None, y_fmt=".0%", height=260, legend=False))
        table(cap.rename(columns={"deal_size": "Size", "days_of_volume": "x ADV", "ret_p50": f"{RET_NAME} med",
                                  "ret_p10": f"{RET_NAME} p10", "pnl_p50": "P&L med", "prob_loss": "P(loss)",
                                  "dilution_p50": "Dilution", "impact_bps_p50": "Impact (bps)",
                                  "price_drag_peak_p50": "Worst drag", "exit_day_p50": "Sold by day"}),
              {"Size": "${:,.0f}", "x ADV": "{:.1f}", f"{RET_NAME} med": "{:.1%}", f"{RET_NAME} p10": "{:.1%}",
               "P&L med": "${:,.0f}", "P(loss)": "{:.1%}", "Dilution": "{:.1%}", "Impact (bps)": "{:,.0f}",
               "Worst drag": "{:.1%}", "Sold by day": "{:,.0f}"}, signed_cols=["P&L med"])

# ---------------------------------------------------------------- notes
with tabs[5]:
    st.markdown(f"""
**Terms.** Started from {preset.name} ({preset.source}). Filled in by the model: {preset.assumed}

**Not modeled: issuer default.** Every cash payment and the repayment at maturity are assumed to be made. For a
company that needs this financing, not getting paid is the main risk, so cash legs are a ceiling.

**Prices and volume.** Zero-drift GBM at the stock's realized 1-year vol ({pct(mkt.sigma, 0)}), no jumps or
delisting. Volume is lognormal around the 60-day average ({mkt.adv_shares / 1e6:,.2f}M shares a day), independent of
price and of the desk.

**Daily VWAP** is the day's price less half the desk's own impact, and every VWAP-based term uses it, so the desk's
selling lowers its own conversion price and can push the stock through the floor.

**Impact.** Selling q shares into daily volume V costs impact strength x daily vol x sqrt(q / V). A share carries
over and part of that fades with a half-life. Placeholders until calibrated on real fills.

**Warrants.** Black-Scholes, vol capped as set, 4% rate. Ignores the ownership blocker, the cost of selling
exercised shares and anti-dilution adjustments.

**Not modeled.** Registration delays, events of default and default interest, acceleration rights, mandatory
redemption on a new financing, signaling from a known seller.

**Lookahead.** Every decision on day t uses data up to day t; the floor test uses VWAPs through day t-1. Option 1
and 2 prices are set after the fact by contract and paid the day after the pricing period.
""".replace("$", r"\$"))

# ---------------------------------------------------------------- export
terms = {**{"ticker": ticker, "structure": preset.name, "source": preset.source},
         **dataclasses.asdict(instr), **({f"warrant_{a}": b for a, b in dataclasses.asdict(warrant).items()} if warrant else {}),
         **{f"execution_{a}": b for a, b in dataclasses.asdict(ex).items()},
         "paths": n_paths, "spot": mkt.s0, "vol": mkt.sigma, "shares_out": mkt.shares_out, "adv_dollars": mkt.adv_dollars}
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as w:
    pd.DataFrame({"input": list(terms), "value": [str(v) if isinstance(v, float) and not np.isfinite(v) else v
                                                  for v in terms.values()]}).to_excel(w, sheet_name="Inputs", index=False)
    cmp.to_excel(w, sheet_name="Summary", index=False)
    pd.DataFrame({"where_the_principal_goes": list(out["mix"]), "share": list(out["mix"].values())}).to_excel(
        w, sheet_name="Payback", index=False)
    rep = run_replay(ticker, instr, warrant, ex)
    if rep is not None:
        daily = rep["daily"].copy()
        if daily.index.tz is not None:
            daily.index = daily.index.tz_localize(None)
        daily.to_excel(w, sheet_name="Replay daily")
    if st.session_state.get("search_key") == (ticker, instr, ex) and "search_df" in st.session_state:
        st.session_state["search_df"].to_excel(w, sheet_name="Optimizer", index=False)
    if st.session_state.get("cap_key") == (ticker, instr, warrant, ex) and "cap_df" in st.session_state:
        st.session_state["cap_df"].to_excel(w, sheet_name="Size sweep", index=False)
export.download_button("Deal results (xlsx)", buf.getvalue(), f"{ticker}_deal.xlsx", width="stretch")
