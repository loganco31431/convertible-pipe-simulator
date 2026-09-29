"""Execution page: how well can the desk sell against VWAP inside the term sheet?

Every trading day in the intraday data is tried as a notice day and five selling strategies are
scored on the same days. The plan tab turns the chosen strategy into a schedule for a notice on
the next trading day.
"""
import datetime as dt
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pipesim.intraday.data import bloomberg_bars, csv_bars, yahoo_bars
from pipesim.intraday.tradeout import STRATEGIES, IntradayImpact, TermSheet, compare, plan
from views.common import (DESK, GRID, MARKET, MUTED, bps, chart, footnote, inject_css, kpis, pct, price, quote,
                          section, shares, signed, style, table, ticker_input, ticker_strip, usd)

STRUCTURES = {
    "Option 1 · same-day VWAP": dict(kind="sepa", option=1, discount=0.04, sell_days=1, pricing_days=1),
    "Option 2 · 3-day low VWAP": dict(kind="sepa", option=2, discount=0.03, sell_days=3, pricing_days=3),
    "Note conversion": dict(kind="note", option=2, discount=0.07, sell_days=5, pricing_days=5),
}
PRICING = {"Lowest daily VWAP": "lowest", "Average of daily VWAPs": "average", "VWAP over the whole period": "period"}
SOURCES = ["Yahoo, 5-min, last 60 days", "Bloomberg export (xlsx / csv)", "Bloomberg terminal (blpapi)"]
NOTICE_TIMES = ["09:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00"]
SHORT = {"twap": "TWAP", "vwap": "VWAP curve", "pov": "POV", "front": "Front-loaded", "strength": "Into strength"}


@st.cache_data(ttl=1800, show_spinner=False)
def load_yahoo(ticker, minutes):
    return yahoo_bars(ticker, minutes)


@st.cache_data(show_spinner=False, max_entries=5)
def load_file(data: bytes, name: str, minutes: int):
    buf = io.BytesIO(data)
    buf.name = name
    return csv_bars(buf, minutes)


@st.cache_data(ttl=1800, show_spinner=False)
def load_bloomberg(security, start, end, minutes):
    return bloomberg_bars(security, pd.Timestamp(start), pd.Timestamp(end), minutes)


@st.cache_data(show_spinner=False, max_entries=20)
def run_all(_bars, bars_key, ts, imp, shares_out):
    """`bars_key` identifies the data for the cache; the arrays themselves aren't hashed."""
    return compare(_bars, ts, imp, shares_out)


@st.cache_data(show_spinner=False, max_entries=40)
def run_plan(_bars, bars_key, ts, imp, strategy, shares_out):
    return plan(_bars, ts, imp, strategy, shares_out)


def buckets(slot_times, minutes, bar_minutes):
    per = max(1, minutes // bar_minutes)
    return per, [slot_times[i] for i in range(0, len(slot_times), per)]


inject_css()

# ---------------------------------------------------------------- controls
top = st.columns([1.1, 3.2, 1.2], vertical_alignment="bottom")
with top[0]:
    ticker = ticker_input("to_ticker")
with top[1]:
    structure = st.segmented_control("Structure", list(STRUCTURES), default=list(STRUCTURES)[0], key="to_struct",
                                     help="Pricing as written in the filed YA II PN agreements (SunPower, Soluna)")
    structure = structure or list(STRUCTURES)[0]
export = top[2].popover("Export", width="stretch")
d0 = STRUCTURES[structure]
sepa, opt1 = d0["kind"] == "sepa", d0["kind"] == "sepa" and d0["option"] == 1
k = structure + "|"

with st.form("tradeout"):
    row = st.columns([1, 1, 1, 1, 1, 0.8], vertical_alignment="bottom")
    size_m = row[0].number_input("Advance ($M)" if sepa else "Conversion ($M)", 0.05, 100.0, 1.0, 0.05, key=k + "size",
                                 help="An advance is set in shares at the prior close")
    discount = row[1].number_input("Discount (%)", 0.0, 30.0, d0["discount"] * 100, 0.5, key=k + "disc") / 100
    sell_days = row[2].number_input("Days to sell", 1, 20, d0["sell_days"], key=k + "sd")
    max_part = row[3].number_input("Max % of bar volume", 1, 50, 15, 1, key=k + "mp") / 100
    notice, vol_thr, map_pct, fixed_x, floor_x, pricing, pricing_days = "09:30", 0.0, 0.0, 0.0, 0.0, "Lowest daily VWAP", 1
    if opt1:
        notice = row[4].selectbox("Notice confirmed", NOTICE_TIMES, key=k + "nt",
                                  help="The pricing period runs from this time to the 4 PM close")
    elif sepa:
        pricing_days = row[4].number_input("Pricing days", 1, 20, d0["pricing_days"], key=k + "pd")
    else:
        pricing_days = row[4].number_input("Lookback days", 1, 20, d0["pricing_days"], key=k + "pd",
                                           help="Days before the notice that set the variable price")
    row[5].form_submit_button("Run", type="primary", width="stretch")

    with st.expander("Term sheet, data and impact model"):
        t1, t2, t3 = st.tabs(["Term sheet", "Data", "Impact model"])
        with t1:
            a, b, c = st.columns(3)
            if opt1:
                vol_thr = a.number_input("Volume threshold", 0.0, 1.0, 0.30, 0.05, key=k + "vt",
                                         help="Advance cut if session volume < advance / threshold. SunPower 0.30, Soluna 0.35")
            elif sepa:
                pricing = a.selectbox("Price set off", list(PRICING), key=k + "pr")
                map_pct = b.number_input("Min acceptable price (x prior close)", 0.0, 1.2, 0.0, 0.05, key=k + "map",
                                         help="A day with VWAP below it is excluded and cuts the advance by one day's share")
            else:
                pricing = a.selectbox("Variable price set off", list(PRICING), key=k + "pr")
                fixed_x = b.number_input("Fixed price (x last)", 0.0, 5.0, 1.25, 0.05, key=k + "fx")
                floor_x = c.number_input("Floor price (x last)", 0.0, 1.0, 0.20, 0.05, key=k + "fl")
            a, b, c = st.columns(3)
            own_cap = a.number_input("Ownership cap", 0.0, 0.25, 0.0499, 0.0001, format="%.4f", key=k + "own")
            adv_cap = b.number_input("Max size (x 5-day ADV)", 0.0, 10.0, 1.0 if sepa else 0.0, 0.25, key=k + "ac")
            lag = c.number_input("Delivery lag (days)", 0, 5, 1, key=k + "lag")
            sell_early = st.checkbox("Can sell committed shares before delivery", value=True, key=k + "se",
                                     help="The filed agreements allow it. If off, selling starts after the delivery lag.")
        with t2:
            a, b, c = st.columns(3)
            source = a.selectbox("Bars", SOURCES)
            minutes = b.selectbox("Bar size (min)", [5, 1, 15], help="Yahoo 1-minute bars go back about 7 days")
            shares_in = c.number_input("Shares out (M, 0 = look up)", 0.0, 1e6, 0.0, 1.0)
            upload = st.file_uploader("Bloomberg export", type=["xlsx", "xls", "csv"],
                                      help="Time column plus OHLC or last price and volume, New York time. "
                                           "A VWAP or value column is used as the bar price.")
            a, b, c = st.columns(3)
            bbg_sec = a.text_input("Bloomberg security", "", placeholder=f"{ticker} US Equity")
            bbg_from = b.date_input("From", dt.date.today() - dt.timedelta(days=200))
            bbg_to = c.date_input("To", dt.date.today())
        with t3:
            a, b, c, d = st.columns(4)
            eta = a.number_input("Impact strength", 0.0, 2.0, 0.5, 0.1, help="Square-root coefficient. Placeholder, not fitted.")
            hl_hours = b.number_input("Recovery half-life (h)", 0.1, 20.0, 1.0, 0.25)
            residual = c.number_input("Share that never fades", 0.0, 1.0, 0.2, 0.05)
            spread_bps = d.number_input("Half spread (bps)", 0.0, 200.0, 25.0, 5.0)

# ---------------------------------------------------------------- data
try:
    q = quote(ticker)
except Exception as e:
    st.error(f"No data for {ticker}: {e}")
    st.stop()
try:
    with st.spinner("Loading intraday bars"):
        if source == SOURCES[0]:
            bars = load_yahoo(ticker, minutes)
        elif source == SOURCES[1]:
            if upload is None:
                st.warning("Upload a Bloomberg export in Term sheet, data and impact model > Data.")
                st.stop()
            bars = load_file(upload.getvalue(), upload.name, minutes)
        else:
            try:
                import blpapi  # noqa: F401
            except ImportError:
                st.error("blpapi isn't installed here. On the terminal PC: `python -m pip install "
                         "--index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi`")
                st.stop()
            bars = load_bloomberg(bbg_sec or f"{ticker} US Equity", bbg_from, bbg_to, minutes)
except Exception as e:
    st.error(f"Couldn't load intraday bars: {e}")
    st.stop()
shares_out = shares_in * 1e6 if shares_in > 0 else q["mkt"].shares_out

ticker_strip(ticker, q, [("Bars", f"{bars.source.split(' ')[0]} {bars.bar_minutes}m",
                          f'<span class="muted">{bars.days[0]:%d %b} to {bars.days[-1]:%d %b}</span>')])

latest = float(bars.close[-1, -1])
ts = TermSheet(kind=d0["kind"], option=d0["option"], size_usd=size_m * 1e6, discount=discount, pricing=PRICING[pricing],
               pricing_days=int(pricing_days), notice_time=notice, volume_threshold=vol_thr, min_price_pct=map_pct,
               fixed_price=fixed_x * latest, floor_price=floor_x * latest, sell_days=int(sell_days),
               sell_before_delivery=sell_early, delivery_lag=int(lag), ownership_cap=own_cap, adv_cap=adv_cap,
               max_participation=max_part)
imp = IntradayImpact(eta=eta, half_life_bars=hl_hours * 60 / bars.bar_minutes, residual=residual,
                     half_spread=spread_bps / 1e4, overnight_bars=390 / bars.bar_minutes)
bars_key = (bars.source, ticker, str(bars.days[0]), str(bars.days[-1]), len(bars.days),
            float(bars.volume.sum()), float(bars.price.sum()))
try:
    with st.spinner("Replaying every notice day"):
        board, runs = run_all(bars, bars_key, ts, imp, shares_out)
except ValueError as e:
    st.error(str(e))
    st.stop()

what = "advance" if sepa else "conversion"
any_run = next(iter(runs.values()))
n_win = int(board["windows"].iloc[0])
adv_usd = float(np.median(bars.daily_volume * bars.daily_vwap))
best = board.iloc[0]
best_vwap = board.sort_values("beat_vwap_bps_median", ascending=False).iloc[0]

# ---------------------------------------------------------------- headline
cards = [
    {"label": "Best strategy", "value": SHORT[best["key"]], "sub": "by median P&L",
     "tip": "Ranked by median profit across every notice day in the data"},
    {"label": "Median P&L", "value": usd(best["profit_median"]),
     "sub": f'bad case {signed(usd(best["profit_p10"]), best["profit_p10"])}',
     "tip": "Median and 10th percentile profit per notice day, best strategy"},
    {"label": "vs VWAP", "value": bps(best["beat_vwap_bps_median"]),
     "sub": f'beat VWAP on {pct(best["win_rate_vs_vwap"], 0)} of days',
     "tip": "Average sale price vs the market VWAP over the selling window, desk's own trades included"},
    {"label": "Discount kept", "value": f'{best["discount_kept_median"]:.2f}x',
     "sub": f"of the {pct(discount)} discount",
     "tip": "Profit margin over the margin the discount alone gives. 1.00x = exactly the discount"},
]
if sepa:
    cards += [
        {"label": "Desk moved price", "value": pct(board["pricing_drag_median"].median(), 2),
         "sub": "on purchase price", "tip": "How much the desk's own sales lowered the VWAP that sets its purchase price"},
        {"label": "Advance cut", "value": pct(board["cut_pct_mean"].mean()),
         "sub": "of shares requested" if opt1 or map_pct else "none at these terms",
         "tip": "Average share of the requested advance never issued"},
    ]
else:
    skipped = any_run.skipped
    cards += [
        {"label": "Size vs ADV", "value": f"{size_m * 1e6 / adv_usd:.2f}x", "sub": f"of {usd(adv_usd)} a day"},
        {"label": "Notice days", "value": f"{n_win}", "sub": f"{skipped} skipped, conversion above market" if skipped else "all tradable"},
    ]
kpis(cards)
footnote(f"Backtest on {n_win} overlapping notice days, {bars.days[0]:%d %b} to {bars.days[-1]:%d %b %Y}, one stock. "
         "Each sale uses only data available at that bar. Impact settings are placeholders. Method in Notes.")

tabs = st.tabs(["Strategies", "Execution plan", "Day replay", "Liquidity", "Notes"])

# ---------------------------------------------------------------- strategies
with tabs[0]:
    lb = board.copy()
    lb["Strategy"] = lb["key"].map(SHORT)
    cols = {"Strategy": "Strategy", "profit_median": "P&L med", "profit_p10": "P&L p10",
            "beat_vwap_bps_median": "vs VWAP med", "beat_vwap_bps_p10": "vs VWAP p10",
            "win_rate_vs_vwap": "Beat VWAP", "beat_pricing_bps_median": "vs pricing VWAP",
            "discount_kept_median": "Disc. kept", "loss_rate": "Losing days", "forced_pct_median": "Dumped at end"}
    if sepa:
        cols["cut_pct_mean"] = "Adv. cut"
    lb = lb[list(cols)].rename(columns=cols)
    table(lb, {"P&L med": "${:,.0f}", "P&L p10": "${:,.0f}", "vs VWAP med": "{:+,.0f}", "vs VWAP p10": "{:+,.0f}",
               "Beat VWAP": "{:.0%}", "vs pricing VWAP": "{:+,.0f}", "Disc. kept": "{:.2f}x", "Losing days": "{:.0%}",
               "Dumped at end": "{:.1%}", "Adv. cut": "{:.1%}"},
          signed_cols=["P&L med", "P&L p10", "vs VWAP med", "vs VWAP p10", "vs pricing VWAP"], highlight_first=True)
    footnote("bps columns are the average sale vs VWAP. 'vs pricing VWAP' is against the measure that sets the purchase "
             "price. p10 = 10th percentile of notice days. 'Dumped at end' = share sold in the last bar because time ran out.")

    metric = st.segmented_control("Spread across notice days", ["vs VWAP (bps)", "P&L ($)"], default="vs VWAP (bps)",
                                  key="to_spread")
    col, fmt = ("beat_vwap_bps", "{:+,.0f}") if metric != "P&L ($)" else ("profit", "${:,.0f}")
    fig = go.Figure()
    for key in board["key"][::-1]:
        v = runs[key].windows[col].dropna()
        lo, med, hi = v.quantile(0.1), v.median(), v.quantile(0.9)
        fig.add_scatter(x=[lo, hi], y=[SHORT[key]] * 2, mode="lines", line=dict(color=MARKET, width=2),
                        hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=[med], y=[SHORT[key]], mode="markers", marker=dict(color=MARKET, size=10), showlegend=False,
                        hovertemplate=f"{SHORT[key]}: median {fmt.format(med)}, 10th to 90th {fmt.format(lo)} to "
                                      f"{fmt.format(hi)}<extra></extra>")
    fig.add_vline(x=0, line=dict(width=1, dash="dot", color=MUTED))
    fig = style(fig, "Median (dot) and 10th to 90th percentile (line)", None, None, height=260, legend=False)
    fig.update_layout(hovermode="closest")
    chart(fig)

# ---------------------------------------------------------------- execution plan
with tabs[1]:
    keys = list(STRATEGIES)
    pk = st.segmented_control("Strategy", keys, default=best["key"], format_func=lambda s: SHORT[s], key="to_plan_s") or best["key"]
    pr, pi = run_plan(bars, bars_key, ts, imp, pk, shares_out)
    if pr is None:
        st.warning(f"No trade: at these terms the conversion price is above the last price ({price(latest)}).")
    else:
        w = pr.windows.iloc[pi]
        qv, vv, fill, raw = pr.q[pi], pr.vol[pi], pr.fill[pi], pr.raw_px[pi]
        per, labels0 = buckets(bars.slot_times, 30, bars.bar_minutes)
        H = len(qv) // pr.slots
        nb = int(np.ceil(pr.slots / per))
        rows = []
        cum = 0.0
        for d in range(H):
            for bi in range(nb):
                sl = slice(d * pr.slots + bi * per, min(d * pr.slots + (bi + 1) * per, (d + 1) * pr.slots))
                qs, vs = qv[sl].sum(), vv[sl].sum()
                if vs <= 0 and qs <= 0:
                    continue
                cum += qs
                cost = 1e4 * (1 - (fill[sl] * qv[sl]).sum() / (raw[sl] * qv[sl]).sum()) if qs > 0 else np.nan
                rows.append({"Day": f"D{d + 1}", "From": labels0[bi], "Market vol": vs, "Sell": qs,
                             "% of vol": qs / vs if vs > 0 else np.nan, "Done": cum / max(w.shares, 1),
                             "Est. cost (bps)": cost})
        sched = pd.DataFrame(rows)
        sched = sched[(sched["Sell"] > 0) | (sched["Done"] < 1)].reset_index(drop=True)
        # the clean-up trade was folded into the last selling bucket; show it on its own line
        left_over = float(w.forced_pct * w.shares)
        sched["Final trade"] = 0.0
        if left_over > 1:
            li = sched.index[sched["Sell"] > 0][-1]
            sched.loc[li, "Sell"] -= left_over
            sched.loc[li, "Final trade"] = left_over
            sched.loc[li, "% of vol"] = sched.loc[li, "Sell"] / sched.loc[li, "Market vol"]
        start = sched[sched["Sell"] > 0].iloc[0] if (sched["Sell"] > 0).any() else None
        arrival_cost = 1e4 * (1 - (fill * qv).sum() / (raw * qv).sum())
        inside = sched[sched["Sell"] > 0]
        last_row = inside.iloc[-1] if len(inside) else sched.iloc[-1]
        kpis([
            {"label": "Shares to sell", "value": shares(w.shares), "sub": f"{usd(w.shares * latest)} at {price(latest)}"
             + (f", cut {pct(w.cut_pct, 0)}" if w.cut_pct > 0.005 else "")},
            {"label": "Start", "value": f"{start['Day']} {start['From']}" if start is not None else "n/a",
             "sub": "notice confirmed" if opt1 else "day of the notice"},
            {"label": "Last bucket", "value": f"{last_row['Day']} {last_row['From']}", "sub": f"{len(inside)} half hours"},
            {"label": "Peak participation", "value": pct(float(inside["% of vol"].max()), 0) if len(inside) else "n/a",
             "sub": f"cap {pct(max_part, 0)} of each bar"},
            {"label": "Left for final trade", "value": pct(w.forced_pct),
             "sub": signed(f"{shares(left_over)} at the close", -1) if left_over > 1 else "finishes inside the cap",
             "tip": "What the participation cap doesn't allow before the window closes"},
            {"label": "Est. cost vs arrival", "value": f"{arrival_cost:,.0f} bps",
             "sub": "spread plus own impact", "tip": "Expected average fill vs the last price, with no market move"},
        ])
        lab = [f"{r.Day} {r.From}" for r in sched.itertuples()]
        fig = go.Figure(go.Bar(x=lab, y=sched["Sell"], marker_color=DESK, name="Inside the cap",
                               customdata=np.c_[sched["% of vol"] * 100, sched["Done"] * 100],
                               hovertemplate="%{x}: %{y:,.0f} sh, %{customdata[0]:.1f}% of volume, "
                                             "%{customdata[1]:.0f}% done<extra></extra>"))
        if left_over > 1:
            fig.add_bar(x=lab, y=sched["Final trade"], name="Final trade, over the cap", marker_color="rgba(0,0,0,0)",
                        marker_line=dict(color=DESK, width=1.5), marker_pattern=dict(shape="/", fgcolor=DESK, size=6),
                        hovertemplate="%{x}: %{y:,.0f} sh left for one final trade<extra></extra>")
            fig.update_layout(barmode="stack")
        fig = style(fig, f"Planned shares per half hour · {SHORT[pk]}", None, "Shares", y_fmt=",.0f", height=270,
                    legend=left_over > 1)
        fig.update_layout(legend_traceorder="normal")
        chart(fig)
        show_cols = ["Day", "From", "Market vol", "Sell", "% of vol", "Done", "Est. cost (bps)"] + \
            (["Final trade"] if left_over > 1 else [])
        table(sched[show_cols], {"Market vol": "{:,.0f}", "Sell": "{:,.0f}", "% of vol": "{:.1%}", "Done": "{:.0%}",
                                 "Est. cost (bps)": "{:,.0f}", "Final trade": lambda v: f"{v:,.0f}" if v > 0 else ""},
              height="content" if len(sched) <= 12 else 460)
        footnote("Expected session: last close held flat, the stock's average half-hour volume pattern and 20-day "
                 "average volume. 'Into strength' reacts to price moves, so it is planned on the volume curve. "
                 "Est. cost is the fill vs the last price: half spread plus the desk's own impact.")
        export.download_button("Execution plan (csv)", sched.to_csv(index=False).encode(), f"{ticker}_plan.csv",
                               width="stretch")

# ---------------------------------------------------------------- day replay
with tabs[2]:
    a, b = st.columns([2, 1])
    rk = a.segmented_control("Strategy ", list(STRATEGIES), default=best["key"], format_func=lambda s: SHORT[s],
                             key="to_replay_s") or best["key"]
    r = runs[rk]
    days = r.windows["notice_day"].dt.strftime("%a %d %b %Y").tolist()
    i = days.index(b.selectbox("Notice day", days, index=len(days) - 1, key="to_day"))
    row = r.windows.iloc[i]
    kpis([
        {"label": "Shares", "value": shares(row.shares),
         "sub": f"{pct(row.shares_pct_out, 2)} of out" + (f", cut {pct(row.cut_pct, 0)}" if row.cut_pct > 0.005 else "")},
        {"label": "Purchase price" if sepa else "Conversion price", "value": price(row.purchase_price)},
        {"label": "Average sale", "value": price(row.avg_sale), "sub": f"VWAP {price(row.interval_vwap)}"},
        {"label": "vs VWAP", "value": bps(row.beat_vwap_bps)},
        {"label": "P&L", "value": usd(row.profit), "sub": signed(pct(abs(row.margin)), row.margin) + " margin"},
        {"label": "Dumped at end", "value": pct(row.forced_pct)},
    ])
    T = r.q.shape[1]
    labels = [f"D{t // r.slots + 1} {bars.slot_times[t % r.slots]}" for t in range(T)]
    vol_us = r.vol[i] + r.q[i]
    run_vwap = np.cumsum(r.px[i] * r.vol[i] + r.fill[i] * r.q[i]) / np.maximum(np.cumsum(vol_us), 1.0)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.04)
    fig.add_scatter(x=labels, y=r.raw_px[i], name="Traded", line=dict(color=MARKET, width=1.6), row=1, col=1)
    fig.add_scatter(x=labels, y=r.px[i], name="With desk selling", line=dict(color=DESK, width=1.6), row=1, col=1)
    fig.add_scatter(x=labels, y=run_vwap, name="Running VWAP", line=dict(color=MUTED, width=1.2, dash="dot"), row=1, col=1)
    fig.add_scatter(x=labels, y=np.full(T, row.purchase_price), name="Purchase price" if sepa else "Conversion price",
                    line=dict(color=MUTED, width=1, dash="dash"), row=1, col=1, hoverinfo="skip")
    if sepa:
        p0, p1 = r.price_bars
        fig.add_vrect(x0=labels[p0], x1=labels[p1], fillcolor=MARKET, opacity=0.06, line_width=0, row=1, col=1)
    share = np.where(r.vol[i] > 0, r.q[i] / np.maximum(r.vol[i], 1.0), 0.0)
    fig.add_bar(x=labels, y=np.minimum(share, 1.0), name="Desk % of volume", marker_color=DESK, row=2, col=1,
                customdata=r.q[i], hovertemplate="%{y:.1%} of volume, %{customdata:,.0f} sh<extra></extra>")
    fig = style(fig, "Price, running VWAP and desk participation" + (" · shaded = pricing period" if sepa else ""),
                None, None, height=460)
    fig.update_yaxes(tickformat="$,.4f" if latest < 1 else "$,.2f", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_xaxes(nticks=12, row=2, col=1)
    chart(fig)

# ---------------------------------------------------------------- liquidity
with tabs[3]:
    m = q["mkt"]
    per, labels0 = buckets(bars.slot_times, 30, bars.bar_minutes)
    recent = bars.volume[-20:]
    share_slot = (recent / np.maximum(recent.sum(axis=1, keepdims=True), 1.0)).mean(axis=0)
    prof = np.array([share_slot[j:j + per].sum() for j in range(0, bars.slots, per)])
    dv = pd.Series(bars.daily_volume * bars.daily_vwap, index=bars.days)
    adv20_sh = float(recent.sum(axis=1).mean())
    size_sh = size_m * 1e6 / latest
    sd = m.sigma_daily
    kpis([
        {"label": "ADV 20D (bars)", "value": f"{adv20_sh / 1e6:,.2f}M sh", "sub": usd(float(dv.tail(20).mean()))},
        {"label": f"{what.capitalize()} size", "value": shares(size_sh), "sub": f"{size_sh / adv20_sh:.2f}x ADV"},
        {"label": "First 30 min", "value": pct(prof[0]), "sub": "of daily volume"},
        {"label": "Last 30 min", "value": pct(prof[-1]), "sub": "of daily volume"},
        {"label": "Daily vol", "value": pct(sd), "sub": f"{pct(m.sigma, 0)} annualized"},
        {"label": "Median 5-min volume", "value": f"{np.median(recent) / 1e3:,.0f}K sh", "sub": "last 20 days"},
    ])
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Bar(x=labels0, y=prof, marker_color=MARKET, name="Share of daily volume",
                               hovertemplate="%{x}: %{y:.1%} of the day<extra></extra>"))
        chart(style(fig, "Intraday volume pattern, last 20 days", None, None, y_fmt=".0%", height=280, legend=False))
    with right:
        fig = go.Figure(go.Bar(x=dv.index, y=dv.values, marker_color=MARKET, name="Dollar volume",
                               hovertemplate="%{x|%d %b}: $%{y:,.0f}<extra></extra>"))
        fig.add_scatter(x=dv.index, y=dv.rolling(20).mean(), name="20-day average", line=dict(color=MUTED, width=1.5))
        chart(style(fig, "Daily dollar volume", None, None, y_fmt="$,.2s", height=280))
    section(f"Time and impact to sell {shares(size_sh)}", "daily square-root law, impact strength "
            f"{eta:g}")
    rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    liq = pd.DataFrame({"Participation": rates,
                        "Shares a day": [p * adv20_sh for p in rates],
                        "Days to sell": [size_sh / (p * adv20_sh) for p in rates],
                        "Est. impact (bps)": [1e4 * eta * sd * np.sqrt(p) for p in rates]})
    table(liq, {"Participation": "{:.0%}", "Shares a day": "{:,.0f}", "Days to sell": "{:,.1f}",
                "Est. impact (bps)": "{:,.0f}"})

# ---------------------------------------------------------------- notes
with tabs[4]:
    bar_price = ("the bar's VWAP (dollars traded / shares traded)" if "bar VWAP" in bars.source
                 else "the bar's typical price, (high + low + close) / 3. The agreements use Bloomberg VWAP, so "
                      "Bloomberg bars make this exact")
    st.markdown(f"""
**Terms.** Option 1: {pct(0.04, 0)} discount to the VWAP from the notice confirmation to the 4 PM close; the advance
is cut to the larger of the threshold x session volume and what the desk sold if volume is below advance / threshold.
Option 2: {pct(0.03, 0)} discount to the lowest daily VWAP over 3 days from the notice; a day below the minimum
acceptable price is excluded and cuts the advance by a third. Conversion: the lower of the fixed price and 93% of the
lowest daily VWAP over the 5 days before the notice, never below the floor. Sources: SunPower 8-K filed 2026-01-30
(exhibits 10.1, 10.2), Soluna 10-K filed 2026-03-30 (exhibit 10.114). Every term is an input.

**Lookahead.** Each sale uses only earlier bars' prices, the running VWAP so far, and the volume pattern and
volatility from the 20 days before the notice. The one same-bar input is the bar's own volume, used as the
participation cap, the way a volume algo fills. When the final size is only known at the end of the pricing period,
what the desk still holds is sold in one clean-up trade after that period closes, outside the pricing VWAP. Notice
days overlap and come from one stretch of one stock, so a few bps between strategies is noise.

**Prices.** Bar price is {bar_price}. Impact: each bar's sale moves the price by a square-root amount, scaled so a
steady day at share pi of volume ends impact strength x daily vol x sqrt(pi) lower; {pct(residual, 0)} stays and the
rest fades with a {hl_hours:g}-hour half-life. These are placeholders until calibrated on real fills.

**Strategies.** TWAP: even by time. VWAP curve: follows the average intraday volume pattern. POV: the maximum allowed
share of every bar. Front-loaded: about 2.5x the VWAP pace early. Into strength: 1.75x the VWAP pace when the last
price is above the running VWAP, 0.5x below.

**Not modeled.** The closing auction as its own venue, dark liquidity, other sellers reacting to a known PIPE
investor, the 19.99% exchange cap across many advances, borrow and hedging, overlapping advances.
""".replace("$", r"\$"))

# ---------------------------------------------------------------- export
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as xw:
    board.drop(columns="key").to_excel(xw, sheet_name="Scoreboard", index=False)
    for key_, rr in runs.items():
        rr.windows.to_excel(xw, sheet_name=key_.upper(), index=False)
export.download_button("Backtest, every notice day (xlsx)", buf.getvalue(), f"{ticker}_tradeout.xlsx", width="stretch")
