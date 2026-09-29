"""Trade-out vs VWAP page: replay the term sheet and five selling strategies on real intraday bars."""
import datetime as dt
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipesim.intraday.data import bloomberg_bars, csv_bars, yahoo_bars
from pipesim.intraday.tradeout import STRATEGIES, IntradayImpact, TermSheet, compare
from views.common import IMPACT, NO_IMPACT, pct, style, tiles, usd

SOURCES = ["Yahoo (free, last 60 days)", "Bloomberg export (Excel or CSV)", "Bloomberg terminal (live)"]
PRICING = {"Lowest daily VWAP": "lowest", "Average of daily VWAPs": "average", "VWAP over the whole period": "period"}
SHEETS = {
    "Equity line, Option 1: same-day VWAP": dict(kind="sepa", option=1, discount=0.04, sell_days=1, pricing_days=1),
    "Equity line, Option 2: lowest VWAP over 3 days": dict(kind="sepa", option=2, discount=0.03, sell_days=3, pricing_days=3),
    "Note or pre-paid advance conversion": dict(kind="note", option=2, discount=0.07, sell_days=5, pricing_days=5),
}
SOURCE_NOTE = ("Defaults are the terms in SunPower's equity line and pre-paid advance with YA II PN (8-K filed "
               "2026-01-30, exhibits 10.1 and 10.2) and Soluna's equity line (10-K filed 2026-03-30, exhibit 10.114).")
NOTICE_TIMES = ["09:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00"]


def bps(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+,.0f} bps"


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


@st.cache_data(ttl=86400, show_spinner=False)
def shares_outstanding(ticker):
    import yfinance as yf

    tk = yf.Ticker(ticker)
    try:
        sh = tk.fast_info.get("shares")
    except Exception:
        sh = None
    return float(sh or tk.info.get("sharesOutstanding") or 0)


@st.cache_data(show_spinner=False, max_entries=20)
def run_all(_bars, bars_key, ts, imp, shares_out):
    """`bars_key` identifies the data for the cache; the arrays themselves aren't hashed."""
    return compare(_bars, ts, imp, shares_out)


# ---------------------------------------------------------------- inputs
st.sidebar.header("Term sheet")
sheet = st.sidebar.selectbox("Pricing rule", list(SHEETS), help=SOURCE_NOTE)
d0 = SHEETS[sheet]
sepa, opt1 = d0["kind"] == "sepa", d0["kind"] == "sepa" and d0["option"] == 1
k = sheet + "|"                       # widget keys: a new pricing rule resets every default

with st.sidebar.form("tradeout"):
    size_m = st.number_input("Advance size ($M at the prior close)" if sepa else "Amount converted ($M)",
                             0.05, 100.0, 1.0, 0.05, key=k + "size")
    discount = st.slider("Discount", 0.0, 0.30, d0["discount"], 0.005, format="%.3f", key=k + "disc")
    notice, vol_thr, map_pct, fixed_x, floor_x, pricing, pricing_days = "09:30", 0.0, 0.0, 0.0, 0.0, "Lowest daily VWAP", 1
    if opt1:
        notice = st.selectbox("Investor confirms the notice at (New York)", NOTICE_TIMES, key=k + "nt",
                              help="The pricing period runs from this time to the 4 PM close")
        vol_thr = st.number_input("Volume threshold (0 = none)", 0.0, 1.0, 0.30, 0.05, key=k + "vt",
                                  help="If volume in the pricing period is below advance / threshold, the advance is "
                                       "cut to the larger of threshold x volume and what the desk sold. "
                                       "SunPower 0.30, Soluna 0.35.")
    elif sepa:
        pricing = st.selectbox("Price set off", list(PRICING), key=k + "pr")
        pricing_days = st.number_input("Pricing period (trading days from the notice)", 1, 20, d0["pricing_days"],
                                       key=k + "pd")
        map_pct = st.number_input("Minimum acceptable price (x prior close, 0 = none)", 0.0, 1.2, 0.0, 0.05, key=k + "map",
                                  help="A day with VWAP below it is excluded from pricing and cuts the advance by one "
                                       "day's share. Shares the desk sold that day are bought at the discount to it.")
    else:
        pricing = st.selectbox("Variable price set off", list(PRICING), key=k + "pr")
        pricing_days = st.number_input("Days before the notice", 1, 20, d0["pricing_days"], key=k + "pd")
        fixed_x = st.number_input("Fixed price (x latest price, 0 = none)", 0.0, 5.0, 1.25, 0.05, key=k + "fx")
        floor_x = st.number_input("Floor price (x latest price, 0 = none)", 0.0, 1.0, 0.20, 0.05, key=k + "fl")
    sell_days = st.number_input("Days the desk gives itself to sell", 1, 20, d0["sell_days"], key=k + "sd")
    sell_early = st.checkbox("Can sell committed shares before delivery", value=True, key=k + "se",
                             help="The filed agreements allow selling shares the investor is unconditionally "
                                  "obligated to buy under a pending notice. If off, selling starts after the lag.")
    lag = st.number_input("Delivery lag (trading days)", 0, 5, 1, disabled=sell_early, key=k + "lag")
    own_cap = st.number_input("Ownership cap (0 = none)", 0.0, 0.25, 0.0499, 0.0001, format="%.4f", key=k + "own")
    adv_cap = st.number_input("Max size (x prior 5-day avg volume, 0 = none)", 0.0, 10.0, 1.0 if sepa else 0.0, 0.25,
                              key=k + "ac")
    max_part = st.slider("Desk never sells more than this share of a bar's volume", 0.01, 0.50, 0.15, 0.01, key=k + "mp")

    st.header("Data")
    source = st.radio("Intraday bars from", SOURCES)
    ticker = st.text_input("Ticker", "OTLK").strip().upper()
    minutes = st.selectbox("Bar size (minutes)", [5, 1, 15], help="Yahoo: 1-minute bars only go back about 7 days")
    upload = st.file_uploader("Bloomberg export", type=["xlsx", "xls", "csv"],
                              help="Needs a time column plus open/high/low/close (or last price) and volume, times in "
                                   "New York local time. A VWAP or value (turnover) column is used as the bar price.")
    bbg_sec = st.text_input("Bloomberg security", "", placeholder=f"{ticker} US Equity")
    bbg_from = st.date_input("From", dt.date.today() - dt.timedelta(days=200))
    bbg_to = st.date_input("To", dt.date.today())
    shares_in = st.number_input("Shares outstanding (millions, 0 = look up)", 0.0, 1e6, 0.0, 1.0)

    st.header("Price impact")
    eta = st.slider("Impact strength", 0.0, 2.0, 0.5, 0.1, help="Same meaning as the deal page. Placeholder, not fitted.")
    hl_hours = st.number_input("Intraday recovery half-life (hours)", 0.1, 20.0, 1.0, 0.25)
    residual = st.slider("Share of impact that never fades", 0.0, 1.0, 0.2, 0.05)
    spread_bps = st.number_input("Half spread paid on each sale (bps)", 0.0, 200.0, 25.0, 5.0)
    go_btn = st.form_submit_button("Run", type="primary", width="stretch")

st.title("Trade-out vs VWAP")
if go_btn:
    st.session_state["to_ran"] = True
if not st.session_state.get("to_ran"):
    st.write("How well can the desk sell against VWAP while sticking to the term sheet? Pick a pricing rule and a "
             "data source on the left and press **Run**. Every trading day in the data is tried as a notice day, and "
             "five selling strategies are scored on the same days.")
    st.markdown("\n".join(f"- **{v}**" for v in STRATEGIES.values()))
    st.caption(SOURCE_NOTE)
    st.stop()

# ---------------------------------------------------------------- data
try:
    with st.spinner("Loading intraday bars..."):
        if source == SOURCES[0]:
            bars = load_yahoo(ticker, minutes)
        elif source == SOURCES[1]:
            if upload is None:
                st.warning("Upload a Bloomberg export first.")
                st.stop()
            bars = load_file(upload.getvalue(), upload.name, minutes)
        else:
            try:
                import blpapi  # noqa: F401
            except ImportError:
                st.error("The Bloomberg API isn't installed on this machine. On a Bloomberg terminal PC run:\n\n"
                         "`python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi`")
                st.stop()
            bars = load_bloomberg(bbg_sec or f"{ticker} US Equity", bbg_from, bbg_to, minutes)
    shares_out = shares_in * 1e6 if shares_in > 0 else shares_outstanding(ticker)
except Exception as e:
    st.error(f"Couldn't load data: {e}")
    st.stop()
if not shares_out:
    st.error("No shares outstanding found; enter it in the sidebar.")
    st.stop()

latest = float(bars.close[-1, -1])
ts = TermSheet(kind=d0["kind"], option=d0["option"], size_usd=size_m * 1e6, discount=discount, pricing=PRICING[pricing],
               pricing_days=int(pricing_days), notice_time=notice, volume_threshold=vol_thr, min_price_pct=map_pct,
               fixed_price=fixed_x * latest, floor_price=floor_x * latest, sell_days=int(sell_days),
               sell_before_delivery=sell_early, delivery_lag=int(lag), ownership_cap=own_cap, adv_cap=adv_cap,
               max_participation=max_part)
imp = IntradayImpact(eta=eta, half_life_bars=hl_hours * 60 / bars.bar_minutes, residual=residual,
                     half_spread=spread_bps / 1e4, overnight_bars=390 / bars.bar_minutes)
try:
    with st.spinner("Replaying every notice day with five strategies..."):
        bars_key = (bars.source, ticker, str(bars.days[0]), str(bars.days[-1]), len(bars.days),
                    float(bars.volume.sum()), float(bars.price.sum()))
        table, runs = run_all(bars, bars_key, ts, imp, shares_out)
except ValueError as e:
    st.error(str(e))
    st.stop()

any_run = next(iter(runs.values()))
n_win = int(table["windows"].iloc[0])
adv_usd = float(np.median(bars.daily_volume * bars.daily_vwap))
what = "advance" if sepa else "conversion"
cap_note = ""
capped = any_run.windows["capped_pct"]
if (capped > 0.005).any():
    cap_note = (f" The ownership and volume caps cut the {what} on {pct((capped > 0.005).mean(), 0)} of notice days, "
                f"by {pct(float(capped[capped > 0.005].mean()), 0)} on average.")
st.caption((f"{sheet}. {ticker}, {bars.source} bars, {bars.days[0]:%d %b %Y} to {bars.days[-1]:%d %b %Y} "
            f"({len(bars.days)} days). {n_win} notice days tested"
            + (f", {any_run.skipped} skipped because converting would have lost money" if any_run.skipped else "")
            + f". Median daily turnover {usd(adv_usd)}, so a {usd(size_m * 1e6, 2)} {what} is "
            f"{size_m * 1e6 / adv_usd:,.2f} days of volume." + cap_note).replace("$", r"\$"))
st.info("**Lookahead check.** Each sale uses only earlier bars' prices, the running VWAP so far, and the volume "
        "pattern and volatility from days before the notice. The one same-bar input is the bar's own volume, used "
        "as a participation cap, the way a volume-participation algo fills. When the final size is only known at the "
        "end of the pricing period, what the desk still holds is sold in one clean-up trade after that period closes. "
        f"The notice days overlap and come from one {len(bars.days)}-day stretch of one stock, so treat differences "
        "of a few bps as noise.")

# ---------------------------------------------------------------- scoreboard
best_profit = table.iloc[0]
best_vwap = table.sort_values("beat_vwap_bps_median", ascending=False).iloc[0]
c = tiles(6 if sepa else 3)
c[0].metric("Most profit", usd(best_profit["profit_median"], 3),
            help=f"Median profit per {what} across notice days")
c[0].caption(best_profit["strategy"])
c[1].metric("Best vs VWAP", bps(best_vwap["beat_vwap_bps_median"]),
            help="Median average sale price vs the market VWAP over the selling days")
c[1].caption(best_vwap["strategy"])
c[2].metric("Discount kept", f"{best_profit['discount_kept_median']:.2f}x",
            help="Profit margin divided by the margin the discount alone would give. 1.00x = exactly the discount.")
c[2].caption(f"of the {pct(discount)} discount, {best_profit['strategy']}")
if sepa:
    c[3].metric("Sale price vs pricing VWAP", bps(best_profit["beat_pricing_bps_median"]),
                help="Average sale price vs the VWAP measure that sets the purchase price. The desk makes money "
                     "when this is above minus the discount.")
    c[3].caption(best_profit["strategy"])
    c[4].metric("Purchase price moved", pct(table["pricing_drag_median"].median(), 2),
                help="How much lower the pricing VWAP ends up because the desk's own sales are in it")
    c[4].caption("by the desk's own selling, median")
    c[5].metric("Advance cut", pct(table["cut_pct_mean"].mean()),
                help="Average share of the requested advance that was never issued, because of the volume threshold "
                     "or excluded days")
    c[5].caption("average across notice days")

show = table.drop(columns=["key", "windows"]).rename(columns={
    "strategy": "Strategy", "beat_vwap_bps_median": "vs VWAP, median (bps)", "beat_vwap_bps_p10": "vs VWAP, bad case (bps)",
    "win_rate_vs_vwap": "Days beating VWAP", "beat_pricing_bps_median": "vs pricing VWAP (bps)",
    "discount_kept_median": "Discount kept", "profit_median": "Profit, median", "profit_p10": "Profit, bad case",
    "loss_rate": "Days losing money", "forced_pct_median": "Dumped at the end", "cut_pct_mean": "Advance cut",
    "pricing_drag_median": "Purchase price moved"})
if not sepa:
    show = show.drop(columns=["Advance cut", "Purchase price moved"])
st.dataframe(show.style.format({"vs VWAP, median (bps)": "{:+,.0f}", "vs VWAP, bad case (bps)": "{:+,.0f}",
                                "Days beating VWAP": "{:.0%}", "vs pricing VWAP (bps)": "{:+,.0f}",
                                "Discount kept": "{:.2f}x", "Profit, median": "${:,.0f}",
                                "Profit, bad case": "${:,.0f}", "Days losing money": "{:.0%}",
                                "Dumped at the end": "{:.1%}", "Advance cut": "{:.1%}",
                                "Purchase price moved": "{:+.2%}"}),
             width="stretch", hide_index=True)
st.caption("vs VWAP: average sale price against the market VWAP over the selling days, with the desk's own trades in "
           "it. vs pricing VWAP: against the VWAP measure that sets the purchase price. Discount kept: profit margin "
           "divided by the margin the discount alone would give. Dumped at the end: share sold in the final bar or "
           "clean-up trade because the desk ran out of time. Bad case = 10th percentile of notice days.")

# spread of outcomes per strategy
w_all = pd.concat({STRATEGIES[key]: r.windows for key, r in runs.items()}, names=["strategy"]).reset_index(level=0)
order = table["strategy"].tolist()[::-1]
fig = go.Figure()
for s in order:
    b = w_all.loc[w_all["strategy"] == s, "beat_vwap_bps"].dropna()
    fig.add_scatter(x=[b.quantile(0.1), b.quantile(0.9)], y=[s, s], mode="lines", line=dict(color=NO_IMPACT, width=2),
                    hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=[b.median()], y=[s], mode="markers", marker=dict(color=NO_IMPACT, size=11),
                    hovertemplate=f"{s}<br>median %{{x:+,.0f}} bps<br>10th to 90th: {b.quantile(0.1):+,.0f} to "
                                  f"{b.quantile(0.9):+,.0f} bps<extra></extra>", showlegend=False)
fig.add_vline(x=0, line=dict(width=1, dash="dot"))
fig = style(fig, "Sale price vs VWAP across notice days (dot = median, line = 10th to 90th percentile)",
            "bps vs VWAP", None, height=320)
fig.update_layout(hovermode="closest")
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- one notice day up close
st.subheader("One notice day up close")
left, right = st.columns(2)
pick_s = left.selectbox("Strategy", list(STRATEGIES.values()), index=list(STRATEGIES.values()).index(best_profit["strategy"]))
key = next(a for a, v in STRATEGIES.items() if v == pick_s)
r = runs[key]
days = r.windows["notice_day"].dt.strftime("%a %d %b %Y").tolist()
i = days.index(right.selectbox("Notice day", days, index=len(days) - 1))
row = r.windows.iloc[i]
c = tiles(6)
c[0].metric("Shares", f"{row.shares / 1e6:,.2f}M")
c[0].caption(f"{pct(row.shares_pct_out, 2)} of shares outstanding"
             + (f", cut {pct(row.cut_pct, 0)} from the request" if row.cut_pct > 0.005 else ""))
c[1].metric("Purchase price" if sepa else "Conversion price", f"${row.purchase_price:,.4f}")
c[2].metric("Average sale", f"${row.avg_sale:,.4f}")
c[3].metric("vs VWAP", bps(row.beat_vwap_bps))
c[4].metric("Profit", usd(row.profit, 3))
c[5].metric("Dumped at the end", pct(row.forced_pct))

T = r.q.shape[1]
labels = [f"D{t // r.slots + 1} {bars.slot_times[t % r.slots]}" for t in range(T)]
fig = go.Figure()
fig.add_scatter(x=labels, y=r.raw_px[i], name="What actually traded", line=dict(color=NO_IMPACT, width=2))
fig.add_scatter(x=labels, y=r.px[i], name="With the desk's selling", line=dict(color=IMPACT, width=2))
fig.add_hline(y=row.purchase_price, line=dict(width=1, dash="dot"),
              annotation_text="purchase price" if sepa else "conversion price", annotation_position="bottom right")
if sepa:
    p0, p1 = r.price_bars
    fig.add_vrect(x0=labels[p0], x1=labels[p1], fillcolor=NO_IMPACT, opacity=0.07, line_width=0,
                  annotation_text="pricing period", annotation_position="top left")
fig = style(fig, "Price through the selling window", None, "Price ($)", y_fmt="$,.4f", height=340)
fig.update_xaxes(nticks=12)
st.plotly_chart(fig, width="stretch")
share = np.where(r.vol[i] > 0, r.q[i] / np.maximum(r.vol[i], 1.0), 0.0)
fig = go.Figure(go.Bar(x=labels, y=np.minimum(share, 1.0), marker_color=IMPACT, name="Desk share of bar volume",
                       customdata=r.q[i], hovertemplate="%{x}: %{y:.1%} of volume, %{customdata:,.0f} shares<extra></extra>"))
fig = style(fig, "Desk's share of each bar's volume (capped at 100% for display)", None, "Share of volume",
            y_fmt=".0%", height=260)
fig.update_xaxes(nticks=12)
st.plotly_chart(fig, width="stretch")

with st.expander("Every notice day for this strategy"):
    st.dataframe(r.windows.style.format({"shares_requested": "{:,.0f}", "capped_pct": "{:.1%}", "shares": "{:,.0f}",
                                         "shares_pct_out": "{:.2%}",
                                         "purchase_price": "${:.4f}", "avg_sale": "${:.4f}", "interval_vwap": "${:.4f}",
                                         "beat_vwap_bps": "{:+,.0f}", "beat_raw_vwap_bps": "{:+,.0f}",
                                         "beat_pricing_bps": "{:+,.0f}", "profit": "${:,.0f}", "margin": "{:.2%}",
                                         "discount_kept": "{:.2f}x", "pricing_drag": "{:+.2%}", "cut_pct": "{:.1%}",
                                         "forced_pct": "{:.1%}", "peak_share_of_volume": "{:.0%}",
                                         "stock_move": "{:+.1%}"}, na_rep="n/a"),
                 width="stretch", hide_index=True)

with st.expander("Assumptions and what's not modeled"):
    bar_price = ("the bar's VWAP (dollars traded divided by shares traded)" if "bar VWAP" in bars.source
                 else "the bar's typical price, (high + low + close) / 3, a proxy for the bar's VWAP. The agreements "
                      "use Bloomberg's VWAP, so Bloomberg bars make this exact")
    st.markdown(f"""
- **Terms.** {SOURCE_NOTE} Every term is an input; the volume threshold, discounts and caps differ by deal.
- **Bar price** is {bar_price}.
- **Impact:** each bar's sale moves the price by a square-root amount, scaled so a steady day of selling at share
  pi of volume lands impact strength x daily vol x sqrt(pi) lower, the same law as the deal page. {pct(residual, 0)} of
  it stays; the rest fades with a {hl_hours:g}-hour half-life. **Placeholders, not fitted.** A desk's own fills are the
  right calibration.
- **Strategies:** TWAP sells evenly by time. VWAP follows the stock's average intraday volume pattern from the 20
  days before the notice. POV sells the maximum allowed share of each bar. Front-loaded sells about 2.5x the VWAP
  pace early. Sell into strength sells 1.75x the VWAP pace when the last price is above the running VWAP and 0.5x
  when below. Anything left at the end of the selling window is sold in the last bar.
- **Sizing:** an advance is set in shares at the prior close; a conversion turns dollars into shares at a price set
  before the notice. Both are capped by the ownership limit and the cap relative to recent volume.
- **Excluded days:** the desk is assumed to take the extra shares it needs to cover what it already sold, at the
  discount to the minimum acceptable price, as the agreements allow.
- **Not modeled:** the closing auction as a separate venue, hidden or dark liquidity, other sellers reacting to a
  known investor, the 19.99% exchange cap across many advances, borrow and hedging, and overlapping advances.
""".replace("$", r"\$"))

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as xw:
    table.drop(columns="key").to_excel(xw, sheet_name="Scoreboard", index=False)
    for a, rr in runs.items():
        rr.windows.to_excel(xw, sheet_name=a.upper(), index=False)
st.sidebar.download_button("Download trade-out results (Excel)", buf.getvalue(),
                           file_name=f"{ticker}_tradeout.xlsx", width="stretch")
