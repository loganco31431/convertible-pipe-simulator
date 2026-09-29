"""Trade-out lab: how well can the desk sell the shares against VWAP while abiding by the term sheet?

The question a PIPE / equity-line desk actually answers every day: shares arrive (or are
committed) at a price set off VWAP, and they have to be sold into a thin stock without giving the
discount away. This module replays that on real intraday bars.

For every possible notice day in the data (a "window"), it

1. sizes the advance or conversion from what was known that morning, within the ownership cap
   and the cap on advance size relative to recent volume;
2. sells the shares bar by bar with one of several strategies, never above the volume cap;
3. moves the price with the desk's own selling: each bar's sale q into bar volume v pushes the
   price down by c x daily vol x sqrt(q / v); a `residual` share of that stays, the rest fades with
   a half-life. c is set so that selling a steady share pi of volume all day leaves the price
   eta x daily vol x sqrt(pi) lower at the close, the same square-root law as the daily model;
4. applies the pricing rule with the desk's own trades in the VWAP, because in a VWAP-priced deal
   the desk's selling helps set its own purchase price;
5. scores the result: average sale price vs VWAP (bps), how much of the discount was kept,
   dollar profit, and how much had to be dumped at the end.

Pricing rules, as written in the filed agreements (see `pipesim/presets.py` for sources):

- SEPA Option 1. Purchase price = (1 - discount) x VWAP from the notice confirmation to the
  4 PM close that day. If total volume in that period is below advance / `volume_threshold`, the
  advance is cut to the larger of `volume_threshold` x volume and what the desk sold.
- SEPA Option 2. Purchase price = (1 - discount) x the lowest daily VWAP over `pricing_days`
  starting on the notice day. A day with VWAP below the minimum acceptable price is excluded from
  pricing and cuts the advance by 1 / `pricing_days`; shares the desk sold that day (or needs to
  cover what it sold) are bought at (1 - discount) x the minimum acceptable price.
- Note. Conversion price = the lower of the fixed price and (1 - discount) x the lowest daily
  VWAP over the days before the notice, the latter never below the floor. Notice days on which
  converting and selling would lose money are skipped.

Lookahead: a strategy's decision at bar t uses only prices from earlier bars, the running VWAP
up to t-1, the intraday volume profile and volatility from days before the notice, and the
desk's own inventory. One same-bar input remains: the bar's volume is used as an execution cap
(a participation algo fills a share of volume as it prints). When the advance size is only fixed
at the end of the pricing period (Option 1's volume threshold, Option 2's excluded days) and the
selling window ends there too, whatever the desk still holds is sold in one clean-up trade after
the pricing period has closed: it is sized from the final, known advance, it pays its own impact,
and it is not counted in the volume or VWAP that set the purchase price. The time loop is needed
because each bar's price depends on the desk's earlier sales; every step is vectorized across
windows.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Bars

STRATEGIES = {
    "twap": "Even through the day (TWAP)",
    "vwap": "Follow the volume curve (VWAP)",
    "pov": "Max share of volume (POV)",
    "front": "Front-loaded",
    "strength": "Sell into strength",
}


@dataclass(frozen=True)
class TermSheet:
    kind: str = "sepa"               # "sepa" or "note"
    option: int = 2                  # SEPA pricing option, 1 or 2
    size_usd: float = 1_000_000.0    # advance (sized in shares at the prior close) or conversion amount
    discount: float = 0.03           # purchase / conversion price = (1 - discount) x pricing measure
    pricing: str = "lowest"          # "lowest" daily VWAP, "average" of daily VWAPs, or "period" VWAP
    pricing_days: int = 3            # Option 2: days from the notice; note: days before the notice
    notice_time: str = "09:30"       # Option 1: when the investor confirms the notice (New York time)
    volume_threshold: float = 0.30   # Option 1 (0 = none)
    min_price_pct: float = 0.0       # Option 2 minimum acceptable price, as a share of the prior close (0 = none)
    fixed_price: float = 0.0         # note: conversion price never above this (0 = none)
    floor_price: float = 0.0         # note: variable price never below this (0 = none)
    sell_days: int = 3               # trading days the desk gives itself to sell
    sell_before_delivery: bool = True  # may sell committed shares before they are delivered
    delivery_lag: int = 1            # trading days until delivery, if selling must wait
    ownership_cap: float = 0.0499    # max share of shares outstanding held at once
    adv_cap: float = 1.0             # advance capped at this multiple of the prior 5-day avg volume (0 = none)
    max_participation: float = 0.15  # desk never sells more than this share of a bar's volume


@dataclass(frozen=True)
class IntradayImpact:
    eta: float = 0.5                 # daily square-root coefficient, same meaning as the daily model
    half_life_bars: float = 12.0     # transient impact half-life, in bars (12 x 5 min = 1 hour)
    residual: float = 0.2            # share of each bar's impact that never fades
    overnight_bars: float = 78.0     # extra decay overnight, expressed in bars
    half_spread: float = 0.0025      # selling at the bid vs the bar's typical price


@dataclass
class TradeOut:
    strategy: str
    windows: pd.DataFrame            # one row per notice day: sizes, prices, scores
    q: np.ndarray                    # (n, T) shares sold per bar
    fill: np.ndarray                 # (n, T) desk's fill price per bar
    px: np.ndarray                   # (n, T) bar price with the desk's impact
    raw_px: np.ndarray               # (n, T) bar price as it actually traded
    vol: np.ndarray                  # (n, T) market volume per bar
    slots: int
    first_bar: int                   # first bar the desk may sell in
    last_bar: int                    # last bar of the selling window
    price_bars: tuple                # (first, last) bar of the pricing period; (-1, -1) for a note
    skipped: int = 0                 # notice days dropped because converting would lose money


def _prior_stats(bars: Bars, starts: np.ndarray, lookback: int = 20):
    """Volume profile, per-bar volatility, last close and 5-day ADV, all from days before each notice."""
    share = bars.volume / np.maximum(bars.volume.sum(axis=1, keepdims=True), 1.0)
    lr = np.diff(np.log(bars.close), axis=1)
    day_var = np.nanvar(lr, axis=1)
    prof, sig, last, adv5 = [], [], [], []
    for d in starts:
        lo = max(0, d - lookback)
        prof.append(share[lo:d].mean(axis=0))
        sig.append(np.sqrt(np.nanmean(day_var[lo:d])))
        last.append(bars.close[d - 1, -1])
        adv5.append(bars.daily_volume[max(0, d - 5):d].mean())
    prof = np.array(prof)
    prof = prof / prof.sum(axis=1, keepdims=True)
    return prof, np.array(sig), np.array(last), np.array(adv5)


def _measure(vwaps: np.ndarray, vols: np.ndarray, how: str, keep: np.ndarray | None = None) -> np.ndarray:
    """Pricing measure across days; `keep` masks out excluded days. Rows with no day kept return 0."""
    if keep is None:
        keep = np.ones_like(vwaps, dtype=bool)
    any_kept = keep.any(axis=1)
    if how == "lowest":
        out = np.where(keep, vwaps, np.inf).min(axis=1)
    elif how == "average":
        out = np.where(keep, vwaps, 0.0).sum(axis=1) / np.maximum(keep.sum(axis=1), 1)
    else:
        w = np.where(keep, vols, 0.0)
        out = (vwaps * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1.0)
    return np.where(any_kept, out, 0.0)


def run_tradeout(bars: Bars, ts: TermSheet, imp: IntradayImpact, strategy: str, shares_out: float,
                 warmup: int = 10) -> TradeOut:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy}")
    sepa = ts.kind == "sepa"
    opt1 = sepa and ts.option == 1
    n_price = (1 if opt1 else ts.pricing_days) if sepa else 0
    S_ = bars.slots
    s0 = 0 if ts.sell_before_delivery else ts.delivery_lag
    H = max(s0 + ts.sell_days, n_price)
    first_day = max(warmup, 5, 0 if sepa else ts.pricing_days)
    starts = np.arange(first_day, len(bars.days) - H + 1)
    if len(starts) == 0:
        raise ValueError("not enough days of data for this term sheet; shorten the pricing or selling period")
    prof, sig, last, adv5 = _prior_stats(bars, starts)

    # ---- size the trade from what is known on the notice morning
    skipped = 0
    conv = None
    if sepa:
        shares = ts.size_usd / last                                     # advances are set in shares
    else:
        pidx = starts[:, None] - np.arange(ts.pricing_days, 0, -1)[None, :]
        var = (1 - ts.discount) * _measure(bars.daily_vwap[pidx], bars.daily_volume[pidx], ts.pricing)
        conv = np.maximum(var, ts.floor_price)
        if ts.fixed_price > 0:
            conv = np.minimum(conv, ts.fixed_price)
        ok = last * (1 - imp.half_spread) > conv                        # otherwise the investor would not convert
        skipped = int((~ok).sum())
        starts, prof, sig, last, adv5, conv = starts[ok], prof[ok], sig[ok], last[ok], adv5[ok], conv[ok]
        if len(starts) == 0:
            raise ValueError("converting would lose money on every notice day in the data at these terms")
        shares = ts.size_usd / conv
    cap_sh = np.full(len(starts), ts.ownership_cap * shares_out if ts.ownership_cap > 0 else np.inf)
    if ts.adv_cap > 0:
        cap_sh = np.minimum(cap_sh, ts.adv_cap * adv5)
    uncapped = shares
    shares = np.minimum(shares, cap_sh)

    n, T = len(starts), H * S_
    idx = starts[:, None] + np.arange(H)[None, :]
    P = bars.price[idx].reshape(n, T)
    V = bars.volume[idx].reshape(n, T)
    tail_prof = np.cumsum(prof[:, ::-1], axis=1)[:, ::-1]            # share of a day's volume still to come

    j0 = 0
    if opt1:
        j0 = next((i for i, s in enumerate(bars.slot_times) if str(s) >= ts.notice_time), S_ - 1)
    first_bar = s0 * S_ + (j0 if s0 == 0 else 0)
    last_bar = (s0 + ts.sell_days) * S_ - 1
    price_first, price_end = (j0, n_price * S_ - 1) if sepa else (-1, -1)
    map_px = ts.min_price_pct * last if (sepa and not opt1) else np.zeros(n)
    resizes = sepa and ((opt1 and ts.volume_threshold > 0) or (not opt1 and ts.min_price_pct > 0))

    decay = 0.5 ** (1.0 / imp.half_life_bars)
    night = decay ** imp.overnight_bars
    sig_d = sig * np.sqrt(S_)                                          # daily vol from prior days' bar returns
    # scale so a steady full day at participation pi ends eta * sig_d * sqrt(pi) below where it started
    c = imp.eta / ((1 - imp.residual) * (1 - decay ** S_) / (1 - decay) + imp.residual * S_)
    typical_bar = np.maximum(adv5 / S_, 1.0)

    target = shares.copy()               # what the desk is working toward; resized at the end of pricing
    sold_tot = np.zeros(n)
    d_keep, d_fade = np.zeros(n), np.zeros(n)
    q_all, fill_all, px_all = np.zeros((n, T)), np.zeros((n, T)), np.zeros((n, T))
    forced = np.zeros(n)
    q_clean, fill_clean = np.zeros(n), np.zeros(n)     # clean-up trade after the pricing period closes
    run_pv, run_v, prev_px = np.zeros(n), np.zeros(n), P[:, 0].copy()
    regular, extra = shares.copy(), np.zeros(n)

    def execute(t, q):
        veff = np.maximum(V[:, t], 0.1 * typical_bar)
        impact = np.clip(c * sig_d * np.sqrt(q / veff), 0.0, 0.5)
        bar_px = P[:, t] * np.exp(-(d_keep + d_fade) - 0.5 * impact)
        return impact, bar_px, bar_px * (1.0 - imp.half_spread)

    def resize(t):
        """Final advance size once the pricing period is over, from the bars up to and including t."""
        on = slice(price_first, t + 1)
        if opt1:
            tot = (V[:, on] + q_all[:, on]).sum(axis=1)
            reg = np.where(tot >= shares / ts.volume_threshold, shares,
                           np.minimum(shares, np.maximum(ts.volume_threshold * tot, sold_tot)))
            return reg, np.zeros(n)
        pv = (px_all[:, :t + 1] * V[:, :t + 1] + fill_all[:, :t + 1] * q_all[:, :t + 1]).reshape(n, n_price, S_)
        vv = (V[:, :t + 1] + q_all[:, :t + 1]).reshape(n, n_price, S_)
        day_vwap = pv.sum(axis=2) / np.maximum(vv.sum(axis=2), 1.0)
        excl = day_vwap < map_px[:, None]
        reg = shares * (1.0 - excl.sum(axis=1) / n_price)
        sold_excl = (q_all[:, :t + 1].reshape(n, n_price, S_).sum(axis=2) * excl).sum(axis=1)
        ext = np.clip(np.maximum(sold_excl, sold_tot - reg), 0.0, shares - reg)
        return reg, ext

    for t in range(T):
        k, j = divmod(t, S_)
        if j == 0:
            run_pv[:], run_v[:] = 0.0, 0.0
            if k > 0:
                d_fade *= night
        v = V[:, t]
        R = np.clip(target - sold_tot, 0.0, None)
        want = np.zeros(n)
        if first_bar <= t <= last_bar:
            left_bars = last_bar - t + 1
            days_after = (last_bar // S_) - k
            exp_left = tail_prof[:, j] + days_after                         # expected volume still to come, in days
            base = R * prof[:, j] / np.maximum(exp_left, 1e-9)
            if strategy == "twap":
                want = R / left_bars
            elif strategy == "vwap":
                want = base
            elif strategy == "pov":
                want = ts.max_participation * v
            elif strategy == "front":
                want = R * np.minimum(1.0, 2.5 * prof[:, j] / np.maximum(exp_left, 1e-9))
            elif strategy == "strength":
                run_vwap = np.where(run_v > 0, run_pv / np.maximum(run_v, 1.0), prev_px)
                m = np.where(j == 0, 1.0, np.where(prev_px > run_vwap, 1.75, 0.5))
                want = base * m
        q = np.minimum(np.minimum(want, ts.max_participation * v), R)
        sizing_now = resizes and t == price_end
        if t == last_bar and not sizing_now:                              # whatever is left goes now
            forced += R - q
            q = R.copy()
        impact, bar_px, fill = execute(t, q)
        q_all[:, t], fill_all[:, t], px_all[:, t] = q, fill, bar_px
        sold_tot = sold_tot + q
        if sizing_now:
            regular, extra = resize(t)
            target = regular + extra
        run_pv += bar_px * v + fill * q
        run_v += v + q
        prev_px = bar_px
        d_keep += imp.residual * impact
        d_fade = d_fade * decay + (1.0 - imp.residual) * impact
        if sizing_now and t >= last_bar:                                   # selling window is over: clean up now
            q_clean = np.clip(target - sold_tot, 0.0, None)
            forced += q_clean
            _, _, fill_clean = execute(t, q_clean)
            sold_tot = sold_tot + q_clean

    # ---- purchase price with and without the desk, scores
    final = target if resizes else shares
    if sepa:
        pv_us = px_all * V + fill_all * q_all
        v_us = V + q_all
        if opt1:
            on = slice(price_first, price_end + 1)
            mkt_px = pv_us[:, on].sum(axis=1) / np.maximum(v_us[:, on].sum(axis=1), 1.0)
            mkt_raw = (P[:, on] * V[:, on]).sum(axis=1) / np.maximum(V[:, on].sum(axis=1), 1.0)
        else:
            N = n_price
            vwap_us = pv_us[:, :N * S_].reshape(n, N, S_).sum(axis=2) / np.maximum(
                v_us[:, :N * S_].reshape(n, N, S_).sum(axis=2), 1.0)
            raw_v = V[:, :N * S_].reshape(n, N, S_).sum(axis=2)
            vwap_raw = (P * V)[:, :N * S_].reshape(n, N, S_).sum(axis=2) / np.maximum(raw_v, 1.0)
            keep = vwap_us >= map_px[:, None]
            mkt_px = _measure(vwap_us, v_us[:, :N * S_].reshape(n, N, S_).sum(axis=2), ts.pricing, keep)
            mkt_raw = _measure(vwap_raw, raw_v, ts.pricing, keep)
        cost = (1 - ts.discount) * (regular * mkt_px + extra * map_px)
    else:
        assert conv is not None
        mkt_px = mkt_raw = conv / (1 - ts.discount)
        cost = final * conv
    # fold the clean-up trade into the last bar for reporting; the pricing above never saw it
    blend = q_all[:, last_bar] + q_clean
    fill_all[:, last_bar] = np.where(blend > 0, (fill_all[:, last_bar] * q_all[:, last_bar] + fill_clean * q_clean)
                                     / np.where(blend > 0, blend, 1.0), fill_all[:, last_bar])
    q_all[:, last_bar] = blend
    sl = slice(first_bar, last_bar + 1)
    int_vwap = (px_all[:, sl] * V[:, sl] + fill_all[:, sl] * q_all[:, sl]).sum(axis=1) / \
        np.maximum((V[:, sl] + q_all[:, sl]).sum(axis=1), 1.0)
    int_vwap_raw = (P[:, sl] * V[:, sl]).sum(axis=1) / np.maximum(V[:, sl].sum(axis=1), 1.0)
    proceeds = (fill_all * q_all).sum(axis=1)
    live = final > 0
    safe = np.where(live, final, 1.0)
    avg = np.where(live, proceeds / safe, np.nan)
    buy = np.where(live, cost / safe, np.nan)
    margin = np.where(live, proceeds / np.where(cost > 0, cost, 1.0) - 1.0, np.nan)
    nominal = 1.0 / (1.0 - ts.discount) - 1.0 if ts.discount > 0 else np.nan
    share_vol = np.where(V > 0, q_all / np.maximum(V, 1.0), 0.0)
    w = pd.DataFrame({
        "notice_day": bars.days[starts],
        "shares_requested": shares,
        "capped_pct": 1.0 - shares / uncapped,
        "shares": final,
        "shares_pct_out": final / shares_out,
        "purchase_price": buy,
        "avg_sale": avg,
        "interval_vwap": int_vwap,
        "beat_vwap_bps": 1e4 * (avg / int_vwap - 1.0),
        "beat_raw_vwap_bps": 1e4 * (avg / int_vwap_raw - 1.0),
        "beat_pricing_bps": 1e4 * (avg / np.where(mkt_px > 0, mkt_px, np.nan) - 1.0),
        "profit": np.where(live, proceeds - cost, 0.0),
        "margin": margin,
        "discount_kept": margin / nominal,
        "pricing_drag": np.where(mkt_raw > 0, mkt_px / np.where(mkt_raw > 0, mkt_raw, 1.0) - 1.0, np.nan),
        "cut_pct": 1.0 - final / shares,
        "forced_pct": forced / np.maximum(safe, 1e-9),
        "peak_share_of_volume": share_vol[:, sl].max(axis=1),
        "stock_move": P[:, -1] / P[:, 0] - 1.0,
    })
    return TradeOut(strategy=strategy, windows=w, q=q_all, fill=fill_all, px=px_all, raw_px=P, vol=V, slots=S_,
                    first_bar=first_bar, last_bar=last_bar, price_bars=(price_first, price_end), skipped=skipped)


def compare(bars: Bars, ts: TermSheet, imp: IntradayImpact, shares_out: float,
            strategies=tuple(STRATEGIES)) -> tuple[pd.DataFrame, dict]:
    """Run every strategy on the same windows; one summary row per strategy, best first."""
    runs = {s: run_tradeout(bars, ts, imp, s, shares_out) for s in strategies}
    rows = []
    for s, r in runs.items():
        w = r.windows
        rows.append({
            "strategy": STRATEGIES[s], "key": s,
            "beat_vwap_bps_median": w["beat_vwap_bps"].median(),
            "beat_vwap_bps_p10": w["beat_vwap_bps"].quantile(0.10),
            "win_rate_vs_vwap": (w["beat_vwap_bps"] > 0).mean(),
            "beat_pricing_bps_median": w["beat_pricing_bps"].median(),
            "discount_kept_median": w["discount_kept"].median(),
            "profit_median": w["profit"].median(),
            "profit_p10": w["profit"].quantile(0.10),
            "loss_rate": (w["profit"] < 0).mean(),
            "forced_pct_median": w["forced_pct"].median(),
            "cut_pct_mean": w["cut_pct"].mean(),
            "pricing_drag_median": w["pricing_drag"].median(),
            "windows": len(w),
        })
    table = pd.DataFrame(rows).sort_values("profit_median", ascending=False).reset_index(drop=True)
    return table, runs


def plan(bars: Bars, ts: TermSheet, imp: IntradayImpact, strategy: str, shares_out: float,
         lookback: int = 20) -> tuple[TradeOut | None, int]:
    """Expected schedule for a notice on the next trading day.

    Appends the days the term sheet needs, each at the last close with the stock's average
    intraday volume pattern and its average daily volume over the last `lookback` days, and runs
    the same engine on them. Prices are flat, so the plan shows sizing, timing and the desk's own
    expected impact, not a price forecast. "Sell into strength" reacts to price moves that a flat
    day doesn't have, so it is planned as the volume curve.

    Returns the run and the row of the planned notice day, or (None, -1) when the trade would not
    happen (a conversion priced above the market).
    """
    strat = "vwap" if strategy == "strength" else strategy
    sepa = ts.kind == "sepa"
    n_price = (1 if ts.option == 1 else ts.pricing_days) if sepa else 0
    s0 = 0 if ts.sell_before_delivery else ts.delivery_lag
    H = max(s0 + ts.sell_days, n_price)
    recent = bars.volume[-lookback:]
    prof = (recent / np.maximum(recent.sum(axis=1, keepdims=True), 1.0)).mean(axis=0)
    prof = prof / prof.sum()
    adv = float(recent.sum(axis=1).mean())
    last = float(bars.close[-1, -1])
    fut = pd.bdate_range(bars.days[-1] + pd.Timedelta(days=1), periods=H)
    flat = np.full((H, bars.slots), last)
    syn = Bars(days=bars.days.append(fut), slot_times=bars.slot_times,
               price=np.vstack([bars.price, flat]), close=np.vstack([bars.close, flat]),
               volume=np.vstack([bars.volume, np.tile(adv * prof, (H, 1))]),
               bar_minutes=bars.bar_minutes, source=bars.source)
    try:
        r = run_tradeout(syn, ts, imp, strat, shares_out)
    except ValueError:
        return None, -1
    hit = np.where(r.windows["notice_day"].to_numpy() == np.datetime64(fut[0]))[0]
    return (r, int(hit[0])) if len(hit) else (None, -1)
