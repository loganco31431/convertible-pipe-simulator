"""Trade-out lab: how well can the desk sell the shares against VWAP while abiding by the term sheet?

The question a PIPE / equity-line desk actually answers every day: shares arrive (or are
committed) at a price set off VWAP, and they have to be sold into a thin stock without giving the
discount away. This module replays that on real intraday bars.

For every possible notice day in the data (a "window"), it

1. sizes the advance or conversion from what was known that morning, within the ownership cap;
2. sells the shares bar by bar with one of several strategies, never above the volume cap;
3. moves the price with the desk's own selling: each bar's sale q into bar volume v pushes the
   price down by c x daily vol x sqrt(q / v); a `residual` share of that stays, the rest fades with
   a half-life. c is set so that selling a steady share pi of volume all day leaves the price
   eta x daily vol x sqrt(pi) lower at the close, the same square-root law as the daily model;
4. recomputes the daily VWAPs with the desk's own trades in them, because in a VWAP-priced deal
   the desk's selling helps set its own purchase price;
5. scores the result: average sale price vs the interval VWAP (bps), how much of the discount
   was kept, dollar profit, and how much had to be dumped at the end.

Lookahead: a strategy's decision at bar t uses only prices from earlier bars, the running VWAP
up to t-1, the intraday volume profile and volatility from days before the notice, and the
desk's own inventory. The one same-bar input is the bar's volume, used as an execution cap
(a participation algo fills a share of volume as it prints). The time loop is needed because each
bar's price depends on the desk's earlier sales; every step is vectorized across windows.
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
    kind: str = "sepa"               # "sepa": priced over days after the notice; "note": priced off days before it
    size_usd: float = 1_000_000.0    # advance (SEPA) or conversion amount (note)
    discount: float = 0.03           # purchase / conversion price = (1 - discount) x pricing measure
    pricing: str = "lowest"          # "lowest" daily VWAP, "average" of daily VWAPs, or "period" VWAP
    pricing_days: int = 3            # trading days in the pricing period
    sell_days: int = 3               # trading days the desk gives itself to sell
    sell_before_delivery: bool = True  # may sell committed shares before they are delivered
    delivery_lag: int = 1            # trading days until delivery, if selling must wait
    ownership_cap: float = 0.0499    # max share of shares outstanding held at once
    adv_cap: float = 0.0             # advance capped at this multiple of the prior 5-day avg volume (0 = none)
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


def _windows(bars: Bars, ts: TermSheet, warmup: int):
    s0 = 0 if ts.sell_before_delivery else ts.delivery_lag
    horizon = max(s0 + ts.sell_days, ts.pricing_days if ts.kind == "sepa" else 0)
    first = max(warmup, ts.pricing_days if ts.kind == "note" else 0, 5)
    starts = np.arange(first, len(bars.days) - horizon + 1)
    return starts, s0, horizon


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


def _measure(vwaps: np.ndarray, vols: np.ndarray, how: str) -> np.ndarray:
    if how == "lowest":
        return vwaps.min(axis=1)
    if how == "average":
        return vwaps.mean(axis=1)
    return (vwaps * vols).sum(axis=1) / np.maximum(vols.sum(axis=1), 1.0)


def run_tradeout(bars: Bars, ts: TermSheet, imp: IntradayImpact, strategy: str, shares_out: float,
                 warmup: int = 10) -> TradeOut:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy}")
    starts, s0, H = _windows(bars, ts, warmup)
    if len(starts) == 0:
        raise ValueError("not enough days of data for this term sheet; shorten the pricing or selling period")
    S_ = bars.slots
    n, T = len(starts), H * S_
    idx = starts[:, None] + np.arange(H)[None, :]
    P = bars.price[idx].reshape(n, T)
    V = bars.volume[idx].reshape(n, T)
    prof, sig, last, adv5 = _prior_stats(bars, starts)
    tail_prof = np.cumsum(prof[:, ::-1], axis=1)[:, ::-1]            # share of a day's volume still to come

    # ---- size the trade from what is known on the notice morning
    if ts.kind == "note":
        pidx = starts[:, None] - np.arange(ts.pricing_days, 0, -1)[None, :]
        conv = (1 - ts.discount) * _measure(bars.daily_vwap[pidx], bars.daily_volume[pidx], ts.pricing)
        shares = ts.size_usd / conv
    else:
        conv = None
        shares = np.full(n, ts.size_usd / 1.0) / last                   # advance in shares at the prior close
    cap_sh = ts.ownership_cap * shares_out
    if ts.adv_cap > 0:
        cap_sh = np.minimum(cap_sh, ts.adv_cap * adv5)
    shares = np.minimum(shares, cap_sh)

    # ---- sell bar by bar
    decay = 0.5 ** (1.0 / imp.half_life_bars)
    night = decay ** imp.overnight_bars
    first_bar, last_bar = s0 * S_, (s0 + ts.sell_days) * S_ - 1
    R = shares.copy()
    d_keep, d_fade = np.zeros(n), np.zeros(n)
    q_all, fill_all, px_all = np.zeros((n, T)), np.zeros((n, T)), np.zeros((n, T))
    forced = np.zeros(n)
    run_pv, run_v, prev_px = np.zeros(n), np.zeros(n), P[:, 0].copy()
    typical_bar = np.maximum(adv5 / S_, 1.0)
    sig_d = sig * np.sqrt(S_)                                          # daily vol from prior days' bar returns
    # scale so a steady full day at participation pi ends eta * sig_d * sqrt(pi) below where it started
    c = imp.eta / ((1 - imp.residual) * (1 - decay ** S_) / (1 - decay) + imp.residual * S_)
    for t in range(T):
        k, j = divmod(t, S_)
        if j == 0:
            run_pv[:], run_v[:] = 0.0, 0.0
            if k > 0:
                d_fade *= night
        v = V[:, t]
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
        if t == last_bar:                                                  # whatever is left goes now
            forced += R - q
            q = R.copy()
        veff = np.maximum(v, 0.1 * typical_bar)
        I = np.clip(c * sig_d * np.sqrt(q / veff), 0.0, 0.5)
        px = P[:, t] * np.exp(-(d_keep + d_fade) - 0.5 * I)
        fill = px * (1.0 - imp.half_spread)
        q_all[:, t], fill_all[:, t], px_all[:, t] = q, fill, px
        R -= q
        run_pv += px * v + fill * q
        run_v += v + q
        prev_px = px
        d_keep += imp.residual * I
        d_fade = d_fade * decay + (1.0 - imp.residual) * I

    # ---- daily VWAPs with and without the desk, purchase price, scores
    pv_day = (px_all * V + fill_all * q_all).reshape(n, H, S_).sum(axis=2)
    v_day = (V + q_all).reshape(n, H, S_).sum(axis=2)
    vwap_us = pv_day / np.maximum(v_day, 1.0)
    raw_v = V.reshape(n, H, S_).sum(axis=2)
    vwap_raw = (P * V).reshape(n, H, S_).sum(axis=2) / np.maximum(raw_v, 1.0)
    if ts.kind == "sepa":
        N = ts.pricing_days
        buy = (1 - ts.discount) * _measure(vwap_us[:, :N], v_day[:, :N], ts.pricing)
        buy_raw = (1 - ts.discount) * _measure(vwap_raw[:, :N], raw_v[:, :N], ts.pricing)
    else:
        assert conv is not None
        buy = buy_raw = conv
    sl = slice(first_bar, last_bar + 1)
    int_vwap = (px_all[:, sl] * V[:, sl] + fill_all[:, sl] * q_all[:, sl]).sum(axis=1) / \
        np.maximum((V[:, sl] + q_all[:, sl]).sum(axis=1), 1.0)
    int_vwap_raw = (P[:, sl] * V[:, sl]).sum(axis=1) / np.maximum(V[:, sl].sum(axis=1), 1.0)
    proceeds = (fill_all * q_all).sum(axis=1)
    avg = proceeds / np.maximum(shares, 1e-9)
    cost = shares * buy
    nominal = 1.0 / (1.0 - ts.discount) - 1.0
    share_vol = np.where(V > 0, q_all / np.maximum(V, 1.0), 0.0)
    w = pd.DataFrame({
        "notice_day": bars.days[starts],
        "shares": shares,
        "shares_pct_out": shares / shares_out,
        "purchase_price": buy,
        "avg_sale": avg,
        "interval_vwap": int_vwap,
        "beat_vwap_bps": 1e4 * (avg / int_vwap - 1.0),
        "beat_raw_vwap_bps": 1e4 * (avg / int_vwap_raw - 1.0),
        "profit": proceeds - cost,
        "margin": proceeds / cost - 1.0,
        "discount_kept": (proceeds / cost - 1.0) / nominal,
        "pricing_drag": buy / buy_raw - 1.0,
        "forced_pct": forced / np.maximum(shares, 1e-9),
        "peak_share_of_volume": share_vol[:, sl].max(axis=1),
        "stock_move": P[:, -1] / P[:, 0] - 1.0,
    })
    return TradeOut(strategy=strategy, windows=w, q=q_all, fill=fill_all, px=px_all, raw_px=P, vol=V, slots=S_)


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
            "discount_kept_median": w["discount_kept"].median(),
            "profit_median": w["profit"].median(),
            "profit_p10": w["profit"].quantile(0.10),
            "forced_pct_median": w["forced_pct"].median(),
            "pricing_drag_median": w["pricing_drag"].median(),
            "windows": len(w),
        })
    table = pd.DataFrame(rows).sort_values("profit_median", ascending=False).reset_index(drop=True)
    return table, runs
