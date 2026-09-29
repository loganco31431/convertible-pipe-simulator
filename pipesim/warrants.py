"""Black-Scholes warrant valuation, vectorized across paths.

Treats the warrant as a European call on the stock with no dividends. It ignores the dilution
from exercise, the 4.99% ownership blocker (which can stop exercise in size), the cost of
selling the exercised shares, and any anti-dilution adjustment. Microcap realized volatility
is often above 100%, where Black-Scholes values a long-dated warrant close to the stock price
itself, so the valuation volatility is capped (`Warrant.vol_cap`); that cap is an assumption.
"""
from math import erf, sqrt

import numpy as np

from .instruments import TRADING_DAYS, Warrant

_erf = np.vectorize(erf, otypes=[float])


def _cdf(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / sqrt(2.0)))


def bs_call(s, k: float, years: float, vol: float, rate: float = 0.0):
    """Call value per share for spot `s` (scalar or array)."""
    s = np.asarray(s, dtype=float)
    if years <= 0 or vol <= 0:
        return np.maximum(s - k * np.exp(-rate * max(years, 0.0)), 0.0)
    sv = vol * sqrt(years)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(np.maximum(s, 1e-12) / k) + (rate + 0.5 * vol**2) * years) / sv
    return s * _cdf(d1) - k * np.exp(-rate * years) * _cdf(d1 - sv)


def warrant_vol(w: Warrant, sigma: float) -> float:
    return w.vol if w.vol > 0 else min(sigma, w.vol_cap)


def value(w: Warrant, s, sigma: float, day: int = 0):
    """Total value of the warrant position on trading day `day` for spot `s`."""
    years = (w.term_days - day) / TRADING_DAYS
    return w.shares * bs_call(s, w.strike, years, warrant_vol(w, sigma), w.rate)
