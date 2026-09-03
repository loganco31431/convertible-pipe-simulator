"""Conversion engine. Everything is vectorized across Monte Carlo paths.

Lookahead note: the conversion price on day t uses only closes up to and including day t
(trailing VWAP), and the investor sells at the day-t close. No future prices enter any
decision, so the simulation has no lookahead by construction.
"""
from dataclasses import dataclass

import numpy as np

from .instruments import ConvertibleNote, StandbyEquityFacility, TRADING_DAYS
from .paths import simulate_prices, trailing_vwap


@dataclass
class SimResult:
    cashflows: np.ndarray      # (n_paths, days+1) investor cash, negative = outflow
    shares_issued: np.ndarray  # (n_paths,) total shares issued by the issuer
    converted: np.ndarray      # (n_paths,) principal converted (or drawn)
    paths: np.ndarray          # simulated closes
    shares_out: float


def simulate(instr, s0: float, sigma: float, shares_out: float, n_paths: int = 10_000,
             mu: float = 0.0, seed: int | None = 0) -> SimResult:
    days = instr.maturity_days
    paths = simulate_prices(s0, sigma, days, n_paths, mu=mu, seed=seed)
    vwap = trailing_vwap(paths, instr.vwap_lookback)

    conv_days = np.arange(instr.cadence, days + 1, instr.cadence)
    px = paths[:, conv_days]                     # market close on conversion days
    conv_px = (1.0 - instr.discount) * vwap[:, conv_days]

    if isinstance(instr, ConvertibleNote):
        conv_px = np.maximum(conv_px, instr.floor_price)
        # Conversion only happens when the investor can sell above the conversion price.
        ok = px * (1.0 - instr.sell_slippage) > conv_px
        per_period = instr.tranche
        budget = instr.principal
        upfront = -instr.principal * (1.0 - instr.oid)
    else:
        ok = px >= instr.min_draw_price
        per_period = instr.advance
        budget = instr.commitment
        upfront = 0.0

    # Dollar amount converted each period: fixed tranche while budget remains, gated by `ok`.
    want = np.where(ok, per_period, 0.0)
    cum = np.cumsum(want, axis=1)
    amount = np.clip(want - np.clip(cum - budget, 0.0, None), 0.0, None)

    shares = amount / conv_px
    proceeds = shares * px * (1.0 - instr.sell_slippage)

    cashflows = np.zeros_like(paths)
    cashflows[:, 0] = upfront
    if isinstance(instr, StandbyEquityFacility):
        # Investor pays the issuer for shares and sells them the same day.
        cashflows[:, conv_days] += proceeds - amount
    else:
        cashflows[:, conv_days] += proceeds
        remaining = instr.principal - amount.sum(axis=1)
        accrued = instr.principal * instr.coupon * days / TRADING_DAYS
        cashflows[:, -1] += remaining + accrued

    return SimResult(cashflows=cashflows, shares_issued=shares.sum(axis=1),
                     converted=amount.sum(axis=1), paths=paths, shares_out=shares_out)
