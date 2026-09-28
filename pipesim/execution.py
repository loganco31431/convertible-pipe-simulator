"""Execution-aware conversion engine: volume limits and price impact.

The frictionless engine in `engine.py` assumes the investor sells every converted share at the
close. Here the investor can only sell a share of each day's volume, holds the rest as
inventory, and moves the price when it sells:

- Temporary impact (square-root law). Selling q shares into daily volume V costs
  `eta * sigma_daily * sqrt(q / V)` of the price on that day's fills. The square-root form and
  a coefficient of order one are the standard empirical result for large orders.
- Carried impact and recovery. A fraction `permanent` of that move carries past the day. Of the
  carried move, a share `residual` stays in the price for good; the rest fades with a half-life of
  `half_life` trading days, so the stock recovers part of the drop once the selling slows. The
  carried move drags down the VWAP that sets future conversion prices, which is the channel
  through which the investor's own selling raises the issuer's dilution. `half_life=inf` means
  nothing recovers (the earlier model); `residual=0` means everything eventually recovers.
- Sizing. On a conversion day the desk converts only what it expects to sell before the next
  conversion (`participation * ADV * cadence` shares, less inventory still held).
- Wind-down. After maturity the desk keeps selling leftover inventory for up to `tail_days`
  and dumps whatever remains on the last day at full impact.

Lookahead: every decision on day t uses closes and volumes up to day t only. Permanent impact
from day t's sales enters prices from day t+1 on. The time loop is needed for exactly that
reason: each day's price depends on the previous days' selling, so days cannot be computed in
one vectorized pass. The loop runs over days; every step is vectorized across paths.
"""
from dataclasses import dataclass

import numpy as np

from .instruments import ConvertibleNote, StandbyEquityFacility, TRADING_DAYS
from .market import MarketInputs


@dataclass
class ExecutionModel:
    participation: float = 0.10   # max share of each day's volume the investor sells
    eta: float = 0.5              # temporary impact coefficient, square-root law
    permanent: float = 0.5        # share of the temporary move that carries past the day
    half_life: float = 10.0       # trading days for the fading part of the carried move to halve
    residual: float = 0.3         # share of the carried move that never fades
    tail_days: int = 63           # trading days allowed after maturity to finish selling


@dataclass
class ExecResult:
    cashflows: np.ndarray       # (n_paths, horizon+1) investor cash
    shares_issued: np.ndarray   # (n_paths,)
    converted: np.ndarray       # (n_paths,) principal converted or drawn
    paths: np.ndarray           # (n_paths, horizon+1) closes after the investor's own impact
    base_paths: np.ndarray      # closes with no investor selling
    sold: np.ndarray            # (n_paths, horizon+1) shares sold each day
    volumes: np.ndarray         # (n_paths, horizon+1) market volume
    impact_cost: np.ndarray     # (n_paths,) dollars lost to temporary impact
    exit_day: np.ndarray        # (n_paths,) last day any shares were sold
    shares_out: float


def simulate_volumes(adv: float, logstd: float, n_paths: int, days: int, seed: int | None) -> np.ndarray:
    """Lognormal daily volume with mean `adv`, independent across days."""
    rng = np.random.default_rng(None if seed is None else seed + 1)
    mu = np.log(adv) - 0.5 * logstd**2
    v = np.exp(mu + logstd * rng.standard_normal((n_paths, days + 1)))
    return v


def run_execution(instr, mkt: MarketInputs, ex: ExecutionModel, n_paths: int = 5_000,
                  mu: float = 0.0, seed: int | None = 0,
                  base_paths: np.ndarray | None = None, volumes: np.ndarray | None = None) -> ExecResult:
    """Run the note or equity line with volume-limited, price-moving sales.

    Pass `base_paths` and `volumes` (shape (n, horizon+1)) to replay real history; otherwise
    zero-drift GBM prices and lognormal volumes are simulated from `mkt`.
    """
    horizon = instr.maturity_days + ex.tail_days
    if base_paths is None:
        from .paths import simulate_prices
        base_paths = simulate_prices(mkt.s0, mkt.sigma, horizon, n_paths, mu=mu, seed=seed)
    if volumes is None:
        volumes = simulate_volumes(mkt.adv_shares, mkt.volume_logstd, base_paths.shape[0], horizon, seed)
    n, T1 = base_paths.shape
    horizon = T1 - 1
    maturity = min(instr.maturity_days, horizon)
    sd = mkt.sigma_daily
    L = instr.vwap_lookback

    is_note = isinstance(instr, ConvertibleNote)
    per_period = instr.tranche if is_note else instr.advance
    budget = instr.principal if is_note else instr.commitment
    cap_shares = ex.participation * mkt.adv_shares * instr.cadence

    px = np.empty((n, T1))
    px[:, 0] = base_paths[:, 0]
    d_keep = np.zeros(n)                 # log price displacement that never fades
    d_fade = np.zeros(n)                 # log price displacement that decays each day
    decay = 0.5 ** (1.0 / ex.half_life) if ex.half_life > 0 else 0.0
    inv = np.zeros(n)                    # shares held, not yet sold
    left = np.full(n, float(budget))     # principal or commitment not yet converted
    cash = np.zeros((n, T1))
    sold = np.zeros((n, T1))
    shares_issued = np.zeros(n)
    impact_cost = np.zeros(n)
    exit_day = np.zeros(n, dtype=int)
    if is_note:
        cash[:, 0] = -instr.principal * (1.0 - instr.oid)

    for t in range(1, T1):
        d_fade *= decay                      # recovery from earlier days' selling
        px[:, t] = base_paths[:, t] * np.exp(-(d_keep + d_fade))
        # 1. convert (or draw) on schedule, sized to what can be sold before the next conversion
        if t <= maturity and t % instr.cadence == 0:
            vwap = px[:, max(0, t - L + 1):t + 1].mean(axis=1)
            cpx = (1.0 - instr.discount) * vwap
            if is_note:
                cpx = np.maximum(cpx, instr.floor_price)
                ok = px[:, t] * (1.0 - instr.sell_slippage) > cpx
            else:
                ok = px[:, t] >= instr.min_draw_price
            room = np.clip(cap_shares - inv, 0.0, None) * cpx
            amt = np.where(ok, np.minimum(np.minimum(per_period, left), room), 0.0)
            new_sh = np.where(cpx > 0, amt / np.where(cpx > 0, cpx, 1.0), 0.0)
            left -= amt
            inv += new_sh
            shares_issued += new_sh
            if not is_note:
                cash[:, t] -= amt            # SEPA investor pays the issuer for the shares
        # 2. sell into today's volume, last day dumps everything
        v = volumes[:, t]
        q = inv if t == horizon else np.minimum(inv, ex.participation * v)
        frac = np.sqrt(q / v)
        temp = np.clip(ex.eta * sd * frac, 0.0, 0.95)
        fill = px[:, t] * (1.0 - instr.sell_slippage - temp)
        cash[:, t] += q * fill
        impact_cost += q * px[:, t] * temp
        sold[:, t] = q
        inv -= q
        exit_day = np.where(q > 0, t, exit_day)
        carried = -np.log1p(-ex.permanent * temp)   # hits prices from tomorrow on
        d_keep += ex.residual * carried
        d_fade += (1.0 - ex.residual) * carried

    if is_note:
        accrued = instr.principal * instr.coupon * instr.maturity_days / TRADING_DAYS
        cash[:, maturity] += left + accrued

    return ExecResult(cashflows=cash, shares_issued=shares_issued, converted=budget - left,
                      paths=px, base_paths=base_paths, sold=sold, volumes=volumes,
                      impact_cost=impact_cost, exit_day=exit_day, shares_out=mkt.shares_out)


def execution_stats(res: ExecResult) -> dict:
    """Execution-specific numbers to report next to `summarize`."""
    share_of_vol = np.where(res.volumes > 0, res.sold / res.volumes, 0.0)
    gross = (res.sold * res.paths).sum(axis=1)
    cost_bps = np.where(gross > 0, 1e4 * res.impact_cost / np.where(gross > 0, gross, 1.0), np.nan)
    gap = 1.0 - res.paths / res.base_paths
    drop = gap[:, -1]
    peak = gap.max(axis=1)

    def q(a, p):
        return float(np.nanpercentile(a, p))

    return {
        "impact_bps_p50": q(cost_bps, 50),
        "price_drag_p50": q(drop, 50),
        "price_drag_peak_p50": q(peak, 50),
        "exit_day_p50": q(res.exit_day, 50),
        "max_share_of_volume": float(np.nanmax(share_of_vol)),
    }
