"""Deal engine: contract terms, volume limits and price impact, one trading day at a time.

This is the single engine behind both the frictionless model (`engine.simulate`, which calls it
with impact switched off and unlimited volume) and the execution-aware model.

Contract terms (see `instruments.py` and `presets.py` for where each comes from):

- Note conversion price = the lower of the fixed price and the variable price; the variable
  price is a discount to the average or lowest daily VWAP and never below the floor.
- Amortization. "scheduled": monthly principal plus premium plus accrued interest from a
  start day. "on_trigger": the same payments, but only after the daily VWAP has been below the
  floor on 5 of 7 consecutive days, and only until it has been above the floor for 10 straight
  days. With `pay_in_shares` the payment is settled by an equity-line advance that offsets the
  note (the pre-paid advance loop) instead of cash.
- Equity line pricing: Option 1 (same-day VWAP, with the volume threshold), Option 2 (lowest
  daily VWAP over the pricing days, with excluded days), or a simple trailing average.
- Ownership cap (shares held at once), exchange cap (shares ever issued), and an advance cap as
  a multiple of recent volume.
- Commitment shares and a structuring fee at signing. Warrants valued at the end of the run.

Execution:

- Temporary impact (square-root law). Selling q shares into daily volume V costs
  `eta * sigma_daily * sqrt(q / V)` of the price on that day's fills.
- Carried impact and recovery. A fraction `permanent` of that move carries past the day. Of the
  carried move, a share `residual` stays in the price for good; the rest fades with a half-life
  of `half_life` trading days. `half_life=inf` means nothing recovers.
- Daily VWAP with the desk in it is approximated as the day's price less half the desk's
  temporary impact. Every VWAP-based term (lowest-VWAP pricing, the floor test) uses it, so the
  desk's own selling moves its own conversion and purchase prices.
- Sizing. The desk converts only what it expects to sell before the next conversion.
- Wind-down. After maturity the desk keeps selling for up to `tail_days` and dumps whatever
  remains on the last day at full impact.

Not modeled: the issuer failing to make a payment (every cash payment is assumed to be made),
registration delays, and events of default. For a loan repaid in cash that is the main risk,
so its returns here are a ceiling.

Lookahead: every decision on day t uses prices and volumes up to day t only; the floor test
uses VWAPs through day t-1. Option 1 and Option 2 purchase prices are set after the fact by
contract, from the pricing days themselves, and the cash is booked the day after the pricing
period ends. Impact from day t's sales enters prices from day t+1 on. The time loop is needed
for exactly that reason: each day's price, the principal left and the shares already issued all
depend on earlier days, so days cannot be computed in one vectorized pass. The loop runs over
days; every step is vectorized across paths.
"""
from dataclasses import dataclass

import numpy as np

from . import warrants
from .instruments import MONTH, TRADING_DAYS, ConvertibleNote, Warrant
from .market import MarketInputs


@dataclass(frozen=True)
class ExecutionModel:
    participation: float = 0.10   # max share of each day's volume the investor sells
    eta: float = 0.5              # temporary impact coefficient, square-root law
    permanent: float = 0.5        # share of the temporary move that carries past the day
    half_life: float = 10.0       # trading days for the fading part of the carried move to halve
    residual: float = 0.3         # share of the carried move that never fades
    tail_days: int = 63           # trading days allowed after maturity to finish selling


@dataclass
class ExecResult:
    cashflows: np.ndarray       # (n_paths, horizon+1) investor cash, warrants excluded
    shares_issued: np.ndarray   # (n_paths,)
    converted: np.ndarray       # (n_paths,) principal converted into shares, or dollars drawn
    paths: np.ndarray           # (n_paths, horizon+1) closes after the investor's own impact
    base_paths: np.ndarray      # closes with no investor selling
    sold: np.ndarray            # (n_paths, horizon+1) shares sold each day
    volumes: np.ndarray         # (n_paths, horizon+1) market volume
    impact_cost: np.ndarray     # (n_paths,) dollars lost to temporary impact
    exit_day: np.ndarray        # (n_paths,) last day any shares were sold
    shares_out: float
    repaid_cash: np.ndarray     # (n_paths,) principal repaid in cash before maturity
    repaid_maturity: np.ndarray # (n_paths,) principal still outstanding and repaid at maturity
    triggered: np.ndarray       # (n_paths,) True if an amortization event occurred
    warrant_cash: np.ndarray | None = None        # (n_paths, horizon+1) warrant payoff or end value
    warrant_value: np.ndarray | None = None       # (n_paths,) with the investor's own impact in the price
    warrant_value_base: np.ndarray | None = None  # (n_paths,) at prices with no investor selling
    warrant_value0: float = 0.0                   # value at closing
    warrant_shares: float = 0.0


def simulate_volumes(adv: float, logstd: float, n_paths: int, days: int, seed: int | None) -> np.ndarray:
    """Lognormal daily volume with mean `adv`, independent across days."""
    rng = np.random.default_rng(None if seed is None else seed + 1)
    mu = np.log(adv) - 0.5 * logstd**2
    v = np.exp(mu + logstd * rng.standard_normal((n_paths, days + 1)))
    return v


def _cap(share: float, shares_out: float) -> float:
    return share * shares_out if share > 0 else np.inf


def run_execution(instr, mkt: MarketInputs, ex: ExecutionModel, n_paths: int = 5_000,
                  mu: float = 0.0, seed: int | None = 0,
                  base_paths: np.ndarray | None = None, volumes: np.ndarray | None = None,
                  warrant: Warrant | None = None) -> ExecResult:
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
    slip = instr.sell_slippage

    is_note = isinstance(instr, ConvertibleNote)
    budget = instr.principal if is_note else instr.commitment
    sizing = ex.participation * mkt.adv_shares * instr.cadence
    own_cap = _cap(instr.ownership_cap, mkt.shares_out)
    exch_cap = _cap(instr.exchange_cap, mkt.shares_out)

    px = np.empty((n, T1))
    px[:, 0] = base_paths[:, 0]
    vw = np.empty((n, T1))               # daily VWAP with the desk's own trades in it
    vw[:, 0] = base_paths[:, 0]
    d_keep = np.zeros(n)                 # log price displacement that never fades
    d_fade = np.zeros(n)                 # log price displacement that decays each day
    decay = 0.5 ** (1.0 / ex.half_life) if ex.half_life > 0 else 0.0
    inv = np.zeros(n)                    # shares held, not yet sold
    left = np.full(n, float(budget))     # principal outstanding, or commitment not yet drawn
    acc = np.zeros(n)                    # accrued interest
    cash = np.zeros((n, T1))
    sold = np.zeros((n, T1))
    shares_issued = np.zeros(n)
    converted = np.zeros(n)
    impact_cost = np.zeros(n)
    repaid_cash = np.zeros(n)
    exit_day = np.zeros(n, dtype=int)
    amort_on = np.zeros(n, dtype=bool)
    triggered = np.zeros(n, dtype=bool)
    next_pay = np.full(n, -1)
    above_run = np.zeros(n, dtype=int)
    # pending equity-line advance (Option 1 or 2): priced after the fact
    pend = np.zeros(n)
    pend_age = np.zeros(n, dtype=int)
    pend_min = np.full(n, np.inf)
    pend_excl = np.zeros(n, dtype=int)
    pend_sold = np.zeros(n)
    pend_sold_excl = np.zeros(n)

    cash[:, 0] += instr.structuring_fee
    if is_note:
        cash[:, 0] -= instr.principal * (1.0 - instr.oid)
    elif instr.commitment_shares > 0:
        inv += instr.commitment_shares
        shares_issued += instr.commitment_shares

    for t in range(1, T1):
        d_fade *= decay                      # recovery from earlier days' selling
        p = base_paths[:, t] * np.exp(-(d_keep + d_fade))
        px[:, t] = p
        v = volumes[:, t]
        new_sh = np.zeros(n)

        if is_note and t <= maturity:
            acc += left * instr.coupon / TRADING_DAYS
            due = np.zeros(n, dtype=bool)
            if instr.payment_mode == "scheduled" and instr.monthly_payment > 0:
                if t >= instr.payment_start_day and (t - instr.payment_start_day) % MONTH == 0:
                    due = left > 0
            elif instr.payment_mode == "on_trigger" and instr.floor_price > 0 and instr.monthly_payment > 0:
                if t > instr.trigger_window:
                    w = vw[:, t - instr.trigger_window:t]
                    start = ((w < instr.floor_price).sum(axis=1) >= instr.trigger_days) & ~amort_on & (left > 0)
                    amort_on |= start
                    triggered |= start
                    next_pay = np.where(start, t + instr.trigger_lag, next_pay)
                    above_run = np.where(start, 0, np.where(vw[:, t - 1] > instr.floor_price, above_run + 1, 0))
                    amort_on &= ~(above_run >= instr.cure_days)
                due = amort_on & (next_pay == t) & (left > 0)
                next_pay = np.where(amort_on & (next_pay == t), t + MONTH, next_pay)
            if due.any():
                pay = np.where(due, np.minimum(instr.monthly_payment, left), 0.0)
                in_shares = np.zeros(n)
                if instr.pay_in_shares:
                    s_px = (1.0 - instr.sepa_discount) * vw[:, t - 1]
                    room = np.clip(np.minimum(own_cap - inv, exch_cap - shares_issued), 0.0, None)
                    in_shares = np.minimum(pay, room * s_px)
                    new_sh += in_shares / s_px
                    converted += in_shares
                in_cash = pay - in_shares
                cash[:, t] += in_cash * (1.0 + instr.payment_premium) + np.where(due, acc, 0.0)
                repaid_cash += in_cash
                left -= pay
                acc = np.where(due, 0.0, acc)

            if instr.tranche > 0 and t % instr.cadence == 0:
                if instr.pricing == "lowest":
                    ref = vw[:, max(0, t - L):t].min(axis=1)
                else:
                    ref = px[:, max(0, t - L + 1):t + 1].mean(axis=1)
                cpx = np.maximum((1.0 - instr.discount) * ref, instr.floor_price)
                if instr.fixed_price > 0:
                    cpx = np.minimum(cpx, instr.fixed_price)
                ok = p * (1.0 - slip) > cpx
                held = inv + new_sh
                room = np.clip(np.minimum(np.minimum(sizing - held, own_cap - held),
                                          exch_cap - shares_issued - new_sh), 0.0, None)
                amt = np.where(ok, np.minimum(np.minimum(instr.tranche, left), room * cpx), 0.0)
                new_sh += amt / cpx
                left -= amt
                converted += amt

        if not is_note and t % instr.cadence == 0:
            n_price = 1 if instr.pricing == "option1" else (L if instr.pricing == "option2" else 0)
            if t + max(n_price - 1, 0) <= maturity:
                can = (left > 0) & (p >= instr.min_draw_price) & (pend == 0)
                room = np.clip(np.minimum(np.minimum(sizing - inv, own_cap - inv), exch_cap - shares_issued), 0.0, None)
                if instr.max_advance_adv > 0:
                    adv5 = volumes[:, max(1, t - 5):t].mean(axis=1) if t > 1 else np.full(n, mkt.adv_shares)
                    room = np.minimum(room, instr.max_advance_adv * adv5)
                if instr.pricing == "trailing":
                    cpx = (1.0 - instr.discount) * px[:, max(0, t - L + 1):t + 1].mean(axis=1)
                    amt = np.where(can, np.minimum(np.minimum(instr.advance, left), room * cpx), 0.0)
                    new_sh += amt / cpx
                    left -= amt
                    converted += amt
                    cash[:, t] -= amt        # investor pays the issuer for the shares
                else:
                    # sized in shares at the prior close; the desk may sell before delivery
                    req = np.minimum(np.minimum(instr.advance, left) / px[:, t - 1], room)
                    req = np.where(can, req, 0.0)
                    new_sh += req
                    fresh = req > 0
                    pend = np.where(fresh, req, pend)
                    pend_age = np.where(fresh, 0, pend_age)
                    pend_min = np.where(fresh, np.inf, pend_min)
                    pend_excl = np.where(fresh, 0, pend_excl)
                    pend_sold = np.where(fresh, 0.0, pend_sold)
                    pend_sold_excl = np.where(fresh, 0.0, pend_sold_excl)

        inv += new_sh
        shares_issued += new_sh

        # sell into today's volume, last day dumps everything
        q = inv if t == horizon else np.minimum(inv, ex.participation * v)
        temp = np.clip(ex.eta * sd * np.sqrt(q / v), 0.0, 0.95)
        fill = p * (1.0 - slip - temp)
        cash[:, t] += q * fill
        impact_cost += q * p * temp
        sold[:, t] = q
        inv -= q
        exit_day = np.where(q > 0, t, exit_day)
        vw[:, t] = p * (1.0 - 0.5 * temp)
        carried = -np.log1p(-ex.permanent * temp)   # hits prices from tomorrow on
        d_keep += ex.residual * carried
        d_fade += (1.0 - ex.residual) * carried

        # price the pending advance once its pricing period is over
        active = pend > 0
        if active.any():
            pend_age += active
            pend_sold += np.where(active, q, 0.0)
            pay_day = min(t + 1, horizon)
            if instr.pricing == "option1":
                final = pend
                if instr.volume_threshold > 0:
                    tot = v + q
                    final = np.where(tot >= pend / instr.volume_threshold, pend,
                                     np.minimum(pend, np.maximum(instr.volume_threshold * tot, q)))
                cost = final * (1.0 - instr.discount) * vw[:, t]
                done = active
            else:
                excl = active & (vw[:, t] < instr.min_draw_price)
                pend_min = np.where(active & ~excl, np.minimum(pend_min, vw[:, t]), pend_min)
                pend_excl += excl
                pend_sold_excl += np.where(excl, q, 0.0)
                done = active & (pend_age >= L)
                regular = pend * (1.0 - pend_excl / L)
                extra = np.clip(np.maximum(pend_sold_excl, pend_sold - regular), 0.0, pend - regular)
                final = regular + extra
                low = np.where(np.isfinite(pend_min), pend_min, 0.0)
                cost = (1.0 - instr.discount) * (regular * low + extra * instr.min_draw_price)
            cut = np.where(done, pend - final, 0.0)   # shares requested but never issued
            inv -= cut
            shares_issued -= cut
            cost = np.where(done, cost, 0.0)
            cash[:, pay_day] -= cost
            left -= cost
            converted += cost
            pend = np.where(done, 0.0, pend)

    repaid_maturity = np.zeros(n)
    if is_note:
        repaid_maturity = left.copy()
        cash[:, maturity] += left + acc

    res = ExecResult(cashflows=cash, shares_issued=shares_issued, converted=converted,
                     paths=px, base_paths=base_paths, sold=sold, volumes=volumes,
                     impact_cost=impact_cost, exit_day=exit_day, shares_out=mkt.shares_out,
                     repaid_cash=repaid_cash, repaid_maturity=repaid_maturity, triggered=triggered)
    if warrant is not None and warrant.shares > 0:
        _value_warrant(res, warrant, mkt.sigma, slip)
    return res


def _value_warrant(res: ExecResult, w: Warrant, sigma: float, slip: float) -> None:
    """Exercise at expiry if it falls inside the run; otherwise mark to Black-Scholes on the last day."""
    n, T1 = res.paths.shape
    wc = np.zeros((n, T1))
    if w.term_days <= T1 - 1:
        day = w.term_days
        val = w.shares * np.maximum(res.paths[:, day] * (1.0 - slip) - w.strike, 0.0)
        base = w.shares * np.maximum(res.base_paths[:, day] * (1.0 - slip) - w.strike, 0.0)
    else:
        day = T1 - 1
        val = warrants.value(w, res.paths[:, day], sigma, day)
        base = warrants.value(w, res.base_paths[:, day], sigma, day)
    wc[:, day] = val
    res.warrant_cash = wc
    res.warrant_value = val
    res.warrant_value_base = base
    res.warrant_value0 = float(warrants.value(w, res.base_paths[0, 0], sigma, 0))
    res.warrant_shares = w.shares


def execution_stats(res: ExecResult) -> dict:
    """Execution-specific numbers to report next to `summarize`."""
    share_of_vol = np.where(res.volumes > 0, res.sold / res.volumes, 0.0)
    gross = (res.sold * res.paths).sum(axis=1)
    cost_bps = np.where(gross > 0, 1e4 * res.impact_cost / np.where(gross > 0, gross, 1.0), np.nan)
    gap = 1.0 - res.paths / res.base_paths
    drop = gap[:, -1]
    peak = gap.max(axis=1)

    def q(a, p):
        a = np.asarray(a, dtype=float)
        return float(np.nanpercentile(a, p)) if np.isfinite(a).any() else float("nan")

    return {
        "impact_bps_p50": q(cost_bps, 50),
        "price_drag_p50": q(drop, 50),
        "price_drag_peak_p50": q(peak, 50),
        "exit_day_p50": q(res.exit_day, 50),
        "max_share_of_volume": float(np.nanmax(share_of_vol)),
    }
