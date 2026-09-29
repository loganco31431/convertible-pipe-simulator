"""Contract terms in the daily engine, each checked on a case with a known answer."""
import dataclasses

import numpy as np

from pipesim import ConvertibleNote, StandbyEquityFacility, Warrant, simulate, summarize
from pipesim.execution import ExecutionModel, run_execution
from pipesim.instruments import MONTH, TRADING_DAYS
from pipesim.market import MarketInputs
from pipesim.paths import simulate_prices, trailing_vwap
from pipesim.presets import PRESETS
from pipesim.warrants import bs_call

NO_CAPS = dict(ownership_cap=0.0, exchange_cap=0.0)
FREE = ExecutionModel(participation=1.0, eta=0.0, permanent=0.0, tail_days=0)


def flat(price, days, n=1):
    return np.full((n, days + 1), float(price))


def run_on(instr, paths, volumes=None, ex=FREE, shares_out=1e9, adv=1e12, warrant=None, sigma=0.5):
    mkt = MarketInputs(s0=float(paths[0, 0]), sigma=sigma, shares_out=shares_out, adv_shares=adv, volume_logstd=0.0)
    vol = np.full(paths.shape, adv) if volumes is None else volumes
    return run_execution(instr, mkt, ex, base_paths=paths, volumes=vol, warrant=warrant)


def legacy_simulate(instr, s0, sigma, n_paths, seed):
    """The original vectorized frictionless engine, kept here as the reference."""
    days = instr.maturity_days
    paths = simulate_prices(s0, sigma, days, n_paths, seed=seed)
    vwap = trailing_vwap(paths, instr.vwap_lookback)
    conv_days = np.arange(instr.cadence, days + 1, instr.cadence)
    px = paths[:, conv_days]
    conv_px = (1.0 - instr.discount) * vwap[:, conv_days]
    if isinstance(instr, ConvertibleNote):
        conv_px = np.maximum(conv_px, instr.floor_price)
        ok = px * (1.0 - instr.sell_slippage) > conv_px
        per_period, budget, upfront = instr.tranche, instr.principal, -instr.principal * (1.0 - instr.oid)
    else:
        ok = px >= instr.min_draw_price
        per_period, budget, upfront = instr.advance, instr.commitment, 0.0
    want = np.where(ok, per_period, 0.0)
    amount = np.clip(want - np.clip(np.cumsum(want, axis=1) - budget, 0.0, None), 0.0, None)
    shares = amount / conv_px
    proceeds = shares * px * (1.0 - instr.sell_slippage)
    cash = np.zeros_like(paths)
    cash[:, 0] = upfront
    if isinstance(instr, StandbyEquityFacility):
        cash[:, conv_days] += proceeds - amount
    else:
        cash[:, conv_days] += proceeds
        cash[:, -1] += instr.principal - amount.sum(axis=1)
    return cash, shares.sum(axis=1)


def test_day_by_day_engine_matches_original_vectorized_engine():
    note = ConvertibleNote(principal=10e6, tranche=1e6, discount=0.10, coupon=0.0, floor_price=1.5, **NO_CAPS)
    sepa = StandbyEquityFacility(commitment=1e7, advance=1e6, discount=0.05, pricing="trailing", **NO_CAPS)
    for instr in (note, sepa):
        cash, shares = legacy_simulate(instr, 2.0, 0.8, 300, seed=3)
        res = simulate(instr, 2.0, 0.8, 1e8, n_paths=300, seed=3)
        assert np.allclose(res.cashflows, cash)
        assert np.allclose(res.shares_issued, shares)


def test_fixed_price_caps_and_floor_lifts_the_conversion_price():
    base = dict(principal=1e6, tranche=1e5, discount=0.10, coupon=0.0, sell_slippage=0.0, cadence=10, **NO_CAPS)
    paths = flat(5.0, TRADING_DAYS)
    plain = run_on(ConvertibleNote(**base), paths)
    capped = run_on(ConvertibleNote(fixed_price=4.0, **base), paths)
    floored = run_on(ConvertibleNote(floor_price=4.8, **base), paths)
    assert np.isclose(plain.shares_issued[0], 1e6 / 4.5)       # 90% of $5
    assert np.isclose(capped.shares_issued[0], 1e6 / 4.0)      # fixed price is lower, so it wins
    assert np.isclose(floored.shares_issued[0], 1e6 / 4.8)     # variable price lifted to the floor


def test_lowest_vwap_uses_days_before_the_notice():
    paths = flat(5.0, 20)
    paths[0, 7] = 4.0       # inside the 5 days before the day-10 notice
    paths[0, 10] = 3.0      # the notice day itself must not be used
    note = ConvertibleNote(principal=1e5, tranche=1e5, discount=0.05, vwap_lookback=5, pricing="lowest", cadence=10,
                           coupon=0.0, sell_slippage=0.0, maturity_days=20, **NO_CAPS)
    res = run_on(note, paths)
    assert res.sold[0, 10] == 0                       # day 10: $3 is below 95% of $4, so no conversion
    paths[0, 10] = 5.0
    res = run_on(note, paths)
    assert np.isclose(res.sold[0, 10], 1e5 / (0.95 * 4.0))


def test_interest_accrues_on_outstanding_principal():
    loan = ConvertibleNote(principal=1e6, tranche=0.0, coupon=0.10, oid=0.0, **NO_CAPS)
    res = run_on(loan, flat(5.0, TRADING_DAYS))
    assert np.isclose(res.cashflows[0, -1], 1e6 * 1.10)
    half = ConvertibleNote(principal=1e6, tranche=0.0, coupon=0.10, oid=0.0, payment_mode="scheduled",
                           monthly_payment=5e5, payment_start_day=126, **NO_CAPS)
    res = run_on(half, flat(5.0, TRADING_DAYS))
    interest = res.cashflows[0, 1:].sum() - 1e6
    assert interest < 1e6 * 0.10                       # less than interest on the full balance all year
    assert np.isclose(interest, 1e6 * 0.10 * 126 / 252 + 5e5 * 0.10 * 21 / 252)


def test_scheduled_payments_repay_the_loan_with_premium():
    loan = ConvertibleNote(principal=12e6, oid=0.05, coupon=0.0, tranche=0.0, maturity_days=273,
                           payment_mode="scheduled", monthly_payment=1.2e6, payment_premium=0.05,
                           payment_start_day=42, **NO_CAPS)
    res = run_on(loan, flat(1.0, 273))
    assert np.isclose(res.repaid_cash[0], 12e6)
    assert np.isclose(res.cashflows[0, 42], 1.2e6 * 1.05)
    assert np.isclose(res.cashflows[0, 1:].sum(), 12e6 * 1.05)
    assert res.repaid_maturity[0] == 0
    assert res.shares_issued[0] == 0


TRIGGER = dict(principal=20e6, oid=0.10, coupon=0.0, discount=0.07, vwap_lookback=5, pricing="lowest", cadence=5,
               tranche=2e6, floor_price=2.0, payment_mode="on_trigger", monthly_payment=2.5e6, payment_premium=0.07,
               sell_slippage=0.0, **NO_CAPS)


def test_amortization_event_starts_payments_and_a_recovery_stops_them():
    paths = flat(1.0, TRADING_DAYS)                    # below the $2 floor from day 1
    res = run_on(ConvertibleNote(**TRIGGER), paths)
    first = 8 + 7                                      # event on day 8 (7 VWAPs seen), first payment 7 days later
    assert res.triggered[0]
    assert res.converted[0] == 0                       # below the floor nothing converts
    assert np.isclose(res.cashflows[0, first], 2.5e6 * 1.07)
    assert np.isclose(res.cashflows[0, first + MONTH], 2.5e6 * 1.07)
    assert res.cashflows[0, 1:first].sum() == 0

    paths[0, 20:] = 3.0                                # back above the floor after one payment
    res = run_on(ConvertibleNote(**TRIGGER), paths)
    assert np.isclose(res.cashflows[0, first], 2.5e6 * 1.07)
    assert np.isclose(res.repaid_cash[0], 2.5e6)       # the second payment never comes due
    assert res.converted[0] > 0                        # and conversions resume


def test_pre_paid_advance_loop_settles_payments_in_shares():
    paths = flat(1.0, TRADING_DAYS)
    res = run_on(ConvertibleNote(pay_in_shares=True, sepa_discount=0.03, **TRIGGER), paths)
    first = 8 + 7
    assert res.repaid_cash[0] == 0
    assert np.isclose(res.sold[0, first], 2.5e6 / 0.97)           # shares at 97% of a $1 VWAP
    assert np.isclose(res.cashflows[0, first], 2.5e6 / 0.97)      # sold at $1, no premium


def test_ownership_and_exchange_caps_bind():
    base = dict(principal=10e6, tranche=5e6, discount=0.10, coupon=0.0, sell_slippage=0.0, cadence=10)
    paths = flat(1.0, TRADING_DAYS)
    own = run_on(ConvertibleNote(ownership_cap=0.0499, exchange_cap=0.0, **base), paths, shares_out=1e7)
    assert np.isclose(own.sold[0, 10], 0.0499 * 1e7)              # each conversion stops at 4.99%
    exch = run_on(ConvertibleNote(ownership_cap=0.0, exchange_cap=0.1999, **base), paths, shares_out=1e7)
    assert np.isclose(exch.shares_issued[0], 0.1999 * 1e7)
    assert exch.repaid_maturity[0] > 0                            # the rest comes back in cash


def test_sepa_option1_earns_the_discount_on_a_flat_stock():
    sepa = StandbyEquityFacility(commitment=1e7, advance=1e6, discount=0.04, pricing="option1", cadence=5,
                                 sell_slippage=0.0, maturity_days=100, **NO_CAPS)
    res = run_on(sepa, flat(2.0, 100))
    assert np.isclose(res.sold[0, 5], 1e6 / 2.0)                  # sized in shares at the prior close
    assert np.isclose(res.cashflows[0, 5], 1e6)                   # sold the day of the notice
    assert np.isclose(res.cashflows[0, 6], -0.96e6)               # paid for the next day
    assert np.isclose(res.cashflows.sum(), (1e7 / 0.96) * 0.04, rtol=1e-6)


def test_sepa_option1_volume_threshold_cuts_the_advance():
    sepa = StandbyEquityFacility(commitment=1e7, advance=1e6, discount=0.04, pricing="option1", cadence=5,
                                 volume_threshold=0.30, sell_slippage=0.0, maturity_days=20, **NO_CAPS)
    paths = flat(2.0, 20)
    vol = np.full(paths.shape, 4e5)                   # advance is 500,000 shares, threshold needs 1.67M traded
    ex = ExecutionModel(participation=0.10, eta=0.0, permanent=0.0, tail_days=0)
    res = run_on(sepa, paths, volumes=vol, ex=ex)
    sold = 0.10 * 4e5
    final = 0.30 * (4e5 + sold)                       # 30% of volume beats the 40,000 shares sold
    assert np.isclose(res.cashflows[0, 6] - res.sold[0, 6] * 2.0, -final * 0.96 * 2.0)


def test_sepa_option2_pays_the_lowest_of_the_pricing_days():
    paths = flat(2.0, 20)
    paths[0, 6] = 1.5                                 # second of the three pricing days (5, 6, 7)
    sepa = StandbyEquityFacility(commitment=1e6, advance=1e6, discount=0.03, vwap_lookback=3, pricing="option2",
                                 cadence=5, sell_slippage=0.0, maturity_days=20, **NO_CAPS)
    res = run_on(sepa, paths)
    shares = 1e6 / 2.0
    assert np.isclose(res.cashflows[0, 5], shares * 2.0)          # sold on the notice day, before delivery
    assert np.isclose(res.cashflows[0, 8], -shares * 0.97 * 1.5)  # priced at the low, paid after the period


def test_sepa_option2_excluded_day_cuts_the_advance_by_a_third():
    paths = flat(2.0, 20)
    paths[0, 7] = 1.0                                 # third pricing day is below the $1.80 minimum
    sepa = StandbyEquityFacility(commitment=1e6, advance=1e6, discount=0.03, vwap_lookback=3, pricing="option2",
                                 cadence=5, min_draw_price=1.8, sell_slippage=0.0, maturity_days=9, **NO_CAPS)
    ex = ExecutionModel(participation=0.10, eta=0.0, permanent=0.0, tail_days=0)
    vol = np.full(paths.shape, 1e6)                   # desk sells 100,000 shares a day
    res = run_on(sepa, paths, volumes=vol, ex=ex)
    regular = (1e6 / 2.0) * (2 / 3)
    extra = 1e5                                       # what it sold on the excluded day, at 97% of the minimum
    cost = 0.97 * (regular * 2.0 + extra * 1.8)
    assert np.isclose(res.cashflows[0, 8] - res.sold[0, 8] * 2.0, -cost)
    assert np.isclose(res.shares_issued[0], regular + extra)


def test_commitment_shares_and_structuring_fee_are_free_money():
    base = dict(commitment=1e6, advance=1e6, discount=0.0, pricing="option1", cadence=5, volume_threshold=0.0,
                sell_slippage=0.0, maturity_days=20, **NO_CAPS)
    plain = run_on(StandbyEquityFacility(**base), flat(2.0, 20))
    fees = run_on(StandbyEquityFacility(commitment_shares=1e4, structuring_fee=5e4, **base), flat(2.0, 20))
    assert np.isclose(fees.cashflows.sum() - plain.cashflows.sum(), 1e4 * 2.0 + 5e4)
    assert np.isclose(fees.shares_issued[0] - plain.shares_issued[0], 1e4)


def test_black_scholes_matches_a_known_value():
    assert abs(float(bs_call(100.0, 100.0, 1.0, 0.20, 0.05)) - 10.4506) < 1e-3
    assert np.isclose(float(bs_call(5.0, 3.0, 0.0, 0.5)), 2.0)


def test_warrants_add_to_the_package_and_lose_value_to_the_desks_own_selling():
    note = ConvertibleNote(principal=10e6, tranche=1e6, discount=0.10)
    w = Warrant.from_coverage(10e6, 0.21, 2.0, term_days=5 * TRADING_DAYS)
    mkt = MarketInputs(s0=2.0, sigma=0.8, shares_out=2e8, adv_shares=3e6)
    res = run_execution(note, mkt, ExecutionModel(eta=0.8), n_paths=500, warrant=w)
    s = summarize(res)
    assert np.isclose(w.shares, 0.21 * 10e6 / 2.0)
    assert s["pkg_pnl_p50"] > s["pnl_p50"]
    assert s["warrant_lost_to_selling_mean"] > 0
    assert (res.warrant_value <= res.warrant_value_base + 1e-9).all()
    assert s["dilution_with_warrants_p50"] > s["dilution_p50"]


def test_warrant_expiring_inside_the_run_is_exercised_for_its_payoff():
    loan = ConvertibleNote(principal=1e6, tranche=0.0, coupon=0.0, sell_slippage=0.0, maturity_days=300, **NO_CAPS)
    w = Warrant(shares=1e5, strike=1.0, term_days=TRADING_DAYS)
    paths = flat(1.5, 300)
    res = run_on(loan, paths, warrant=w)
    assert np.isclose(res.warrant_cash[0, TRADING_DAYS], 1e5 * 0.5)
    assert np.isclose(res.warrant_cash.sum(), 1e5 * 0.5)


def test_every_preset_runs_and_stays_inside_its_caps():
    mkt = MarketInputs(s0=1.0, sigma=1.2, shares_out=2e8, adv_shares=5e6)
    for name, build in PRESETS.items():
        p = build(mkt.s0)
        instr = p.note or p.sepa
        instr = dataclasses.replace(instr, maturity_days=min(instr.maturity_days, TRADING_DAYS))
        res = run_execution(instr, mkt, ExecutionModel(), n_paths=200, warrant=p.warrant)
        s = summarize(res)
        assert np.isfinite(s["pnl_p50"]), name
        assert (res.shares_issued <= instr.exchange_cap * mkt.shares_out * (1 + 1e-9)).all(), name
        assert (res.sold.sum(axis=1) <= res.shares_issued * (1 + 1e-9) + 1e-6).all(), name
