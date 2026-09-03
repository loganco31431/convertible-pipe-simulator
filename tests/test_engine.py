import numpy as np

from pipesim import ConvertibleNote, StandbyEquityFacility, irr, simulate, summarize
from pipesim.paths import trailing_vwap


def test_trailing_vwap_matches_rolling_mean():
    p = np.random.default_rng(1).uniform(1, 10, size=(3, 30))
    v = trailing_vwap(p, 5)
    for t in range(30):
        lo = max(0, t - 4)
        assert np.allclose(v[:, t], p[:, lo:t + 1].mean(axis=1))


def test_irr_recovers_known_rate():
    # -100 today, +110 in one year -> IRR 10%
    cf = np.zeros((1, 253))
    cf[0, 0] = -100
    cf[0, 252] = 110
    assert abs(irr(cf)[0] - 0.10) < 1e-6


def test_zero_vol_note_irr_is_discount_driven():
    # With sigma=0 the stock is flat, so every conversion captures exactly the discount
    # (net of slippage) and the OID, and the whole note converts.
    note = ConvertibleNote(principal=1e6, tranche=1e5, discount=0.10, oid=0.05, coupon=0.0,
                           sell_slippage=0.0, cadence=10)
    res = simulate(note, s0=5.0, sigma=0.0, shares_out=1e8, n_paths=10)
    s = summarize(res)
    assert np.isclose(s["converted_p50"], 1e6)
    # proceeds per tranche = tranche / 0.9 ; total in = 1e6/0.9 ; total out = 0.95e6
    assert np.isclose(s["moic_p50"], (1e6 / 0.9) / 0.95e6)


def test_floor_above_market_blocks_conversion():
    note = ConvertibleNote(principal=1e6, tranche=1e5, floor_price=50.0, coupon=0.0)
    res = simulate(note, s0=5.0, sigma=0.5, shares_out=1e8, n_paths=50)
    assert np.all(res.converted == 0)
    assert np.all(res.shares_issued == 0)


def test_sepa_dilution_positive_and_pnl_near_discount():
    sepa = StandbyEquityFacility(commitment=1e7, advance=1e6, discount=0.05, sell_slippage=0.0)
    res = simulate(sepa, s0=2.0, sigma=0.0, shares_out=1e8, n_paths=5)
    s = summarize(res)
    assert s["dilution_p50"] > 0
    assert np.isclose(s["pnl_p50"], 1e7 / 0.95 - 1e7)
