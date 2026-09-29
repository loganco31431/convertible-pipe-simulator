import numpy as np
import pandas as pd

from pipesim.intraday.data import Bars, _normalize, to_grid
from pipesim.intraday.tradeout import STRATEGIES, IntradayImpact, TermSheet, run_tradeout

S = 78


def synthetic(days=30, seed=0, flat=False):
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(rng.normal(0, 0.01, (days, S)), axis=1))
    price = np.ones((days, S)) if flat else close
    u = 1.0 + 1.5 * np.cos(np.linspace(0, 2 * np.pi, S)) ** 2               # U-shaped volume
    vol = 1e5 * u * np.exp(rng.normal(0, 0.3, (days, S)))
    return Bars(days=pd.bdate_range("2026-01-05", periods=days), slot_times=list(range(S)), price=price,
                close=close, volume=vol, bar_minutes=5, source="test")


def test_calibration_matches_daily_square_root_law():
    b = synthetic(flat=True)
    b.volume[:] = 1e5
    ts = TermSheet(kind="note", size_usd=1e12, sell_days=1, pricing_days=1, max_participation=0.04, discount=0.0)
    r = run_tradeout(b, ts, IntradayImpact(half_spread=0.0), "pov", shares_out=1e15)
    sig_d = np.sqrt(np.nanvar(np.diff(np.log(b.close), axis=1), axis=1).mean()) * np.sqrt(S)
    target = 0.5 * sig_d * np.sqrt(0.04)
    assert abs((1 - r.px[0, -2]) / target - 1) < 0.05


def test_no_impact_no_spread_fills_at_bar_price():
    b = synthetic()
    r = run_tradeout(b, TermSheet(), IntradayImpact(eta=0.0, half_spread=0.0), "vwap", shares_out=1e9)
    sold = r.q > 0
    assert np.allclose(r.fill[sold], r.raw_px[sold])


def test_participation_cap_until_final_bar():
    b = synthetic()
    ts = TermSheet(max_participation=0.05)
    for s in STRATEGIES:
        r = run_tradeout(b, ts, IntradayImpact(), s, shares_out=1e9)
        share = r.q[:, :-1] / r.vol[:, :-1]
        assert share.max() <= 0.05 + 1e-12, s
        assert np.allclose(r.q.sum(axis=1), r.windows["shares"])          # everything gets sold


def test_ownership_cap():
    b = synthetic()
    r = run_tradeout(b, TermSheet(size_usd=1e9), IntradayImpact(), "vwap", shares_out=1e7)
    assert (r.windows["shares"] <= 0.0499 * 1e7 + 1e-6).all()


def test_no_lookahead():
    # Changing prices and volumes after bar k of the first window must not change any sale up to bar k.
    b = synthetic()
    ts = TermSheet(sell_days=3, pricing_days=3)
    base = {s: run_tradeout(b, ts, IntradayImpact(), s, shares_out=1e9) for s in STRATEGIES}
    d0 = base["vwap"].windows.index[0]
    start_day = np.where(b.days == base["vwap"].windows.loc[d0, "notice_day"])[0][0]
    k = S + 40                                                          # day 2, bar 40 of the window
    shocked = synthetic()
    shocked.price[start_day + 1, 41:] *= 0.5
    shocked.price[start_day + 2:] *= 1.7
    shocked.volume[start_day + 1, 41:] *= 3.0
    shocked.volume[start_day + 2:] *= 0.2
    for s in STRATEGIES:
        a = base[s]
        c = run_tradeout(shocked, ts, IntradayImpact(), s, shares_out=1e9)
        assert np.allclose(a.q[0, :k + 1], c.q[0, :k + 1]), s
        assert np.allclose(a.fill[0, :k + 1], c.fill[0, :k + 1]), s


def test_own_selling_lowers_sepa_purchase_price():
    b = synthetic()
    r = run_tradeout(b, TermSheet(kind="sepa", size_usd=5e4), IntradayImpact(eta=1.0), "front", shares_out=1e9)
    assert (r.windows["pricing_drag"] <= 1e-12).all()
    assert r.windows["pricing_drag"].median() < 0


def test_lowest_vwap_rule_never_above_average():
    b = synthetic()
    lo = run_tradeout(b, TermSheet(pricing="lowest"), IntradayImpact(), "vwap", shares_out=1e9)
    av = run_tradeout(b, TermSheet(pricing="average"), IntradayImpact(), "vwap", shares_out=1e9)
    assert (lo.windows["purchase_price"] <= av.windows["purchase_price"] + 1e-12).all()


def test_csv_style_export_is_parsed():
    t = pd.date_range("2026-03-02 09:30", periods=S, freq="5min")
    t = t.append(pd.date_range("2026-03-03 09:30", periods=S, freq="5min"))
    df = pd.DataFrame({"Dates": t, "Open": 1.0, "High": 1.1, "Low": 0.9, "LAST_PRICE": 1.0, "VOLUME": 1000})
    b = to_grid(_normalize(df, "America/New_York"), 5, "test")
    assert b.price.shape == (2, S)
    assert np.allclose(b.price, 1.0) and np.allclose(b.volume, 1000)
