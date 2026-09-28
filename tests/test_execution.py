import numpy as np

from pipesim import ConvertibleNote, simulate, summarize
from pipesim.execution import ExecutionModel, run_execution
from pipesim.market import MarketInputs

MKT = MarketInputs(s0=2.0, sigma=0.8, shares_out=2e8, adv_shares=3e6, volume_logstd=0.5)
NOTE = ConvertibleNote(principal=10e6, tranche=1e6, discount=0.10, cadence=10)


def test_no_friction_matches_original_engine():
    # Unlimited participation, zero impact, no wind-down: same cash as the frictionless engine.
    ex = ExecutionModel(participation=1e9, eta=0.0, permanent=0.0, tail_days=0)
    a = run_execution(NOTE, MKT, ex, n_paths=300, seed=3)
    b = simulate(NOTE, MKT.s0, MKT.sigma, MKT.shares_out, n_paths=300, seed=3)
    assert np.allclose(a.cashflows, b.cashflows)
    assert np.allclose(a.shares_issued, b.shares_issued)


def test_impact_lowers_irr_and_raises_dilution():
    free = summarize(run_execution(NOTE, MKT, ExecutionModel(eta=0.0, permanent=0.0), n_paths=1_000))
    real = summarize(run_execution(NOTE, MKT, ExecutionModel(eta=0.8, permanent=0.5), n_paths=1_000))
    assert real["irr_p50"] < free["irr_p50"]
    assert real["dilution_p50"] > free["dilution_p50"]


def test_daily_sales_respect_participation_cap():
    ex = ExecutionModel(participation=0.05)
    r = run_execution(NOTE, MKT, ex, n_paths=200)
    last = r.sold.shape[1] - 1
    share = r.sold[:, 1:last] / r.volumes[:, 1:last]
    assert share.max() <= 0.05 + 1e-12


def test_no_lookahead():
    # Changing prices after day k must not change anything the desk did up to day k.
    ex = ExecutionModel()
    rng = np.random.default_rng(0)
    T = NOTE.maturity_days + ex.tail_days
    base = MKT.s0 * np.exp(np.cumsum(rng.normal(0, 0.03, (1, T + 1)), axis=1))
    base[:, 0] = MKT.s0
    vol = np.full((1, T + 1), MKT.adv_shares)
    k = 120
    shocked = base.copy()
    shocked[:, k + 1:] *= 0.3
    a = run_execution(NOTE, MKT, ex, base_paths=base, volumes=vol)
    b = run_execution(NOTE, MKT, ex, base_paths=shocked, volumes=vol)
    assert np.allclose(a.cashflows[:, :k + 1], b.cashflows[:, :k + 1])
    assert np.allclose(a.sold[:, :k + 1], b.sold[:, :k + 1])


def test_no_recovery_when_half_life_infinite():
    # With nothing fading, how the carried move splits between residual and fading is irrelevant.
    a = run_execution(NOTE, MKT, ExecutionModel(half_life=np.inf, residual=0.0), n_paths=200)
    b = run_execution(NOTE, MKT, ExecutionModel(half_life=np.inf, residual=1.0), n_paths=200)
    assert np.allclose(a.paths, b.paths)
    assert np.allclose(a.cashflows, b.cashflows)


def test_recovery_lifts_price_and_irr():
    stuck = run_execution(NOTE, MKT, ExecutionModel(half_life=np.inf), n_paths=1_000)
    heals = run_execution(NOTE, MKT, ExecutionModel(half_life=5.0, residual=0.3), n_paths=1_000)
    gap_stuck = 1 - stuck.paths[:, -1] / stuck.base_paths[:, -1]
    gap_heals = 1 - heals.paths[:, -1] / heals.base_paths[:, -1]
    assert np.median(gap_heals) < np.median(gap_stuck)
    assert summarize(heals)["irr_p50"] > summarize(stuck)["irr_p50"]


def test_fading_part_decays_to_residual():
    # One big sale on day 1, then nothing: the gap to the base price shrinks toward the residual.
    ex = ExecutionModel(participation=1.0, eta=1.0, permanent=1.0, half_life=3.0, residual=0.25, tail_days=0)
    note = ConvertibleNote(principal=1e6, tranche=1e6, discount=0.10, cadence=1, maturity_days=40)
    base = np.full((1, 41), MKT.s0)
    vol = np.full((1, 41), 1e6)
    r = run_execution(note, MKT, ex, base_paths=base, volumes=vol)
    assert r.sold[0, 1] > 0 and r.sold[0, 2:].sum() == 0    # all selling happens on day 1
    total = -np.log(r.paths[0, 2] / base[0, 2])           # log gap on day 2, after one day of decay
    decay = 0.5 ** (1 / ex.half_life)
    keep = ex.residual * total / (ex.residual + (1 - ex.residual) * decay)
    gap39 = -np.log(r.paths[0, 39] / base[0, 39])
    assert np.isclose(gap39, keep + (total - keep) * decay ** 37, rtol=1e-9)
    assert keep < gap39 < total
