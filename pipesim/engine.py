"""Frictionless model: the investor sells every converted share at that day's close, for free.

This is the deal engine in `execution.py` with price impact switched off and unlimited volume,
so both models apply exactly the same contract terms and differ only in how shares get sold.
"""
import numpy as np

from .execution import ExecResult, ExecutionModel, run_execution
from .instruments import Warrant
from .market import MarketInputs
from .paths import simulate_prices

SimResult = ExecResult
_UNLIMITED = 1e15       # shares a day; large enough that no volume limit ever binds


def simulate(instr, s0: float, sigma: float, shares_out: float, n_paths: int = 10_000,
             mu: float = 0.0, seed: int | None = 0, warrant: Warrant | None = None) -> ExecResult:
    paths = simulate_prices(s0, sigma, instr.maturity_days, n_paths, mu=mu, seed=seed)
    mkt = MarketInputs(s0=s0, sigma=sigma, shares_out=shares_out, adv_shares=_UNLIMITED, volume_logstd=0.0)
    ex = ExecutionModel(participation=1.0, eta=0.0, permanent=0.0, tail_days=0)
    return run_execution(instr, mkt, ex, base_paths=paths, volumes=np.full(paths.shape, _UNLIMITED),
                         warrant=warrant)
