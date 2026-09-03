import dataclasses

import pandas as pd

from .engine import simulate
from .metrics import summarize


def grid(instr, s0, sigma_grid, discount_grid, shares_out, n_paths=5_000, seed=0) -> pd.DataFrame:
    """Median IRR, median dilution and loss probability across a volatility x discount grid."""
    rows = []
    for sig in sigma_grid:
        for d in discount_grid:
            inst = dataclasses.replace(instr, discount=d)
            s = summarize(simulate(inst, s0, sig, shares_out, n_paths=n_paths, seed=seed))
            rows.append({"sigma": sig, "discount": d, "irr_p50": s["irr_p50"],
                         "dilution_p50": s["dilution_p50"], "prob_loss": s["prob_loss"]})
    return pd.DataFrame(rows)


def stress(instr, s0, sigma, shares_out, cadences, floor_pcts, n_paths=5_000, seed=0) -> pd.DataFrame:
    """IRR percentiles and dilution across conversion cadence and floor price (as a share of spot)."""
    rows = []
    for c in cadences:
        for f in floor_pcts:
            inst = dataclasses.replace(instr, cadence=c, floor_price=f * s0)
            s = summarize(simulate(inst, s0, sigma, shares_out, n_paths=n_paths, seed=seed))
            rows.append({"cadence_days": c, "floor_pct_spot": f, "irr_p10": s["irr_p10"],
                         "irr_p50": s["irr_p50"], "irr_p90": s["irr_p90"],
                         "prob_loss": s["prob_loss"], "dilution_p50": s["dilution_p50"],
                         "converted_p50": s["converted_p50"]})
    return pd.DataFrame(rows)
