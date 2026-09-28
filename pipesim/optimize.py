"""Search conversion cadence, tranche size and selling speed for the best way to run a note."""
import dataclasses
import itertools

import pandas as pd

from .execution import ExecutionModel, execution_stats, run_execution
from .metrics import summarize


def search(instr, mkt, ex: ExecutionModel, cadences=(5, 10, 21), tranche_pcts=(0.05, 0.10, 0.20),
           participations=(0.05, 0.10, 0.20), n_paths=2_000, seed=0, max_loss=0.10) -> pd.DataFrame:
    """Every combination, ranked by median dollar profit among setups with P(loss) <= `max_loss`.

    Ranking by IRR alone always picks the fastest schedule, because capital comes back sooner,
    even when the profit is the same or lower. Profit is the fairer yardstick; IRR is shown
    alongside.

    Tranche size is a share of principal (or of the commitment for an equity line). Every
    setup runs on the same simulated prices and volumes (same seed), so differences come from
    the terms, not from noise.
    """
    size = instr.principal if hasattr(instr, "principal") else instr.commitment
    size_field = "tranche" if hasattr(instr, "tranche") else "advance"
    rows = []
    for c, tp, p in itertools.product(cadences, tranche_pcts, participations):
        inst = dataclasses.replace(instr, cadence=c, **{size_field: tp * size})
        e = dataclasses.replace(ex, participation=p)
        res = run_execution(inst, mkt, e, n_paths=n_paths, seed=seed)
        s, x = summarize(res), execution_stats(res)
        rows.append({"cadence_days": c, "tranche_pct": tp, "participation": p,
                     "irr_p10": s["irr_p10"], "irr_p50": s["irr_p50"], "prob_loss": s["prob_loss"],
                     "pnl_p50": s["pnl_p50"], "moic_p50": s["moic_p50"], "dilution_p50": s["dilution_p50"],
                     "converted_p50": s["converted_p50"], "impact_bps_p50": x["impact_bps_p50"],
                     "exit_day_p50": x["exit_day_p50"]})
    df = pd.DataFrame(rows)
    df["eligible"] = df["prob_loss"] <= max_loss
    return df.sort_values(["eligible", "pnl_p50", "irr_p50"], ascending=[False, False, False]).reset_index(drop=True)
