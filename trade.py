"""Put in a ticker, see how a convertible note trades out once volume and price impact are real.

    python trade.py --ticker GPRO
    python trade.py --ticker LCID --principal 25e6 --discount 0.12
    python trade.py --ticker GPRO --no-optimize --paths 3000

Prints four things:
  1. market inputs (price, vol, ADV) and how big the note is relative to daily volume
  2. frictionless model vs execution-aware model on the same terms
  3. the best cadence / tranche / selling-speed combination from a grid search
  4. a replay of the same note on the stock's actual last year of prices and volume
"""
import argparse
import dataclasses

import numpy as np

from pipesim import ConvertibleNote, simulate, summarize
from pipesim.execution import ExecutionModel, execution_stats, run_execution
from pipesim.market import calibrate_market
from pipesim.optimize import search


def pct(x):
    return f"{100 * x:,.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--principal", type=float, default=10e6)
    ap.add_argument("--discount", type=float, default=0.10)
    ap.add_argument("--cadence", type=int, default=10)
    ap.add_argument("--tranche-pct", type=float, default=0.10)
    ap.add_argument("--participation", type=float, default=0.10)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--half-life", type=float, default=10.0,
                    help="trading days for the fading part of the price impact to halve; inf = never recovers")
    ap.add_argument("--residual", type=float, default=0.3,
                    help="share of the carried impact that never fades")
    ap.add_argument("--paths", type=int, default=5_000)
    ap.add_argument("--no-optimize", action="store_true")
    a = ap.parse_args()

    mkt = calibrate_market(a.ticker)
    t = a.ticker.upper()
    note = ConvertibleNote(principal=a.principal, tranche=a.tranche_pct * a.principal,
                           discount=a.discount, cadence=a.cadence)
    ex = ExecutionModel(participation=a.participation, eta=a.eta, half_life=a.half_life, residual=a.residual)

    # 1. market
    print(f"\n{t}: spot ${mkt.s0:,.2f}   vol {pct(mkt.sigma)}   shares out {mkt.shares_out / 1e6:,.1f}M")
    print(f"  ADV {mkt.adv_shares / 1e6:,.2f}M shares = ${mkt.adv_dollars / 1e6:,.2f}M a day")
    print(f"  ${a.principal / 1e6:,.0f}M note = {a.principal / mkt.adv_dollars:,.1f} days of total volume;"
          f" at {pct(a.participation)} of volume it takes ~{a.principal / (a.participation * mkt.adv_dollars):,.0f}"
          f" trading days just to sell the shares")

    # 2. frictionless vs execution-aware
    s0 = summarize(simulate(note, mkt.s0, mkt.sigma, mkt.shares_out, n_paths=a.paths))
    res = run_execution(note, mkt, ex, n_paths=a.paths)
    s1, x1 = summarize(res), execution_stats(res)
    print(f"\n{'':24}{'frictionless':>14}{'with impact':>14}")
    for label, k in (("median IRR", "irr_p50"), ("p10 IRR", "irr_p10"), ("P(loss)", "prob_loss"),
                     ("median dilution", "dilution_p50")):
        print(f"  {label:22}{pct(s0[k]):>14}{pct(s1[k]):>14}")
    print(f"  {'converted (median)':22}{'$' + format(s0['converted_p50'] / 1e6, ',.1f') + 'M':>14}"
          f"{'$' + format(s1['converted_p50'] / 1e6, ',.1f') + 'M':>14}")
    print(f"  impact cost {x1['impact_bps_p50']:,.0f} bps of sales, own selling pushes the price down"
          f" {pct(x1['price_drag_peak_p50'])} at worst and leaves it {pct(x1['price_drag_p50'])} lower at the end,"
          f" shares fully sold by day {x1['exit_day_p50']:,.0f}")

    # 3. best way to run it
    if not a.no_optimize:
        print("\nSearching cadence x tranche size x selling speed (P(loss) <= 10%) ...")
        df = search(note, mkt, ex, n_paths=max(1_000, a.paths // 3))
        best = df.iloc[0]
        show = df.head(5).copy()
        for c in ("tranche_pct", "participation", "irr_p10", "irr_p50", "prob_loss", "dilution_p50"):
            show[c] = show[c].map(pct)
        show["converted_p50"] = (show["converted_p50"] / 1e6).map(lambda v: f"${v:,.1f}M")
        show["pnl_p50"] = (show["pnl_p50"] / 1e6).map(lambda v: f"${v:,.2f}M")
        show["impact_bps_p50"] = show["impact_bps_p50"].map(lambda v: f"{v:,.0f}")
        cols = ["cadence_days", "tranche_pct", "participation", "pnl_p50", "irr_p50", "irr_p10", "prob_loss",
                "dilution_p50", "converted_p50", "impact_bps_p50", "exit_day_p50"]
        print(show[cols].to_string(index=False))
        print(f"  best: convert {pct(best.tranche_pct)} of principal every {best.cadence_days} days,"
              f" sell at {pct(best.participation)} of volume")

    # 4. replay on real history
    horizon = note.maturity_days + ex.tail_days
    h = mkt.history.tail(horizon + 1)
    print("\nReplay on actual prices and volume, last "
          f"{len(h) - 1} trading days ({h.index[0]:%Y-%m-%d} to {h.index[-1]:%Y-%m-%d}).")
    print("  Lookahead check: each conversion and sale uses only closes and volume up to that day.")
    print("  The leak risk is in the inputs, not the decisions: vol and ADV are calibrated on this")
    print("  same window, and the replay prices are what actually traded without this investor")
    print("  selling, so the replay adds the investor's own impact on top of real prices.")
    if len(h) - 1 < horizon:
        print("  not enough history to replay the full term; skipped")
    else:
        rep = dataclasses.replace(ex)
        rr = run_execution(note, mkt, rep, base_paths=h["Close"].to_numpy()[None, :],
                           volumes=h["Volume"].to_numpy(dtype=float)[None, :])
        sr, xr = summarize(rr), execution_stats(rr)
        print(f"  IRR {pct(sr['irr_p50'])}   P&L ${sr['pnl_p50'] / 1e6:,.2f}M   converted "
              f"${sr['converted_p50'] / 1e6:,.1f}M   dilution {pct(sr['dilution_p50'])}   "
              f"impact {xr['impact_bps_p50']:,.0f} bps   price drag {pct(xr['price_drag_peak_p50'])} peak,"
              f" {pct(xr['price_drag_p50'])} at end")
        stock_ret = h["Close"].iloc[note.maturity_days] / h["Close"].iloc[0] - 1
        print(f"  (stock itself over the term: {pct(stock_ret)})")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
