"""Run the simulator for one ticker and write charts and tables to output/.

Examples
    python run.py --ticker GPRO
    python run.py --ticker LCID --principal 25e6 --discount 0.12 --floor-pct 0.5
    python run.py --s0 3.20 --sigma 0.85 --shares-out 150e6   # offline, manual inputs
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipesim import ConvertibleNote, StandbyEquityFacility, historical_vol, simulate, summarize
from pipesim.scenarios import grid, stress


def pct(x):
    return f"{100 * x:,.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--s0", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--shares-out", type=float, default=None)
    ap.add_argument("--principal", type=float, default=10e6)
    ap.add_argument("--discount", type=float, default=0.10)
    ap.add_argument("--floor-pct", type=float, default=0.0, help="floor price as a share of spot")
    ap.add_argument("--paths", type=int, default=10_000)
    ap.add_argument("--out", default="output")
    a = ap.parse_args()

    if a.ticker:
        sigma, s0, shares_out = historical_vol(a.ticker)
        label = a.ticker.upper()
    else:
        if None in (a.s0, a.sigma, a.shares_out):
            ap.error("give --ticker, or all of --s0 --sigma --shares-out")
        sigma, s0, shares_out, label = a.sigma, a.s0, a.shares_out, "manual"
    os.makedirs(a.out, exist_ok=True)

    note = ConvertibleNote(principal=a.principal, tranche=a.principal / 10, discount=a.discount,
                           floor_price=a.floor_pct * s0)
    sepa = StandbyEquityFacility(commitment=a.principal * 5, advance=a.principal / 5, discount=0.05)

    print(f"\n{label}: spot ${s0:,.2f}  realized vol {pct(sigma)}  shares out {shares_out / 1e6:,.1f}M")
    print(f"{'-' * 72}")
    print(f"Convertible note  ${a.principal / 1e6:,.0f}M  discount {pct(a.discount)}  floor {pct(a.floor_pct)} of spot")
    res = simulate(note, s0, sigma, shares_out, n_paths=a.paths)
    s = summarize(res)
    print(f"  investor IRR  p10 {pct(s['irr_p10'])}   p50 {pct(s['irr_p50'])}   p90 {pct(s['irr_p90'])}")
    print(f"  MOIC p50 {s['moic_p50']:.2f}x   P(loss) {pct(s['prob_loss'])}   converted p50 ${s['converted_p50'] / 1e6:,.1f}M")
    print(f"  issuer dilution  p50 {pct(s['dilution_p50'])}   p90 {pct(s['dilution_p90'])}")

    print(f"\nSEPA  ${sepa.commitment / 1e6:,.0f}M commitment  discount {pct(sepa.discount)}")
    rs = simulate(sepa, s0, sigma, shares_out, n_paths=a.paths)
    ss = summarize(rs)
    print(f"  investor P&L p50 ${ss['pnl_p50'] / 1e6:,.2f}M on ${ss['converted_p50'] / 1e6:,.1f}M drawn")
    print(f"  issuer dilution  p50 {pct(ss['dilution_p50'])}   p90 {pct(ss['dilution_p90'])}")

    # 1. IRR distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    r = s["irr"][np.isfinite(s["irr"])]
    ax.hist(np.clip(r, -1, 3), bins=80, color="#2f5d8a")
    for p, c in ((s["irr_p10"], "#c0392b"), (s["irr_p50"], "black"), (s["irr_p90"], "#27ae60")):
        ax.axvline(min(p, 3), color=c, ls="--")
    ax.set_title(f"{label}: investor IRR, ${a.principal / 1e6:,.0f}M note at {pct(a.discount)} to VWAP")
    ax.set_xlabel("annualized IRR (clipped at 300%)")
    ax.set_ylabel("paths")
    fig.tight_layout()
    fig.savefig(f"{a.out}/{label}_irr_hist.png", dpi=130)

    # 2. vol x discount heatmaps
    sig_grid = [0.3, 0.5, 0.8, 1.2]
    disc_grid = [0.05, 0.10, 0.15, 0.20]
    g = grid(note, s0, sig_grid, disc_grid, shares_out, n_paths=max(2_000, a.paths // 4))
    g.to_csv(f"{a.out}/{label}_grid.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, title, fmt in ((axes[0], "irr_p50", "median investor IRR", "{:.0%}"),
                                (axes[1], "dilution_p50", "median issuer dilution", "{:.1%}")):
        m = g.pivot(index="sigma", columns="discount", values=col)
        ax.imshow(m.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(disc_grid)), [pct(d) for d in disc_grid])
        ax.set_yticks(range(len(sig_grid)), [pct(v) for v in sig_grid])
        ax.set_xlabel("discount to VWAP")
        ax.set_ylabel("annualized vol")
        ax.set_title(title)
        for i in range(len(sig_grid)):
            for j in range(len(disc_grid)):
                ax.text(j, i, fmt.format(m.values[i, j]), ha="center", va="center", color="white", fontsize=9)
    fig.suptitle(f"{label}: ${a.principal / 1e6:,.0f}M convertible note, volatility vs discount")
    fig.tight_layout()
    fig.savefig(f"{a.out}/{label}_grid.png", dpi=130)

    # 3. stress: cadence x floor
    st = stress(note, s0, sigma, shares_out, cadences=[5, 10, 21], floor_pcts=[0.0, 0.5, 0.75],
                n_paths=max(2_000, a.paths // 4))
    st.to_csv(f"{a.out}/{label}_stress.csv", index=False)
    print("\nStress: conversion cadence x floor price")
    show = st.copy()
    for c in ("floor_pct_spot", "irr_p10", "irr_p50", "irr_p90", "prob_loss", "dilution_p50"):
        show[c] = show[c].map(pct)
    show["converted_p50"] = (show["converted_p50"] / 1e6).map(lambda v: f"${v:,.1f}M")
    print(show.to_string(index=False))

    # 4. dilution fan
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pcts = np.percentile(res.paths, [10, 50, 90], axis=0)
    ax.fill_between(range(pcts.shape[1]), pcts[0], pcts[2], color="#2f5d8a", alpha=0.25, label="p10 to p90 price")
    ax.plot(pcts[1], color="#2f5d8a", label="median price")
    ax.axhline(note.floor_price, color="#c0392b", ls="--", label="floor") if note.floor_price else None
    ax.set_xlabel("trading day")
    ax.set_ylabel("share price")
    ax.set_title(f"{label}: simulated price fan, vol {pct(sigma)}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{a.out}/{label}_price_fan.png", dpi=130)
    print(f"\ncharts and tables written to {a.out}/")


if __name__ == "__main__":
    main()
