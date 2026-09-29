import numpy as np

from .instruments import TRADING_DAYS


def irr(cashflows: np.ndarray, lo: float = -0.99, hi: float = 20.0, iters: int = 80) -> np.ndarray:
    """Annualized IRR per path by vectorized bisection on daily cashflows, shape (n_paths,).

    Paths with no sign change in NPV over [lo, hi] (for example a SEPA that never drew)
    return nan.
    """
    t = np.arange(cashflows.shape[1]) / TRADING_DAYS

    def npv(r):
        return (cashflows * (1.0 + r[:, None]) ** (-t)).sum(axis=1)

    lo_v = np.full(cashflows.shape[0], lo)
    hi_v = np.full(cashflows.shape[0], hi)
    has_root = np.sign(npv(lo_v)) != np.sign(npv(hi_v))
    for _ in range(iters):
        mid = 0.5 * (lo_v + hi_v)
        go_left = np.sign(npv(mid)) == np.sign(npv(lo_v))
        lo_v = np.where(go_left, mid, lo_v)
        hi_v = np.where(go_left, hi_v, mid)
    out = 0.5 * (lo_v + hi_v)
    return np.where(has_root, out, np.nan)


def summarize(res) -> dict:
    """Percentiles of the investor's return and the issuer's dilution.

    `irr`, `pnl` and `moic` are cash only. When the deal has warrants, the `pkg_` numbers add
    the warrant payoff (or its Black-Scholes value on the last day) to the cash.

    `margin` is profit over the money put to work: the cash funded up front for a note, the
    dollars drawn for an equity line. An equity line has no meaningful IRR, because the
    investor sells the shares before paying for them and never has capital tied up.
    """
    cf = res.cashflows
    invested = -cf[:, 0]
    pnl = cf.sum(axis=1)
    moic = np.where(invested > 0, (pnl + invested) / np.where(invested > 0, invested, 1.0), np.nan)
    r = irr(cf)
    dilution = res.shares_issued / res.shares_out
    capital = np.where(invested > 0, invested, res.converted)
    margin = np.where(capital > 0, pnl / np.where(capital > 0, capital, 1.0), np.nan)

    def q(a, p):
        a = np.asarray(a, dtype=float)
        return float(np.nanpercentile(a, p)) if np.isfinite(a).any() else float("nan")

    out = {
        "irr_p10": q(r, 10), "irr_p50": q(r, 50), "irr_p90": q(r, 90),
        "moic_p50": q(moic, 50),
        "pnl_p10": q(pnl, 10), "pnl_p50": q(pnl, 50), "pnl_mean": float(np.mean(pnl)),
        "margin_p10": q(margin, 10), "margin_p50": q(margin, 50),
        "prob_loss": float(np.nanmean(pnl < 0)),
        "dilution_p50": q(dilution, 50), "dilution_p90": q(dilution, 90),
        "converted_p50": q(res.converted, 50),
        "repaid_cash_p50": q(getattr(res, "repaid_cash", np.zeros(1)), 50),
        "repaid_maturity_p50": q(getattr(res, "repaid_maturity", np.zeros(1)), 50),
        "prob_trigger": float(np.mean(getattr(res, "triggered", np.zeros(1, dtype=bool)))),
        "irr": r, "pnl": pnl, "dilution": dilution, "margin": margin,
    }
    wc = getattr(res, "warrant_cash", None)
    if wc is not None:
        pkg = cf + wc
        r_pkg = irr(pkg)
        pnl_pkg = pkg.sum(axis=1)
        margin_pkg = np.where(capital > 0, pnl_pkg / np.where(capital > 0, capital, 1.0), np.nan)
        out.update({
            "pkg_irr_p10": q(r_pkg, 10), "pkg_irr_p50": q(r_pkg, 50),
            "pkg_pnl_p10": q(pnl_pkg, 10), "pkg_pnl_p50": q(pnl_pkg, 50), "pkg_pnl_mean": float(np.mean(pnl_pkg)),
            "pkg_margin_p10": q(margin_pkg, 10), "pkg_margin_p50": q(margin_pkg, 50), "pkg_margin": margin_pkg,
            "pkg_prob_loss": float(np.nanmean(pnl_pkg < 0)),
            "warrant_value0": res.warrant_value0,
            "warrant_value_p50": q(res.warrant_value, 50), "warrant_value_mean": float(np.mean(res.warrant_value)),
            "warrant_lost_to_selling_mean": float(np.mean(res.warrant_value_base - res.warrant_value)),
            "warrant_in_money": float(np.mean(res.warrant_value > 0)) if wc.any() else 0.0,
            "dilution_with_warrants_p50": q(dilution + res.warrant_shares / res.shares_out, 50),
            "pkg_irr": r_pkg, "pkg_pnl": pnl_pkg,
        })
    return out
