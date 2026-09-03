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
    cf = res.cashflows
    invested = -cf[:, 0]
    pnl = cf.sum(axis=1)
    moic = np.where(invested > 0, (pnl + invested) / np.where(invested > 0, invested, 1.0), np.nan)
    r = irr(cf)
    dilution = res.shares_issued / res.shares_out

    def q(a, p):
        return float(np.nanpercentile(a, p))

    return {
        "irr_p10": q(r, 10), "irr_p50": q(r, 50), "irr_p90": q(r, 90),
        "moic_p50": q(moic, 50),
        "pnl_p50": q(pnl, 50),
        "prob_loss": float(np.nanmean(pnl < 0)),
        "dilution_p50": q(dilution, 50), "dilution_p90": q(dilution, 90),
        "converted_p50": q(res.converted, 50),
        "irr": r, "pnl": pnl, "dilution": dilution,
    }
