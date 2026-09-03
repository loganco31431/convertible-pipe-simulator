import numpy as np

TRADING_DAYS = 252


def simulate_prices(s0: float, sigma: float, days: int, n_paths: int, mu: float = 0.0,
                    seed: int | None = 0) -> np.ndarray:
    """Geometric Brownian motion daily closes, shape (n_paths, days + 1). Column 0 is s0."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS
    z = rng.standard_normal((n_paths, days))
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    paths = np.empty((n_paths, days + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(np.cumsum(log_ret, axis=1))
    return paths


def trailing_vwap(paths: np.ndarray, lookback: int) -> np.ndarray:
    """Trailing mean of the last `lookback` closes, as a VWAP proxy (volume is not simulated).

    Vectorized via cumulative sums. For the first `lookback` days the window is truncated.
    """
    csum = np.cumsum(paths, axis=1)
    n = paths.shape[1]
    idx = np.arange(n)
    start = np.clip(idx - lookback + 1, 0, None)
    window = (idx - start + 1).astype(float)
    prev_idx = np.broadcast_to(np.clip(start - 1, 0, None), paths.shape)
    prev = np.where(start > 0, np.take_along_axis(csum, prev_idx, axis=1), 0.0)
    return (csum - prev) / window
