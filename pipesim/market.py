"""Market inputs for the execution model: price, volatility, shares outstanding and trading volume.

`calibrate_market` pulls everything from Yahoo Finance for a ticker. `MarketInputs` can also be
built by hand for offline runs.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .instruments import TRADING_DAYS


@dataclass
class MarketInputs:
    s0: float                       # last close
    sigma: float                    # annualized close-to-close volatility
    shares_out: float               # shares outstanding
    adv_shares: float               # average daily volume, shares
    volume_logstd: float = 0.5      # day-to-day dispersion of log volume
    history: pd.DataFrame | None = field(default=None, repr=False)  # Close, Volume (for replay)

    @property
    def sigma_daily(self) -> float:
        return self.sigma / np.sqrt(TRADING_DAYS)

    @property
    def adv_dollars(self) -> float:
        return self.adv_shares * self.s0


def calibrate_market(ticker: str, lookback_days: int = 252, adv_days: int = 60) -> MarketInputs:
    """Volatility from the last `lookback_days` of closes, ADV from the last `adv_days` of volume.

    The full two-year Close/Volume history is kept on the result so the note can be replayed
    on real data.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    hist = tk.history(period="2y", auto_adjust=True)
    if hist.empty:
        raise RuntimeError(f"no price history for {ticker}")
    hist = hist[["Close", "Volume"]].dropna()
    hist = hist[hist["Volume"] > 0]

    closes = hist["Close"].tail(lookback_days + 1)
    log_ret = np.log(closes / closes.shift(1)).dropna()
    sigma = float(log_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))

    vol = hist["Volume"].tail(adv_days)
    adv = float(vol.mean())
    logstd = float(np.log(hist["Volume"].tail(lookback_days)).std(ddof=1))

    shares = None
    try:
        shares = tk.fast_info.get("shares")
    except Exception:
        pass
    if not shares:
        shares = tk.info.get("sharesOutstanding")
    if not shares:
        raise RuntimeError(f"no shares outstanding for {ticker}")

    return MarketInputs(s0=float(closes.iloc[-1]), sigma=sigma, shares_out=float(shares),
                        adv_shares=adv, volume_logstd=logstd, history=hist)
