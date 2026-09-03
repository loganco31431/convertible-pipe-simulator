import numpy as np

from .instruments import TRADING_DAYS


def historical_vol(ticker: str, lookback_days: int = 252) -> tuple[float, float, float]:
    """Annualized close-to-close vol, last close, and shares outstanding from Yahoo Finance.

    Returns (sigma, last_price, shares_outstanding). Raises if the download fails so the
    caller can fall back to manual inputs.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    hist = tk.history(period="2y", auto_adjust=True)
    if hist.empty:
        raise RuntimeError(f"no price history for {ticker}")
    closes = hist["Close"].tail(lookback_days + 1)
    log_ret = np.log(closes / closes.shift(1)).dropna()
    sigma = float(log_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    last = float(closes.iloc[-1])
    shares = None
    try:
        shares = tk.fast_info.get("shares")
    except Exception:
        pass
    if not shares:
        shares = tk.info.get("sharesOutstanding")
    if not shares:
        raise RuntimeError(f"no shares outstanding for {ticker}")
    return sigma, last, float(shares)
