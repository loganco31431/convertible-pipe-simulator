"""Intraday bars from any source, normalized to one grid the trade-out engine can use.

Three sources, one output:

- `yahoo_bars`      free, 5-minute bars for about the last 60 trading days (1-minute for ~7 days).
- `csv_bars`        a file exported from Bloomberg (Excel intraday bars, BDIB/IntradayBar
                    output) or any other vendor: needs a time column plus open/high/low/close
                    (or last price) and volume. Column names are matched loosely.
- `bloomberg_bars`  live IntradayBarRequest through the Desktop API (`blpapi`), for use on a
                    terminal. Written against the public blpapi reference; NOT yet run on a
                    terminal, so treat the first run as a test.

`to_grid` turns any of them into `Bars`: regular-session slots (09:30 to 16:00 New York) laid out
as (n_days, slots) arrays, so every trading day lines up bar for bar.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

NY = "America/New_York"


@dataclass
class Bars:
    days: pd.DatetimeIndex     # trading dates, oldest first
    slot_times: list           # "09:30", "09:35", ...
    price: np.ndarray          # (n_days, slots) typical price (h + l + c) / 3, a proxy for the bar's VWAP
    close: np.ndarray          # (n_days, slots) last trade in the bar
    volume: np.ndarray         # (n_days, slots) shares traded in the bar, 0 where no trade printed
    bar_minutes: int
    source: str

    @property
    def slots(self) -> int:
        return self.price.shape[1]

    @property
    def daily_volume(self) -> np.ndarray:
        return self.volume.sum(axis=1)

    @property
    def daily_vwap(self) -> np.ndarray:
        v = self.volume.sum(axis=1)
        return np.where(v > 0, (self.price * self.volume).sum(axis=1) / np.where(v > 0, v, 1), self.close[:, -1])


def _normalize(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Lower-case OHLCV columns and a New York DatetimeIndex."""
    cols = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols)
    aliases = {"last_price": "close", "last": "close", "px_last": "close", "price": "close",
               "open_price": "open", "px_open": "open", "high_price": "high", "px_high": "high",
               "low_price": "low", "px_low": "low", "px_volume": "volume", "vol": "volume"}
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})
    if not isinstance(df.index, pd.DatetimeIndex):
        tcol = next((c for c in df.columns if c in ("time", "datetime", "date", "dates", "timestamp", "date_time")), None)
        if tcol is None:
            raise ValueError(f"no time column found in {list(df.columns)}")
        df = df.set_index(pd.to_datetime(df.pop(tcol)))
    if "close" not in df or "volume" not in df:
        raise ValueError(f"need at least a close/last price and a volume column, got {list(df.columns)}")
    for c in ("open", "high", "low"):
        if c not in df:
            df[c] = df["close"]
    idx = df.index
    df.index = idx.tz_localize(tz) if idx.tz is None else idx.tz_convert(NY)
    if tz != NY:
        df.index = df.index.tz_convert(NY)
    return df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()


def to_grid(df: pd.DataFrame, bar_minutes: int, source: str, min_fill: float = 0.6) -> Bars:
    """Lay regular-session bars on a fixed (day, slot) grid.

    Slots with no bar get zero volume and the last known price. Days with fewer than `min_fill`
    of their slots present (half days, data gaps) are dropped so days stay comparable.
    """
    df = df[(df.index.time >= pd.Timestamp("09:30").time()) & (df.index.time < pd.Timestamp("16:00").time())]
    slots = pd.date_range("09:30", "16:00", freq=f"{bar_minutes}min", inclusive="left").strftime("%H:%M").tolist()
    df = df.assign(day=df.index.normalize().tz_localize(None), slot=df.index.strftime("%H:%M"))
    df = df[df["slot"].isin(slots)]
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    df = df.assign(typical=typical)
    keep = df.groupby("day").size() >= min_fill * len(slots)
    df = df[df["day"].isin(keep[keep].index)]
    if df.empty:
        raise ValueError("no complete regular-session days in the data")
    piv = lambda col: df.pivot_table(index="day", columns="slot", values=col, aggfunc="last").reindex(columns=slots)
    vol = piv("volume").fillna(0.0)
    close = piv("close").ffill(axis=1)
    price = piv("typical").ffill(axis=1)
    # a day that opens with missing bars: back-fill from the first printed bar of that day
    close, price = close.bfill(axis=1), price.bfill(axis=1)
    return Bars(days=pd.DatetimeIndex(vol.index), slot_times=slots, price=price.to_numpy(),
                close=close.to_numpy(), volume=vol.to_numpy(), bar_minutes=bar_minutes, source=source)


def yahoo_bars(ticker: str, bar_minutes: int = 5) -> Bars:
    import yfinance as yf

    period = {1: "7d", 2: "60d", 5: "60d", 15: "60d", 30: "60d"}.get(bar_minutes, "60d")
    h = yf.Ticker(ticker).history(period=period, interval=f"{bar_minutes}m", prepost=False, auto_adjust=False)
    if h.empty:
        raise RuntimeError(f"no intraday bars for {ticker} from Yahoo")
    return to_grid(_normalize(h, NY), bar_minutes, f"Yahoo {bar_minutes}-minute")


def csv_bars(path_or_buffer, bar_minutes: int = 5, tz: str = NY) -> Bars:
    """Load an intraday export. Excel (.xlsx) and CSV both work; times are read as `tz` local."""
    name = str(getattr(path_or_buffer, "name", path_or_buffer)).lower()
    df = pd.read_excel(path_or_buffer) if name.endswith((".xlsx", ".xls")) else pd.read_csv(path_or_buffer)
    return to_grid(_normalize(df, tz), bar_minutes, f"file ({bar_minutes}-minute)")


def bloomberg_bars(security: str, start: pd.Timestamp, end: pd.Timestamp, bar_minutes: int = 5,
                   host: str = "localhost", port: int = 8194) -> Bars:
    """Intraday TRADE bars from a Bloomberg terminal via the Desktop API.

    `security` is a Bloomberg ticker such as "OTLK US Equity". Bloomberg returns bar times in
    UTC (GMT). Intraday bar history on the terminal goes back roughly 140 business days.
    Untested on a live terminal.
    """
    import blpapi

    opts = blpapi.SessionOptions()
    opts.setServerHost(host)
    opts.setServerPort(port)
    session = blpapi.Session(opts)
    if not session.start() or not session.openService("//blp/refdata"):
        raise RuntimeError("could not connect to Bloomberg; is the terminal running and logged in?")
    req = session.getService("//blp/refdata").createRequest("IntradayBarRequest")
    req.set("security", security)
    req.set("eventType", "TRADE")
    req.set("interval", int(bar_minutes))
    req.set("startDateTime", pd.Timestamp(start).tz_localize(NY).tz_convert("UTC").to_pydatetime())
    req.set("endDateTime", pd.Timestamp(end).tz_localize(NY).tz_convert("UTC").to_pydatetime())
    session.sendRequest(req)
    rows = []
    try:
        while True:
            ev = session.nextEvent(5000)
            for msg in ev:
                if msg.hasElement("responseError"):
                    raise RuntimeError(str(msg.getElement("responseError")))
                if not msg.hasElement("barData"):
                    continue
                ticks = msg.getElement("barData").getElement("barTickData")
                for i in range(ticks.numValues()):
                    b = ticks.getValueAsElement(i)
                    rows.append({"time": b.getElementAsDatetime("time"), "open": b.getElementAsFloat("open"),
                                 "high": b.getElementAsFloat("high"), "low": b.getElementAsFloat("low"),
                                 "close": b.getElementAsFloat("close"), "volume": b.getElementAsInteger("volume")})
            if ev.eventType() == blpapi.Event.RESPONSE:
                break
    finally:
        session.stop()
    if not rows:
        raise RuntimeError(f"Bloomberg returned no bars for {security}")
    df = pd.DataFrame(rows).set_index("time")
    return to_grid(_normalize(df, "UTC"), bar_minutes, f"Bloomberg {bar_minutes}-minute")
