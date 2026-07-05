"""
data.py — Load OHLCV history for one or many tickers.

Same yfinance-first, synthetic-fallback pattern as stock-forecast-bench's
data.py, extended with a multi-ticker convenience loader since clustering
is inherently a many-tickers operation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_ohlc(ticker: str, period: str = "2y") -> pd.DataFrame:
    try:
        import yfinance as yf

        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise RuntimeError("yfinance returned no data")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index.name = "Date"
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[data.py] Live fetch failed for {ticker} ({exc}). Using synthetic fallback.")
        return _synthetic_ohlc(seed=_stable_seed(ticker))


def load_many(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    """Load OHLCV for many tickers, skipping any that fail entirely."""
    out = {}
    for t in tickers:
        try:
            df = load_ohlc(t, period=period)
            if len(df) >= 60:  # need enough history for the feature window
                out[t] = df
        except Exception as exc:  # noqa: BLE001
            print(f"[data.py] Skipping {t}: {exc}")
    return out


def _stable_seed(ticker: str) -> int:
    """
    Deterministic seed from a ticker string. sum(ord(c) for c in ticker)
    looks deterministic too, but collides on any anagram (e.g. 'GS' and
    'KO' both sum to 154), which would silently give two different tickers
    identical synthetic price paths. crc32 over the actual byte sequence
    avoids that.
    """
    import zlib

    return zlib.crc32(ticker.encode()) % (2**32)


def _synthetic_ohlc(n_days: int = 500, seed: int = 0, start_price: float = 100.0) -> pd.DataFrame:
    """
    Deterministic synthetic OHLCV for offline development/testing. Each
    seed gets a different drift/volatility regime so synthetic "tickers"
    are still distinguishable from each other when clustering.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    drift = rng.uniform(-0.0003, 0.0009)
    vol = rng.uniform(0.010, 0.035)
    daily_returns = rng.normal(loc=drift, scale=vol, size=n_days)
    close = start_price * np.exp(np.cumsum(daily_returns))

    open_ = np.empty(n_days)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.003, size=n_days - 1))

    intraday_range = np.abs(rng.normal(0.006, 0.004, size=n_days)) * close
    high = np.maximum(open_, close) + intraday_range
    low = np.minimum(open_, close) - intraday_range
    volume = rng.integers(500_000, 50_000_000, size=n_days)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df
