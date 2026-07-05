"""
features.py — Turn raw OHLCV history into a per-ticker feature vector
suitable for clustering.

Every feature here is something a human analyst would recognize and could
sanity-check by eye — that matters more than squeezing in exotic factors,
since the whole point of clustering tickers is to explain *why* two stocks
ended up in the same group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

FEATURE_NAMES = [
    "ann_return",
    "ann_volatility",
    "sharpe_like",
    "max_drawdown",
    "momentum_3m",
    "momentum_6m",
    "momentum_12m",
    "avg_dollar_volume_log",
    "volume_volatility",
]


def compute_features(df: pd.DataFrame) -> dict[str, float]:
    """Compute one feature vector (as a dict) from a single ticker's OHLCV history."""
    close = df["Close"]
    returns = close.pct_change().dropna()

    ann_return = float((1 + returns.mean()) ** TRADING_DAYS_PER_YEAR - 1)
    ann_vol = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe_like = ann_return / ann_vol if ann_vol > 1e-9 else 0.0

    running_max = close.cummax()
    drawdown = (close - running_max) / running_max
    max_drawdown = float(drawdown.min())  # negative number; more negative = worse

    def trailing_return(n_days: int) -> float:
        if len(close) <= n_days:
            return 0.0
        return float(close.iloc[-1] / close.iloc[-n_days] - 1)

    momentum_3m = trailing_return(63)
    momentum_6m = trailing_return(126)
    momentum_12m = trailing_return(252)

    dollar_volume = (df["Close"] * df["Volume"]).replace(0, np.nan).dropna()
    avg_dollar_volume_log = float(np.log10(dollar_volume.mean())) if len(dollar_volume) else 0.0
    volume_volatility = float(df["Volume"].pct_change().dropna().std()) if len(df) > 1 else 0.0

    return {
        "ann_return": ann_return,
        "ann_volatility": ann_vol,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_drawdown,
        "momentum_3m": momentum_3m,
        "momentum_6m": momentum_6m,
        "momentum_12m": momentum_12m,
        "avg_dollar_volume_log": avg_dollar_volume_log,
        "volume_volatility": volume_volatility,
    }


def build_feature_matrix(ticker_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    ticker_data: {ticker: ohlcv_dataframe}
    Returns a DataFrame indexed by ticker, one column per feature in
    FEATURE_NAMES, with any failures dropped (and printed) rather than
    silently producing NaNs downstream.
    """
    rows = {}
    for ticker, df in ticker_data.items():
        try:
            rows[ticker] = compute_features(df)
        except Exception as exc:  # noqa: BLE001
            print(f"[features.py] Skipping {ticker}: {exc}")

    feature_df = pd.DataFrame(rows).T
    feature_df = feature_df[FEATURE_NAMES]
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna()
    return feature_df
