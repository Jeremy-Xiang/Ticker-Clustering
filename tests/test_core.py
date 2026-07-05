"""
Tests for ticker-clustering. Run: pytest tests/ -v

Encodes the real bug found during development: sum-of-ordinals seeding
gave anagram tickers (GS/KO) byte-identical synthetic price paths, which
put two unrelated tickers on the exact same clustering coordinates.
"""

import numpy as np
import pytest

from src.clustering import run_clustering, silhouette_curve
from src.data import _stable_seed, _synthetic_ohlc, load_many
from src.features import FEATURE_NAMES, build_feature_matrix, compute_features


TICKERS = ["AAPL", "MSFT", "GOOGL", "GS", "KO", "XOM", "CVX", "JNJ", "PG", "TSLA"]


@pytest.fixture(scope="module")
def feature_df():
    data = load_many(TICKERS, period="2y")
    return build_feature_matrix(data)


def test_anagram_tickers_get_distinct_data():
    """The GS/KO bug: anagrams must not share a synthetic price path."""
    gs = _synthetic_ohlc(seed=_stable_seed("GS"))
    ko = _synthetic_ohlc(seed=_stable_seed("KO"))
    assert not np.allclose(gs["Close"].values, ko["Close"].values)


def test_features_complete_and_finite(feature_df):
    assert list(feature_df.columns) == FEATURE_NAMES
    assert len(feature_df) == len(TICKERS)
    assert np.isfinite(feature_df.to_numpy()).all()


def test_feature_values_sane():
    df = _synthetic_ohlc(n_days=500, seed=1)
    f = compute_features(df)
    assert f["ann_volatility"] > 0
    assert f["max_drawdown"] <= 0  # drawdown is never positive
    assert f["avg_dollar_volume_log"] > 0


def test_kmeans_and_hierarchical_same_shape(feature_df):
    for method in ("kmeans", "hierarchical"):
        r = run_clustering(feature_df, n_clusters=3, method=method)
        assert len(r.labels) == len(feature_df)
        assert r.pca_coords.shape == (len(feature_df), 2)
        assert set(r.labels) == {0, 1, 2}
        assert -1.0 <= r.silhouette <= 1.0


def test_clustering_deterministic(feature_df):
    a = run_clustering(feature_df, n_clusters=3, method="kmeans")
    b = run_clustering(feature_df, n_clusters=3, method="kmeans")
    assert (a.labels == b.labels).all()


def test_too_few_tickers_raises(feature_df):
    with pytest.raises(ValueError):
        run_clustering(feature_df.head(2), n_clusters=5)


def test_silhouette_curve_shape(feature_df):
    curve = silhouette_curve(feature_df, k_range=range(2, 5))
    assert list(curve["k"]) == [2, 3, 4]
    assert curve["silhouette"].between(-1, 1).all()
