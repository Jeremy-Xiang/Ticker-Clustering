# ticker-clustering

Clusters tickers by how they actually trade rather than what sector they're labeled. A high-growth energy name and a high-growth tech name often have more in common behaviorally than two utilities in the same sector. K-means or hierarchical clustering on nine standardized features, with PCA projection so the result is interpretable, not just a label assignment.

## Features

Nine per-ticker features computed over the lookback window:

`ann_return`, `ann_volatility`, `sharpe_like` (return / vol — a ranking signal, not a real Sharpe ratio), `max_drawdown`, `momentum_3m`, `momentum_6m`, `momentum_12m`, `avg_dollar_volume_log` (log-scaled so mega-caps don't dominate distance calculations), `volume_volatility`.

Every feature is something you could sanity-check on a price chart. That matters for clustering outputs to be believable — if two tickers end up in the same cluster, you should be able to point at the features and explain why.

## Choosing k

`silhouette_curve()` scores k from 2 to a configurable max. Higher is better, and the peak is the principled starting point. On the default 19-ticker universe, k=4 sits at the silhouette peak.

The silhouette score updates when you change k via the API's `n_clusters` parameter — it's in the response so a slider in a frontend can show the score changing in real time.

## The anagram seed bug

The synthetic offline fallback originally seeded its random generator with `sum(ord(c) for c in ticker)`. That's deterministic but not collision-resistant: `"GS"` and `"KO"` both sum to 154. Both tickers got byte-identical synthetic price histories and landed on exactly the same point on the scatter plot.

Fix: `zlib.crc32(ticker.encode()) % (2**32)`. crc32 operates on the actual byte sequence, not just the sum of its components. GS and KO now get distinct random streams.

This is in `tests/test_core.py` as `test_anagram_tickers_get_distinct_data`. If the seeding scheme ever reverts, the test fails before anything else breaks.

## Running it

```bash
pip install -r requirements.txt

python run_demo.py --basket --n-clusters 4 --method kmeans
python run_demo.py --tickers AAPL,MSFT,GOOGL,XOM,CVX,JNJ --method hierarchical
python run_demo.py --tickers-file tickers.txt --period 1y
```

Saves three plots: `ticker_clusters.png` (PCA scatter colored by cluster), `pc_loadings.png` (what each axis actually means, feature by feature), `silhouette_curve.png` (k selection guide).

Or as an API:

```bash
uvicorn app:app --port 8001
curl -X POST http://localhost:8001/cluster \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL","MSFT","GOOGL","XOM","JNJ"], "n_clusters": 3, "method": "kmeans", "period": "2y"}'

curl -X POST http://localhost:8001/silhouette \
  -d '{"tickers": [...], "period": "2y", "max_k": 8}'
```

The `/cluster` response includes `pca_coords` and `labels` directly — ready for a Recharts `ScatterChart`. `pc_loadings` gives the feature contributions to each axis for an "explain this axis" panel.

## Running the tests

```bash
pytest tests/ -v
```

Seven tests. The anagram collision test, feature completeness, both clustering methods produce matching output shape, determinism, error handling on too-few-tickers for the requested k.

## Structure

```
ticker-clustering/
├── run_demo.py   # CLI: clustering + three plots
├── app.py        # FastAPI service
├── src/
│   ├── features.py   # OHLCV → 9-feature vector per ticker
│   ├── clustering.py # KMeans/hierarchical, PCA, silhouette curve
│   └── data.py       # yfinance loader + synthetic fallback
└── tests/test_core.py
```
