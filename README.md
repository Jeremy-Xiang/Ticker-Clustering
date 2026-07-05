# ticker-clustering

Cluster a basket of tickers by *behavior* — return, volatility, momentum,
liquidity, drawdown risk — rather than by sector label, and see it on a 2D
map you can actually explain. Built as a FastAPI service with tunable
parameters (number of clusters, clustering method, lookback period) so it
can be wired straight into a React frontend, in the same spirit as
`stardex`'s "cluster your GitHub stars" pattern — here applied to a stock
universe instead.

## Why behavioral clustering instead of sector labels

Sector tags ("Technology", "Energy") tell you what a company *does*, not how
its stock actually *behaves*. A high-growth, high-volatility energy name can
trade more like a tech growth stock than like a sleepy utility in its own
sector. Clustering on the actual return/volatility/momentum/liquidity profile
surfaces groupings sector labels miss — which is the more useful lens if
you're trying to understand correlation and diversification, not just GICS
classification.

## Features (one vector per ticker)

| Feature | What it captures |
|---|---|
| `ann_return` | Annualized return over the lookback window |
| `ann_volatility` | Annualized volatility of daily returns |
| `sharpe_like` | Return / volatility (not risk-free-adjusted — a rough ranking signal, not a real Sharpe ratio) |
| `max_drawdown` | Worst peak-to-trough decline in the window |
| `momentum_3m` / `momentum_6m` / `momentum_12m` | Trailing return over 63/126/252 trading days |
| `avg_dollar_volume_log` | log10(price × volume) — liquidity, log-scaled so mega-caps don't dominate the distance metric |
| `volume_volatility` | How erratic daily volume is — a rough proxy for "how news-driven is this stock" |

Every feature here is something a human analyst would recognize and could
sanity-check by eye on a price chart. That matters more than squeezing in
exotic factors, since the whole point of clustering is to explain *why* two
stocks ended up in the same group — see "what each axis means" below.

## Methodology

1. **Standardize** all features (`StandardScaler`) — otherwise `ann_return`
   (values like 0.15) would be swamped by `avg_dollar_volume_log` (values
   like 9.5) in any distance calculation.
2. **PCA to 2 components**, purely for visualization — clustering itself
   runs on the full standardized feature space, not the 2D projection.
3. **Cluster** via K-means (fast, assumes roughly spherical clusters) or
   agglomerative/hierarchical (no shape assumption, but no `predict()` for
   new points later) — exposed as a simple method switch, since neither is
   strictly "more correct" and it's a legitimate thing to let a user toggle.
4. **Pick k via silhouette score**, not by eye. `silhouette_curve()` scores
   every k from 2 to a configurable max; the peak is a principled
   starting point (see `plots/silhouette_curve.png`).

### What each axis actually means

PCA components are linear combinations of the original features, and on the
ticker basket above they came out cleanly interpretable:
- **PC1** loads positively on return, Sharpe-like, and liquidity, and
  negatively on momentum and volume volatility — reads roughly as a
  "steady quality" axis.
- **PC2** loads positively on drawdown risk and trailing momentum, and
  negatively on volatility — reads roughly as "recent momentum vs.
  risk taken to get there."

`plots/pc_loadings.png` plots every feature's contribution to both axes —
check that plot before trusting any "these stocks are similar" claim from
the scatter plot, since the axes' meaning isn't guaranteed to repeat
identically on a different basket or lookback window.

## A real bug found while building this

The synthetic offline fallback originally seeded its random generator with
`sum(ord(c) for c in ticker)` — deterministic, but it collides on any
anagram: `"GS"` and `"KO"` both sum to 154, so two unrelated tickers
silently got *identical* synthetic price histories and landed on exactly the
same point on the scatter plot. Fixed by seeding with `zlib.crc32()` over
the ticker's bytes instead. Worth knowing about if you ever reach for a
"deterministic seed from a string" trick: sum-of-character-codes is not a
hash function, and the failure mode (silently identical fake data for two
different inputs) is the kind of thing that looks fine until you happen to
pick the two tickers that collide.

## Running it

### Standalone demo (no server)

```bash
pip install -r requirements.txt

python run_demo.py --basket --n-clusters 4 --method kmeans
python run_demo.py --tickers AAPL,MSFT,GOOGL,XOM,CVX,JNJ --n-clusters 3
python run_demo.py --tickers-file my_tickers.txt --n-clusters 5
```

Saves three plots to `plots/`:
- `ticker_clusters.png` — PCA scatter, colored by cluster, every ticker labeled
- `pc_loadings.png` — what PC1/PC2 actually mean, feature by feature
- `silhouette_curve.png` — silhouette score for k = 2..8, for picking n_clusters

### As an API (for wiring into a frontend)

```bash
uvicorn app:app --reload --port 8001
```

```bash
curl -X POST http://localhost:8001/cluster \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL","MSFT","GOOGL","XOM","CVX","JNJ"], "n_clusters": 3, "method": "kmeans", "period": "2y"}'
```

Returns ticker labels, 2D PCA coordinates (ready to scatter-plot directly),
explained variance per axis, PC loadings (for an "explain this axis" panel),
and the silhouette score for the chosen k.

`POST /silhouette` takes a ticker list and returns the silhouette-vs-k curve
as JSON, for a "suggest a good k" control in the UI.

## Wiring this into THESIS

This is built to slot in next to THESIS's existing FastAPI backend rather
than run as a separate service long-term:

1. Copy `src/features.py` and `src/clustering.py` into the THESIS backend.
2. Mount `app.py`'s router under THESIS's existing FastAPI app instead of
   running a second `uvicorn` process — same data layer, one less moving
   part. THESIS's existing yfinance/Cohere data pipeline can replace
   `src/data.py` directly, since you already have live price history loaded
   for the 53 tracked tickers.
3. New React tab: a ticker multi-select (or just "use all 53 THESIS
   tickers" by default), a `k` slider (2-10), a method dropdown
   (kmeans/hierarchical), and a scatter chart (Recharts `ScatterChart` maps
   directly onto `pca_coords` + `labels` from the response) with the
   silhouette score shown next to the slider so changing `k` has immediate,
   legible feedback instead of just changing colors on a scatter plot for no
   stated reason.

## Project structure

```
ticker-clustering/
├── app.py              # FastAPI service
├── run_demo.py          # standalone CLI + plots, no server needed
├── src/
│   ├── data.py          # yfinance loader (+ offline synthetic fallback)
│   ├── features.py      # OHLCV -> per-ticker feature vector
│   └── clustering.py    # standardize -> PCA -> KMeans/hierarchical
├── plots/                # generated PNGs from run_demo.py
└── requirements.txt
```

## Running the tests

```bash
pytest tests/ -v
```

The suite pins the behaviors that actually caught bugs during development
(see the sections above), not ceremony coverage — every test encodes a
check where the wrong answer was at some point the actual behavior.

## Possible next steps

- Pull in the sentiment scores THESIS already computes per ticker as
  additional features — would directly test whether sentiment adds a
  distinct axis or just rides along with momentum.
- Cache `build_feature_matrix()` output with a short TTL in the FastAPI
  layer; recomputing 9 features over 2 years of daily data for the same
  53 tickers on every slider change is wasted work.
- Re-run clustering on a rolling basis (e.g. monthly) and track which
  tickers *change* cluster over time — a ticker jumping from the
  "quality" cluster to the "high drawdown risk" cluster is a more
  interesting signal than its static cluster membership on any single day.
