"""
run_demo.py — Standalone demo: cluster a basket of tickers and save plots,
no API server required.

Usage:
    python run_demo.py --tickers AAPL,MSFT,GOOGL,XOM,CVX,JNJ,PG,TSLA,JPM,BAC
    python run_demo.py --basket --n-clusters 4 --method kmeans
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from src.clustering import run_clustering, silhouette_curve
from src.data import load_many
from src.features import build_feature_matrix

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

# A sector-diverse default basket, same spirit as stock-forecast-bench's,
# extended a bit since clustering wants more points to be meaningful.
DEFAULT_BASKET = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",  # tech / growth
    "JPM", "BAC", "GS",                                 # financials
    "XOM", "CVX",                                       # energy
    "JNJ", "PG", "KO", "PEP",                            # defensive / staples
    "TSLA", "RIVN",                                     # high-vol growth
    "WMT", "COST",                                       # retail
]


def plot_clusters(result, feature_df, ticker: str = "") -> str:
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    for cluster_id in sorted(set(result.labels)):
        mask = result.labels == cluster_id
        coords = result.pca_coords[mask]
        tickers_in_cluster = [t for t, m in zip(result.tickers, mask) if m]
        ax.scatter(coords[:, 0], coords[:, 1], color=cmap(cluster_id % 10), label=f"Cluster {cluster_id}", s=80)
        for (x, y), t in zip(coords, tickers_in_cluster):
            ax.annotate(t, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=9)

    ev = result.explained_variance_ratio
    ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}% of variance)")
    ax.set_title(
        f"Ticker clusters ({result.method}, k={result.n_clusters}, silhouette={result.silhouette:.2f})"
    )
    ax.legend(loc="best", fontsize=8)
    ax.axhline(0, color="lightgray", linewidth=0.8)
    ax.axvline(0, color="lightgray", linewidth=0.8)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "ticker_clusters.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_loadings(result) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    loadings = result.pc_loadings
    y_pos = np.arange(len(loadings))
    ax.barh(y_pos - 0.2, loadings["PC1"], height=0.4, label="PC1")
    ax.barh(y_pos + 0.2, loadings["PC2"], height=0.4, label="PC2")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(loadings.index)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Loading (contribution to principal component)")
    ax.set_title("What each axis actually means")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "pc_loadings.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_silhouette(feature_df) -> str:
    curve = silhouette_curve(feature_df)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["k"], curve["silhouette"], marker="o")
    ax.set_xlabel("n_clusters (k)")
    ax.set_ylabel("Silhouette score (higher = better-separated clusters)")
    ax.set_title("Silhouette score by k — how to actually pick n_clusters")
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "silhouette_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers")
    parser.add_argument("--tickers-file", default=None)
    parser.add_argument("--basket", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--method", choices=["kmeans", "hierarchical"], default="kmeans")
    parser.add_argument("--period", default="2y")
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    if args.tickers_file:
        with open(args.tickers_file) as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = DEFAULT_BASKET

    print(f"Loading {len(tickers)} tickers...")
    ticker_data = load_many(tickers, period=args.period)
    print(f"Got usable history for {len(ticker_data)}/{len(tickers)} tickers.\n")

    feature_df = build_feature_matrix(ticker_data)
    print("Feature matrix:")
    print(feature_df.round(3))
    print()

    result = run_clustering(feature_df, n_clusters=args.n_clusters, method=args.method)

    print(f"Method: {result.method}   k={result.n_clusters}   silhouette={result.silhouette:.3f}\n")
    for cluster_id in sorted(set(result.labels)):
        members = [t for t, l in zip(result.tickers, result.labels) if l == cluster_id]
        print(f"Cluster {cluster_id}: {', '.join(members)}")

    p1 = plot_clusters(result, feature_df)
    p2 = plot_loadings(result)
    p3 = plot_silhouette(feature_df)
    print(f"\nSaved: {p1}\nSaved: {p2}\nSaved: {p3}")


if __name__ == "__main__":
    main()
