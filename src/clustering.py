"""
clustering.py — Standardize features, project to 2D via PCA for plotting,
and cluster via either K-means or hierarchical (agglomerative) clustering.

Both clustering methods are exposed with the same output shape so the API
layer (and a future frontend tab) can offer "method" as a simple dropdown
without changing anything else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterResult:
    tickers: list[str]
    labels: np.ndarray
    pca_coords: np.ndarray  # shape (n_tickers, 2)
    explained_variance_ratio: np.ndarray  # shape (2,)
    pc_loadings: pd.DataFrame  # feature loadings on PC1/PC2, for explaining the axes
    silhouette: float
    method: str
    n_clusters: int


def run_clustering(
    feature_df: pd.DataFrame,
    n_clusters: int = 4,
    method: str = "kmeans",
    random_state: int = 42,
) -> ClusterResult:
    if len(feature_df) < n_clusters:
        raise ValueError(f"Need at least {n_clusters} tickers to form {n_clusters} clusters, got {len(feature_df)}.")

    tickers = feature_df.index.tolist()
    X = feature_df.to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_df.columns,
        columns=["PC1", "PC2"],
    )

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(X_scaled)
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'kmeans' or 'hierarchical'.")

    sil = float(silhouette_score(X_scaled, labels)) if n_clusters > 1 and n_clusters < len(X_scaled) else 0.0

    return ClusterResult(
        tickers=tickers,
        labels=labels,
        pca_coords=coords,
        explained_variance_ratio=pca.explained_variance_ratio_,
        pc_loadings=loadings,
        silhouette=sil,
        method=method,
        n_clusters=n_clusters,
    )


def silhouette_curve(feature_df: pd.DataFrame, k_range=range(2, 9), random_state: int = 42) -> pd.DataFrame:
    """
    Silhouette score for each k in k_range — a quick, principled way to pick
    n_clusters instead of guessing. Higher is better (max 1.0).
    """
    X_scaled = StandardScaler().fit_transform(feature_df.to_numpy())
    rows = []
    for k in k_range:
        if k >= len(feature_df):
            continue
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        rows.append({"k": k, "silhouette": score})
    return pd.DataFrame(rows)
