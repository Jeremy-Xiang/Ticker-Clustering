"""
app.py — FastAPI service for ticker clustering.

Designed to slot in next to THESIS's existing FastAPI backend as a new
router (see README for how to mount it under the same app instead of
running standalone), or to run standalone for local development:

    uvicorn app:app --reload --port 8001

Endpoints:
    GET  /health
    POST /cluster   — run K-means or hierarchical clustering on a basket
                       of tickers, with tunable n_clusters and lookback period
    GET  /silhouette — silhouette score per k, to help pick n_clusters
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.clustering import run_clustering, silhouette_curve
from src.data import load_many
from src.features import build_feature_matrix

app = FastAPI(title="Ticker Clustering API", version="1.0")

# Permissive CORS for local development against a separate frontend dev
# server. Tighten this (or remove it and rely on THESIS's existing CORS
# config) before deploying for real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClusterRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=2, description="Ticker symbols to cluster")
    n_clusters: int = Field(4, ge=2, le=20)
    method: Literal["kmeans", "hierarchical"] = "kmeans"
    period: str = Field("2y", description="yfinance lookback period, e.g. '1y', '2y', '5y'")


class ClusterResponse(BaseModel):
    tickers: list[str]
    labels: list[int]
    pca_coords: list[list[float]]
    explained_variance_ratio: list[float]
    pc_loadings: dict[str, dict[str, float]]
    silhouette: float
    method: str
    n_clusters: int
    skipped_tickers: list[str]


class SilhouetteRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=3)
    period: str = "2y"
    max_k: int = Field(8, ge=3, le=15)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/cluster", response_model=ClusterResponse)
def cluster_tickers(req: ClusterRequest):
    ticker_data = load_many(req.tickers, period=req.period)
    skipped = [t for t in req.tickers if t.upper() not in ticker_data and t not in ticker_data]

    feature_df = build_feature_matrix(ticker_data)
    if len(feature_df) < req.n_clusters:
        raise HTTPException(
            status_code=400,
            detail=f"Only {len(feature_df)} tickers had usable data — need at least {req.n_clusters} for "
            f"{req.n_clusters} clusters. Try fewer clusters or check your ticker symbols.",
        )

    result = run_clustering(feature_df, n_clusters=req.n_clusters, method=req.method)

    return ClusterResponse(
        tickers=result.tickers,
        labels=[int(l) for l in result.labels],
        pca_coords=[[float(x), float(y)] for x, y in result.pca_coords],
        explained_variance_ratio=[float(v) for v in result.explained_variance_ratio],
        pc_loadings=result.pc_loadings.to_dict(),
        silhouette=result.silhouette,
        method=result.method,
        n_clusters=result.n_clusters,
        skipped_tickers=skipped,
    )


@app.post("/silhouette")
def silhouette(req: SilhouetteRequest):
    ticker_data = load_many(req.tickers, period=req.period)
    feature_df = build_feature_matrix(ticker_data)
    curve = silhouette_curve(feature_df, k_range=range(2, req.max_k + 1))
    return curve.to_dict(orient="records")
