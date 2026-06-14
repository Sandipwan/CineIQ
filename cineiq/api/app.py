"""
api/app.py
===========
FastAPI REST API for CINEIQ.

Endpoints:
  GET  /health                  — liveness check
  POST /recommend               — hybrid personalised recommendations
  POST /similar                 — content-similar movies for a given movie

The app lazily loads all ML models on startup and keeps them in module-level
singletons so each request doesn't re-load from disk.

Run with:
    uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from src.data_processing import load_data
from src.explainability import Explainer
from src.models.collaborative import CollaborativeModel
from src.models.content_based import ContentBasedModel
from src.models.hybrid_ensemble import HybridEnsemble
from src.models.sentiment import SentimentReRanker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application state (module-level singletons)
# ---------------------------------------------------------------------------

class AppState:
    movies_df: Optional[pd.DataFrame] = None
    ratings_df: Optional[pd.DataFrame] = None
    reviews_df: Optional[pd.DataFrame] = None
    collab: Optional[CollaborativeModel] = None
    content: Optional[ContentBasedModel] = None
    ensemble: Optional[HybridEnsemble] = None
    reranker: Optional[SentimentReRanker] = None
    explainer: Optional[Explainer] = None
    ready: bool = False


STATE = AppState()


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all models on startup; clean up on shutdown.
    """
    logger.info("CINEIQ API starting — loading data and models ...")

    try:
        movies_df, ratings_df, reviews_df = load_data(n_movies=500, n_users=300)
        STATE.movies_df = movies_df
        STATE.ratings_df = ratings_df
        STATE.reviews_df = reviews_df
        logger.info(
            "Data loaded: %d movies | %d ratings | %d reviews",
            len(movies_df), len(ratings_df), len(reviews_df),
        )

        STATE.collab = CollaborativeModel()
        STATE.collab.fit(ratings_df)
        logger.info("CollaborativeModel trained.")

        STATE.content = ContentBasedModel()
        STATE.content.fit(movies_df)
        logger.info("ContentBasedModel trained.")

        STATE.ensemble = HybridEnsemble(STATE.collab, STATE.content, alpha=0.6, beta=0.4)

        STATE.reranker = SentimentReRanker(backend="vader")
        STATE.reranker.fit(reviews_df)
        logger.info("SentimentReRanker fitted.")

        STATE.explainer = Explainer(movies_df, ratings_df)

        STATE.ready = True
        logger.info("CINEIQ API is ready.")
    except Exception as exc:
        logger.exception("Failed to initialise models: %s", exc)
        STATE.ready = False

    yield

    logger.info("CINEIQ API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CINEIQ — Explainable Movie Recommendation API",
    description=(
        "Hybrid recommendation engine combining SVD collaborative filtering, "
        "TF-IDF content-based filtering, and VADER sentiment re-ranking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=1, description="User ID to generate recommendations for.")
    top_n: int = Field(10, ge=1, le=50, description="Number of recommendations to return.")
    alpha: float = Field(
        0.6, ge=0.0, le=1.0,
        description="Weight for collaborative filtering (0 = pure content, 1 = pure collaborative).",
    )
    use_sentiment: bool = Field(True, description="Apply sentiment re-ranking boost.")

    @field_validator("alpha")
    @classmethod
    def alpha_valid(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0")
        return round(v, 4)


class SimilarRequest(BaseModel):
    movie_id: int = Field(..., ge=1, description="Seed movie ID to find similars for.")
    top_n: int = Field(10, ge=1, le=50, description="Number of similar movies to return.")
    exclude_ids: list[int] = Field(
        default_factory=list, description="Movie IDs to exclude from results."
    )


class MovieResult(BaseModel):
    movie_id: int
    title: str
    hybrid_score: float
    final_score: float
    svd_norm: float
    content_norm: float
    sentiment_score: float
    explanation: str


class RecommendResponse(BaseModel):
    user_id: int
    alpha: float
    top_n: int
    use_sentiment: bool
    latency_ms: float
    recommendations: list[MovieResult]


class SimilarMovieResult(BaseModel):
    movie_id: int
    title: str
    content_score: float


class SimilarResponse(BaseModel):
    seed_movie_id: int
    seed_title: str
    top_n: int
    similar_movies: list[SimilarMovieResult]


class HealthResponse(BaseModel):
    status: str
    models_ready: bool
    n_movies: int
    n_users: int
    n_ratings: int


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_ready() -> None:
    if not STATE.ready:
        raise HTTPException(
            status_code=503,
            detail="Models are not yet loaded. Please try again shortly.",
        )


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if not (v != v) else default   # NaN guard
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """
    Returns API liveness status and basic dataset statistics.
    """
    return HealthResponse(
        status="ok" if STATE.ready else "initialising",
        models_ready=STATE.ready,
        n_movies=len(STATE.movies_df) if STATE.movies_df is not None else 0,
        n_users=(
            int(STATE.ratings_df["user_id"].nunique())
            if STATE.ratings_df is not None else 0
        ),
        n_ratings=len(STATE.ratings_df) if STATE.ratings_df is not None else 0,
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
def recommend(body: RecommendRequest) -> RecommendResponse:
    """
    Generate personalised hybrid movie recommendations for a user.

    - Blends SVD collaborative filtering with TF-IDF content similarity.
    - Optionally applies VADER sentiment re-ranking.
    - Returns structured JSON with scores and natural-language explanations.
    """
    _require_ready()

    start = time.perf_counter()

    # Validate user_id exists (graceful: cold-start users are supported)
    user_exists = (
        body.user_id in STATE.ratings_df["user_id"].values  # type: ignore[union-attr]
    )
    if not user_exists:
        logger.info(
            "User %d is new (cold-start) — using popularity-weighted fallback.", body.user_id
        )

    try:
        rec_df = STATE.ensemble.recommend(  # type: ignore[union-attr]
            user_id=body.user_id,
            ratings_df=STATE.ratings_df,  # type: ignore[arg-type]
            movies_df=STATE.movies_df,  # type: ignore[arg-type]
            top_n=body.top_n,
            alpha_override=body.alpha,
        )
    except Exception as exc:
        logger.exception("Ensemble recommendation failed for user %d.", body.user_id)
        raise HTTPException(status_code=500, detail=f"Recommendation engine error: {exc}")

    if body.use_sentiment:
        try:
            rec_df = STATE.reranker.rerank(rec_df)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("Sentiment re-ranking failed (non-fatal): %s", exc)
            rec_df["sentiment_score"] = 0.0
            rec_df["final_score"] = rec_df["hybrid_score"]
    else:
        rec_df["sentiment_score"] = 0.0
        rec_df["final_score"] = rec_df["hybrid_score"]

    try:
        rec_df = STATE.explainer.explain(body.user_id, rec_df)  # type: ignore[union-attr]
    except Exception as exc:
        logger.warning("Explainability layer failed (non-fatal): %s", exc)
        rec_df["explanation"] = "Recommended by the CINEIQ hybrid engine."

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    results: list[MovieResult] = []
    for _, row in rec_df.iterrows():
        results.append(
            MovieResult(
                movie_id=int(row["movie_id"]),
                title=str(row.get("title", f"Movie {row['movie_id']}")),
                hybrid_score=round(_safe_float(row.get("hybrid_score")), 4),
                final_score=round(_safe_float(row.get("final_score")), 4),
                svd_norm=round(_safe_float(row.get("svd_norm")), 4),
                content_norm=round(_safe_float(row.get("content_norm")), 4),
                sentiment_score=round(_safe_float(row.get("sentiment_score")), 4),
                explanation=str(row.get("explanation", "")),
            )
        )

    logger.info(
        "Recommend: user=%d | top_n=%d | alpha=%.2f | sentiment=%s | latency=%.1fms",
        body.user_id, body.top_n, body.alpha, body.use_sentiment, latency_ms,
    )

    return RecommendResponse(
        user_id=body.user_id,
        alpha=body.alpha,
        top_n=body.top_n,
        use_sentiment=body.use_sentiment,
        latency_ms=latency_ms,
        recommendations=results,
    )


@app.post("/similar", response_model=SimilarResponse, tags=["Recommendations"])
def similar(body: SimilarRequest) -> SimilarResponse:
    """
    Return the top-N most content-similar movies to a seed movie.

    Uses cosine similarity over TF-IDF feature vectors (genres, cast,
    director, keywords).
    """
    _require_ready()

    movies_df = STATE.movies_df
    seed_row = movies_df[movies_df["movie_id"] == body.movie_id]  # type: ignore[index]
    if seed_row.empty:
        raise HTTPException(
            status_code=404, detail=f"movie_id={body.movie_id} not found in catalogue."
        )
    seed_title = str(seed_row.iloc[0]["title"])

    try:
        sim_df = STATE.ensemble.similar_movies(  # type: ignore[union-attr]
            movie_id=body.movie_id,
            top_n=body.top_n,
            exclude_ids=body.exclude_ids or [],
        )
    except Exception as exc:
        logger.exception("Similar-movie lookup failed for movie_id=%d.", body.movie_id)
        raise HTTPException(status_code=500, detail=f"Similarity engine error: {exc}")

    similar_results: list[SimilarMovieResult] = []
    for _, row in sim_df.iterrows():
        similar_results.append(
            SimilarMovieResult(
                movie_id=int(row["movie_id"]),
                title=str(row.get("title", f"Movie {row['movie_id']}")),
                content_score=round(_safe_float(row.get("content_score")), 4),
            )
        )

    logger.info(
        "Similar: seed_movie=%d ('%s') | top_n=%d | returned %d results.",
        body.movie_id, seed_title, body.top_n, len(similar_results),
    )

    return SimilarResponse(
        seed_movie_id=body.movie_id,
        seed_title=seed_title,
        top_n=body.top_n,
        similar_movies=similar_results,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
