"""
src/models/hybrid_ensemble.py
==============================
Hybrid Recommendation Engine that blends Collaborative Filtering (SVD)
and Content-Based Filtering scores via a weighted ensemble:

    Final_Score = (w_coll * SVD_score_normalised)
                + (w_cont * Content_score_normalised)

The weights w_coll + w_cont should sum to 1.0. Both score columns are
min-max normalised to [0, 1] before blending to ensure comparability.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models.collaborative import CollaborativeModel
from src.models.content_based import ContentBasedModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default ensemble weights
# ---------------------------------------------------------------------------

DEFAULT_ALPHA: float = 0.6   # collaborative weight
DEFAULT_BETA: float = 0.4    # content-based weight


def _min_max_normalise(values: list[float]) -> list[float]:
    """
    Min-max normalise a list of floats to [0, 1].

    Parameters
    ----------
    values : list[float]

    Returns
    -------
    list[float]
    """
    if not values:
        return values
    arr = np.array(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return ((arr - lo) / (hi - lo)).tolist()


class HybridEnsemble:
    """
    Weighted ensemble combining SVD-based collaborative filtering and
    TF-IDF content-based filtering.

    Parameters
    ----------
    collab_model : CollaborativeModel
    content_model : ContentBasedModel
    alpha : float
        Weight for collaborative filtering score.  Default 0.6.
    beta : float
        Weight for content-based score.  Default 0.4.
        alpha + beta must equal 1.0.
    """

    def __init__(
        self,
        collab_model: CollaborativeModel,
        content_model: ContentBasedModel,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
    ) -> None:
        if not np.isclose(alpha + beta, 1.0, atol=1e-6):
            raise ValueError(f"alpha + beta must equal 1.0. Got alpha={alpha}, beta={beta}.")
        self.collab_model = collab_model
        self.content_model = content_model
        self.alpha = alpha
        self.beta = beta
        logger.info(
            "HybridEnsemble initialised: alpha (collab)=%.2f | beta (content)=%.2f",
            alpha, beta,
        )

    # ------------------------------------------------------------------
    # Core recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: int,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        top_n: int = 10,
        alpha_override: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Generate top-N hybrid recommendations for a user.

        Parameters
        ----------
        user_id : int
        ratings_df : pd.DataFrame
            Historical ratings used to determine already-seen movies and
            to build the user content profile.
        movies_df : pd.DataFrame
            Full movie catalogue.
        top_n : int
        alpha_override : float, optional
            Override the instance-level alpha for this call only.

        Returns
        -------
        pd.DataFrame
            Columns: movie_id, title, svd_score, content_score,
                     svd_norm, content_norm, hybrid_score
            Sorted descending by hybrid_score.
        """
        alpha = alpha_override if alpha_override is not None else self.alpha
        beta = 1.0 - alpha

        user_seen = set(
            ratings_df[ratings_df["user_id"] == user_id]["movie_id"].astype(int).tolist()
        )
        all_movie_ids = movies_df["movie_id"].astype(int).tolist()
        candidates = [mid for mid in all_movie_ids if mid not in user_seen]

        if not candidates:
            logger.warning("User %d has rated all movies — returning global popularity.", user_id)
            candidates = all_movie_ids

        # --- Collaborative scores ---
        logger.debug("Computing SVD scores for user %d, %d candidates.", user_id, len(candidates))
        collab_list = self.collab_model.predict_top_n(user_id, candidates, top_n=len(candidates))
        collab_map: dict[int, float] = {r["movie_id"]: r["svd_score"] for r in collab_list}

        # --- Content-based scores ---
        logger.debug("Computing content scores for user %d.", user_id)
        content_list = self.content_model.get_user_content_scores(
            user_id, ratings_df, candidates, top_n=len(candidates)
        )
        content_map: dict[int, float] = {r["movie_id"]: r["content_score"] for r in content_list}

        # --- Build combined DataFrame ---
        records = []
        for mid in candidates:
            records.append(
                {
                    "movie_id": mid,
                    "svd_score": collab_map.get(mid, self.collab_model.global_mean),
                    "content_score": content_map.get(mid, 0.0),
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame(
                columns=["movie_id", "title", "svd_score", "content_score",
                         "svd_norm", "content_norm", "hybrid_score"]
            )

        # --- Normalise ---
        df["svd_norm"] = _min_max_normalise(df["svd_score"].tolist())
        df["content_norm"] = _min_max_normalise(df["content_score"].tolist())

        # --- Ensemble formula ---
        df["hybrid_score"] = alpha * df["svd_norm"] + beta * df["content_norm"]

        # --- Join movie metadata ---
        df = df.merge(movies_df[["movie_id", "title"]], on="movie_id", how="left")

        df = df.sort_values("hybrid_score", ascending=False).head(top_n).reset_index(drop=True)
        logger.info(
            "HybridEnsemble: generated %d recommendations for user %d (alpha=%.2f).",
            len(df), user_id, alpha,
        )
        return df

    # ------------------------------------------------------------------
    # Similar-movie lookup (content-only)
    # ------------------------------------------------------------------

    def similar_movies(
        self, movie_id: int, top_n: int = 10, exclude_ids: Optional[list[int]] = None
    ) -> pd.DataFrame:
        """
        Return the top-N most content-similar movies to a given movie.

        Parameters
        ----------
        movie_id : int
        top_n : int
        exclude_ids : list[int], optional

        Returns
        -------
        pd.DataFrame
            Columns: movie_id, content_score (sorted descending).
        """
        similar = self.content_model.get_similar_movies(movie_id, top_n=top_n, exclude_ids=exclude_ids)
        df = pd.DataFrame(similar)
        if df.empty:
            return df

        movies_df = self.content_model._movies_df
        if movies_df is not None:
            df = df.merge(movies_df[["movie_id", "title"]], on="movie_id", how="left")

        return df.sort_values("content_score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Dynamic weight adjustment (optional — e.g., for A/B testing)
    # ------------------------------------------------------------------

    def set_weights(self, alpha: float, beta: float) -> None:
        """
        Update ensemble weights.

        Parameters
        ----------
        alpha : float
            New collaborative weight.
        beta : float
            New content weight.
        """
        if not np.isclose(alpha + beta, 1.0, atol=1e-6):
            raise ValueError("alpha + beta must equal 1.0.")
        self.alpha = alpha
        self.beta = beta
        logger.info("HybridEnsemble weights updated: alpha=%.2f | beta=%.2f", alpha, beta)
