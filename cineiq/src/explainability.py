"""
src/explainability.py
=====================
Explainability layer for CINEIQ.

Generates human-readable rationale strings for every recommendation by
combining three signals:

  1. Rule-based templates — drawn from feature values (genres, director,
     cast, shared keywords, decade affinity, rating similarity).
  2. Score decomposition tags — shows which component (collaborative /
     content / sentiment) drove the final score.
  3. LIME hooks — optional local linear explanation around the hybrid
     score function (requires the ``lime`` package).
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template bank
# ---------------------------------------------------------------------------

GENRE_TEMPLATES: list[str] = [
    "Recommended because you enjoy {genre} films.",
    "Matches your strong preference for {genre} movies.",
    "Based on your history with {genre} titles.",
]

DIRECTOR_TEMPLATES: list[str] = [
    "You have watched and enjoyed films by {director} before.",
    "Directed by {director}, who aligns with your taste profile.",
    "Your ratings suggest you appreciate {director}'s work.",
]

CAST_TEMPLATES: list[str] = [
    "Features {actor}, an actor you have rated highly in the past.",
    "Stars {actor}, consistent with your cast preferences.",
    "{actor} appears, who you tend to enjoy.",
]

COLLAB_TEMPLATES: list[str] = [
    "Users with similar tastes to yours gave this film high ratings.",
    "Highly rated by viewers who share your cinematic preferences.",
    "Strong collaborative signal: your peer group loved this movie.",
]

SENTIMENT_TEMPLATES: list[str] = [
    "Audience reviews are overwhelmingly positive.",
    "Strong real-world reception supports this recommendation.",
    "Critics and viewers alike have praised this film.",
]

DECADE_TEMPLATES: list[str] = [
    "Released in the {decade}s, a decade you frequently watch.",
    "From the {decade}s, which aligns with your era preferences.",
]

KEYWORD_TEMPLATES: list[str] = [
    "Shares the theme of '{keyword}' with movies you love.",
    "The '{keyword}' motif resonates with your watch history.",
]

GENERIC_TEMPLATES: list[str] = [
    "Selected by the hybrid engine as a strong match for your profile.",
    "A high-confidence recommendation across multiple signals.",
    "Ranked in your top picks based on combined ML scoring.",
]


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class Explainer:
    """
    Generates natural-language explanations for recommendations.

    Parameters
    ----------
    movies_df : pd.DataFrame
        Full movie catalogue with genre, cast, director, keyword columns.
    ratings_df : pd.DataFrame
        Historical ratings to infer user preferences.
    seed : int
        Random seed for template selection reproducibility.
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        seed: int = 42,
    ) -> None:
        self._movies_df = movies_df.copy()
        self._ratings_df = ratings_df.copy()
        self._rng = random.Random(seed)
        self._movie_lookup: dict[int, pd.Series] = {
            int(row["movie_id"]): row
            for _, row in movies_df.iterrows()
        }
        logger.info("Explainer initialised with %d movies.", len(self._movie_lookup))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def explain(
        self,
        user_id: int,
        rec_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Attach an 'explanation' column to a recommendations DataFrame.

        Parameters
        ----------
        user_id : int
        rec_df : pd.DataFrame
            Must have columns: movie_id, hybrid_score, svd_norm, content_norm.
            Optionally: sentiment_score, final_score.

        Returns
        -------
        pd.DataFrame
            Input frame with an 'explanation' column appended.
        """
        if rec_df.empty:
            return rec_df

        # Pre-compute user profile once
        user_genres = self._get_user_top_genres(user_id, top_k=3)
        user_directors = self._get_user_top_directors(user_id, top_k=2)
        user_actors = self._get_user_top_actors(user_id, top_k=3)
        user_decades = self._get_user_top_decades(user_id)

        explanations: list[str] = []
        for _, row in rec_df.iterrows():
            mid = int(row["movie_id"])
            explanation = self._build_explanation(
                user_id, mid, row,
                user_genres, user_directors, user_actors, user_decades,
            )
            explanations.append(explanation)

        rec_df = rec_df.copy()
        rec_df["explanation"] = explanations
        return rec_df

    # ------------------------------------------------------------------
    # Per-row explanation builder
    # ------------------------------------------------------------------

    def _build_explanation(
        self,
        user_id: int,
        movie_id: int,
        row: pd.Series,
        user_genres: list[str],
        user_directors: list[str],
        user_actors: list[str],
        user_decades: list[int],
    ) -> str:
        """
        Compose up to 3 explanation sentences for a single recommendation.
        """
        parts: list[str] = []
        movie = self._movie_lookup.get(movie_id)

        if movie is None:
            return self._rng.choice(GENERIC_TEMPLATES)

        movie_genres = [g.strip() for g in str(movie.get("genres", "")).split("|") if g.strip()]
        movie_director = str(movie.get("director", "")).strip()
        movie_cast = [a.strip() for a in str(movie.get("cast", "")).split("|") if a.strip()]
        movie_keywords = [k.strip() for k in str(movie.get("keywords", "")).split("|") if k.strip()]
        movie_decade = int(movie.get("decade", 2000))

        # 1. Genre match
        shared_genres = [g for g in movie_genres if g in user_genres]
        if shared_genres:
            genre = self._rng.choice(shared_genres)
            parts.append(self._rng.choice(GENRE_TEMPLATES).format(genre=genre))

        # 2. Director match
        if movie_director and movie_director in user_directors:
            parts.append(
                self._rng.choice(DIRECTOR_TEMPLATES).format(director=movie_director)
            )

        # 3. Cast match
        shared_actors = [a for a in movie_cast if a in user_actors]
        if shared_actors:
            actor = self._rng.choice(shared_actors)
            parts.append(self._rng.choice(CAST_TEMPLATES).format(actor=actor))

        # 4. Decade match
        if movie_decade in user_decades:
            parts.append(
                self._rng.choice(DECADE_TEMPLATES).format(decade=movie_decade)
            )

        # 5. Keyword match (pick one randomly)
        if movie_keywords:
            keyword = self._rng.choice(movie_keywords)
            parts.append(self._rng.choice(KEYWORD_TEMPLATES).format(keyword=keyword))

        # 6. Score-based tags
        svd_norm = float(row.get("svd_norm", 0.0))
        content_norm = float(row.get("content_norm", 0.0))
        sentiment = float(row.get("sentiment_score", 0.0))

        if svd_norm > 0.7 and content_norm < 0.4:
            parts.append(self._rng.choice(COLLAB_TEMPLATES))
        elif content_norm > 0.7 and svd_norm < 0.4:
            parts.append("Strong content similarity to movies already in your favourites.")
        elif svd_norm > 0.5 and content_norm > 0.5:
            parts.append("Both your taste profile and content similarity point strongly here.")

        if sentiment > 0.3:
            parts.append(self._rng.choice(SENTIMENT_TEMPLATES))
        elif sentiment < -0.3:
            parts.append("Note: audience reviews are mixed — recommended primarily on fit score.")

        # 7. Fallback
        if not parts:
            parts.append(self._rng.choice(GENERIC_TEMPLATES))

        # Deduplicate while preserving order; cap at 3 sentences
        seen: set[str] = set()
        unique_parts: list[str] = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                unique_parts.append(p)

        return " ".join(unique_parts[:3])

    # ------------------------------------------------------------------
    # User profile helpers
    # ------------------------------------------------------------------

    def _get_user_rated_movies(self, user_id: int, min_rating: float = 3.5) -> pd.DataFrame:
        """Return movies rated ≥ min_rating by user_id."""
        user_df = self._ratings_df[
            (self._ratings_df["user_id"] == user_id)
            & (self._ratings_df["rating"] >= min_rating)
        ]
        if user_df.empty:
            user_df = self._ratings_df[self._ratings_df["user_id"] == user_id]
        return user_df.merge(self._movies_df, on="movie_id", how="left")

    def _get_user_top_genres(self, user_id: int, top_k: int = 3) -> list[str]:
        merged = self._get_user_rated_movies(user_id)
        if merged.empty or "genres" not in merged.columns:
            return []
        genre_counts: dict[str, int] = {}
        for genres_str in merged["genres"].fillna(""):
            for g in genres_str.split("|"):
                g = g.strip()
                if g:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
        return sorted(genre_counts, key=genre_counts.get, reverse=True)[:top_k]  # type: ignore

    def _get_user_top_directors(self, user_id: int, top_k: int = 2) -> list[str]:
        merged = self._get_user_rated_movies(user_id)
        if merged.empty or "director" not in merged.columns:
            return []
        counts: dict[str, int] = {}
        for d in merged["director"].fillna(""):
            d = d.strip()
            if d:
                counts[d] = counts.get(d, 0) + 1
        return sorted(counts, key=counts.get, reverse=True)[:top_k]  # type: ignore

    def _get_user_top_actors(self, user_id: int, top_k: int = 3) -> list[str]:
        merged = self._get_user_rated_movies(user_id)
        if merged.empty or "cast" not in merged.columns:
            return []
        counts: dict[str, int] = {}
        for cast_str in merged["cast"].fillna(""):
            for a in cast_str.split("|"):
                a = a.strip()
                if a:
                    counts[a] = counts.get(a, 0) + 1
        return sorted(counts, key=counts.get, reverse=True)[:top_k]  # type: ignore

    def _get_user_top_decades(self, user_id: int) -> list[int]:
        merged = self._get_user_rated_movies(user_id)
        if merged.empty or "decade" not in merged.columns:
            return []
        counts: dict[int, int] = {}
        for d in merged["decade"].fillna(0).astype(int):
            counts[d] = counts.get(d, 0) + 1
        return sorted(counts, key=counts.get, reverse=True)[:3]  # type: ignore


# ---------------------------------------------------------------------------
# Optional LIME integration
# ---------------------------------------------------------------------------

def lime_explain(
    user_id: int,
    movie_id: int,
    hybrid_fn,
    feature_names: list[str],
    instance_features: np.ndarray,
    num_samples: int = 300,
    num_features: int = 6,
) -> Optional[dict]:
    """
    Generate a LIME explanation for a single (user, movie) prediction.

    Parameters
    ----------
    user_id : int
    movie_id : int
    hybrid_fn : callable
        A function f(X: np.ndarray) → np.ndarray that returns predicted
        scores for a batch of feature vectors.
    feature_names : list[str]
    instance_features : np.ndarray
        1-D feature vector for the specific movie.
    num_samples : int
        Number of LIME perturbation samples.
    num_features : int
        Number of top features to report.

    Returns
    -------
    dict or None
        {'feature': weight, ...} or None if lime is unavailable.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        logger.warning("lime package not installed — skipping LIME explanation.")
        return None

    explainer = LimeTabularExplainer(
        training_data=np.random.rand(num_samples, len(feature_names)),
        feature_names=feature_names,
        mode="regression",
        verbose=False,
        random_state=42,
    )

    try:
        exp = explainer.explain_instance(
            data_row=instance_features,
            predict_fn=hybrid_fn,
            num_features=num_features,
        )
        return dict(exp.as_list())
    except Exception as exc:
        logger.error("LIME explanation failed for movie_id=%d: %s", movie_id, exc)
        return None
