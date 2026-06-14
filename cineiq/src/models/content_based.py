"""
src/models/content_based.py
============================
Content-Based Filtering using TF-IDF vectorisation over a movie's
'content soup' (genres, cast, director, keywords) and cosine similarity.

Features:
  - Builds a TF-IDF matrix from pre-engineered content_soup column.
  - Computes pairwise cosine similarity between movies.
  - Recommends N most similar movies given a seed movie_id.
  - Returns content-based affinity scores for a given user based on
    their rating history.
  - Cold-start: falls back to global popularity when a movie or user
    is not found.
"""

from __future__ import annotations

import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_TFIDF_PARAMS: dict = {
    "min_df": 1,
    "max_df": 0.95,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "analyzer": "word",
}


class ContentBasedModel:
    """
    TF-IDF + Cosine Similarity content-based recommender.

    Parameters
    ----------
    tfidf_params : dict, optional
        Parameters forwarded to ``sklearn.TfidfVectorizer``.
    """

    def __init__(self, tfidf_params: Optional[dict] = None) -> None:
        self.tfidf_params: dict = tfidf_params or DEFAULT_TFIDF_PARAMS.copy()
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None           # scipy sparse (n_movies × vocab)
        self._cosine_sim: Optional[np.ndarray] = None   # dense (n_movies × n_movies)
        self._movie_index: dict[int, int] = {}          # movie_id → row index
        self._index_movie: dict[int, int] = {}          # row index → movie_id
        self._movies_df: Optional[pd.DataFrame] = None
        self._global_popularity: dict[int, float] = {}  # movie_id → score
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, movies_df: pd.DataFrame) -> "ContentBasedModel":
        """
        Build TF-IDF matrix and cosine similarity from ``movies_df``.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Must contain columns: movie_id, content_soup.
            Optionally: vote_average for popularity fallback.

        Returns
        -------
        ContentBasedModel
            Self, for chaining.
        """
        if movies_df.empty:
            raise ValueError("movies_df must not be empty.")

        if "content_soup" not in movies_df.columns:
            raise KeyError("movies_df must have a 'content_soup' column. Run clean_movies() first.")

        self._movies_df = movies_df.reset_index(drop=True).copy()

        # Build lookup indices
        self._movie_index = {
            int(row["movie_id"]): idx
            for idx, row in self._movies_df.iterrows()
        }
        self._index_movie = {v: k for k, v in self._movie_index.items()}

        # TF-IDF fit
        self._vectorizer = TfidfVectorizer(**self.tfidf_params)
        self._tfidf_matrix = self._vectorizer.fit_transform(
            self._movies_df["content_soup"].fillna("")
        )
        logger.info(
            "TF-IDF matrix built: shape=%s, vocab_size=%d.",
            self._tfidf_matrix.shape,
            len(self._vectorizer.vocabulary_),
        )

        # Dense cosine similarity — feasible for ≤ 10k movies; for larger sets
        # we'd compute on-the-fly per query.
        n_movies = self._tfidf_matrix.shape[0]
        if n_movies <= 5000:
            self._cosine_sim = cosine_similarity(self._tfidf_matrix, self._tfidf_matrix)
            logger.info("Cosine similarity matrix computed: shape=%s.", self._cosine_sim.shape)
        else:
            self._cosine_sim = None
            logger.info(
                "Dataset > 5000 movies; cosine similarity will be computed on-the-fly per query."
            )

        # Popularity fallback
        if "vote_average" in self._movies_df.columns:
            self._global_popularity = (
                self._movies_df.set_index("movie_id")["vote_average"].to_dict()
            )

        self._trained = True
        return self

    # ------------------------------------------------------------------
    # Similar movie lookup
    # ------------------------------------------------------------------

    def get_similar_movies(
        self, movie_id: int, top_n: int = 10, exclude_ids: Optional[list[int]] = None
    ) -> list[dict]:
        """
        Return the ``top_n`` most content-similar movies to ``movie_id``.

        Parameters
        ----------
        movie_id : int
        top_n : int
        exclude_ids : list[int], optional
            Movie IDs to exclude (e.g., movies the user already rated).

        Returns
        -------
        list[dict]
            [{'movie_id': int, 'content_score': float}, ...] sorted descending.
        """
        if not self._trained:
            raise RuntimeError("Call fit() before get_similar_movies().")

        exclude = set(exclude_ids or []) | {movie_id}

        if movie_id not in self._movie_index:
            logger.warning("movie_id=%d not in index — returning popularity fallback.", movie_id)
            return self._popularity_fallback(exclude, top_n)

        idx = self._movie_index[movie_id]

        if self._cosine_sim is not None:
            sim_scores = list(enumerate(self._cosine_sim[idx]))
        else:
            # On-the-fly computation
            query_vec = self._tfidf_matrix[idx]
            sims = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
            sim_scores = list(enumerate(sims))

        sim_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, score in sim_scores:
            mid = self._index_movie[i]
            if mid in exclude:
                continue
            results.append({"movie_id": mid, "content_score": float(score)})
            if len(results) >= top_n:
                break

        return results

    # ------------------------------------------------------------------
    # User-level content affinity (using rating history)
    # ------------------------------------------------------------------

    def get_user_content_scores(
        self,
        user_id: int,
        ratings_df: pd.DataFrame,
        candidate_movie_ids: list[int],
        top_n: int = 10,
    ) -> list[dict]:
        """
        Compute content-based scores for candidate movies by averaging the
        TF-IDF vectors of the user's highest-rated movies.

        Parameters
        ----------
        user_id : int
        ratings_df : pd.DataFrame
            Must have columns: user_id, movie_id, rating.
        candidate_movie_ids : list[int]
        top_n : int

        Returns
        -------
        list[dict]
            [{'movie_id': int, 'content_score': float}, ...] sorted descending.
        """
        if not self._trained:
            raise RuntimeError("Call fit() before get_user_content_scores().")

        user_ratings = ratings_df[ratings_df["user_id"] == user_id].copy()

        if user_ratings.empty:
            logger.warning(
                "No ratings found for user_id=%d — content cold-start fallback.", user_id
            )
            return self._popularity_fallback(set(), top_n)

        # Focus on well-rated movies (≥ 3.5) for the user profile vector
        liked = user_ratings[user_ratings["rating"] >= 3.5].copy()
        if liked.empty:
            liked = user_ratings.nlargest(5, "rating")

        liked_ids = [mid for mid in liked["movie_id"].tolist() if mid in self._movie_index]

        if not liked_ids:
            return self._popularity_fallback(set(), top_n)

        # Build user profile vector as rating-weighted mean of TF-IDF rows
        liked_with_ratings = liked[liked["movie_id"].isin(liked_ids)]
        indices = [self._movie_index[mid] for mid in liked_ids]
        weights = liked_with_ratings.set_index("movie_id").loc[liked_ids, "rating"].values
        weights = weights / weights.sum()   # normalise

        user_profile = np.zeros((1, self._tfidf_matrix.shape[1]))
        for idx, weight in zip(indices, weights):
            user_profile += weight * self._tfidf_matrix[idx].toarray()

        # Score each candidate
        exclude_ids = set(user_ratings["movie_id"].tolist())
        results = []
        for mid in candidate_movie_ids:
            if mid not in self._movie_index:
                results.append({"movie_id": mid, "content_score": 0.0})
                continue
            if mid in exclude_ids:
                continue
            cand_vec = self._tfidf_matrix[self._movie_index[mid]]
            score = float(cosine_similarity(user_profile, cand_vec.toarray())[0][0])
            results.append({"movie_id": mid, "content_score": score})

        results.sort(key=lambda x: x["content_score"], reverse=True)
        return results[:top_n]

    # ------------------------------------------------------------------
    # Cold-start fallback
    # ------------------------------------------------------------------

    def _popularity_fallback(
        self, exclude: set[int], top_n: int
    ) -> list[dict]:
        """
        Return movies sorted by vote_average for cold-start situations.

        Parameters
        ----------
        exclude : set[int]
        top_n : int

        Returns
        -------
        list[dict]
        """
        if self._movies_df is None:
            return []

        ranked = self._movies_df[~self._movies_df["movie_id"].isin(exclude)].copy()
        if "vote_average" in ranked.columns:
            ranked = ranked.sort_values("vote_average", ascending=False)
        else:
            ranked = ranked.sample(frac=1, random_state=0)

        results = []
        for _, row in ranked.head(top_n).iterrows():
            results.append(
                {
                    "movie_id": int(row["movie_id"]),
                    "content_score": float(row.get("vote_average", 5.0)) / 10.0,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str = "models/content_model.pkl") -> None:
        """Persist model artifacts to disk."""
        from pathlib import Path
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self._vectorizer,
                "tfidf_matrix": self._tfidf_matrix,
                "cosine_sim": self._cosine_sim,
                "movie_index": self._movie_index,
                "index_movie": self._index_movie,
                "movies_df": self._movies_df,
                "global_popularity": self._global_popularity,
                "tfidf_params": self.tfidf_params,
            },
            target,
        )
        logger.info("ContentBasedModel saved to %s.", target)

    def load(self, path: str = "models/content_model.pkl") -> "ContentBasedModel":
        """Load model artifacts from disk."""
        from pathlib import Path
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Content model file not found: {source}")
        payload = joblib.load(source)
        self._vectorizer = payload["vectorizer"]
        self._tfidf_matrix = payload["tfidf_matrix"]
        self._cosine_sim = payload["cosine_sim"]
        self._movie_index = payload["movie_index"]
        self._index_movie = payload["index_movie"]
        self._movies_df = payload["movies_df"]
        self._global_popularity = payload["global_popularity"]
        self.tfidf_params = payload["tfidf_params"]
        self._trained = True
        logger.info("ContentBasedModel loaded from %s.", source)
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def n_movies(self) -> int:
        return len(self._movie_index)
