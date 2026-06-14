"""
src/models/collaborative.py
============================
SVD-based Collaborative Filtering using the Surprise library.

Features:
  - Trains an SVD model on MovieLens-style rating data.
  - Evaluates with RMSE and MAE via cross-validation.
  - Saves / loads trained models via joblib.
  - Handles cold-start (unknown user or item) by falling back to
    global popularity averages.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

try:
    from surprise import SVD, Dataset, Reader, accuracy
    from surprise.model_selection import cross_validate, train_test_split
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "scikit-surprise not installed. CollaborativeModel will return popularity fallbacks only."
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_SVD_PARAMS: dict = {
    "n_factors": 100,
    "n_epochs": 20,
    "lr_all": 0.005,
    "reg_all": 0.02,
    "random_state": 42,
    "verbose": False,
}


class CollaborativeModel:
    """
    Wrapper around Surprise's SVD algorithm with utility methods for training,
    evaluation, prediction, and serialisation.

    Parameters
    ----------
    svd_params : dict, optional
        Hyper-parameters passed directly to ``surprise.SVD``.
    model_path : str, optional
        File path for saving / loading the trained model.
    """

    def __init__(
        self,
        svd_params: Optional[dict] = None,
        model_path: str = "models/svd_model.pkl",
    ) -> None:
        self.svd_params: dict = svd_params or DEFAULT_SVD_PARAMS.copy()
        self.model_path: Path = Path(model_path)
        self._algo: Optional["SVD"] = None  # type: ignore[name-defined]
        self._trainset = None
        self._global_mean: float = 3.5
        self._popularity: dict[int, float] = {}   # movie_id → weighted score
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, ratings_df: pd.DataFrame) -> "CollaborativeModel":
        """
        Train the SVD model on a ratings DataFrame.

        Parameters
        ----------
        ratings_df : pd.DataFrame
            Must have columns: user_id, movie_id, rating.

        Returns
        -------
        CollaborativeModel
            Self, for chaining.
        """
        if ratings_df.empty:
            logger.error("Ratings DataFrame is empty — cannot train collaborative model.")
            raise ValueError("ratings_df must not be empty.")

        # Compute global mean and popularity for cold-start fallback
        self._global_mean = float(ratings_df["rating"].mean())
        pop = (
            ratings_df.groupby("movie_id")["rating"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_rating", "count": "n_ratings"})
        )
        # Bayesian average to smooth popularity scores
        C = pop["n_ratings"].quantile(0.60)
        m = self._global_mean
        pop["bayesian_avg"] = (pop["n_ratings"] * pop["avg_rating"] + C * m) / (pop["n_ratings"] + C)
        self._popularity = pop["bayesian_avg"].to_dict()

        if not SURPRISE_AVAILABLE:
            logger.warning("Surprise not available — using popularity fallback only.")
            self._trained = False
            return self

        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings_df[["user_id", "movie_id", "rating"]], reader)

        self._algo = SVD(**self.svd_params)
        full_trainset = data.build_full_trainset()
        self._algo.fit(full_trainset)
        self._trainset = full_trainset
        self._trained = True
        logger.info("SVD model trained on %d ratings.", len(ratings_df))
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, ratings_df: pd.DataFrame, cv_folds: int = 3
    ) -> dict[str, float]:
        """
        Run k-fold cross-validation and return mean RMSE and MAE.

        Parameters
        ----------
        ratings_df : pd.DataFrame
        cv_folds : int

        Returns
        -------
        dict[str, float]
            Keys: 'rmse', 'mae'.
        """
        if not SURPRISE_AVAILABLE:
            logger.warning("Surprise unavailable — skipping evaluation.")
            return {"rmse": float("nan"), "mae": float("nan")}

        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings_df[["user_id", "movie_id", "rating"]], reader)
        algo = SVD(**self.svd_params)

        results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=cv_folds, verbose=False)
        rmse = float(np.mean(results["test_rmse"]))
        mae = float(np.mean(results["test_mae"]))
        logger.info("CV evaluation — RMSE: %.4f | MAE: %.4f (folds=%d)", rmse, mae, cv_folds)
        return {"rmse": rmse, "mae": mae}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict a user's rating for a single movie.

        Falls back to the Bayesian popularity average if the user or movie
        is unknown (cold-start), or if Surprise is unavailable.

        Parameters
        ----------
        user_id : int
        movie_id : int

        Returns
        -------
        float
            Predicted rating in [0.5, 5.0].
        """
        if self._trained and self._algo is not None:
            pred = self._algo.predict(str(user_id), str(movie_id))
            if pred.details.get("was_impossible", False):
                return self._cold_start_score(movie_id)
            return float(np.clip(pred.est, 0.5, 5.0))
        return self._cold_start_score(movie_id)

    def predict_top_n(
        self, user_id: int, candidate_movie_ids: list[int], top_n: int = 10
    ) -> list[dict]:
        """
        Predict ratings for a set of candidate movies and return the top-N.

        Parameters
        ----------
        user_id : int
        candidate_movie_ids : list[int]
        top_n : int

        Returns
        -------
        list[dict]
            Sorted list of {'movie_id': int, 'svd_score': float}, descending.
        """
        if not candidate_movie_ids:
            return []

        predictions = []
        for mid in candidate_movie_ids:
            score = self.predict_rating(user_id, mid)
            predictions.append({"movie_id": mid, "svd_score": score})

        predictions.sort(key=lambda x: x["svd_score"], reverse=True)
        return predictions[:top_n]

    # ------------------------------------------------------------------
    # Cold-start
    # ------------------------------------------------------------------

    def _cold_start_score(self, movie_id: int) -> float:
        """
        Return a Bayesian popularity score for unknown users / items.

        Parameters
        ----------
        movie_id : int

        Returns
        -------
        float
        """
        score = self._popularity.get(movie_id, self._global_mean)
        return float(np.clip(score, 0.5, 5.0))

    def get_popularity_ranking(self, movie_ids: list[int], top_n: int = 10) -> list[dict]:
        """
        Return movies ranked purely by Bayesian popularity (for cold-start users).

        Parameters
        ----------
        movie_ids : list[int]
        top_n : int

        Returns
        -------
        list[dict]
        """
        scored = [{"movie_id": mid, "svd_score": self._cold_start_score(mid)} for mid in movie_ids]
        scored.sort(key=lambda x: x["svd_score"], reverse=True)
        return scored[:top_n]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Persist the trained model to disk."""
        target = Path(path) if path else self.model_path
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "algo": self._algo,
                "global_mean": self._global_mean,
                "popularity": self._popularity,
                "svd_params": self.svd_params,
                "trained": self._trained,
            },
            target,
        )
        logger.info("CollaborativeModel saved to %s.", target)

    def load(self, path: Optional[str] = None) -> "CollaborativeModel":
        """Load a persisted model from disk."""
        source = Path(path) if path else self.model_path
        if not source.exists():
            raise FileNotFoundError(f"Model file not found: {source}")
        payload = joblib.load(source)
        self._algo = payload["algo"]
        self._global_mean = payload["global_mean"]
        self._popularity = payload["popularity"]
        self.svd_params = payload["svd_params"]
        self._trained = payload["trained"]
        logger.info("CollaborativeModel loaded from %s.", source)
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def global_mean(self) -> float:
        return self._global_mean
