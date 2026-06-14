"""
src/tracking.py
================
MLflow experiment tracking pipeline for CINEIQ.

Provides:
  - run_training_experiment() — trains collaborative + content models,
    logs params / metrics / artefacts to MLflow.
  - log_recommendation_event() — lightweight per-request logging.
  - load_best_run() — retrieves the best run's artefacts from the
    MLflow tracking server.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.models.collaborative import CollaborativeModel
from src.models.content_based import ContentBasedModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENT_NAME: str = "CINEIQ_RecommendationEngine"
TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "mlruns")


def _setup_mlflow(tracking_uri: str = TRACKING_URI) -> None:
    """Configure MLflow tracking URI and ensure the experiment exists."""
    mlflow.set_tracking_uri(tracking_uri)
    existing = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if existing is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
        logger.info("MLflow experiment '%s' created.", EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)


# ---------------------------------------------------------------------------
# Training experiment runner
# ---------------------------------------------------------------------------

def run_training_experiment(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    collab_params: Optional[dict] = None,
    tfidf_params: Optional[dict] = None,
    cv_folds: int = 3,
    alpha: float = 0.6,
    sentiment_backend: str = "vader",
    run_name: Optional[str] = None,
) -> str:
    """
    Train both models, evaluate, and log everything to MLflow.

    Parameters
    ----------
    movies_df : pd.DataFrame
    ratings_df : pd.DataFrame
    collab_params : dict, optional
        SVD hyper-parameters.
    tfidf_params : dict, optional
        TF-IDF hyper-parameters.
    cv_folds : int
        Number of cross-validation folds for the collaborative model.
    alpha : float
        Ensemble weight for the collaborative model.
    sentiment_backend : str
        'vader' or 'distilbert'.
    run_name : str, optional
        Custom name for the MLflow run.

    Returns
    -------
    str
        The MLflow run_id.
    """
    _setup_mlflow()

    with mlflow.start_run(run_name=run_name or "cineiq_training_run") as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: %s", run_id)

        # --- Log dataset stats ---
        mlflow.log_param("n_movies", len(movies_df))
        mlflow.log_param("n_users", ratings_df["user_id"].nunique())
        mlflow.log_param("n_ratings", len(ratings_df))
        mlflow.log_param("ensemble_alpha", alpha)
        mlflow.log_param("ensemble_beta", round(1.0 - alpha, 4))
        mlflow.log_param("sentiment_backend", sentiment_backend)
        mlflow.log_param("cv_folds", cv_folds)

        # --- Collaborative model ---
        logger.info("Training CollaborativeModel...")
        collab = CollaborativeModel(svd_params=collab_params)

        svd_p = collab.svd_params
        mlflow.log_param("svd_n_factors", svd_p.get("n_factors", 100))
        mlflow.log_param("svd_n_epochs", svd_p.get("n_epochs", 20))
        mlflow.log_param("svd_lr_all", svd_p.get("lr_all", 0.005))
        mlflow.log_param("svd_reg_all", svd_p.get("reg_all", 0.02))

        collab.fit(ratings_df)

        eval_metrics = collab.evaluate(ratings_df, cv_folds=cv_folds)
        mlflow.log_metric("collab_rmse", eval_metrics["rmse"])
        mlflow.log_metric("collab_mae", eval_metrics["mae"])
        logger.info("Collaborative metrics — RMSE: %.4f | MAE: %.4f",
                    eval_metrics["rmse"], eval_metrics["mae"])

        # --- Content model ---
        logger.info("Training ContentBasedModel...")
        content = ContentBasedModel(tfidf_params=tfidf_params)
        content.fit(movies_df)

        tf_p = content.tfidf_params
        mlflow.log_param("tfidf_min_df", tf_p.get("min_df", 1))
        mlflow.log_param("tfidf_max_df", tf_p.get("max_df", 0.95))
        mlflow.log_param("tfidf_ngram_range", str(tf_p.get("ngram_range", (1, 2))))
        mlflow.log_param("tfidf_sublinear_tf", tf_p.get("sublinear_tf", True))
        mlflow.log_metric("content_vocab_size", len(content._vectorizer.vocabulary_))  # type: ignore
        mlflow.log_metric("content_n_movies", content.n_movies)

        # --- Coverage metric ---
        rated_movies = set(ratings_df["movie_id"].astype(int).tolist())
        catalogue_movies = set(movies_df["movie_id"].astype(int).tolist())
        catalogue_coverage = len(rated_movies & catalogue_movies) / max(len(catalogue_movies), 1)
        mlflow.log_metric("catalogue_coverage", round(catalogue_coverage, 4))

        # --- Rating distribution stats ---
        mlflow.log_metric("rating_mean", round(float(ratings_df["rating"].mean()), 4))
        mlflow.log_metric("rating_std", round(float(ratings_df["rating"].std()), 4))
        mlflow.log_metric("rating_sparsity",
                          round(1.0 - catalogue_coverage * len(ratings_df) /
                                max(ratings_df["user_id"].nunique() * len(catalogue_movies), 1), 4))

        # --- Save model artefacts ---
        with tempfile.TemporaryDirectory() as tmpdir:
            collab_path = str(Path(tmpdir) / "svd_model.pkl")
            content_path = str(Path(tmpdir) / "content_model.pkl")

            collab.save(collab_path)
            content.save(content_path)

            mlflow.log_artifact(collab_path, artifact_path="models")
            mlflow.log_artifact(content_path, artifact_path="models")

        logger.info("MLflow run %s completed successfully.", run_id)
        return run_id


# ---------------------------------------------------------------------------
# Recommendation event logging
# ---------------------------------------------------------------------------

def log_recommendation_event(
    run_id: str,
    user_id: int,
    recommendations: list[dict],
    alpha: float,
    latency_ms: float,
) -> None:
    """
    Log a single recommendation request event to an existing MLflow run.

    Parameters
    ----------
    run_id : str
    user_id : int
    recommendations : list[dict]
        Each dict should have at least 'movie_id' and 'final_score'.
    alpha : float
    latency_ms : float
        End-to-end recommendation latency in milliseconds.
    """
    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("recommendation_latency_ms", latency_ms)
            mlflow.log_metric("n_recommendations_returned", len(recommendations))
            avg_score = float(
                np.mean([r.get("final_score", r.get("hybrid_score", 0.0)) for r in recommendations])
            ) if recommendations else 0.0
            mlflow.log_metric("avg_recommendation_score", round(avg_score, 4))
    except Exception as exc:
        logger.warning("MLflow event logging failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Best run loader
# ---------------------------------------------------------------------------

def load_best_run_metrics(metric: str = "collab_rmse") -> Optional[dict[str, Any]]:
    """
    Find the best MLflow run in the experiment by a given metric (lower is better).

    Parameters
    ----------
    metric : str
        The MLflow metric name to optimise.

    Returns
    -------
    dict or None
        {'run_id': str, 'metrics': dict, 'params': dict} or None.
    """
    _setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            logger.warning("Experiment '%s' not found.", EXPERIMENT_NAME)
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} ASC"],
            max_results=1,
        )
        if not runs:
            return None

        best = runs[0]
        return {
            "run_id": best.info.run_id,
            "metrics": dict(best.data.metrics),
            "params": dict(best.data.params),
        }
    except Exception as exc:
        logger.error("Failed to query MLflow for best run: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Quick report
# ---------------------------------------------------------------------------

def print_experiment_summary() -> None:
    """Print a tabular summary of all tracked runs to stdout."""
    _setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"No experiment named '{EXPERIMENT_NAME}' found.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=20,
    )
    if not runs:
        print("No runs recorded yet.")
        return

    rows = []
    for r in runs:
        rows.append(
            {
                "run_id": r.info.run_id[:8],
                "run_name": r.info.run_name or "",
                "status": r.info.status,
                "rmse": round(r.data.metrics.get("collab_rmse", float("nan")), 4),
                "mae": round(r.data.metrics.get("collab_mae", float("nan")), 4),
                "alpha": r.data.params.get("ensemble_alpha", "?"),
                "n_movies": r.data.params.get("n_movies", "?"),
            }
        )

    summary_df = pd.DataFrame(rows)
    print("\n=== CINEIQ MLflow Experiment Summary ===")
    print(summary_df.to_string(index=False))
    print()
