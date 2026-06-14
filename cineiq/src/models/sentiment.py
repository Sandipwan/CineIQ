"""
src/models/sentiment.py
========================
Sentiment-Aware Re-Ranker.

Uses two sentiment backends (configurable at runtime):
  1. VADER   — lightweight lexicon-based, fast, no GPU required.
  2. DistilBERT — HuggingFace transformer, more accurate, requires torch.

Workflow:
  1. Aggregate per-movie review texts.
  2. Score each review with the chosen backend.
  3. Compute a composite sentiment_score ∈ [-1, 1] per movie.
  4. Apply a bounded boost to the hybrid score:
         final_score = hybrid_score + SENTIMENT_BOOST * sentiment_score

This preserves the relative ranking from the ensemble while nudging
well-received titles upward and critically panned ones downward.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Maximum absolute adjustment to the hybrid score from sentiment
SENTIMENT_BOOST: float = 0.15

# ---------------------------------------------------------------------------
# VADER Backend
# ---------------------------------------------------------------------------

class VADERSentimentScorer:
    """
    Wrapper around vaderSentiment for fast lexicon-based scoring.
    """

    def __init__(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER SentimentIntensityAnalyzer initialised.")
        except ImportError:
            self._analyzer = None
            logger.warning("vaderSentiment not installed. VADER scorer will return 0.0.")

    def score(self, text: str) -> float:
        """
        Return compound VADER score ∈ [-1, 1] for a text.

        Parameters
        ----------
        text : str

        Returns
        -------
        float
        """
        if self._analyzer is None:
            return 0.0
        try:
            scores = self._analyzer.polarity_scores(str(text))
            return float(scores["compound"])
        except Exception as exc:
            logger.debug("VADER scoring error: %s", exc)
            return 0.0

    def score_batch(self, texts: list[str]) -> list[float]:
        """Score a batch of texts."""
        return [self.score(t) for t in texts]


# ---------------------------------------------------------------------------
# DistilBERT Backend
# ---------------------------------------------------------------------------

class DistilBERTSentimentScorer:
    """
    HuggingFace DistilBERT-based sentiment scorer.

    Uses ``distilbert-base-uncased-finetuned-sst-2-english`` which outputs
    POSITIVE / NEGATIVE labels with confidence scores.
    """

    MODEL_NAME: str = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self, device: str = "cpu", batch_size: int = 32) -> None:
        self._pipeline = None
        self._device = device
        self._batch_size = batch_size
        self._loaded = False
        self._load_model()

    def _load_model(self) -> None:
        """Lazy-load the HuggingFace pipeline."""
        try:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self.MODEL_NAME,
                device=-1 if self._device == "cpu" else 0,
                truncation=True,
                max_length=512,
            )
            self._loaded = True
            logger.info("DistilBERT sentiment pipeline loaded: %s.", self.MODEL_NAME)
        except Exception as exc:
            logger.warning(
                "Could not load DistilBERT pipeline (%s). Falling back to VADER.", exc
            )
            self._loaded = False

    def score(self, text: str) -> float:
        """
        Return a compound score ∈ [-1, 1] for a text.

        Maps POSITIVE → [0, 1] and NEGATIVE → [-1, 0].

        Parameters
        ----------
        text : str

        Returns
        -------
        float
        """
        if not self._loaded or self._pipeline is None:
            return 0.0
        try:
            result = self._pipeline(str(text)[:512])[0]
            confidence = float(result["score"])
            label = result["label"].upper()
            return confidence if label == "POSITIVE" else -confidence
        except Exception as exc:
            logger.debug("DistilBERT scoring error: %s", exc)
            return 0.0

    def score_batch(self, texts: list[str]) -> list[float]:
        """Score texts in mini-batches for throughput."""
        if not self._loaded or self._pipeline is None:
            return [0.0] * len(texts)

        truncated = [str(t)[:512] for t in texts]
        scores: list[float] = []

        for i in range(0, len(truncated), self._batch_size):
            batch = truncated[i : i + self._batch_size]
            try:
                results = self._pipeline(batch)
                for r in results:
                    c = float(r["score"])
                    lbl = r["label"].upper()
                    scores.append(c if lbl == "POSITIVE" else -c)
            except Exception as exc:
                logger.warning("DistilBERT batch error at index %d: %s", i, exc)
                scores.extend([0.0] * len(batch))

        return scores


# ---------------------------------------------------------------------------
# Sentiment Re-Ranker
# ---------------------------------------------------------------------------

class SentimentReRanker:
    """
    Aggregates review sentiment per movie and applies a bounded re-ranking
    boost to hybrid recommendation scores.

    Parameters
    ----------
    backend : Literal['vader', 'distilbert']
        Sentiment scoring backend to use.
    sentiment_boost : float
        Maximum absolute adjustment added to hybrid scores (default: 0.15).
    device : str
        Device for DistilBERT ('cpu' or 'cuda').
    """

    def __init__(
        self,
        backend: Literal["vader", "distilbert"] = "vader",
        sentiment_boost: float = SENTIMENT_BOOST,
        device: str = "cpu",
    ) -> None:
        self.backend_name = backend
        self.sentiment_boost = sentiment_boost
        self._movie_sentiment: dict[int, float] = {}   # movie_id → compound score

        if backend == "distilbert":
            self._scorer: VADERSentimentScorer | DistilBERTSentimentScorer = \
                DistilBERTSentimentScorer(device=device)
            # Fallback to VADER if DistilBERT failed to load
            if not self._scorer._loaded:  # type: ignore[attr-defined]
                logger.info("Falling back to VADER.")
                self._scorer = VADERSentimentScorer()
                self.backend_name = "vader"
        else:
            self._scorer = VADERSentimentScorer()

        logger.info("SentimentReRanker initialised with backend='%s'.", self.backend_name)

    # ------------------------------------------------------------------
    # Pre-compute sentiment per movie
    # ------------------------------------------------------------------

    def fit(self, reviews_df: pd.DataFrame) -> "SentimentReRanker":
        """
        Score all reviews and aggregate per movie.

        Parameters
        ----------
        reviews_df : pd.DataFrame
            Must have columns: movie_id, review_text.

        Returns
        -------
        SentimentReRanker
            Self, for chaining.
        """
        if reviews_df.empty:
            logger.warning("Empty reviews_df — sentiment scores will default to 0.0.")
            return self

        texts = reviews_df["review_text"].fillna("").tolist()
        logger.info(
            "Scoring %d reviews with backend='%s' ...", len(texts), self.backend_name
        )
        raw_scores = self._scorer.score_batch(texts)
        reviews_with_scores = reviews_df[["movie_id"]].copy()
        reviews_with_scores["raw_score"] = raw_scores

        agg = reviews_with_scores.groupby("movie_id")["raw_score"].mean()
        self._movie_sentiment = agg.to_dict()

        logger.info(
            "Sentiment computed for %d movies. Mean=%.4f, Std=%.4f.",
            len(self._movie_sentiment),
            float(np.mean(list(self._movie_sentiment.values()))),
            float(np.std(list(self._movie_sentiment.values()))),
        )
        return self

    # ------------------------------------------------------------------
    # Re-rank
    # ------------------------------------------------------------------

    def rerank(self, recommendations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sentiment boost to a recommendations DataFrame.

        Parameters
        ----------
        recommendations_df : pd.DataFrame
            Must have columns: movie_id, hybrid_score.
            The function adds columns: sentiment_score, final_score.

        Returns
        -------
        pd.DataFrame
            Same frame with sentiment_score and final_score columns,
            sorted descending by final_score.
        """
        if recommendations_df.empty:
            return recommendations_df

        if "hybrid_score" not in recommendations_df.columns:
            raise KeyError("recommendations_df must have a 'hybrid_score' column.")

        df = recommendations_df.copy()
        df["sentiment_score"] = df["movie_id"].apply(
            lambda mid: self._movie_sentiment.get(int(mid), 0.0)
        )
        df["final_score"] = (
            df["hybrid_score"] + self.sentiment_boost * df["sentiment_score"]
        )
        df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
        logger.debug("Re-ranked %d recommendations via sentiment.", len(df))
        return df

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_movie_sentiment(self, movie_id: int) -> float:
        """Return the pre-computed sentiment score for a movie, or 0.0."""
        return self._movie_sentiment.get(movie_id, 0.0)

    def score_text(self, text: str) -> float:
        """Score a single review text on the fly."""
        return self._scorer.score(text)
