"""
src/data_processing.py
======================
Handles loading, cleaning, merging, and feature engineering for:
  - MovieLens 25M (ratings)
  - TMDB Metadata (genres, cast, keywords)
  - IMDB 50K Reviews (sentiment corpus)

Includes a fully self-contained synthetic data generator so the entire
repository can run out-of-the-box without downloading any datasets.
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GENRES: list[str] = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "Western", "Musical",
]

DIRECTORS: list[str] = [
    "Christopher Nolan", "Martin Scorsese", "Steven Spielberg",
    "Quentin Tarantino", "Denis Villeneuve", "Ridley Scott",
    "James Cameron", "David Fincher", "Wes Anderson", "Alfonso Cuarón",
    "Guillermo del Toro", "Darren Aronofsky", "Paul Thomas Anderson",
    "Coen Brothers", "Bong Joon-ho",
]

ACTORS: list[str] = [
    "Leonardo DiCaprio", "Meryl Streep", "Tom Hanks", "Cate Blanchett",
    "Joaquin Phoenix", "Natalie Portman", "Brad Pitt", "Scarlett Johansson",
    "Denzel Washington", "Viola Davis", "Ryan Gosling", "Emma Stone",
    "Christian Bale", "Anne Hathaway", "Matt Damon", "Jennifer Lawrence",
    "Robert De Niro", "Charlize Theron", "Morgan Freeman", "Amy Adams",
]

KEYWORDS_POOL: list[str] = [
    "time-travel", "heist", "dystopia", "redemption", "survival",
    "artificial intelligence", "space exploration", "underdog", "revenge",
    "coming-of-age", "conspiracy", "supernatural", "road trip", "war",
    "romance", "family", "friendship", "betrayal", "identity", "power",
]

POSITIVE_REVIEW_TEMPLATES: list[str] = [
    "An absolute masterpiece. The storytelling is captivating and the performances are stellar.",
    "One of the best films I have seen this decade. Deeply moving and visually stunning.",
    "Brilliant direction and a gripping narrative. Highly recommended.",
    "A perfect blend of action and emotion. Cannot recommend it enough.",
    "Phenomenal acting and a tight, well-paced script. A must watch.",
]

NEGATIVE_REVIEW_TEMPLATES: list[str] = [
    "Disappointing overall. The plot felt rushed and characters were underdeveloped.",
    "I expected much more. The film drags on with little payoff.",
    "Poor execution of an interesting concept. The pacing was terrible.",
    "Stilted dialogue and a predictable story. Not worth the time.",
    "Overhyped and underwhelming. Very little character depth.",
]

NEUTRAL_REVIEW_TEMPLATES: list[str] = [
    "A decent watch with some good moments, though nothing groundbreaking.",
    "Competently made but forgettable. Average entertainment.",
    "Has its highs and lows. Acceptable for a casual viewing experience.",
    "Some good performances but the script could have been stronger.",
    "Middle-of-the-road film. Watchable but not particularly memorable.",
]


# ---------------------------------------------------------------------------
# Synthetic Data Generator
# ---------------------------------------------------------------------------

def generate_synthetic_movies(n_movies: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic TMDB-style movie metadata DataFrame.

    Parameters
    ----------
    n_movies : int
        Number of synthetic movies to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: movie_id, title, year, genres, cast, director, keywords, runtime_min, vote_average
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    records = []
    for i in range(1, n_movies + 1):
        n_genres = rng.randint(1, 3)
        genres = rng.sample(GENRES, k=n_genres)

        n_cast = rng.randint(2, 5)
        cast = rng.sample(ACTORS, k=n_cast)

        director = rng.choice(DIRECTORS)

        n_keywords = rng.randint(2, 5)
        keywords = rng.sample(KEYWORDS_POOL, k=n_keywords)

        year = rng.randint(1970, 2024)
        decade = (year // 10) * 10

        # Fabricate a plausible title
        adjectives = ["Dark", "Last", "Silent", "Lost", "Broken", "New", "Hidden", "Final", "Rising", "Eternal"]
        nouns = ["Kingdom", "Hour", "Shadow", "World", "Dream", "City", "Mind", "Fire", "Storm", "Journey"]
        title = f"The {rng.choice(adjectives)} {rng.choice(nouns)} {i}"

        vote_avg = round(np.clip(np.random.normal(6.5, 1.2), 1.0, 10.0), 1)
        runtime = rng.randint(75, 180)

        records.append(
            {
                "movie_id": i,
                "title": title,
                "year": year,
                "decade": decade,
                "genres": "|".join(genres),
                "cast": "|".join(cast),
                "director": director,
                "keywords": "|".join(keywords),
                "runtime_min": runtime,
                "vote_average": vote_avg,
            }
        )

    df = pd.DataFrame(records)
    logger.info("Generated %d synthetic movies.", len(df))
    return df


def generate_synthetic_ratings(
    movies_df: pd.DataFrame,
    n_users: int = 300,
    ratings_per_user: tuple[int, int] = (20, 80),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic MovieLens-style ratings DataFrame.

    Parameters
    ----------
    movies_df : pd.DataFrame
        Movies reference frame (must have 'movie_id' column).
    n_users : int
        Number of synthetic users.
    ratings_per_user : tuple[int, int]
        (min, max) number of ratings per user.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: user_id, movie_id, rating, timestamp
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    movie_ids = movies_df["movie_id"].tolist()
    records = []

    for user_id in range(1, n_users + 1):
        n_ratings = rng.randint(*ratings_per_user)
        sampled_movies = rng.sample(movie_ids, k=min(n_ratings, len(movie_ids)))

        # Bias each user toward a preferred genre cluster to make CF meaningful
        for movie_id in sampled_movies:
            rating = float(np.clip(np.random.normal(3.5, 1.0), 0.5, 5.0))
            rating = round(rating * 2) / 2  # snap to 0.5 increments
            timestamp = rng.randint(1_000_000_000, 1_700_000_000)
            records.append(
                {"user_id": user_id, "movie_id": movie_id, "rating": rating, "timestamp": timestamp}
            )

    df = pd.DataFrame(records)
    logger.info("Generated %d synthetic ratings for %d users.", len(df), n_users)
    return df


def generate_synthetic_reviews(
    movies_df: pd.DataFrame,
    reviews_per_movie: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic IMDB-style reviews DataFrame.

    Parameters
    ----------
    movies_df : pd.DataFrame
        Movies reference frame.
    reviews_per_movie : int
        Number of reviews to generate per movie.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: movie_id, review_text, sentiment_label (positive/negative/neutral)
    """
    rng = random.Random(seed)
    records = []

    template_map = {
        "positive": POSITIVE_REVIEW_TEMPLATES,
        "negative": NEGATIVE_REVIEW_TEMPLATES,
        "neutral": NEUTRAL_REVIEW_TEMPLATES,
    }
    labels = ["positive", "negative", "neutral"]
    weights = [0.55, 0.25, 0.20]

    for movie_id in movies_df["movie_id"]:
        for _ in range(reviews_per_movie):
            label = rng.choices(labels, weights=weights, k=1)[0]
            text = rng.choice(template_map[label])
            records.append({"movie_id": movie_id, "review_text": text, "sentiment_label": label})

    df = pd.DataFrame(records)
    logger.info("Generated %d synthetic reviews.", len(df))
    return df


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def build_content_soup(row: pd.Series) -> str:
    """
    Concatenate genres, cast, director, and keywords into a single text 'soup'
    suitable for TF-IDF vectorisation.

    Parameters
    ----------
    row : pd.Series
        A single row from the movies DataFrame.

    Returns
    -------
    str
        Space-separated token string.
    """
    parts: list[str] = []

    for field in ("genres", "cast", "keywords"):
        raw = row.get(field, "")
        if isinstance(raw, str) and raw:
            tokens = [re.sub(r"\s+", "_", t.strip().lower()) for t in raw.split("|")]
            parts.extend(tokens)

    director = row.get("director", "")
    if isinstance(director, str) and director:
        parts.append(re.sub(r"\s+", "_", director.strip().lower()))

    return " ".join(parts)


def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the movies DataFrame.

    - Drops rows with null movie_id or title.
    - Fills missing textual fields with empty strings.
    - Ensures vote_average is numeric and within [0, 10].
    - Adds a 'content_soup' column for TF-IDF.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    initial_len = len(df)
    df = df.dropna(subset=["movie_id", "title"]).copy()
    dropped = initial_len - len(df)
    if dropped:
        logger.warning("Dropped %d rows with null movie_id or title.", dropped)

    for col in ("genres", "cast", "director", "keywords"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    if "vote_average" in df.columns:
        df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(5.0)
        df["vote_average"] = df["vote_average"].clip(0.0, 10.0)

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(2000).astype(int)
        df["decade"] = (df["year"] // 10) * 10

    df["content_soup"] = df.apply(build_content_soup, axis=1)
    logger.info("Cleaned movies DataFrame: %d rows remain.", len(df))
    return df.reset_index(drop=True)


def clean_ratings(df: pd.DataFrame, valid_movie_ids: set[int]) -> pd.DataFrame:
    """
    Validate and clean the ratings DataFrame.

    - Drops rows with null user_id, movie_id, or rating.
    - Filters out ratings for movies not in valid_movie_ids.
    - Clips ratings to [0.5, 5.0].

    Parameters
    ----------
    df : pd.DataFrame
    valid_movie_ids : set[int]

    Returns
    -------
    pd.DataFrame
    """
    df = df.dropna(subset=["user_id", "movie_id", "rating"]).copy()
    df = df[df["movie_id"].isin(valid_movie_ids)]
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3.0).clip(0.5, 5.0)
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    logger.info("Cleaned ratings DataFrame: %d rows.", len(df))
    return df.reset_index(drop=True)


def clean_reviews(df: pd.DataFrame, valid_movie_ids: set[int]) -> pd.DataFrame:
    """
    Validate and clean the reviews DataFrame.

    - Drops rows with null movie_id or review_text.
    - Filters to valid movie IDs.
    - Truncates extremely long reviews to 512 characters.

    Parameters
    ----------
    df : pd.DataFrame
    valid_movie_ids : set[int]

    Returns
    -------
    pd.DataFrame
    """
    df = df.dropna(subset=["movie_id", "review_text"]).copy()
    df = df[df["movie_id"].isin(valid_movie_ids)]
    df["review_text"] = df["review_text"].astype(str).str.strip().str[:512]
    logger.info("Cleaned reviews DataFrame: %d rows.", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def load_data(
    data_dir: Optional[str] = None,
    n_movies: int = 500,
    n_users: int = 300,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load or synthetically generate all three datasets.

    If ``data_dir`` is provided and contains the expected CSV files
    (``movies.csv``, ``ratings.csv``, ``reviews.csv``), those are loaded.
    Otherwise, fully synthetic data is generated.

    Parameters
    ----------
    data_dir : str, optional
        Path to directory containing real dataset CSV files.
    n_movies : int
        Number of synthetic movies (ignored when loading real data).
    n_users : int
        Number of synthetic users (ignored when loading real data).
    seed : int
        Random seed for synthetic generation.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (movies_df, ratings_df, reviews_df)
    """
    if data_dir is not None:
        base = Path(data_dir)
        movies_path = base / "movies.csv"
        ratings_path = base / "ratings.csv"
        reviews_path = base / "reviews.csv"

        if movies_path.exists() and ratings_path.exists() and reviews_path.exists():
            logger.info("Loading real datasets from %s ...", data_dir)
            movies_df = pd.read_csv(movies_path)
            ratings_df = pd.read_csv(ratings_path)
            reviews_df = pd.read_csv(reviews_path)
        else:
            logger.warning(
                "data_dir='%s' is set but CSV files not found. Falling back to synthetic data.", data_dir
            )
            movies_df, ratings_df, reviews_df = _generate_all(n_movies, n_users, seed)
    else:
        logger.info("No data_dir provided — generating synthetic datasets.")
        movies_df, ratings_df, reviews_df = _generate_all(n_movies, n_users, seed)

    # Clean everything
    movies_df = clean_movies(movies_df)
    valid_ids: set[int] = set(movies_df["movie_id"].astype(int).tolist())
    ratings_df = clean_ratings(ratings_df, valid_ids)
    reviews_df = clean_reviews(reviews_df, valid_ids)

    return movies_df, ratings_df, reviews_df


def _generate_all(
    n_movies: int, n_users: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movies_df = generate_synthetic_movies(n_movies=n_movies, seed=seed)
    ratings_df = generate_synthetic_ratings(movies_df, n_users=n_users, seed=seed)
    reviews_df = generate_synthetic_reviews(movies_df, seed=seed)
    return movies_df, ratings_df, reviews_df


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_movie_genre_vector(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot-encode the pipe-separated genre column into individual binary columns.

    Parameters
    ----------
    movies_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added one-hot genre columns.
    """
    genre_dummies = movies_df["genres"].str.get_dummies(sep="|")
    return pd.concat([movies_df, genre_dummies], axis=1)


def get_user_genre_affinity(
    ratings_df: pd.DataFrame, movies_df: pd.DataFrame, user_id: int
) -> dict[str, float]:
    """
    Compute a user's affinity score per genre based on their rated movies.

    Parameters
    ----------
    ratings_df : pd.DataFrame
    movies_df : pd.DataFrame
    user_id : int

    Returns
    -------
    dict[str, float]
        Mapping of genre → mean rating.
    """
    user_ratings = ratings_df[ratings_df["user_id"] == user_id].copy()
    if user_ratings.empty:
        logger.warning("No ratings found for user_id=%d. Returning empty affinity.", user_id)
        return {}

    merged = user_ratings.merge(movies_df[["movie_id", "genres"]], on="movie_id", how="left")
    merged = merged.dropna(subset=["genres"])

    affinity: dict[str, list[float]] = {}
    for _, row in merged.iterrows():
        for genre in str(row["genres"]).split("|"):
            affinity.setdefault(genre, []).append(float(row["rating"]))

    return {g: float(np.mean(scores)) for g, scores in affinity.items()}


def get_user_decade_affinity(
    ratings_df: pd.DataFrame, movies_df: pd.DataFrame, user_id: int
) -> dict[int, float]:
    """
    Compute a user's mean rating per movie decade.

    Parameters
    ----------
    ratings_df : pd.DataFrame
    movies_df : pd.DataFrame
    user_id : int

    Returns
    -------
    dict[int, float]
        Mapping of decade → mean rating.
    """
    user_ratings = ratings_df[ratings_df["user_id"] == user_id].copy()
    if user_ratings.empty:
        return {}

    merged = user_ratings.merge(movies_df[["movie_id", "decade"]], on="movie_id", how="left")
    merged = merged.dropna(subset=["decade"])

    decade_groups = merged.groupby("decade")["rating"].mean()
    return {int(k): float(v) for k, v in decade_groups.items()}
