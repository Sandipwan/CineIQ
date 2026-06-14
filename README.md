# 🎬 CINEIQ — Explainable Hybrid Movie Recommendation Engine

CINEIQ is a production-ready, open, and explainable movie recommendation system that combines
hybrid ML strategies with a sentiment-aware re-ranker and an interactive user taste dashboard.

---

## 🧠 Architecture Overview

```
MovieLens 25M + TMDB Metadata + IMDB Reviews
           │
           ▼
   [ data_processing.py ]  ← cleans, merges, generates synthetic data
           │
     ┌─────┴──────┐
     ▼            ▼
[collaborative] [content_based]   ← SVD + TF-IDF/Cosine
     └─────┬──────┘
           ▼
   [hybrid_ensemble.py]   ← Final_Score = w_coll*SVD + w_cont*Content
           │
           ▼
    [sentiment.py]        ← VADER / DistilBERT re-ranking boost
           │
           ▼
  [explainability.py]     ← Human-readable rationale per recommendation
           │
     ┌─────┴──────┐
     ▼            ▼
 [api/app.py]  [app/dashboard.py]
 FastAPI        Streamlit + Plotly
```

---

## 🚀 Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI server
```bash
uvicorn api.app:app --reload --port 8000
```

### 3. Run the Streamlit dashboard
```bash
streamlit run app/dashboard.py
```

### 4. API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/recommend` | POST | Get hybrid recommendations for a user |
| `/similar` | POST | Get similar movies to a given movie |

#### Example `/recommend` request:
```json
{
  "user_id": 42,
  "top_n": 10,
  "alpha": 0.6,
  "use_sentiment": true
}
```

#### Example `/similar` request:
```json
{
  "movie_id": 1,
  "top_n": 5
}
```

---

## 📦 Module Descriptions

| Module | Purpose |
|---|---|
| `src/data_processing.py` | Synthetic + real data loading, cleaning, feature engineering |
| `src/models/collaborative.py` | SVD-based Matrix Factorization via Surprise |
| `src/models/content_based.py` | TF-IDF vectorizer + Cosine Similarity matrix |
| `src/models/hybrid_ensemble.py` | Weighted ensemble blending both models |
| `src/models/sentiment.py` | VADER/DistilBERT sentiment scoring & re-ranking |
| `src/explainability.py` | Rule-based explanation templates + LIME hooks |
| `src/tracking.py` | MLflow experiment logging pipeline |
| `api/app.py` | FastAPI REST API |
| `app/dashboard.py` | Streamlit interactive dashboard |

---

## 🧪 MLflow Experiment Tracking
```bash
mlflow ui --port 5000
```
Navigate to `http://localhost:5000` to view logged runs, metrics (RMSE, MAE), and parameters.

---

## 📊 Datasets

- **MovieLens 25M** — https://grouplens.org/datasets/movielens/25m/
- **TMDB Metadata** — Kaggle: `tmdb-movie-metadata`
- **IMDB 50K Reviews** — Kaggle: `imdb-dataset-of-50k-movie-reviews`

> **Note:** The system ships with a built-in synthetic data generator in `src/data_processing.py`
> so the entire repo runs out-of-the-box without downloading any datasets.

---

## ⚙️ Configuration

Tune ensemble weights and sentiment boost in `src/models/hybrid_ensemble.py`:
```python
ALPHA = 0.6          # Weight for collaborative filtering score
BETA  = 0.4          # Weight for content-based score
SENTIMENT_BOOST = 0.15  # Max sentiment re-ranking adjustment
```

---

## 👥 Authors
- **ARNAV GUPTA** — 7982390939
