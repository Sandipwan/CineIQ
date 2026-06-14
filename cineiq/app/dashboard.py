"""
app/dashboard.py
=================
CINEIQ Interactive User Taste Dashboard — built with Streamlit + Plotly.

Features:
  - Sidebar: user ID picker, alpha weight slider, sentiment toggle.
  - Genre Radar Chart: user's taste profile across all genres.
  - Decade Bar Chart: distribution of rated movies by decade.
  - Director & Actor Affinity: top-N cards.
  - Recommendation Cards: styled cards showing title, scores, explanation.
  - Live calls to the FastAPI backend (if running) or direct in-process
    computation as fallback.

Run with:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.data_processing import (
    load_data,
    get_user_genre_affinity,
    get_user_decade_affinity,
)
from src.models.collaborative import CollaborativeModel
from src.models.content_based import ContentBasedModel
from src.models.hybrid_ensemble import HybridEnsemble
from src.models.sentiment import SentimentReRanker
from src.explainability import Explainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CINEIQ — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
  /* ── Global ── */
  body { background-color: #0e0e0e; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* ── Header banner ── */
  .cineiq-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
  }
  .cineiq-header h1 {
    color: #e94560;
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: 0.08em;
    margin: 0;
  }
  .cineiq-header p {
    color: #a8b2c1;
    font-size: 0.95rem;
    margin: 0.3rem 0 0;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    text-align: center;
  }
  .metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #e94560;
  }
  .metric-card .label {
    font-size: 0.78rem;
    color: #7a8899;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
  }

  /* ── Recommendation card ── */
  .rec-card {
    background: #16213e;
    border: 1px solid #2a3a5a;
    border-left: 4px solid #e94560;
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.9rem;
    transition: border-color 0.2s;
  }
  .rec-card:hover { border-left-color: #f5a623; }
  .rec-card .rank {
    font-size: 0.75rem;
    color: #e94560;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .rec-card .title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e8eaf0;
    margin: 0.2rem 0 0.3rem;
  }
  .rec-card .scores {
    display: flex; gap: 1rem; flex-wrap: wrap;
    margin-bottom: 0.55rem;
  }
  .score-badge {
    background: #0f3460;
    color: #7ec8e3;
    font-size: 0.72rem;
    padding: 0.2rem 0.55rem;
    border-radius: 20px;
    font-weight: 600;
  }
  .rec-card .explanation {
    font-size: 0.83rem;
    color: #9aa3b0;
    line-height: 1.5;
    font-style: italic;
  }

  /* ── Section titles ── */
  .section-title {
    color: #e8eaf0;
    font-size: 1.15rem;
    font-weight: 700;
    margin: 1.6rem 0 0.8rem;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #e94560;
    display: inline-block;
  }

  /* ── Sidebar ── */
  .css-1d391kg { background-color: #0a0a1a !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data + Model caching
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading data and training models — please wait...")
def load_all() -> tuple:
    movies_df, ratings_df, reviews_df = load_data(n_movies=500, n_users=300)

    collab = CollaborativeModel()
    collab.fit(ratings_df)

    content = ContentBasedModel()
    content.fit(movies_df)

    ensemble = HybridEnsemble(collab, content, alpha=0.6, beta=0.4)

    reranker = SentimentReRanker(backend="vader")
    reranker.fit(reviews_df)

    explainer = Explainer(movies_df, ratings_df)

    return movies_df, ratings_df, reviews_df, ensemble, reranker, explainer


movies_df, ratings_df, reviews_df, ensemble, reranker, explainer = load_all()

ALL_GENRES = sorted(
    {g for genres_str in movies_df["genres"].fillna("") for g in genres_str.split("|") if g.strip()}
)
ALL_USER_IDS = sorted(ratings_df["user_id"].unique().tolist())


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🎬 CINEIQ")
    st.markdown("---")
    st.markdown("### 🎛 User Settings")

    user_id = st.selectbox(
        "Select User ID",
        options=ALL_USER_IDS,
        index=0,
        help="Choose the user whose taste profile and recommendations to display.",
    )

    top_n = st.slider("Number of Recommendations", min_value=5, max_value=30, value=10, step=1)

    alpha = st.slider(
        "Ensemble Weight (α)",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="α controls the blend: 1.0 = pure collaborative, 0.0 = pure content-based.",
    )
    st.caption(f"Collaborative: {alpha:.0%}  |  Content: {1-alpha:.0%}")

    use_sentiment = st.toggle("Enable Sentiment Re-ranking", value=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "CINEIQ uses **SVD** collaborative filtering, **TF-IDF** content similarity, "
        "and **VADER** sentiment analysis to generate explainable movie recommendations.",
        unsafe_allow_html=False,
    )
    st.markdown("---")

    if st.button("🔄 Refresh Recommendations", use_container_width=True):
        st.cache_data.clear()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="cineiq-header">
      <h1>CINEIQ</h1>
      <p>Open &amp; Explainable Hybrid Movie Recommendation Engine</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------

user_ratings = ratings_df[ratings_df["user_id"] == user_id]
n_rated = len(user_ratings)
avg_rating = round(float(user_ratings["rating"].mean()), 2) if n_rated else 0.0
genre_affinity = get_user_genre_affinity(ratings_df, movies_df, user_id)
top_genre = max(genre_affinity, key=genre_affinity.get) if genre_affinity else "—"  # type: ignore

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f'<div class="metric-card"><div class="value">{n_rated}</div>'
        f'<div class="label">Movies Rated</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="metric-card"><div class="value">{avg_rating}</div>'
        f'<div class="label">Avg Rating</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="metric-card"><div class="value">{top_genre}</div>'
        f'<div class="label">Top Genre</div></div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f'<div class="metric-card"><div class="value">User {user_id}</div>'
        f'<div class="label">Active Profile</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Visualisation row
# ---------------------------------------------------------------------------

viz_col1, viz_col2 = st.columns([1, 1])

# ── Genre Radar Chart ──────────────────────────────────────────────────────
with viz_col1:
    st.markdown('<p class="section-title">🎯 Genre Taste Radar</p>', unsafe_allow_html=True)

    radar_genres = ALL_GENRES[:12]   # cap for readability
    radar_values = [round(genre_affinity.get(g, 0.0), 2) for g in radar_genres]
    radar_values_closed = radar_values + [radar_values[0]]
    radar_genres_closed = radar_genres + [radar_genres[0]]

    fig_radar = go.Figure(
        data=go.Scatterpolar(
            r=radar_values_closed,
            theta=radar_genres_closed,
            fill="toself",
            fillcolor="rgba(233, 69, 96, 0.20)",
            line=dict(color="#e94560", width=2),
            marker=dict(color="#e94560", size=6),
            name="Genre Affinity",
        )
    )
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#1a1a2e",
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                tickfont=dict(color="#7a8899", size=9),
                gridcolor="#2a2a4a",
                linecolor="#2a2a4a",
            ),
            angularaxis=dict(
                tickfont=dict(color="#c8d0da", size=10),
                gridcolor="#2a2a4a",
                linecolor="#2a2a4a",
            ),
        ),
        paper_bgcolor="#16213e",
        plot_bgcolor="#16213e",
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30),
        height=340,
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ── Decade Affinity Bar Chart ──────────────────────────────────────────────
with viz_col2:
    st.markdown('<p class="section-title">📅 Decade Preferences</p>', unsafe_allow_html=True)

    decade_aff = get_user_decade_affinity(ratings_df, movies_df, user_id)
    if decade_aff:
        decade_df = (
            pd.DataFrame(list(decade_aff.items()), columns=["decade", "avg_rating"])
            .sort_values("decade")
        )
        decade_df["decade_label"] = decade_df["decade"].astype(str) + "s"

        fig_bar = go.Figure(
            go.Bar(
                x=decade_df["decade_label"],
                y=decade_df["avg_rating"],
                marker=dict(
                    color=decade_df["avg_rating"],
                    colorscale=[[0, "#0f3460"], [0.5, "#e94560"], [1, "#f5a623"]],
                    showscale=False,
                ),
                text=decade_df["avg_rating"].round(2),
                textposition="outside",
                textfont=dict(color="#c8d0da", size=10),
            )
        )
        fig_bar.update_layout(
            paper_bgcolor="#16213e",
            plot_bgcolor="#16213e",
            xaxis=dict(
                tickfont=dict(color="#c8d0da"),
                gridcolor="#2a2a4a",
                linecolor="#2a2a4a",
            ),
            yaxis=dict(
                range=[0, 5.5],
                tickfont=dict(color="#c8d0da"),
                gridcolor="#2a2a4a",
                title="Avg Rating",
                title_font=dict(color="#7a8899"),
            ),
            margin=dict(l=40, r=20, t=20, b=30),
            height=340,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Not enough data to visualise decade preferences.")


# ---------------------------------------------------------------------------
# Director & Actor affinity section
# ---------------------------------------------------------------------------

aff_col1, aff_col2 = st.columns(2)

with aff_col1:
    st.markdown('<p class="section-title">🎬 Top Directors</p>', unsafe_allow_html=True)
    liked = user_ratings[user_ratings["rating"] >= 3.5].merge(
        movies_df[["movie_id", "director"]], on="movie_id", how="left"
    )
    if not liked.empty and "director" in liked.columns:
        dir_counts = (
            liked["director"].fillna("")
            .str.strip()
            .value_counts()
            .head(5)
            .reset_index()
        )
        dir_counts.columns = ["Director", "Films Rated"]
        st.dataframe(
            dir_counts,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No director data available.")


with aff_col2:
    st.markdown('<p class="section-title">⭐ Top Actors</p>', unsafe_allow_html=True)
    liked_cast = user_ratings[user_ratings["rating"] >= 3.5].merge(
        movies_df[["movie_id", "cast"]], on="movie_id", how="left"
    )
    if not liked_cast.empty and "cast" in liked_cast.columns:
        actor_counts: dict[str, int] = {}
        for cast_str in liked_cast["cast"].fillna(""):
            for a in cast_str.split("|"):
                a = a.strip()
                if a:
                    actor_counts[a] = actor_counts.get(a, 0) + 1
        actor_df = (
            pd.DataFrame(list(actor_counts.items()), columns=["Actor", "Films Rated"])
            .sort_values("Films Rated", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )
        st.dataframe(actor_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No cast data available.")


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-title">🍿 Your Recommendations</p>', unsafe_allow_html=True)

with st.spinner("Generating personalised recommendations..."):
    try:
        rec_df = ensemble.recommend(
            user_id=user_id,
            ratings_df=ratings_df,
            movies_df=movies_df,
            top_n=top_n,
            alpha_override=alpha,
        )

        if use_sentiment:
            rec_df = reranker.rerank(rec_df)
        else:
            rec_df["sentiment_score"] = 0.0
            rec_df["final_score"] = rec_df["hybrid_score"]

        rec_df = explainer.explain(user_id, rec_df)

    except Exception as exc:
        st.error(f"Recommendation engine error: {exc}")
        rec_df = pd.DataFrame()

if rec_df.empty:
    st.warning("No recommendations could be generated. Try adjusting the parameters.")
else:
    rec_left, rec_right = st.columns(2)

    for i, (_, row) in enumerate(rec_df.iterrows()):
        col = rec_left if i % 2 == 0 else rec_right
        rank_str = f"#{i + 1}"
        title = str(row.get("title", f"Movie {row['movie_id']}"))
        hybrid = round(float(row.get("hybrid_score", 0.0)), 3)
        final = round(float(row.get("final_score", 0.0)), 3)
        svd_n = round(float(row.get("svd_norm", 0.0)), 3)
        cnt_n = round(float(row.get("content_norm", 0.0)), 3)
        senti = round(float(row.get("sentiment_score", 0.0)), 3)
        explanation = str(row.get("explanation", ""))

        sentiment_emoji = "😊" if senti > 0.1 else ("😐" if senti >= -0.1 else "😕")

        with col:
            st.markdown(
                f"""
                <div class="rec-card">
                  <div class="rank">Rank {rank_str}</div>
                  <div class="title">{title}</div>
                  <div class="scores">
                    <span class="score-badge">🎯 Final {final}</span>
                    <span class="score-badge">🤝 Collab {svd_n}</span>
                    <span class="score-badge">📝 Content {cnt_n}</span>
                    <span class="score-badge">{sentiment_emoji} Sentiment {senti}</span>
                  </div>
                  <div class="explanation">💡 {explanation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Score distribution chart
# ---------------------------------------------------------------------------

if not rec_df.empty:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p class="section-title">📊 Score Decomposition</p>', unsafe_allow_html=True
    )

    chart_df = rec_df[["title", "svd_norm", "content_norm", "final_score"]].copy()
    chart_df["title"] = chart_df["title"].str[:30]

    fig_decomp = go.Figure()
    fig_decomp.add_trace(
        go.Bar(
            name="Collaborative (SVD)",
            x=chart_df["title"],
            y=chart_df["svd_norm"] * alpha,
            marker_color="#e94560",
        )
    )
    fig_decomp.add_trace(
        go.Bar(
            name="Content (TF-IDF)",
            x=chart_df["title"],
            y=chart_df["content_norm"] * (1 - alpha),
            marker_color="#0f3460",
        )
    )
    fig_decomp.update_layout(
        barmode="stack",
        paper_bgcolor="#16213e",
        plot_bgcolor="#16213e",
        legend=dict(font=dict(color="#c8d0da")),
        xaxis=dict(
            tickfont=dict(color="#c8d0da", size=9),
            tickangle=-35,
            gridcolor="#2a2a4a",
        ),
        yaxis=dict(
            tickfont=dict(color="#c8d0da"),
            gridcolor="#2a2a4a",
            title="Score Contribution",
            title_font=dict(color="#7a8899"),
        ),
        margin=dict(l=40, r=20, t=20, b=100),
        height=360,
    )
    st.plotly_chart(fig_decomp, use_container_width=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#7a8899; font-size:0.8rem;'>"
    "CINEIQ &nbsp;·&nbsp; Explainable Hybrid Movie Recommendation Engine &nbsp;·&nbsp; "
    "SVD + TF-IDF + VADER &nbsp;·&nbsp; Built with Streamlit &amp; Plotly"
    "</div>",
    unsafe_allow_html=True,
)
