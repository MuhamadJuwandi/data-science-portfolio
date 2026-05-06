"""
MAL Hybrid Recommender - Streamlit Cloud Entry Point.

This file serves as the main module for Streamlit Cloud deployment.
It sets up the correct paths and delegates to the dashboard application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# PATH SETUP — Critical for Streamlit Cloud where this runs from a subfolder
# ---------------------------------------------------------------------------
# Get the directory where THIS file lives (the project root for mal-recommender-hybrid)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hybrid_model import HybridRecommender

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MAL Hybrid Recommender — Insight Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(100, 100, 255, 0.15);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        text-align: center;
    }
    .metric-card h3 {
        color: #a0a0ff;
        font-size: 14px;
        font-weight: 500;
        margin: 0 0 6px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
    }

    /* Anime card */
    .anime-card {
        background: linear-gradient(145deg, #1e1e30 0%, #262640 100%);
        border: 1px solid rgba(120, 120, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .anime-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(100, 100, 255, 0.15);
    }
    .anime-title {
        font-size: 15px;
        font-weight: 600;
        color: #e0e0ff;
        margin-top: 10px;
        height: 44px;
        overflow: hidden;
        line-height: 1.4;
    }
    .anime-score {
        color: #f0ad4e;
        font-weight: 700;
        font-size: 14px;
    }
    .anime-genre {
        color: #888;
        font-size: 12px;
        margin-top: 4px;
    }

    /* Section header */
    .section-header {
        background: linear-gradient(90deg, rgba(100,100,255,0.1) 0%, transparent 100%);
        border-left: 3px solid #6666ff;
        padding: 12px 20px;
        border-radius: 0 8px 8px 0;
        margin: 24px 0 16px 0;
    }
    .section-header h2 {
        color: #c0c0ff;
        margin: 0;
        font-size: 20px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="🔧 Loading recommender model…")
def load_recommender():
    """Load and initialise the hybrid recommender."""
    data_path = os.path.join(PROJECT_ROOT, "data", "processed")

    required_files = ["anime_processed.parquet", "train.parquet"]
    for f in required_files:
        fpath = os.path.join(data_path, f)
        if not os.path.exists(fpath):
            st.error(f"❌ Missing required file: `{f}` in `{data_path}`")
            return None

    recommender = HybridRecommender(data_path=data_path)
    recommender.load_data()
    recommender.build_cf_model()
    recommender.build_cb_model()
    return recommender


@st.cache_data(show_spinner=False)
def compute_eda_stats(_recommender):
    """Pre-compute EDA statistics so they don't re-run on every interaction."""
    stats = {}

    anime_df = _recommender.anime_df
    train_df = _recommender.train_df

    # Basic counts
    stats["n_anime"] = len(anime_df)
    stats["n_users"] = train_df["username"].nunique()
    stats["n_interactions"] = len(train_df)
    stats["avg_rating"] = train_df["my_score"].mean()
    stats["median_rating"] = train_df["my_score"].median()

    # Rating distribution
    stats["rating_counts"] = train_df["my_score"].value_counts().sort_index()

    # Genre stats
    genre_series = anime_df["genre"].dropna().str.split(", ").explode()
    stats["genre_counts"] = genre_series.value_counts().head(15)

    # Anime type distribution
    if "type" in anime_df.columns:
        stats["type_counts"] = anime_df["type"].value_counts()
    else:
        stats["type_counts"] = pd.Series(dtype=int)

    # Top rated anime (with minimum voters)
    well_known = anime_df[anime_df["scored_by"] >= 1000].copy() if "scored_by" in anime_df.columns else anime_df.copy()
    stats["top_rated"] = well_known.sort_values("score", ascending=False).head(10)

    # Most popular (most rated)
    if "scored_by" in anime_df.columns:
        stats["most_popular"] = anime_df.sort_values("scored_by", ascending=False).head(10)
    else:
        stats["most_popular"] = anime_df.head(10)

    # Ratings per user distribution
    user_counts = train_df.groupby("username").size()
    stats["user_activity_mean"] = user_counts.mean()
    stats["user_activity_median"] = user_counts.median()
    stats["user_activity_hist"] = user_counts

    # Ratings per anime distribution
    anime_counts = train_df.groupby("anime_id").size()
    stats["anime_popularity_hist"] = anime_counts

    # Score vs popularity correlation
    merged = train_df.groupby("anime_id").agg(
        avg_user_score=("my_score", "mean"),
        n_ratings=("my_score", "count"),
    ).reset_index()
    merged = merged.merge(anime_df[["anime_id", "score", "title"]], on="anime_id", how="left")
    stats["score_pop_df"] = merged

    return stats


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
def render_sidebar(recommender):
    """Render sidebar controls."""
    st.sidebar.markdown("## 🎬 MAL Recommender")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### 🔍 Get Recommendations")

    # Pick example users for convenience
    example_users = recommender.train_df["username"].value_counts().head(10).index.tolist()
    selected_user = st.sidebar.selectbox(
        "Select a user (or type below)",
        options=["— Custom —"] + example_users,
    )

    if selected_user == "— Custom —":
        user_id = st.sidebar.text_input("Enter Username", value=example_users[0])
    else:
        user_id = selected_user

    n_recs = st.sidebar.slider("Number of recommendations", 5, 20, 10)

    get_recs = st.sidebar.button("🎯 Get Recommendations", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This dashboard uses a **Hybrid Recommender** combining:\n"
        "- **Collaborative Filtering** (TruncatedSVD)\n"
        "- **Content-Based Filtering** (TF-IDF on genres)\n\n"
        "Data sourced from MyAnimeList."
    )

    return user_id, n_recs, get_recs


# ---------------------------------------------------------------------------
# RECOMMENDATION DISPLAY
# ---------------------------------------------------------------------------
def display_recommendations(recommender, user_id, n_recs):
    """Generate and display anime recommendations."""
    with st.spinner("🧠 Generating personalised recommendations…"):
        try:
            recs = recommender.recommend(user_id, n_recommendations=n_recs)
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            return

    if recs is None or recs.empty:
        st.warning("No recommendations found for this user.")
        return

    # Check if user is known or cold-start
    is_known = user_id in recommender.user_mapper
    if is_known:
        st.markdown(
            f'<div class="section-header"><h2>🎯 Top {n_recs} for <em>{user_id}</em></h2></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="section-header"><h2>🌟 Popular Picks (new user: {user_id})</h2></div>',
            unsafe_allow_html=True,
        )
        st.info("This user has no history — showing popular anime instead (cold-start fallback).")

    # Grid display
    cols_per_row = 5
    for row_start in range(0, len(recs), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= len(recs):
                break
            row = recs.iloc[idx]
            with col:
                img_url = row.get("image_url", "")
                if not img_url or pd.isna(img_url):
                    img_url = "https://via.placeholder.com/225x318?text=No+Image"
                st.image(img_url, use_container_width=True)
                title = row.get("title", "Unknown")
                st.markdown(f"<div class='anime-title'>{title}</div>", unsafe_allow_html=True)

                score_val = row.get("score", 0)
                hybrid_val = row.get("hybrid_score", None)
                if hybrid_val is not None and not pd.isna(hybrid_val):
                    st.markdown(f"⭐ {score_val:.1f} &nbsp;|&nbsp; 🎯 {hybrid_val:.2f}")
                else:
                    st.markdown(f"⭐ {score_val:.1f}")

                genre = row.get("genre", "")
                if genre and not pd.isna(genre):
                    short_genre = ", ".join(genre.split(", ")[:2])
                    st.markdown(f"<div class='anime-genre'>{short_genre}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# EDA PANELS
# ---------------------------------------------------------------------------
def render_eda(stats):
    """Render all EDA insight panels."""

    # ── Metric cards ──
    st.markdown('<div class="section-header"><h2>📊 Dataset Overview</h2></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><h3>Anime Titles</h3><p class="value">{stats["n_anime"]:,}</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><h3>Active Users</h3><p class="value">{stats["n_users"]:,}</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><h3>Interactions</h3><p class="value">{stats["n_interactions"]:,}</p></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><h3>Avg Rating</h3><p class="value">{stats["avg_rating"]:.2f}</p></div>',
            unsafe_allow_html=True,
        )

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Rating Distribution",
        "🎭 Genre Analysis",
        "🏆 Top Anime",
        "👥 User Activity",
    ])

    # TAB 1 — Rating Distribution
    with tab1:
        st.subheader("Distribution of User Ratings")
        rc = stats["rating_counts"]
        fig = px.bar(
            x=rc.index,
            y=rc.values,
            labels={"x": "Rating (1-10)", "y": "Count"},
            color=rc.values,
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(dtick=1),
            coloraxis_showscale=False,
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Mean Rating", f"{stats['avg_rating']:.2f}")
        with col_b:
            st.metric("Median Rating", f"{stats['median_rating']:.1f}")

        st.markdown(
            "> **Insight:** The distribution reveals user rating tendencies. "
            "A left-skewed distribution (peak at 7-8) is typical for MAL, indicating users "
            "tend to rate anime they liked and drop those they didn't."
        )

    # TAB 2 — Genre Analysis
    with tab2:
        st.subheader("Top 15 Anime Genres")
        gc = stats["genre_counts"]
        fig = px.bar(
            x=gc.values,
            y=gc.index,
            orientation="h",
            labels={"x": "Number of Titles", "y": "Genre"},
            color=gc.values,
            color_continuous_scale="Plasma",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Type distribution pie
        if not stats["type_counts"].empty:
            st.subheader("Anime Type Distribution")
            tc = stats["type_counts"]
            fig2 = px.pie(
                names=tc.index,
                values=tc.values,
                color_discrete_sequence=px.colors.sequential.Plasma_r,
                hole=0.4,
            )
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            "> **Insight:** Comedy, Action, and Adventure dominate MAL catalogues, reflecting "
            "the mainstream appeal of these genres in the anime industry."
        )

    # TAB 3 — Top Anime
    with tab3:
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("🏅 Highest Rated")
            top = stats["top_rated"][["title", "score", "genre"]].reset_index(drop=True)
            top.index += 1
            st.dataframe(top, use_container_width=True, height=420)

        with col_right:
            st.subheader("🔥 Most Popular (by # ratings)")
            if "scored_by" in stats["most_popular"].columns:
                pop = stats["most_popular"][["title", "scored_by", "score"]].reset_index(drop=True)
                pop.index += 1
                pop = pop.rename(columns={"scored_by": "voters"})
                st.dataframe(pop, use_container_width=True, height=420)

        # Score vs Popularity scatter
        st.subheader("Score vs Popularity")
        spdf = stats["score_pop_df"]
        fig = px.scatter(
            spdf,
            x="n_ratings",
            y="avg_user_score",
            hover_data=["title"],
            opacity=0.45,
            color="avg_user_score",
            color_continuous_scale="Turbo",
            labels={"n_ratings": "Number of User Ratings", "avg_user_score": "Average User Score"},
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "> **Insight:** Highly popular anime tend to cluster around 7-8 average scores. "
            "Niche titles can have extreme ratings (very high or low) due to smaller, more "
            "opinionated audiences."
        )

    # TAB 4 — User Activity
    with tab4:
        st.subheader("User Activity Distribution")

        uah = stats["user_activity_hist"]
        fig = px.histogram(
            uah,
            nbins=50,
            labels={"value": "Ratings per User", "count": "Number of Users"},
            color_discrete_sequence=["#7c4dff"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        col_x, col_y = st.columns(2)
        with col_x:
            st.metric("Avg Ratings / User", f"{stats['user_activity_mean']:.1f}")
        with col_y:
            st.metric("Median Ratings / User", f"{stats['user_activity_median']:.0f}")

        st.subheader("Anime Popularity Distribution")
        aph = stats["anime_popularity_hist"]
        fig2 = px.histogram(
            aph,
            nbins=50,
            labels={"value": "Ratings per Anime", "count": "Number of Anime"},
            color_discrete_sequence=["#ff6d00"],
        )
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            "> **Insight:** Most users rate between 20-100 anime (by our sampling filter). "
            "The anime popularity follows a long-tail distribution — a small number of titles "
            "receive the majority of ratings, while most titles have relatively few."
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    # Header
    st.markdown(
        "<h1 style='text-align:center; color:#c0c0ff;'>🎬 MyAnimeList Hybrid Recommender</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#888; font-size:16px;'>"
        "Personalised Anime Recommendations using <strong>SVD × TF-IDF</strong> Hybrid Model"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Load model
    recommender = load_recommender()

    if recommender is None:
        st.error(
            "⚠️ Could not load the recommender model. "
            "Make sure processed data files exist in `data/processed/`."
        )
        st.stop()

    # Sidebar
    user_id, n_recs, get_recs = render_sidebar(recommender)

    # Recommendations section
    if get_recs:
        display_recommendations(recommender, user_id, n_recs)
    else:
        st.info("👈 Select a user and click **Get Recommendations** in the sidebar to start!")

    st.markdown("---")

    # EDA section
    stats = compute_eda_stats(recommender)
    render_eda(stats)


if __name__ == "__main__":
    main()
