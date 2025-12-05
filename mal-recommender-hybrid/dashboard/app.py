import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hybrid_model import HybridRecommender

# Page Config
st.set_page_config(
    page_title="MAL Hybrid Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .anime-title {
        font-size: 16px;
        font-weight: bold;
        color: #ffffff;
        margin-top: 10px;
        height: 50px;
        overflow: hidden;
    }
    .anime-score {
        color: #f0ad4e;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Recommender
@st.cache_resource
def load_recommender():
    # Dynamic path relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed")
    recommender = HybridRecommender(data_path=data_path)
    # Check if data exists
    if not os.path.exists(os.path.join(data_path, "anime_processed.parquet")):
        st.error("Processed data not found. Please run preprocessing first.")
        return None
    
    recommender.load_data()
    recommender.build_cf_model()
    recommender.build_cb_model()
    return recommender

st.title("🎬 MyAnimeList Hybrid Recommender")
st.markdown("### Personalized Recommendations using SVD + TF-IDF")

recommender = load_recommender()

if recommender:
    # Sidebar
    st.sidebar.header("User Input")
    
    # Get list of users for dropdown (optional, or just text input)
    # For demo, text input is fine, maybe show a random user example
    example_user = recommender.train_df['username'].iloc[0] if recommender.train_df is not None else "User123"
    
    user_id = st.sidebar.text_input("Enter User ID (Username)", value=example_user)
    n_recs = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
    
    if st.sidebar.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            recs = recommender.recommend(user_id, n_recommendations=n_recs)
            
            st.subheader(f"Top {n_recs} Recommendations for **{user_id}**")
            
            # Display in a grid
            cols = st.columns(5)
            for i, row in recs.iterrows():
                col = cols[i % 5]
                with col:
                    # Use placeholder if image_url is missing or invalid
                    img_url = row['image_url'] if row['image_url'] else "https://via.placeholder.com/225x318?text=No+Image"
                    
                    st.image(img_url, use_column_width=True)
                    st.markdown(f"<div class='anime-title'>{row['title']}</div>", unsafe_allow_html=True)
                    st.markdown(f"⭐ {row['score']:.2f} | 🎯 {row['hybrid_score']:.2f}")
                    st.caption(row['genre'].split(',')[0] if row['genre'] else "")

    # Tabs for EDA
    st.markdown("---")
    tab1, tab2 = st.tabs(["📊 EDA: Ratings", "📈 EDA: Genres"])
    
    with tab1:
        st.header("Rating Distribution")
        if recommender.train_df is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(recommender.train_df['my_score'], bins=10, kde=False, ax=ax, color='#ff6b6b')
            ax.set_title("Distribution of User Ratings")
            st.pyplot(fig)
            
    with tab2:
        st.header("Popular Genres")
        if recommender.anime_df is not None:
            # Simple genre counting
            genres = recommender.anime_df['genre'].str.split(', ', expand=True).stack().value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=genres.values, y=genres.index, ax=ax, palette="viridis")
            ax.set_title("Top 10 Anime Genres")
            st.pyplot(fig)

else:
    st.info("Please wait for the model to load or run preprocessing.")
