"""
Hybrid Recommender Model Module.

Combines Collaborative Filtering (SVD) and Content-Based Filtering (TF-IDF)
to provide personalized anime recommendations.
Uses scikit-learn for SVD to avoid compilation issues with Surprise on Windows.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridRecommender:
    def __init__(self, data_path=None):
        """
        Initialize the Hybrid Recommender.
        
        Args:
            data_path (str): Path to processed data directory.
        """
        self.data_path = data_path if data_path else os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
        self.anime_df = None
        self.train_df = None
        self.cf_model = None
        self.tfidf_matrix = None
        self.indices = None
        self.user_mapper = None
        self.anime_mapper = None
        self.user_inv_mapper = None
        self.anime_inv_mapper = None
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        
    def load_data(self):
        """Load processed parquet files."""
        logger.info("Loading data for model...")
        self.anime_df = pd.read_parquet(os.path.join(self.data_path, "anime_processed.parquet"))
        self.train_df = pd.read_parquet(os.path.join(self.data_path, "train.parquet"))
        
        # Create a mapping of anime_id to index for content-based
        self.indices = pd.Series(self.anime_df.index, index=self.anime_df['anime_id']).drop_duplicates()
        
    def build_cf_model(self):
        """Build and train Collaborative Filtering model (TruncatedSVD)."""
        logger.info("Building Collaborative Filtering model (SVD)...")
        
        # Create User-Item Matrix
        # Map IDs to indices
        unique_users = self.train_df['username'].unique()
        unique_anime = self.train_df['anime_id'].unique()
        
        self.user_mapper = {user: i for i, user in enumerate(unique_users)}
        self.anime_mapper = {anime: i for i, anime in enumerate(unique_anime)}
        self.user_inv_mapper = {i: user for user, i in self.user_mapper.items()}
        self.anime_inv_mapper = {i: anime for anime, i in self.anime_mapper.items()}
        
        user_indices = self.train_df['username'].map(self.user_mapper)
        anime_indices = self.train_df['anime_id'].map(self.anime_mapper)
        
        self.user_item_matrix = csr_matrix(
            (self.train_df['my_score'], (user_indices, anime_indices)),
            shape=(len(unique_users), len(unique_anime))
        )
        
        # Train SVD
        self.cf_model = TruncatedSVD(n_components=20, random_state=42)
        self.cf_model.fit(self.user_item_matrix)
        
        self.user_factors = self.cf_model.transform(self.user_item_matrix)
        self.item_factors = self.cf_model.components_.T
        
        logger.info("CF model trained.")
        
    def build_cb_model(self):
        """Build Content-Based model using TF-IDF on Genres."""
        logger.info("Building Content-Based model...")
        # Combine genre and title (since synopsis is missing)
        self.anime_df['content'] = self.anime_df['genre'] + " " + self.anime_df['title']
        
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = tfidf.fit_transform(self.anime_df['content'])
        
        logger.info("CB model built (TF-IDF matrix created).")

    def recommend(self, user_id, n_recommendations=10, alpha=0.6):
        """
        Generate hybrid recommendations.
        """
        # Cold Start Check
        if user_id not in self.user_mapper:
            logger.info(f"User {user_id} not found. Returning popular recommendations.")
            return self.get_popular_recommendations(n_recommendations)
            
        # 1. Collaborative Filtering Candidates
        user_idx = self.user_mapper[user_id]
        
        # Predict scores: User Vector dot Item Matrix
        # user_vector = self.user_factors[user_idx]
        # scores = user_vector.dot(self.item_factors.T)
        
        # More efficient: self.cf_model.inverse_transform([self.user_factors[user_idx]])
        # But that gives reconstructed matrix row.
        cf_scores = np.dot(self.user_factors[user_idx], self.item_factors.T)
        
        # Get indices of anime user has watched
        watched_indices = self.user_item_matrix[user_idx].indices
        
        # Filter out watched
        # We want to keep indices that are NOT in watched_indices
        # But cf_scores corresponds to anime in self.anime_mapper (0 to N_anime)
        
        candidate_indices = [i for i in range(len(self.anime_mapper)) if i not in watched_indices]
        
        # Create DataFrame for candidates
        # Map internal index back to anime_id
        candidate_anime_ids = [self.anime_inv_mapper[i] for i in candidate_indices]
        candidate_scores = cf_scores[candidate_indices]
        
        recs_df = pd.DataFrame({'anime_id': candidate_anime_ids, 'cf_score': candidate_scores})
        
        # Normalize CF scores to 0-1 or keep as is? 
        # SVD reconstruction approximates original ratings (1-10).
        
        # 2. Content-Based Adjustment
        # For simplicity in this fix, we'll use CF primarily and boost with CB if possible,
        # but calculating CB for all candidates is slow.
        # Let's just return CF results for now to ensure it works.
        # Or implement the profile method if we have time.
        
        # Let's do a simple re-ranking of top 100 CF results using CB
        recs_df = recs_df.sort_values('cf_score', ascending=False).head(100)
        
        # Calculate CB score for these 100 items against user profile
        # User Profile: Mean of TF-IDF vectors of watched items
        # We need to map watched_indices (internal CF index) -> anime_id -> TF-IDF index
        watched_anime_ids = [self.anime_inv_mapper[i] for i in watched_indices]
        watched_tfidf_indices = [self.indices[aid] for aid in watched_anime_ids if aid in self.indices]
        
        if watched_tfidf_indices:
            user_profile = np.asarray(np.mean(self.tfidf_matrix[watched_tfidf_indices], axis=0))
            
            # Get TF-IDF indices for candidates
            candidate_tfidf_indices = [self.indices[aid] for aid in recs_df['anime_id'] if aid in self.indices]
            
            # Note: some candidates might not be in anime_df (if filtered out in preprocessing but present in interactions? 
            # Preprocessing ensured consistency, so should be fine).
            
            if len(candidate_tfidf_indices) == len(recs_df):
                cb_scores = linear_kernel(user_profile, self.tfidf_matrix[candidate_tfidf_indices]).flatten()
                
                # Scale CB (0-1) to match CF (approx 1-10)
                recs_df['cb_score'] = 1 + 9 * cb_scores
                recs_df['hybrid_score'] = alpha * recs_df['cf_score'] + (1 - alpha) * recs_df['cb_score']
            else:
                 recs_df['hybrid_score'] = recs_df['cf_score']
        else:
            recs_df['hybrid_score'] = recs_df['cf_score']
            
        top_recs = recs_df.sort_values('hybrid_score', ascending=False).head(n_recommendations)
        
        # Merge with anime details
        result = top_recs.merge(self.anime_df, on='anime_id')
        return result[['anime_id', 'title', 'genre', 'score', 'hybrid_score', 'image_url']]

    def get_popular_recommendations(self, n=10):
        """Return top popular anime based on score and member count."""
        popular = self.anime_df.sort_values(['scored_by', 'score'], ascending=False).head(n)
        return popular[['anime_id', 'title', 'genre', 'score', 'image_url']]

if __name__ == "__main__":
    # Example usage
    recommender = HybridRecommender()
    recommender.load_data()
    recommender.build_cf_model()
    recommender.build_cb_model()
    
    # Test with a user
    sample_user = recommender.train_df['username'].iloc[0]
    print(f"Recommendations for user: {sample_user}")
    recs = recommender.recommend(sample_user)
    print(recs[['title', 'hybrid_score']])
