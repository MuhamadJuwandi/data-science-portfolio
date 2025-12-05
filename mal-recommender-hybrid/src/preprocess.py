"""
Data Preprocessing Module for MAL Recommender System.

This module handles loading raw data, performing sampling to manage memory usage,
cleaning missing values, and splitting data into training and testing sets.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
RAW_DATA_PATH = r"c:\Users\muham\.gemini\antigravity\scratch\dataset myanime"
PROCESSED_DATA_PATH = r"c:\Users\muham\.gemini\antigravity\scratch\mal-recommender-hybrid\data\processed"

def load_data():
    """
    Load raw datasets from CSV files.
    
    Returns:
        tuple: (anime_df, user_anime_df)
    """
    logger.info("Loading raw data...")
    anime_path = os.path.join(RAW_DATA_PATH, "AnimeList.csv")
    user_anime_path = os.path.join(RAW_DATA_PATH, "UserAnimeList.csv")
    
    # Load Anime List
    anime_df = pd.read_csv(anime_path)
    logger.info(f"Loaded {len(anime_df)} anime records.")
    
    # Load User Anime List (Interaction Data)
    # Using iterator or reading a subset could be better for huge files, 
    # but for this step we'll read and then sample immediately if memory allows.
    # If memory is an issue, we would use chunksize.
    # Given 16GB RAM constraint and 5GB file, we might need to be careful.
    # Let's read specific columns to save memory.
    user_anime_df = pd.read_csv(
        user_anime_path, 
        usecols=['username', 'anime_id', 'my_score'],
        dtype={'my_score': 'int8', 'anime_id': 'int32'}
    )
    logger.info(f"Loaded {len(user_anime_df)} interaction records.")
    
    return anime_df, user_anime_df

def clean_anime_data(anime_df):
    """
    Clean anime dataset.
    
    Args:
        anime_df (pd.DataFrame): Raw anime dataframe.
        
    Returns:
        pd.DataFrame: Cleaned anime dataframe.
    """
    logger.info("Cleaning anime data...")
    
    # Select relevant columns
    cols = ['anime_id', 'title', 'genre', 'image_url', 'type', 'score', 'scored_by']
    anime_df = anime_df[cols].copy()
    
    # Drop anime with missing titles or genres (essential for content-based)
    anime_df.dropna(subset=['title', 'genre'], inplace=True)
    
    # Fill missing values if any
    # anime_df['synopsis'] = anime_df['synopsis'].fillna('')
    
    # Filter out anime with very few ratings (optional, but good for quality)
    # anime_df = anime_df[anime_df['scored_by'] > 50]
    
    logger.info(f"Anime data cleaned. Rows: {len(anime_df)}")
    return anime_df

def sample_interactions(user_anime_df, n_users=10000, min_interactions=20):
    """
    Sample interactions to manage memory and ensure data quality.
    
    Args:
        user_anime_df (pd.DataFrame): Raw interaction dataframe.
        n_users (int): Number of top active users to keep.
        min_interactions (int): Minimum interactions per user to be considered.
        
    Returns:
        pd.DataFrame: Sampled interaction dataframe.
    """
    logger.info(f"Sampling interactions (Top {n_users} users)...")
    
    # Filter only rated items (score > 0 usually implies watched and rated in MAL datasets, 
    # sometimes 0 means watched but not rated. We focus on explicit ratings for now or implicit if 0 is kept)
    # For this hybrid model, we'll focus on explicit ratings (1-10).
    user_anime_df = user_anime_df[user_anime_df['my_score'] > 0]
    
    # Count interactions per user
    user_counts = user_anime_df['username'].value_counts()
    
    # Filter users with enough interactions
    active_users = user_counts[user_counts >= min_interactions]
    
    # Take top N users
    top_users = active_users.head(n_users).index
    
    # Filter dataframe
    sampled_df = user_anime_df[user_anime_df['username'].isin(top_users)].copy()
    
    logger.info(f"Sampled {len(sampled_df)} interactions from {len(top_users)} users.")
    return sampled_df

def preprocess_pipeline():
    """
    Execute the full preprocessing pipeline.
    """
    # Create processed data directory
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    
    # 1. Load Data
    anime_df, user_anime_df = load_data()
    
    # 2. Clean Anime Data
    anime_clean = clean_anime_data(anime_df)
    
    # 3. Sample Interactions
    interactions_sample = sample_interactions(user_anime_df, n_users=10000)
    
    # 4. Filter Anime in Interactions
    # Only keep interactions for anime that exist in our cleaned anime list
    valid_anime_ids = anime_clean['anime_id'].unique()
    interactions_sample = interactions_sample[interactions_sample['anime_id'].isin(valid_anime_ids)]
    
    # 5. Train/Test Split
    # Stratified split based on user to ensure every user is in both sets (if possible)
    # or simple random split. Given the density, random split is usually fine for collaborative filtering
    # but let's try to be careful.
    logger.info("Splitting data...")
    train_df, test_df = train_test_split(interactions_sample, test_size=0.2, random_state=42, stratify=interactions_sample['username'])
    
    # 6. Save Data
    logger.info("Saving processed data...")
    anime_clean.to_parquet(os.path.join(PROCESSED_DATA_PATH, "anime_processed.parquet"), index=False)
    interactions_sample.to_parquet(os.path.join(PROCESSED_DATA_PATH, "interactions_full_sample.parquet"), index=False)
    train_df.to_parquet(os.path.join(PROCESSED_DATA_PATH, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(PROCESSED_DATA_PATH, "test.parquet"), index=False)
    
    logger.info("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_pipeline()
