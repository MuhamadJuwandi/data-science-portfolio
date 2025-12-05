# %% [markdown]
# # MAL Recommender Hybrid - EDA & Training
# 
# This notebook covers:
# 1. Exploratory Data Analysis (EDA) of the processed data.
# 2. Training the Hybrid Recommender System.
# 3. Evaluating the model.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join('..')))

from src.hybrid_model import HybridRecommender

# %% [markdown]
# ## 1. Load Data

# %%
DATA_PATH = r"c:\Users\muham\.gemini\antigravity\scratch\mal-recommender-hybrid\data\processed"

anime_df = pd.read_parquet(os.path.join(DATA_PATH, "anime_processed.parquet"))
train_df = pd.read_parquet(os.path.join(DATA_PATH, "train.parquet"))

print(f"Anime shape: {anime_df.shape}")
print(f"Train interactions shape: {train_df.shape}")

# %% [markdown]
# ## 2. EDA

# %%
# Rating Distribution
plt.figure(figsize=(10, 6))
sns.histplot(train_df['my_score'], bins=10, kde=False)
plt.title('Distribution of User Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# %%
# Top Popular Anime
top_anime = anime_df.sort_values('scored_by', ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x='scored_by', y='title', data=top_anime)
plt.title('Top 10 Most Rated Anime')
plt.xlabel('Number of Ratings')
plt.show()

# %% [markdown]
# ## 3. Model Training

# %%
recommender = HybridRecommender(data_path=DATA_PATH)
recommender.load_data()

# %%
# Train Collaborative Filtering
recommender.build_cf_model()

# %%
# Train Content-Based
recommender.build_cb_model()

# %% [markdown]
# ## 4. Evaluation / Testing

# %%
# Test with a random user from the test set (or train set for now)
sample_user = train_df['username'].iloc[0]
print(f"Testing recommendations for user: {sample_user}")

recommendations = recommender.recommend(sample_user, n_recommendations=10)
print(recommendations[['title', 'genre', 'hybrid_score']])

# %%
# Check Cold Start
print("Testing Cold Start...")
cold_recs = recommender.recommend("non_existent_user_123")
print(cold_recs[['title', 'score']])
