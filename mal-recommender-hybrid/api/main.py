from fastapi import FastAPI, HTTPException
import pandas as pd
import sys
import os
from typing import List, Optional
from pydantic import BaseModel

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hybrid_model import HybridRecommender

app = FastAPI(
    title="MAL Recommender API",
    description="API for Hybrid Anime Recommendation System",
    version="1.0.0"
)

# Global recommender instance
recommender = None

@app.on_event("startup")
async def startup_event():
    global recommender
    # Dynamic path relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed")
    if os.path.exists(os.path.join(data_path, "anime_processed.parquet")):
        recommender = HybridRecommender(data_path=data_path)
        recommender.load_data()
        recommender.build_cf_model()
        recommender.build_cb_model()
    else:
        print("WARNING: Processed data not found. API will not function correctly.")

class AnimeRecommendation(BaseModel):
    anime_id: int
    title: str
    genre: Optional[str] = None
    score: Optional[float] = None
    hybrid_score: Optional[float] = None
    image_url: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to MAL Recommender API. Go to /docs for usage."}

@app.get("/recommend/{user_id}", response_model=List[AnimeRecommendation])
def get_recommendations(user_id: str, n: int = 10):
    """
    Get personalized recommendations for a user.
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        recs = recommender.recommend(user_id, n_recommendations=n)
        return recs.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cold-start", response_model=List[AnimeRecommendation])
def get_cold_start(n: int = 10):
    """
    Get popular recommendations for cold start.
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    recs = recommender.get_popular_recommendations(n=n)
    return recs.to_dict(orient="records")
