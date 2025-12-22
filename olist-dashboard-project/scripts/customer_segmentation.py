
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Paths
INPUT_FILE = 'dataset/cleaned_data.pkl'
VISUALS_DIR = 'visuals'
OUTPUT_FILE = 'dataset/customer_segmentation.pkl'

def load_data():
    return pd.read_pickle(INPUT_FILE)

def calculate_rfm(df):
    # Filter for necessary columns
    df_rfm = df[['customer_unique_id', 'order_purchase_timestamp', 'price']].copy()
    
    # Calculate Recency
    max_date = df_rfm['order_purchase_timestamp'].max()
    recency_df = df_rfm.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
    recency_df['Recency'] = (max_date - recency_df['order_purchase_timestamp']).dt.days
    
    # Calculate Frequency
    frequency_df = df_rfm.groupby('customer_unique_id')['order_purchase_timestamp'].count().reset_index()
    frequency_df.columns = ['customer_unique_id', 'Frequency']
    
    # Calculate Monetary
    monetary_df = df_rfm.groupby('customer_unique_id')['price'].sum().reset_index()
    monetary_df.columns = ['customer_unique_id', 'Monetary']
    
    # Merge RFM
    rfm = recency_df[['customer_unique_id', 'Recency']].merge(frequency_df, on='customer_unique_id')
    rfm = rfm.merge(monetary_df, on='customer_unique_id')
    
    return rfm

def plot_rfm_distributions(rfm):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.histplot(rfm['Recency'], bins=50)
    plt.title('Recency Distribution')
    
    plt.subplot(1, 3, 2)
    sns.histplot(rfm['Frequency'], bins=50)
    plt.title('Frequency Distribution')
    
    plt.subplot(1, 3, 3)
    sns.histplot(rfm['Monetary'], bins=50)
    plt.title('Monetary Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'rfm_distributions.png'))
    plt.close()

def perform_clustering(rfm):
    # Preprocessing
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    # Elbow Method (simplified: using 4 clusters based on common practice)
    # Ideally should loop and plot inertia, but for automation we will pick a reasonable number: 4
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    return rfm

def plot_clusters(rfm):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='viridis', alpha=0.6)
    plt.title('Customer Segments (Recency vs Monetary)')
    plt.savefig(os.path.join(VISUALS_DIR, 'cluster_scatter.png'))
    plt.close()
    
def run_segmentation():
    print("Loading data...")
    df = load_data()
    
    print("Calculating RFM...")
    rfm = calculate_rfm(df)
    
    print("Plotting RFM distributions...")
    plot_rfm_distributions(rfm)
    
    print("Performing Clustering...")
    rfm = perform_clustering(rfm)
    
    print("Plotting Clusters...")
    plot_clusters(rfm)
    
    print(f"Saving segmentation result to {OUTPUT_FILE}...")
    rfm.to_pickle(OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    run_segmentation()
