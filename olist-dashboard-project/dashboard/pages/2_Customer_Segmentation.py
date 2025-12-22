
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")

st.title("👥 Customer Segmentation (RFM)")

@st.cache_data
def load_segmentation_data():
    file_path = 'dataset/customer_segmentation.pkl'
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    return None

rfm = load_segmentation_data()

if rfm is not None:
    st.markdown("### RFM Analysis & Clustering")
    st.dataframe(rfm.head())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cluster Distribution")
        fig, ax = plt.subplots()
        sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='viridis', alpha=0.6, ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Cluster Stats")
        cluster_stats = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        st.dataframe(cluster_stats)
        
    st.subheader("Distributions")
    tab1, tab2, tab3 = st.tabs(["Recency", "Frequency", "Monetary"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(rfm['Recency'], bins=50, ax=ax)
        st.pyplot(fig)
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(rfm['Frequency'], bins=50, ax=ax)
        st.pyplot(fig)
    with tab3:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(rfm['Monetary'], bins=50, ax=ax)
        st.pyplot(fig)

else:
    st.error("Segmentation data not found. Please run Step 3 script first.")
