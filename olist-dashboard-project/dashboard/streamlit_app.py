
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Page Config
st.set_page_config(
    page_title="Olist E-Commerce Dashboard",
    page_icon="🛒",
    layout="wide"
)

# Title
st.title("🛒 Olist E-Commerce Analytics Dashboard")
st.markdown("### Comprehensive analysis of sales, customers, and product performance.")

# Load Data
@st.cache_data
def load_data():
    file_path = 'dataset/cleaned_data.pkl'
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    return None

df = load_data()

if df is not None:
    # Sidebar Filters
    st.sidebar.header("Filters")
    min_date = df['order_purchase_timestamp'].min()
    max_date = df['order_purchase_timestamp'].max()
    
    start_date, end_date = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter Data
    filtered_df = df[
        (df['order_purchase_timestamp'].dt.date >= start_date) & 
        (df['order_purchase_timestamp'].dt.date <= end_date)
    ]
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"R$ {filtered_df['price'].sum():,.2f}")
    with col2:
        st.metric("Total Orders", f"{filtered_df['order_id'].nunique():,}")
    with col3:
        st.metric("Total Customers", f"{filtered_df['customer_unique_id'].nunique():,}")
    with col4:
        st.metric("Avg Ticket Size", f"R$ {filtered_df['price'].mean():,.2f}")
    
    st.divider()
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales Over Time")
        sales_over_time = filtered_df.groupby(filtered_df['order_purchase_timestamp'].dt.to_period('M'))['price'].sum().reset_index()
        sales_over_time['order_purchase_timestamp'] = sales_over_time['order_purchase_timestamp'].astype(str)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=sales_over_time, x='order_purchase_timestamp', y='price', marker='o', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Top 10 Product Categories")
        top_products = filtered_df['product_category_name_english'].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top_products.values, y=top_products.index, palette='viridis', ax=ax)
        st.pyplot(fig)
        
    # Additional Info
    st.markdown("---")
    st.markdown("Use the sidebar to filter data by date range. Navigate to other pages for detailed analysis.")

else:
    st.error("Data not found. Please ensure the dataset is cleaned and saved in 'dataset/cleaned_data.pkl'.")
