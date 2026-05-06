
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Path Setup ---
# Get the absolute path of the project root (parent of 'dashboard/')
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')

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
    file_path = os.path.join(DATASET_DIR, 'cleaned_data.pkl')
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    return None

df = load_data()

if df is not None:
    # Ensure datetime column
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')

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
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with col2:
        st.subheader("Top 10 Product Categories")
        top_products = filtered_df['product_category_name_english'].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top_products.values, y=top_products.index, palette='viridis', ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- Additional Insights ---
    st.divider()
    
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Payment Method Distribution")
        if 'payment_type' in filtered_df.columns:
            payment_counts = filtered_df['payment_type'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=payment_counts.index, y=payment_counts.values, palette='Blues_d', ax=ax)
            ax.set_xlabel("Payment Type")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col4:
        st.subheader("Delivery Performance")
        if 'order_delivered_customer_date' in filtered_df.columns:
            delivered = filtered_df.dropna(subset=['order_delivered_customer_date']).copy()
            delivered['delivery_days'] = (
                pd.to_datetime(delivered['order_delivered_customer_date']) - 
                pd.to_datetime(delivered['order_purchase_timestamp'])
            ).dt.days
            delivered = delivered[delivered['delivery_days'] >= 0]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(delivered['delivery_days'], bins=50, kde=True, color='purple', ax=ax)
            ax.set_xlabel("Delivery Time (Days)")
            ax.set_ylabel("Frequency")
            ax.set_xlim(0, 60)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            avg_delivery = delivered['delivery_days'].mean()
            median_delivery = delivered['delivery_days'].median()
            st.info(f"📦 Average delivery: **{avg_delivery:.1f} days** | Median: **{median_delivery:.1f} days**")

    # Additional Info
    st.markdown("---")
    st.markdown("Use the sidebar to filter data by date range. Navigate to other pages for detailed analysis.")

else:
    st.error("❌ Data not found. Please ensure the dataset is cleaned and saved in 'dataset/cleaned_data.pkl'.")
    st.info(f"Expected path: `{os.path.join(DATASET_DIR, 'cleaned_data.pkl')}`")
