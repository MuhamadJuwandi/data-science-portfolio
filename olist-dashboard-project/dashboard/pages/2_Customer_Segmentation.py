
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Path Setup ---
# __file__ is in dashboard/pages/ -> go up 2 levels to project root
PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.dirname(PAGES_DIR)
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")

st.title("👥 Customer Segmentation (RFM)")

@st.cache_data
def load_segmentation_data():
    file_path = os.path.join(DATASET_DIR, 'customer_segmentation.pkl')
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    return None

rfm = load_segmentation_data()

if rfm is not None:
    st.markdown("### RFM Analysis & Clustering")
    
    # --- Cluster Summary with Business Labels ---
    st.subheader("📊 Cluster Overview")
    cluster_stats = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].agg(['mean', 'count']).reset_index()
    
    # Simplified cluster stats for display
    cluster_summary = rfm.groupby('Cluster').agg(
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Customer_Count=('Recency', 'count')
    ).reset_index()
    cluster_summary = cluster_summary.round(2)
    st.dataframe(cluster_summary, use_container_width=True)
    
    # --- Cluster Size Pie Chart ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cluster Distribution (Scatter)")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = sns.scatterplot(
            data=rfm, x='Recency', y='Monetary', 
            hue='Cluster', palette='viridis', alpha=0.6, ax=ax
        )
        ax.set_xlabel("Recency (Days)")
        ax.set_ylabel("Monetary Value (R$)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with col2:
        st.subheader("Cluster Proportions")
        cluster_counts = rfm['Cluster'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette('viridis', n_colors=len(cluster_counts))
        ax.pie(
            cluster_counts.values, 
            labels=[f"Cluster {i}" for i in cluster_counts.index],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax.set_title("Customer Distribution by Cluster")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- Detailed Cluster Comparison ---
    st.subheader("📈 RFM Distributions by Cluster")
    tab1, tab2, tab3 = st.tabs(["Recency", "Frequency", "Monetary"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=rfm, x='Cluster', y='Recency', palette='viridis', ax=ax)
        ax.set_title("Recency Distribution by Cluster")
        ax.set_ylabel("Days Since Last Purchase")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with tab2:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=rfm, x='Cluster', y='Frequency', palette='viridis', ax=ax)
        ax.set_title("Frequency Distribution by Cluster")
        ax.set_ylabel("Number of Purchases")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with tab3:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=rfm, x='Cluster', y='Monetary', palette='viridis', ax=ax)
        ax.set_title("Monetary Distribution by Cluster")
        ax.set_ylabel("Total Spending (R$)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # --- Business Insights ---
    st.divider()
    st.subheader("💡 Business Insights & Recommendations")
    
    # Auto-generate insights from cluster data
    best_cluster = cluster_summary.loc[cluster_summary['Avg_Monetary'].idxmax()]
    risk_cluster = cluster_summary.loc[cluster_summary['Avg_Recency'].idxmax()]
    biggest_cluster = cluster_summary.loc[cluster_summary['Customer_Count'].idxmax()]
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.success(f"🏆 **Best Value Cluster**: Cluster {int(best_cluster['Cluster'])}")
        st.write(f"Avg Spending: R$ {best_cluster['Avg_Monetary']:,.2f}")
        st.write(f"Customers: {int(best_cluster['Customer_Count']):,}")
        st.caption("→ Offer loyalty programs & exclusive deals")
        
    with col_b:
        st.warning(f"⚠️ **At-Risk Cluster**: Cluster {int(risk_cluster['Cluster'])}")
        st.write(f"Avg Recency: {risk_cluster['Avg_Recency']:.0f} days")
        st.write(f"Customers: {int(risk_cluster['Customer_Count']):,}")
        st.caption("→ Run win-back campaigns & special discounts")
        
    with col_c:
        st.info(f"📊 **Largest Cluster**: Cluster {int(biggest_cluster['Cluster'])}")
        st.write(f"Customers: {int(biggest_cluster['Customer_Count']):,}")
        st.write(f"Avg Frequency: {biggest_cluster['Avg_Frequency']:.1f}")
        st.caption("→ Analyze for targeted upselling opportunities")

    # Sample data preview
    st.divider()
    st.subheader("🔍 Sample Customer Data")
    st.dataframe(rfm.head(20), use_container_width=True)

else:
    st.error("❌ Segmentation data not found.")
    st.info(f"Expected path: `{os.path.join(DATASET_DIR, 'customer_segmentation.pkl')}`")
    st.markdown("Please run the customer segmentation script first: `python scripts/customer_segmentation.py`")
