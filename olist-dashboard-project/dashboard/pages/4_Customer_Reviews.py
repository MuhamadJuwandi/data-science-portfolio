
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud

# --- Path Setup ---
# __file__ is in dashboard/pages/ -> go up 2 levels to project root
PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.dirname(PAGES_DIR)
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')

st.set_page_config(page_title="Customer Reviews", page_icon="⭐", layout="wide")

st.title("⭐ Customer Reviews Analysis")

@st.cache_data
def load_data():
    file_path = os.path.join(DATASET_DIR, 'cleaned_data.pkl')
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    return None

df = load_data()

if df is not None:
    # Ensure numeric review_score
    df['review_score'] = pd.to_numeric(df['review_score'], errors='coerce')

    # --- KPI Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_score = df['review_score'].mean()
        st.metric("Average Rating", f"⭐ {avg_score:.2f}")
    with col2:
        total_reviews = df['review_score'].notna().sum()
        st.metric("Total Reviews", f"{total_reviews:,}")
    with col3:
        pct_positive = (df['review_score'] >= 4).sum() / total_reviews * 100
        st.metric("Positive Reviews (≥4★)", f"{pct_positive:.1f}%")
    with col4:
        pct_negative = (df['review_score'] <= 2).sum() / total_reviews * 100
        st.metric("Negative Reviews (≤2★)", f"{pct_negative:.1f}%")
    
    st.divider()

    # --- Review Score Distribution ---
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Review Score Distribution")
        review_counts = df['review_score'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#d32f2f', '#f57c00', '#fbc02d', '#7cb342', '#2e7d32']
        bars = ax.bar(
            review_counts.index.astype(int), 
            review_counts.values, 
            color=colors[:len(review_counts)]
        )
        ax.set_xlabel("Review Score", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Distribution of Review Scores")
        # Add value labels on bars
        for bar, count in zip(bars, review_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col_b:
        st.subheader("Score Distribution (%)")
        fig, ax = plt.subplots(figsize=(10, 5))
        review_pct = (review_counts / review_counts.sum() * 100).round(1)
        colors_pie = ['#d32f2f', '#f57c00', '#fbc02d', '#7cb342', '#2e7d32']
        ax.pie(
            review_pct.values,
            labels=[f"{int(k)}★ ({v}%)" for k, v in zip(review_pct.index, review_pct.values)],
            colors=colors_pie[:len(review_pct)],
            startangle=90,
            autopct='',
        )
        ax.set_title("Review Score Proportions")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # --- Review Score by Category ---
    st.subheader("📦 Average Rating by Top 10 Categories")
    if 'product_category_name_english' in df.columns:
        top_cats = df['product_category_name_english'].value_counts().head(10).index
        cat_ratings = df[df['product_category_name_english'].isin(top_cats)].groupby(
            'product_category_name_english'
        )['review_score'].mean().sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.barh(cat_ratings.index, cat_ratings.values, color=sns.color_palette('RdYlGn', len(cat_ratings)))
        ax.set_xlabel("Average Review Score")
        ax.set_xlim(0, 5)
        ax.axvline(x=avg_score, color='red', linestyle='--', alpha=0.7, label=f'Overall Avg ({avg_score:.2f})')
        ax.legend()
        # Add value labels
        for bar, val in zip(bars, cat_ratings.values):
            ax.text(val + 0.05, bar.get_y() + bar.get_height()/2., 
                    f'{val:.2f}', va='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # --- Review Score vs Delivery Time ---
    st.subheader("🚚 Review Score vs Delivery Time")
    if 'order_delivered_customer_date' in df.columns:
        reviews_delivery = df.dropna(subset=['order_delivered_customer_date', 'review_score']).copy()
        reviews_delivery['delivery_days'] = (
            pd.to_datetime(reviews_delivery['order_delivered_customer_date']) - 
            pd.to_datetime(reviews_delivery['order_purchase_timestamp'])
        ).dt.days
        reviews_delivery = reviews_delivery[
            (reviews_delivery['delivery_days'] >= 0) & (reviews_delivery['delivery_days'] <= 60)
        ]
        
        avg_delivery_by_score = reviews_delivery.groupby('review_score')['delivery_days'].mean()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        colors_bar = ['#d32f2f', '#f57c00', '#fbc02d', '#7cb342', '#2e7d32']
        bars = ax.bar(
            avg_delivery_by_score.index.astype(int), 
            avg_delivery_by_score.values,
            color=colors_bar[:len(avg_delivery_by_score)]
        )
        ax.set_xlabel("Review Score", fontsize=12)
        ax.set_ylabel("Average Delivery Time (Days)", fontsize=12)
        ax.set_title("Correlation: Lower Reviews = Longer Delivery Times")
        for bar, val in zip(bars, avg_delivery_by_score.values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.1f}d', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.info("💡 **Key Insight**: Lower review scores correlate with longer delivery times, suggesting logistics is a primary driver of customer dissatisfaction.")

    st.divider()

    # --- Comments WordCloud ---
    st.subheader("💬 Review Comments WordCloud")
    if 'review_comment_message' in df.columns:
        comments = df['review_comment_message'].dropna().astype(str)
        all_comments = ' '.join(comments)
        
        if len(all_comments.strip()) > 0:
            tab_all, tab_pos, tab_neg = st.tabs(["All Reviews", "Positive (4-5★)", "Negative (1-2★)"])
            
            with tab_all:
                wordcloud = WordCloud(
                    width=800, height=400, 
                    background_color='white',
                    max_words=100,
                    colormap='viridis'
                ).generate(all_comments)
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                ax.set_title("All Reviews")
                st.pyplot(fig)
                plt.close(fig)
            
            with tab_pos:
                pos_comments = df[df['review_score'] >= 4]['review_comment_message'].dropna().astype(str)
                pos_text = ' '.join(pos_comments)
                if len(pos_text.strip()) > 0:
                    wc_pos = WordCloud(
                        width=800, height=400, 
                        background_color='white',
                        max_words=100,
                        colormap='Greens'
                    ).generate(pos_text)
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.imshow(wc_pos, interpolation='bilinear')
                    ax.axis("off")
                    ax.set_title("Positive Reviews (4-5★)")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Not enough positive review comments.")
            
            with tab_neg:
                neg_comments = df[df['review_score'] <= 2]['review_comment_message'].dropna().astype(str)
                neg_text = ' '.join(neg_comments)
                if len(neg_text.strip()) > 0:
                    wc_neg = WordCloud(
                        width=800, height=400, 
                        background_color='white',
                        max_words=100,
                        colormap='Reds'
                    ).generate(neg_text)
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.imshow(wc_neg, interpolation='bilinear')
                    ax.axis("off")
                    ax.set_title("Negative Reviews (1-2★)")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Not enough negative review comments.")
        else:
            st.info("No review comments available.")
    
    # Sample Reviews
    st.divider()
    st.subheader("🔍 Sample Reviews")
    review_cols = ['review_score', 'review_comment_title', 'review_comment_message']
    available_cols = [c for c in review_cols if c in df.columns]
    if available_cols:
        sample_reviews = df[available_cols].dropna().head(20)
        st.dataframe(sample_reviews, use_container_width=True)

else:
    st.error("❌ Data not found.")
    st.info(f"Expected path: `{os.path.join(DATASET_DIR, 'cleaned_data.pkl')}`")
