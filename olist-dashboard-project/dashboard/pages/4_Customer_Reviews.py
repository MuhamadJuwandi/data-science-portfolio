
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud

st.set_page_config(page_title="Customer Reviews", page_icon="⭐", layout="wide")

st.title("⭐ Customer Reviews Analysis")

@st.cache_data
def load_data():
    file_path = 'dataset/cleaned_data.pkl'
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    return None

df = load_data()

if df is not None:
    # Review Score Distribution
    st.subheader("Review Score Distribution")
    review_counts = df['review_score'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=review_counts.index, y=review_counts.values, palette='RdYlGn', ax=ax)
    st.pyplot(fig)
    
    # Comments WordCloud
    st.subheader("Review Comments WordCloud")
    comments = df['review_comment_message'].dropna().astype(str).str.cat(sep=' ')
    
    if comments:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comments)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No review comments available.")
        
    # Sample Reviews
    st.subheader("Sample Reviews")
    st.dataframe(df[['review_score', 'review_comment_title', 'review_comment_message']].dropna().head(20))

else:
    st.error("Data not found.")
