
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import os

st.set_page_config(page_title="Demand Forecasting", page_icon="📈", layout="wide")

st.title("📈 Demand Forecasting")

@st.cache_data
def load_data():
    file_path = 'dataset/cleaned_data.pkl'
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    return None

df = load_data()

if df is not None:
    # Select Category
    categories = df['product_category_name_english'].value_counts().head(5).index.tolist()
    selected_category = st.selectbox("Select Product Category", categories)
    
    # Prepare Data
    category_df = df[df['product_category_name_english'] == selected_category].copy()
    category_df['order_purchase_timestamp'] = pd.to_datetime(category_df['order_purchase_timestamp'])
    weekly_sales = category_df.groupby(pd.Grouper(key='order_purchase_timestamp', freq='W'))['price'].sum().reset_index()
    
    if len(weekly_sales) > 10:
        st.write(f"Displaying forecast for: **{selected_category}**")
        
        # Prophet Model
        prophet_df = weekly_sales.rename(columns={'order_purchase_timestamp': 'ds', 'price': 'y'})
        
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        m.fit(prophet_df)
        
        future = m.make_future_dataframe(periods=12, freq='W')
        forecast = m.predict(future)
        
        # Plot
        st.subheader("Forecast Plot")
        fig = plot_plotly(m, forecast)
        st.plotly_chart(fig)
        
        st.subheader("Forecast Components")
        st.line_chart(forecast.set_index('ds')[['trend', 'weekly', 'yearly']])
    else:
        st.warning("Not enough data to generate a forecast for this category.")
else:
    st.error("Data not found.")
