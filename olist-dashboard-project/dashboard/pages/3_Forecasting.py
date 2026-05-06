
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import os
import logging

# Suppress verbose logging from prophet/cmdstanpy
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

# --- Path Setup ---
# __file__ is in dashboard/pages/ -> go up 2 levels to project root
PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.dirname(PAGES_DIR)
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')

st.set_page_config(page_title="Demand Forecasting", page_icon="📈", layout="wide")

st.title("📈 Demand Forecasting")

# --- Try importing Prophet with graceful fallback ---
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly
    PROPHET_AVAILABLE = True
except ImportError:
    pass

@st.cache_data
def load_data():
    file_path = os.path.join(DATASET_DIR, 'cleaned_data.pkl')
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    return None

df = load_data()

if df is not None:
    # Ensure datetime
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')
    
    # Select Category
    categories = df['product_category_name_english'].value_counts().head(10).index.tolist()
    selected_category = st.selectbox("Select Product Category", categories)
    
    # Prepare Data
    category_df = df[df['product_category_name_english'] == selected_category].copy()
    weekly_sales = category_df.groupby(
        pd.Grouper(key='order_purchase_timestamp', freq='W')
    )['price'].sum().reset_index()
    
    if len(weekly_sales) > 10:
        st.write(f"Displaying forecast for: **{selected_category}**")
        
        # --- Always show historical data first ---
        st.subheader("📊 Historical Weekly Sales")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=weekly_sales['order_purchase_timestamp'],
            y=weekly_sales['price'],
            mode='lines+markers',
            name='Actual Sales',
            line=dict(color='#1f77b4'),
            marker=dict(size=4)
        ))
        fig_hist.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales (R$)",
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # --- Historical Statistics ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Weeks", f"{len(weekly_sales)}")
        with col2:
            st.metric("Avg Weekly Sales", f"R$ {weekly_sales['price'].mean():,.2f}")
        with col3:
            st.metric("Peak Week Sales", f"R$ {weekly_sales['price'].max():,.2f}")
        with col4:
            st.metric("Total Revenue", f"R$ {weekly_sales['price'].sum():,.2f}")
        
        st.divider()
        
        # --- Prophet Forecasting ---
        if PROPHET_AVAILABLE:
            st.subheader("🔮 Prophet Forecast")
            
            with st.spinner("Training Prophet model..."):
                prophet_df = weekly_sales.rename(
                    columns={'order_purchase_timestamp': 'ds', 'price': 'y'}
                )
                
                m = Prophet(
                    yearly_seasonality=True, 
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05
                )
                m.fit(prophet_df)
                
                forecast_periods = st.slider("Forecast Weeks Ahead", min_value=4, max_value=52, value=12)
                future = m.make_future_dataframe(periods=forecast_periods, freq='W')
                forecast = m.predict(future)
            
            # Forecast Plot
            fig = plot_plotly(m, forecast)
            fig.update_layout(
                title=f"Sales Forecast - {selected_category}",
                xaxis_title="Date",
                yaxis_title="Sales (R$)",
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast Components
            st.subheader("📉 Forecast Components")
            
            # Trend
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['trend'],
                mode='lines', name='Trend',
                line=dict(color='#ff7f0e', width=2)
            ))
            fig_trend.update_layout(
                title="Overall Trend",
                xaxis_title="Date", yaxis_title="Trend Value (R$)",
                template='plotly_white', height=350
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Yearly seasonality if available
            if 'yearly' in forecast.columns:
                fig_yearly = go.Figure()
                fig_yearly.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yearly'],
                    mode='lines', name='Yearly Seasonality',
                    line=dict(color='#2ca02c', width=2)
                ))
                fig_yearly.update_layout(
                    title="Yearly Seasonality",
                    xaxis_title="Date", yaxis_title="Seasonal Effect (R$)",
                    template='plotly_white', height=350
                )
                st.plotly_chart(fig_yearly, use_container_width=True)
            
            # Forecast Summary Table
            st.subheader("📋 Forecast Summary (Next Periods)")
            future_only = forecast[forecast['ds'] > weekly_sales['order_purchase_timestamp'].max()][
                ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            ].copy()
            future_only.columns = ['Date', 'Predicted Sales', 'Lower Bound', 'Upper Bound']
            future_only = future_only.round(2)
            st.dataframe(future_only, use_container_width=True)
            
            # Insight
            avg_forecast = future_only['Predicted Sales'].mean()
            avg_actual = weekly_sales['price'].mean()
            growth = ((avg_forecast - avg_actual) / avg_actual) * 100
            
            if growth > 0:
                st.success(f"📈 Forecast shows **{growth:.1f}% growth** compared to historical average for **{selected_category}**.")
            else:
                st.warning(f"📉 Forecast shows **{abs(growth):.1f}% decline** compared to historical average for **{selected_category}**.")
        else:
            st.warning("⚠️ Prophet library is not available. Showing statistical forecast instead.")
            st.subheader("📊 Statistical Trend Analysis (Fallback)")
            
            # Simple Moving Average Forecast as fallback
            weekly_sales_sorted = weekly_sales.sort_values('order_purchase_timestamp')
            weekly_sales_sorted['SMA_4'] = weekly_sales_sorted['price'].rolling(window=4).mean()
            weekly_sales_sorted['SMA_8'] = weekly_sales_sorted['price'].rolling(window=8).mean()
            
            fig_sma = go.Figure()
            fig_sma.add_trace(go.Scatter(
                x=weekly_sales_sorted['order_purchase_timestamp'],
                y=weekly_sales_sorted['price'],
                mode='lines', name='Actual', opacity=0.5
            ))
            fig_sma.add_trace(go.Scatter(
                x=weekly_sales_sorted['order_purchase_timestamp'],
                y=weekly_sales_sorted['SMA_4'],
                mode='lines', name='4-Week Moving Avg',
                line=dict(color='#ff7f0e', width=2)
            ))
            fig_sma.add_trace(go.Scatter(
                x=weekly_sales_sorted['order_purchase_timestamp'],
                y=weekly_sales_sorted['SMA_8'],
                mode='lines', name='8-Week Moving Avg',
                line=dict(color='#2ca02c', width=2)
            ))
            fig_sma.update_layout(
                title=f"Sales Trend Analysis - {selected_category}",
                xaxis_title="Date", yaxis_title="Sales (R$)",
                template='plotly_white', height=500
            )
            st.plotly_chart(fig_sma, use_container_width=True)
            
            # Trend insight
            recent_avg = weekly_sales_sorted['price'].tail(8).mean()
            overall_avg = weekly_sales_sorted['price'].mean()
            trend_pct = ((recent_avg - overall_avg) / overall_avg) * 100
            
            if trend_pct > 0:
                st.success(f"📈 Recent 8-week average is **{trend_pct:.1f}% higher** than overall average.")
            else:
                st.warning(f"📉 Recent 8-week average is **{abs(trend_pct):.1f}% lower** than overall average.")
    else:
        st.warning("Not enough data to generate a forecast for this category.")
else:
    st.error("❌ Data not found.")
    st.info(f"Expected path: `{os.path.join(DATASET_DIR, 'cleaned_data.pkl')}`")
