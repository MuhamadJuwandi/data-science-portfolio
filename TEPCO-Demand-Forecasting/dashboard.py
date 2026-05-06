import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Resolve the base directory relative to this script file,
# so that all paths work correctly on Streamlit Cloud as well as locally.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    st.set_page_config(page_title="TEPCO Demand Forecasting", layout="wide")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "EDA & Insights", "Model Performance", "Simulation"])
    
    # Logo
    logo_path = os.path.join(BASE_DIR, "assets", "logo.jpg")
    if os.path.exists(logo_path):
        image = Image.open(logo_path)
        st.sidebar.image(image, use_container_width=True)
    
    # Load Data
    @st.cache_data
    def load_data():
        data_path = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")
        df = pd.read_csv(data_path)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        return df

    try:
        df = load_data()
    except Exception:
        st.error("Data not found. Please run ETL scripts first.")
        return

    st.title("Time Series: Prediksi Konsumsi Listrik (TEPCO)")
    st.markdown("### Project by: User")

    if page == "Overview":
        st.header("Project Overview")
        st.write("""
        This project aims to predict electricity consumption in the Kanto region (TEPCO area) using historical demand and weather data.
        
        **Objectives:**
        - Analyze seasonal trends and weather sensitivity.
        - Forecast future demand using LSTM.
        - Simulate peak shaving scenarios.
        """)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col1.metric("Start Date", str(df['Datetime'].min().date()))
        col1.metric("End Date", str(df['Datetime'].max().date()))
        
        st.subheader("Recent Data")
        st.dataframe(df.tail(10))

    elif page == "EDA & Insights":
        st.header("Exploratory Data Analysis")
        
        st.subheader("Overall Demand Trend")
        st.line_chart(df.set_index('Datetime')['Demand'])
        
        st.subheader("Visualizations")
        viz_files = [
            "visualizations/01_overall_demand.png",
            "visualizations/02_temp_vs_demand.png",
            "visualizations/03_monthly_boxplot.png",
            "visualizations/04_hourly_heatmap.png"
        ]
        
        for v in viz_files:
            viz_path = os.path.join(BASE_DIR, v)
            if os.path.exists(viz_path):
                st.image(viz_path, caption=os.path.basename(v), use_container_width=True)
            else:
                st.warning(f"Visualization {v} not found.")

    elif page == "Model Performance":
        st.header("Model Evaluation")
        
        st.write("### LSTM Forecast (PyTorch)")
        lstm_forecast_path = os.path.join(BASE_DIR, "visualizations", "lstm_forecast.png")
        if os.path.exists(lstm_forecast_path):
            st.image(lstm_forecast_path, caption="LSTM Forecast vs Actual", use_container_width=True)
            
        lstm_loss_path = os.path.join(BASE_DIR, "visualizations", "lstm_loss.png")
        if os.path.exists(lstm_loss_path):
            st.image(lstm_loss_path, caption="Training Loss", use_container_width=True)
            
        st.info("Metrics (MAE, RMSE, MAPE) are calculated during training and logged.")

        st.write("### Prophet Forecast (Baseline)")
        prophet_path = os.path.join(BASE_DIR, "visualizations", "prophet_forecast.png")
        if os.path.exists(prophet_path):
             st.image(prophet_path, caption="Prophet Forecast", use_container_width=True)
        else:
            st.warning("Prophet model results not available (Installation issue).")

    elif page == "Simulation":
        st.header("Peak Shaving Simulation")
        
        st.write("Simulate the effect of setting a maximum grid capacity.")
        
        max_demand = int(df['Demand'].max())
        min_demand = int(df['Demand'].min())
        
        capacity_limit = st.slider("Max Capacity Limit (10k kW)", min_demand, max_demand, int(max_demand * 0.9))
        
        over_limit = df[df['Demand'] > capacity_limit]
        hours_over = len(over_limit)
        total_hours = len(df)
        percent_over = (hours_over / total_hours) * 100
        
        col1, col2 = st.columns(2)
        col1.metric("Hours Exceeding Limit", hours_over)
        col2.metric("Percentage Exceeding", f"{percent_over:.2f}%")
        
        st.subheader("Demand Duration Curve")
        sorted_demand = np.sort(df['Demand'].values)[::-1]
        fig, ax = plt.subplots()
        ax.plot(sorted_demand)
        ax.axhline(capacity_limit, color='r', linestyle='--', label='Limit')
        ax.set_ylabel("Demand")
        ax.set_xlabel("Hours")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
