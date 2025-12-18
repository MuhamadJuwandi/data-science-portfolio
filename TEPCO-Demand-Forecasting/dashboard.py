import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def main():
    st.set_page_config(page_title="TEPCO Demand Forecasting", layout="wide")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "EDA & Insights", "Model Performance", "Simulation"])
    
    # Logo
    if os.path.exists("assets/logo.jpg"):
        image = Image.open("assets/logo.jpg")
        st.sidebar.image(image, use_column_width=True)
    
    # Load Data
    @st.cache_data
    def load_data():
        df = pd.read_csv("data/processed/final_dataset.csv")
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        return df

    try:
        df = load_data()
    except:
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
            if os.path.exists(v):
                st.image(v, caption=os.path.basename(v), use_column_width=True)
            else:
                st.warning(f"Visualization {v} not found.")

    elif page == "Model Performance":
        st.header("Model Evaluation")
        
        st.write("### LSTM Forecast (PyTorch)")
        if os.path.exists("visualizations/lstm_forecast.png"):
            st.image("visualizations/lstm_forecast.png", caption="LSTM Forecast vs Actual", use_column_width=True)
            
        if os.path.exists("visualizations/lstm_loss.png"):
            st.image("visualizations/lstm_loss.png", caption="Training Loss", use_column_width=True)
            
        st.info("Metrics (MAE, RMSE, MAPE) are calculated during training and logged.")

        st.write("### Prophet Forecast (Baseline)")
        if os.path.exists("visualizations/prophet_forecast.png"):
             st.image("visualizations/prophet_forecast.png", caption="Prophet Forecast", use_column_width=True)
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
