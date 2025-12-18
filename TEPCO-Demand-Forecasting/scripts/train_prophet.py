import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np

def train_prophet():
    input_file = "data/processed/final_dataset.csv"
    model_output = "models/prophet_model.pkl"
    
    print("Loading data for Prophet...")
    df = pd.read_csv(input_file)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Prepare for Prophet: ds, y
    prophet_df = df[['Datetime', 'Demand', 'Temperature']].rename(columns={'Datetime': 'ds', 'Demand': 'y'})
    
    # Split Train/Test (Last 30 days as test for demo speed, or 2024?)
    # Let's start with a reasonable split. 2020-2023 Train, 2024-2025 Test.
    split_date = '2024-01-01'
    train = prophet_df[prophet_df['ds'] < split_date]
    test = prophet_df[prophet_df['ds'] >= split_date]
    
    print(f"Train size: {train.shape}, Test size: {test.shape}")
    
    # Model
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.add_country_holidays(country_name='JP')
    model.add_regressor('Temperature')
    
    print("Training Prophet model (this may take a while)...")
    model.fit(train)
    
    # Predict on Test
    print("Predicting...")
    future = test[['ds', 'Temperature']]
    forecast = model.predict(future)
    
    # Metrics
    y_true = test['y'].values
    y_pred = forecast['yhat'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Save Model
    joblib.dump(model, model_output)
    print(f"Model saved to {model_output}")
    
    # Plot forecast
    fig1 = model.plot(forecast)
    plt.savefig("visualizations/prophet_forecast.png")
    
    fig2 = model.plot_components(forecast)
    plt.savefig("visualizations/prophet_components.png")
    
    print("Visualizations saved.")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    train_prophet()
