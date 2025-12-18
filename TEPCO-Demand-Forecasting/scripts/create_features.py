import pandas as pd
import numpy as np
import holidays
import os

def create_features(input_file, output_file):
    print("Creating features...")
    df = pd.read_csv(input_file)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime')
    
    # 1. Time Features
    df['Hour'] = df['Datetime'].dt.hour
    df['Month'] = df['Datetime'].dt.month
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'] >= 5
    
    # 2. Holidays (Japan)
    jp_holidays = holidays.JP()
    df['IsHoliday'] = df['Datetime'].dt.date.apply(lambda x: x in jp_holidays)
    
    # 3. Lag Features
    # Lag 1 hour, 24 hours (1 day), 168 hours (1 week)
    df['Lag_1h'] = df['Demand'].shift(1)
    df['Lag_24h'] = df['Demand'].shift(24)
    df['Lag_168h'] = df['Demand'].shift(168)
    
    # 4. Rolling Features
    # Rolling mean for past 24 hours
    df['RollingMean_24h'] = df['Demand'].shift(1).rolling(window=24).mean()
    
    # 5. Weather Interactions
    # Temp > 28 (Hot)
    df['Temp_Above_28'] = df['Temperature'].apply(lambda x: max(0, x - 28))
    
    # Drop NaNs created by lags
    initial_shape = df.shape
    df = df.dropna()
    print(f"Dropped {initial_shape[0] - df.shape[0]} rows due to lags.")
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"Saved featured dataset to {output_file}")
    print(df.head())
    print("Columns:", df.columns.tolist())

if __name__ == "__main__":
    input_path = "data/processed/final_dataset.csv"
    output_path = "data/processed/featured_dataset.csv"
    create_features(input_path, output_path)
