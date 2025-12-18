import pandas as pd
import os

def merge_data(tepco_path, weather_path, output_path):
    print("Merging TEPCO and Weather data...")
    
    if not os.path.exists(tepco_path):
        print(f"Error: {tepco_path} not found.")
        return
    if not os.path.exists(weather_path):
        print(f"Error: {weather_path} not found.")
        return

    # Load data
    df_tepco = pd.read_csv(tepco_path)
    df_weather = pd.read_csv(weather_path)
    
    # Convert Datetime
    df_tepco['Datetime'] = pd.to_datetime(df_tepco['Datetime'])
    df_weather['Datetime'] = pd.to_datetime(df_weather['Datetime'])
    
    # Sort just in case
    df_tepco = df_tepco.sort_values('Datetime')
    df_weather = df_weather.sort_values('Datetime')
    
    # Merge
    # We want to keep all TEPCO data and match weather to it.
    # Weather data is hourly, TEPCO is hourly.
    
    merged_df = pd.merge_asof(
        df_tepco, 
        df_weather, 
        on='Datetime', 
        direction='nearest',
        tolerance=pd.Timedelta('10min')
    )
    
    # Metrics
    print(f"TEPCO shape: {df_tepco.shape}")
    print(f"Weather shape: {df_weather.shape}")
    print(f"Merged shape: {merged_df.shape}")
    
    # Check for NaNs
    missing = merged_df.isnull().sum()
    print("Missing values after merge:")
    print(missing[missing > 0])
    
    # Save
    merged_df.to_csv(output_path, index=False)
    print(f"Saved final dataset to {output_path}")
    print(merged_df.head())

if __name__ == "__main__":
    tepco_file = "data/processed/tepco_merged.csv"
    weather_file = "data/external/weather_tokyo.csv"
    output_file = "data/processed/final_dataset.csv"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merge_data(tepco_file, weather_file, output_file)
