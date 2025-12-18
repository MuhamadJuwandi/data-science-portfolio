import pandas as pd
import requests
import os
from datetime import datetime

def fetch_weather_data(output_file):
    print("Fetching Tokyo Weather Data from Open-Meteo...")
    
    # Tokyo Coordinates
    lat = 35.6895
    lon = 139.6917
    
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m&timezone=Asia%2FTokyo"
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch data: {response.status_code}")
        return

    data = response.json()
    
    hourly = data.get('hourly', {})
    if not hourly:
        print("No hourly data found in response")
        return

    df = pd.DataFrame({
        'Datetime': hourly['time'],
        'Temperature': hourly['temperature_2m'],
        'Humidity': hourly['relative_humidity_2m']
    })
    
    # Open-Meteo returns ISO8601, clean it
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"Saved weather data to {output_file}")
    print(df.head())

if __name__ == "__main__":
    output_path = "data/external/weather_tokyo.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fetch_weather_data(output_path)
