import pandas as pd
import os

def process_tepco_data(raw_data_dir, output_file):
    print("Starting TEPCO data processing (v2)...")
    dfs = []
    
    # 1. Process 2020-2022 type files (juyo-*.csv)
    print("Looking for standard CSV files (2020-2022)...")
    for root, dirs, files in os.walk(raw_data_dir):
        for file in files:
            if file.endswith(".csv") and "juyo" in file:
                file_path = os.path.join(root, file)
                print(f"Reading {file_path}...")
                try:
                    df = pd.read_csv(file_path, encoding='shift_jis')
                    if len(df.columns) < 3:
                         df = pd.read_csv(file_path, encoding='shift_jis', skiprows=1)
                    
                    # Take first 3 columns
                    df = df.iloc[:, :3]
                    df.columns = ['Date', 'Time', 'Demand']
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    # 2. Process 2023+ type files (*_power_usage.csv) - Daily files
    print("Looking for daily power usage files (2023+)...")
    daily_files = []
    for root, dirs, files in os.walk(raw_data_dir):
        for file in files:
            if file.endswith("power_usage.csv"):
                daily_files.append(os.path.join(root, file))
    
    print(f"Found {len(daily_files)} daily power usage files.")
    
    count = 0
    for file in daily_files:
        try:
            with open(file, 'r', encoding='shift_jis') as f:
                lines = f.readlines()
            
            hourly_data = []
            capture = False
            for line in lines:
                line = line.strip()
                if not line:
                    if capture: break
                    continue
                
                if "DATE,TIME" in line and "当日実績(万kW)" in line:
                    capture = True
                    continue
                
                if capture:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        # parts[0]=Date, [1]=Time, [2]=Actual Demand
                        hourly_data.append({'Date': parts[0], 'Time': parts[1], 'Demand': parts[2]})
            
            if hourly_data:
                df_daily = pd.DataFrame(hourly_data)
                dfs.append(df_daily)
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} daily files...")
                    
        except Exception as e:
            print(f"Error reading daily file {file}: {e}")

    if not dfs:
        print("No data found!")
        return

    print("Concatenating all data...")
    full_df = pd.concat(dfs, ignore_index=True)
    
    print("Preprocessing...")
    # Clean Date and Time
    full_df['Datetime'] = pd.to_datetime(full_df['Date'] + ' ' + full_df['Time'], errors='coerce')
    full_df = full_df.dropna(subset=['Datetime'])
    
    # Handle Demand
    full_df['Demand'] = pd.to_numeric(full_df['Demand'], errors='coerce')
    
    # Remove duplicates if any
    full_df = full_df.drop_duplicates(subset=['Datetime'])
    
    # Sort
    full_df = full_df.sort_values('Datetime').reset_index(drop=True)
    
    # Save
    full_df.to_csv(output_file, index=False)
    print(f"Saved merged TEPCO data to {output_file}")
    print(f"Total rows: {len(full_df)}")
    print(full_df.head())
    print(full_df.tail())

if __name__ == "__main__":
    raw_dir = "data/raw"
    output_path = "data/processed/tepco_merged.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    process_tepco_data(raw_dir, output_path)
