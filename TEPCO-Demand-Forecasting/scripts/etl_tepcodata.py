import pandas as pd
import os
import glob

def process_tepco_data(raw_data_dir, output_file):
    print("Starting TEPCO data processing...")
    all_files = []
    # Recursively find all csv files in the raw data directory
    for root, dirs, files in os.walk(raw_data_dir):
        for file in files:
            if file.endswith(".csv") and "juyo" in file:
                all_files.append(os.path.join(root, file))
    
    if not all_files:
        print("No TEPCO data files found!")
        return

    dfs = []
    for file in all_files:
        print(f"Reading {file}...")
        try:
            # TEPCO CSVs usually have header at different lines or English/Japanese mess.
            # Inspecting standard TEPCO format: Date, Time, Demand
            # We will use encoding shift-jis sometimes for Japanese CSVs
            # Based on inspection, let's assume standard format but be robust.
            df = pd.read_csv(file, encoding='shift_jis') 
            
            # Check if columns need renaming
            # Expected: DATE, TIME, KW (or similar)
            # Usually column 0 is Date, 1 is Time, 2 is Demand
            if len(df.columns) < 3:
                # Try skipping rows if header is buried
                df = pd.read_csv(file, encoding='shift_jis', skiprows=1) # generic retry
            
            # Standardization
            # We assume the first 3 columns are what we need if names don't match
            df = df.iloc[:, :3]
            df.columns = ['Date', 'Time', 'Demand']
            
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not dfs:
        return

    full_df = pd.concat(dfs, ignore_index=True)
    
    # Preprocessing
    print("Preprocessing...")
    # Clean Date and Time
    full_df['Datetime'] = pd.to_datetime(full_df['Date'] + ' ' + full_df['Time'], errors='coerce')
    full_df = full_df.dropna(subset=['Datetime'])
    
    # Handle Demand
    full_df['Demand'] = pd.to_numeric(full_df['Demand'], errors='coerce')
    
    # Sort
    full_df = full_df.sort_values('Datetime').reset_index(drop=True)
    
    # Save
    full_df.to_csv(output_file, index=False)
    print(f"Saved merged TEPCO data to {output_file}")
    print(full_df.head())

if __name__ == "__main__":
    raw_dir = "data/raw"
    output_path = "data/processed/tepco_merged.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    process_tepco_data(raw_dir, output_path)
