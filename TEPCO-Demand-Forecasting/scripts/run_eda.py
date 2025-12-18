import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(input_file, output_dir):
    print("Running EDA...")
    df = pd.read_csv(input_file)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    
    # Feature extraction for EDA
    df['Hour'] = df.index.hour
    df['Month'] = df.index.month
    df['DayOfWeek'] = df.index.dayofweek
    df['Year'] = df.index.year
    
    # 1. Overall Time Series
    plt.figure(figsize=(15, 6))
    plt.plot(df['Demand'], label='Hourly Demand', alpha=0.5)
    plt.title('TEPCO Electricity Demand (2020-2025)')
    plt.ylabel('Demand (10k kW)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_overall_demand.png'))
    plt.close()
    
    # 2. Temperature vs Demand (Correlation)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Temperature', y='Demand', data=df, alpha=0.1, s=10)
    plt.title('Temperature vs Electricity Demand')
    plt.savefig(os.path.join(output_dir, '02_temp_vs_demand.png'))
    plt.close()
    
    # 3. Monthly Distributio (Boxplot)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Month', y='Demand', data=df)
    plt.title('Monthly Demand Distribution')
    plt.savefig(os.path.join(output_dir, '03_monthly_boxplot.png'))
    plt.close()
    
    # 4. Hourly Profile (Heatmap)
    # Pivot table: Index=Hour, Columns=Month, Values=Mean Demand
    pivot = df.pivot_table(index='Hour', columns='Month', values='Demand', aggfunc='mean')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, cmap='viridis')
    plt.title('Average Demand by Hour and Month')
    plt.savefig(os.path.join(output_dir, '04_hourly_heatmap.png'))
    plt.close()
    
    print(f"EDA plots saved to {output_dir}")

if __name__ == "__main__":
    input_path = "data/processed/final_dataset.csv"
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    run_eda(input_path, output_dir)
