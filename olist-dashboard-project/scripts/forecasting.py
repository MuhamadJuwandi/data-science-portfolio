
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

# Paths
INPUT_FILE = 'dataset/cleaned_data.pkl'
VISUALS_DIR = 'visuals'

# Mute warnings
import logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

def load_data():
    return pd.read_pickle(INPUT_FILE)

def calculate_weekly_sales(df, category):
    # Filter by category
    category_df = df[df['product_category_name_english'] == category].copy()
    
    # Group by week
    category_df['order_purchase_timestamp'] = pd.to_datetime(category_df['order_purchase_timestamp'])
    weekly_sales = category_df.groupby(pd.Grouper(key='order_purchase_timestamp', freq='W'))['price'].sum().reset_index()
    return weekly_sales

def run_prophet(df_sales, category):
    # Prepare for Prophet
    prophet_df = df_sales.rename(columns={'order_purchase_timestamp': 'ds', 'price': 'y'})
    
    # Model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(prophet_df)
    
    # Forecast
    future = model.make_future_dataframe(periods=12, freq='W')
    forecast = model.predict(future)
    
    # Plot
    fig1 = model.plot(forecast)
    plt.title(f'Sales Forecast for {category}')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, f'forecast_{category}.png'))
    plt.close()
    
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, f'forecast_components_{category}.png'))
    plt.close()

def run_forecasting():
    print("Loading data...")
    df = load_data()
    
    # Identify top categories
    top_categories = df['product_category_name_english'].value_counts().head(3).index.tolist()
    print(f"Top categories: {top_categories}")
    
    for category in top_categories:
        print(f"Forecasting for {category}...")
        try:
            weekly_sales = calculate_weekly_sales(df, category)
            # Ensure enough data points
            if len(weekly_sales) > 10:
                run_prophet(weekly_sales, category)
            else:
                print(f"Not enough data for {category}")
        except Exception as e:
            print(f"Error forecasting for {category}: {e}")
            
    print("Forecasting completed.")

if __name__ == "__main__":
    run_forecasting()
