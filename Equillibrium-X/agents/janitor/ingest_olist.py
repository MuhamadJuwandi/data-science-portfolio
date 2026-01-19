
import pandas as pd
import os
from datetime import timedelta

def ingest_olist_data():
    """
    Ingests Olist data, performs merging, and features engineering relative to 
    competitor pricing and logistics.
    """
    base_path = "data/raw/Olist"
    output_path = "data/processed/olist_processed.csv"
    
    print("[Janitor] Starting Olist Data Ingestion...")

    # 1. Load Data
    try:
        items = pd.read_csv(os.path.join(base_path, "olist_order_items_dataset.csv"))
        orders = pd.read_csv(os.path.join(base_path, "olist_orders_dataset.csv"))
        products = pd.read_csv(os.path.join(base_path, "olist_products_dataset.csv"))
        sellers = pd.read_csv(os.path.join(base_path, "olist_sellers_dataset.csv"))
        print("[OK] Data Loaded Successfully.")
    except FileNotFoundError as e:
        print(f"[ERROR] Error loading data: {e}")
        return

    # 2. Merge Data (Star Schema -> Flat Table)
    # Merge Items + Orders
    df = items.merge(orders, on="order_id", how="left")
    
    # Merge + Products
    df = df.merge(products, on="product_id", how="left")
    
    # Merge + Sellers
    df = df.merge(sellers, on="seller_id", how="left")
    
    print(f"[INFO] Merged DataFrame Shape: {df.shape}")

    # 3. Type Conversion
    time_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in time_cols:
        df[col] = pd.to_datetime(df[col])

    # 4. Feature Engineering: Logistics
    # Delivery Days (Actual)
    df['delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    
    # Clean negative or huge delivery days (outliers)
    df = df[(df['delivery_days'] >= 0) & (df['delivery_days'] < 100)]

    # 5. Feature Engineering: Competitor Price (The Core Task)
    print("[INFO] Calculating Competitor Prices (this might take a moment)...")
    
    # Create a 'month_year' granularity to find competitors in the same time window
    df['month_year'] = df['order_purchase_timestamp'].dt.to_period('M')

    # Group by Product and Month to find the 'Market Price'
    # We want the average price of this product in this month sold by ANYONE
    market_prices = df.groupby(['product_id', 'month_year'])['price'].mean().reset_index()
    market_prices.rename(columns={'price': 'market_avg_price'}, inplace=True)

    # Merge market price back
    df = df.merge(market_prices, on=['product_id', 'month_year'], how="left")

    # Now, logic for 'Avg Competitor Price':
    # If I am the only seller, competitor price is NULL (or we can use my price as proxy, but let's keep it clean first)
    # A simplified heuristic: avg_competitor_price ~= market_avg_price
    # (Refining this to exclude 'self' is expensive O(N^2), so we use market_avg for this MVP phase)
    
    df['avg_competitor_price'] = df['market_avg_price']
    
    # Create 'num_competitors' proxy
    competitor_counts = df.groupby(['product_id', 'month_year'])['seller_id'].nunique().reset_index()
    competitor_counts.rename(columns={'seller_id': 'num_active_sellers'}, inplace=True)
    df = df.merge(competitor_counts, on=['product_id', 'month_year'], how="left")

    # 6. Select Columns
    cols_to_keep = [
        'order_id', 'product_id', 'seller_id', 
        'order_purchase_timestamp', 'month_year',
        'price', 'freight_value', 
        'delivery_days', 
        'product_category_name',
        'avg_competitor_price', 'num_active_sellers'
    ]
    
    final_df = df[cols_to_keep]

    # 7. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Processed Data Saved to: {output_path}")

    print(final_df.head())

if __name__ == "__main__":
    ingest_olist_data()
