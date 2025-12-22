
import pandas as pd
import os

# Paths
INPUT_DIR = 'dataset'
OUTPUT_FILE = os.path.join(INPUT_DIR, 'cleaned_data.pkl')

def load_data():
    orders = pd.read_csv(os.path.join(INPUT_DIR, 'olist_orders_dataset.csv'))
    items = pd.read_csv(os.path.join(INPUT_DIR, 'olist_order_items_dataset.csv'))
    customers = pd.read_csv(os.path.join(INPUT_DIR, 'olist_customers_dataset.csv'))
    products = pd.read_csv(os.path.join(INPUT_DIR, 'olist_products_dataset.csv'))
    reviews = pd.read_csv(os.path.join(INPUT_DIR, 'olist_order_reviews_dataset.csv'))
    payments = pd.read_csv(os.path.join(INPUT_DIR, 'olist_order_payments_dataset.csv'))
    translation = pd.read_csv(os.path.join(INPUT_DIR, 'product_category_name_translation.csv'))
    return orders, items, customers, products, reviews, payments, translation

def clean_data():
    print("Loading data...")
    orders, items, customers, products, reviews, payments, translation = load_data()

    print("Merging data...")
    # Merge Orders with Items
    df = orders.merge(items, on='order_id', how='left')
    
    # Merge with Customers
    df = df.merge(customers, on='customer_id', how='left')
    
    # Merge with Reviews
    df = df.merge(reviews, on='order_id', how='left')
    
    # Merge with Payments
    df = df.merge(payments, on='order_id', how='left')
    
    # Merge with Products
    df = df.merge(products, on='product_id', how='left')
    
    # Translate Product Categories
    print("Translating categories...")
    product_category_map = dict(zip(translation['product_category_name'], translation['product_category_name_english']))
    df['product_category_name_english'] = df['product_category_name'].map(product_category_map)
    
    # Standardize Dates
    print("Standardizing dates...")
    date_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 
                 'order_delivered_customer_date', 'order_estimated_delivery_date', 'review_creation_date', 
                 'review_answer_timestamp']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Handle Missing Values
    print("Handling missing values...")
    # Drop rows where essential IDs are missing (should be rare/none for left joins on main tables)
    df.dropna(subset=['order_id', 'customer_id'], inplace=True)
    
    # For delivery date, if missing, it might be not delivered yet or cancelled.
    # We will keep them but user can filter by status 'delivered' later.
    
    # Fill missing category names
    df['product_category_name_english'].fillna('unknown', inplace=True)

    print(f"Data shape after cleaning: {df.shape}")
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_pickle(OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    clean_data()
