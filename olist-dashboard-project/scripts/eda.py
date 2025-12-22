
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
INPUT_FILE = 'dataset/cleaned_data.pkl'
VISUALS_DIR = 'visuals'

# Setup
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    return pd.read_pickle(INPUT_FILE)

def plot_sales_trends(df):
    df['month_year'] = df['order_purchase_timestamp'].dt.to_period('M')
    sales_per_month = df.groupby('month_year')['price'].sum().reset_index()
    sales_per_month['month_year'] = sales_per_month['month_year'].astype(str)
    
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=sales_per_month, x='month_year', y='price', marker='o')
    plt.xticks(rotation=45)
    plt.title('Total Sales per Month', fontsize=16)
    plt.xlabel('Month-Year')
    plt.ylabel('Sales (BRL)')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'sales_per_month.png'))
    plt.close()

def plot_best_selling_products(df):
    top_products = df['product_category_name_english'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
    plt.title('Top 10 Best Selling Product Categories', fontsize=16)
    plt.xlabel('Number of Orders')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'best_selling_products.png'))
    plt.close()

def plot_payment_methods(df):
    payment_counts = df['payment_type'].value_counts()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=payment_counts.index, y=payment_counts.values, palette='Blues_d')
    plt.title('Most Popular Payment Methods', fontsize=16)
    plt.xlabel('Payment Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'payment_methods.png'))
    plt.close()

def plot_delivery_time(df):
    df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df['delivery_time'], bins=50, kde=True, color='purple')
    plt.title('Distribution of Delivery Time (Days)', fontsize=16)
    plt.xlabel('Days')
    plt.xlim(0, 50) # Limit outlier for better view
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'delivery_time_distribution.png'))
    plt.close()

def plot_review_scores(df):
    review_counts = df['review_score'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=review_counts.index, y=review_counts.values, palette='RdYlGn')
    plt.title('Distribution of Review Scores', fontsize=16)
    plt.xlabel('Review Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'review_score_distribution.png'))
    plt.close()

def run_eda():
    if not os.path.exists(VISUALS_DIR):
        os.makedirs(VISUALS_DIR)
        
    print("Loading data...")
    df = load_data()
    
    print("Generating visuals...")
    plot_sales_trends(df)
    plot_best_selling_products(df)
    plot_payment_methods(df)
    plot_delivery_time(df)
    plot_review_scores(df)
    
    print("EDA completed. Visuals saved to visuals/ folder.")

if __name__ == "__main__":
    run_eda()
