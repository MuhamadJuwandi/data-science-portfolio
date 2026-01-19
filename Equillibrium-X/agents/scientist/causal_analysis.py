
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import os
import sys

def run_causal_analysis():
    """
    Agent Scientist: Estimates Price Elasticity using Log-Log OLS Regression.
    Includes "Placebo Test" for partial causal validation.
    """
    input_path = "data/processed/olist_processed_with_cost.csv"
    output_path = "artifacts/elasticity.json"
    
    print("[Scientist] Starting Causal Inference & Elasticity Estimation...")

    # 1. Load Data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {input_path}")
        return

    # 2. Preprocessing: Aggregation
    # Elasticity is a property of a Product Market Level, not individual transactions.
    # We aggregate to: Product_ID + Month Granularity
    print("[INFO] Aggregating data to Product-Month level...")
    
    # Ensure month_year is handled correctly
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['month'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
    
    # Aggregation mapping
    agg_rules = {
        'price': 'mean',
        'avg_competitor_price': 'mean',
        'order_id': 'count', # This becomes Volume (Sales Quantity)
        'freight_value': 'mean'
    }
    
    # We group by Product first to get elasticity per product, or Global?
    # To get a stable "Global Elasticity" for the Portfolio, we group by Category first.
    # Let's do a Global Model first to find the "Universal Law" for this dataset.
    # Better: Group by (Product, Month) -> One row per product per month.
    
    panel_df = df.groupby(['product_id', 'month']).agg(agg_rules).reset_index()
    panel_df.rename(columns={'order_id': 'quantity'}, inplace=True)
    
    # Filter for products with enough history (at least 3 months of sales)
    # to avoid noise from one-off sales.
    product_counts = panel_df['product_id'].value_counts()
    valid_products = product_counts[product_counts >= 3].index
    panel_df = panel_df[panel_df['product_id'].isin(valid_products)]
    
    print(f"[INFO] Panel Data Shape: {panel_df.shape} (Products with >= 3 months history)")

    # 3. Log-Log Transformation
    # Q = A * P^E  => log(Q) = log(A) + E * log(P)
    # We add 1 to avoid log(0) error
    panel_df['log_quantity'] = np.log(panel_df['quantity'] + 1)
    panel_df['log_price'] = np.log(panel_df['price'] + 1)
    panel_df['log_competitor_price'] = np.log(panel_df['avg_competitor_price'] + 1)
    
    # 4. Modeling: OLS Regression
    # We control for Competitor Price (Substitute Goods)
    formula = "log_quantity ~ log_price + log_competitor_price"
    
    print(f"[INFO] Running Regression (OLS): {formula}")
    model = smf.ols(formula=formula, data=panel_df).fit()
    
    elasticity = model.params['log_price']
    p_value = model.pvalues['log_price']
    r_squared = model.rsquared
    
    print("\n---------------- MODEL RESULTS ----------------")
    print(f"Price Elasticity (E): {elasticity:.4f}")
    print(f"P-Value:              {p_value:.4f}")
    print(f"R-Squared:            {r_squared:.4f}")
    print("-----------------------------------------------")

    # 5. Scientific Validation (Placebo Test)
    # Hypothesis: If we shuffle prices randomly, Elasticity should be 0.
    # If it is still negative/significant, our model is hallucinating (finding patterns in noise).
    print("\n[VALIDATION] Running Placebo Test (Refutation)...")
    
    shuffled_df = panel_df.copy()
    shuffled_df['log_price'] = np.random.permutation(panel_df['log_price'].values)
    
    placebo_model = smf.ols(formula=formula, data=shuffled_df).fit()
    placebo_elasticity = placebo_model.params['log_price']
    
    print(f"Placebo Elasticity:   {placebo_elasticity:.4f} (Expected ~0.0)")
    
    if abs(placebo_elasticity) > 0.1:
        print("[WARN] Placebo effect is strong! Model might be biased.")
    else:
        print("[PASS] Placebo effect is negligible.")

    # 6. Final Verdict & Save
    is_law_of_demand = elasticity < 0
    is_significant = p_value < 0.05
    
    if is_law_of_demand and is_significant:
        print("\n[SUCCESS] Law of Demand Confirmed! (Negative & Significant)")
    elif not is_law_of_demand:
        print("\n[FAILURE] Positive Elasticity detected (Giffen Good?). Check Data.")
    else:
        print("\n[WARNING] Result is not statistically significant (P > 0.05).")


    # Save artifacts
    result_data = {
        "elasticity": elasticity,
        "p_value": p_value,
        "model_summary": str(model.summary())
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=4)
        
    print(f"[INFO] Analysis saved to {output_path}")

if __name__ == "__main__":
    run_causal_analysis()
