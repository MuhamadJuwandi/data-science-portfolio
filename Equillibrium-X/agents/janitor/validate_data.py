
import pandas as pd
import sys

def validate_olist_data():
    """
    Validates the augmented Olist dataset using Strict Pandas Assertions.
    (Plan B: Replacing Great Expectations for stability).
    """
    data_path = "data/processed/olist_processed_with_cost.csv"
    
    print("[Janitor] Starting Data Validation (Pandas Strict Mode)...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(data_path)
        print(f"[INFO] Data loaded: {data_path} | Rows: {len(df)}")
    except Exception as e:
        print(f"[CRITICAL] Could not load data: {e}")
        sys.exit(1)

    failures = []


    # 2. Rule A: Price Integrity
    # Logic: No nulls, must be > 0.01
    invalid_prices = df[ (df['price'].isnull()) | (df['price'] < 0.01) ]
    if not invalid_prices.empty:
        failures.append(f"Rule A Failed: {len(invalid_prices)} rows have invalid Price (<= 0 or NaNs).")
    else:
        print("[PASS] Rule A (Price Integrity)")

    # 3. Rule B: Cost Integrity
    invalid_costs = df[ (df['unit_cost'].isnull()) | (df['unit_cost'] < 0.01) ]
    if not invalid_costs.empty:
        failures.append(f"Rule B Failed: {len(invalid_costs)} rows have invalid Unit Cost.")
    else:
        print("[PASS] Rule B (Cost Integrity)")

    # 4. Rule C: Margin Sanity
    # We warn if margins are wild (< -50% or > 90%). 
    # Note: Since we engineered this, it should pass, but this protects against calculation errors.
    wild_margins = df[ (df['synthetic_margin_pct'] < -50) | (df['synthetic_margin_pct'] > 90) ]
    if not wild_margins.empty:
        # This might be a warning, not a hard fail? Let's make it a warning.
        print(f"[WARN] Rule C (Margin Sanity): WARNING - {len(wild_margins)} rows have suspicious margins.")
    else:
        print("[PASS] Rule C (Margin Sanity)")

    # Final Verdict
    if not failures:
        print("\n[SUCCESS] All Critical Validations Passed. Data is Clean.")
        sys.exit(0)
    else:
        print("\n[FAILURE] Validation Failed. See errors below:")
        for fail in failures:
            print(f"   - {fail}")
        sys.exit(1)


if __name__ == "__main__":
    validate_olist_data()
