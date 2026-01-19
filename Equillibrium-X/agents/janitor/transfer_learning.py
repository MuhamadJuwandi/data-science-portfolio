
import pandas as pd
import numpy as np
import os

def transfer_margins():
    """
    Extracts margin distribution from Dominick's Data (DFF) and applies it to Olist
    to estimate Unit Cost.
    """
    dff_path = "data/raw/Movement File.csv"
    olist_path = "data/processed/olist_processed.csv"
    output_path = "data/processed/olist_processed_with_cost.csv"

    print("[Janitor] Starting Domain Transfer (DFF -> Olist)...")

    # 1. Extract Margins from Dominick's (Robust Loading)
    print(f"[INFO] Loading Dominick's Data from {dff_path}...")
    
    # We only need PRICE and PROFIT columns. Reading in chunks to be safe with RAM.
    # DFF 'PROFIT' column is Gross Margin % according to our Blueprint.
    try:
        # Read a sample to infer types or check existance
        margin_samples = []
        chunk_size = 100000
        
        # We will read the first 1M rows which is sufficient for a robust distribution estimate
        # Reading the whole 400MB is fine, but sampling is faster for this demo
        for chunk in pd.read_csv(dff_path, usecols=['PRICE', 'PROFIT'], chunksize=chunk_size):
            # Clean Data
            # Remove negatives, zeros, and unrealistic margins (> 100% or < 0%)
            valid_chunk = chunk[
                (chunk['PROFIT'] > 0) & 
                (chunk['PROFIT'] < 100) & 
                (chunk['PRICE'] > 0)
            ]
            margin_samples.append(valid_chunk['PROFIT'])
            
            if len(margin_samples) * chunk_size > 1000000: # Stop after ~1M rows
                break
        
        all_margins = pd.concat(margin_samples)
        
        # Calculate Stats
        avg_margin = all_margins.mean()
        median_margin = all_margins.median()
        std_margin = all_margins.std()
        
        print(f"[INSIGHT] Learned Margins from Dominick's:")
        print(f"   - Mean Margin:   {avg_margin:.2f}%")
        print(f"   - Median Margin: {median_margin:.2f}%")
        print(f"   - Std Dev:       {std_margin:.2f}")
        
    except Exception as e:
        print(f"[ERROR] Failed to read Dominick's data: {e}")
        return

    # 2. Apply to Olist
    print(f"[INFO] Loading Olist Data from {olist_path}...")
    try:
        df_olist = pd.read_csv(olist_path)
    except FileNotFoundError:
        print("[ERROR] Olist processed data not found. Run ingest_olist.py first.")
        return

    # 3. Engineering 'unit_cost'
    # Strategy: We apply the MEDIAN margin as a safe baseline.
    # Blueprint Formula: COST = PRICE / (1 + (PROFIT/100))
    # Note: PROFIT here is treated as Markup % in the formula context given in PDF.
    
    # Let's use the Median Margin we found
    markup_factor = 1 + (median_margin / 100.0)
    
    print(f"[ACTION] Applying Market Markup Factor: {markup_factor:.4f}")
    
    df_olist['unit_cost'] = df_olist['price'] / markup_factor
    
    # Calculate synthetic profit for verification
    df_olist['synthetic_profit'] = df_olist['price'] - df_olist['unit_cost']
    df_olist['synthetic_margin_pct'] = (df_olist['synthetic_profit'] / df_olist['price']) * 100

    # 4. Verification
    # Check if Cost < Price
    invalid_costs = df_olist[df_olist['unit_cost'] >= df_olist['price']]
    if not invalid_costs.empty:
        print(f"[WARNING] Found {len(invalid_costs)} items with Cost >= Price. This shouldn't happen with the formula.")
    else:
        print("[PASS] Verification Successful: All Unit Costs are less than Price.")

    # 5. Save
    df_olist.to_csv(output_path, index=False)
    print(f"[SUCCESS] Augmented Dataset Saved to: {output_path}")
    print(df_olist[['price', 'unit_cost', 'synthetic_margin_pct']].head())

if __name__ == "__main__":
    transfer_margins()
