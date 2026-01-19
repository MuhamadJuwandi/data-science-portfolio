
import nashpy as nash
import numpy as np
import pandas as pd
import json
import os

def run_game_theory_simulation():
    """
    Agent Strategist: Determines Prices using Game Theory (Nash Equilibrium).
    Simulates a 2-Player Game (Us vs Competitor).
    """
    elasticity_path = "artifacts/elasticity.json"
    data_path = "data/processed/olist_processed_with_cost.csv"
    output_path = "artifacts/strategy_recommendation.json"
    
    print("[Strategist] Starting Game Theory Simulation...")

    # 1. Load Inputs (Elasticity & Baseline Specs)
    try:
        with open(elasticity_path, 'r') as f:
            elasticity_data = json.load(f)
            elasticity = elasticity_data['elasticity']
            print(f"[INFO] Loaded Elasticity: {elasticity:.4f}")
            
        df = pd.read_csv(data_path)
        # Use median values for a "Representative Product"
        baseline_price = df['price'].median()
        baseline_cost = df['unit_cost'].median()
        baseline_quantity = 1000 # Hypothetical monthly volume at baseline
        
        print(f"[INFO] Baseline Scenario: Price=${baseline_price:.2f}, Cost=${baseline_cost:.2f}")
        
    except Exception as e:
        print(f"[ERROR] Input Data Missing: {e}")
        return

    # 2. Define Strategies
    # Action Space: 0=Discount (-10%), 1=Normal (0%), 2=Premium (+10%)
    strategies = ["Discount (-10%)", "Normal (0%)", "Premium (+10%)"]
    price_multipliers = [0.90, 1.00, 1.10]
    
    # 3. Build Payoff Matrix
    # We need to calculate Profit for Player A (Us) and Player B (Competitor)
    # for every combination of strategies.
    
    payoff_matrix_A = np.zeros((3, 3))
    payoff_matrix_B = np.zeros((3, 3))
    
    # Assumptions for Cross-Price Elasticity (Substitutability)
    # If Competitor drops price, we lose volume. 
    # Usually Cross Elasticity is positive. Let's assume Cross E = 0.5 * |Own E|
    # Why? People switch, but not perfectly.
    cross_elasticity = 0.5 * abs(elasticity) 
    
    print(f"[INFO] Simulation Params: Own_E={elasticity:.4f}, Cross_E={cross_elasticity:.4f}")

    for i, strat_A in enumerate(price_multipliers):     # Our Move
        for j, strat_B in enumerate(price_multipliers): # Opponent Move
            
            # --- Player A (Us) Calculations ---
            pct_change_price_A = strat_A - 1.0
            pct_change_price_B = strat_B - 1.0
            
            # Demand Function: Q = Q0 * (1 + E*dP + CrossE*dP_comp)
            # Note: Elasticity is usually negative, so E*dP reduces Q if Price goes up.
            q_change_A = (elasticity * pct_change_price_A) + (cross_elasticity * pct_change_price_B)
            new_q_A = baseline_quantity * (1 + q_change_A)
            
            # Profit = (Price - Cost) * Q
            new_price_A = baseline_price * strat_A
            profit_A = (new_price_A - baseline_cost) * new_q_A
            
            # --- Player B (Competitor) Calculations ---
            # Symmetric Game assumption: They have same cost/elasticity structure
            q_change_B = (elasticity * pct_change_price_B) + (cross_elasticity * pct_change_price_A)
            new_q_B = baseline_quantity * (1 + q_change_B)
            
            new_price_B = baseline_price * strat_B
            profit_B = (new_price_B - baseline_cost) * new_q_B
            
            payoff_matrix_A[i, j] = profit_A
            payoff_matrix_B[i, j] = profit_B

    print("\n[MATRIX] Payoff Matrix (Our Profit):")
    print(payoff_matrix_A)

    # 4. Find Nash Equilibrium
    game = nash.Game(payoff_matrix_A, payoff_matrix_B)
    equilibria = list(game.support_enumeration())
    
    print(f"\n[SOLVER] Found {len(equilibria)} Nash Equilibrium(s).")
    
    recommended_strategy = None
    equilibrium_profit = 0
    is_prisoners_dilemma = False
    
    for eq in equilibria:
        # eq returns a tuple of probability vectors (strategy mix) for P1 and P2
        strat_A_prob, strat_B_prob = eq
        
        # Determine dominant strategy (highest probability)
        idx_A = np.argmax(strat_A_prob)
        idx_B = np.argmax(strat_B_prob)
        
        eq_strategy_A = strategies[idx_A]
        eq_strategy_B = strategies[idx_B]
        
        equilibrium_profit = payoff_matrix_A[idx_A, idx_B]
        
        print(f"   >>> Equilibrium Strategy: Us='{eq_strategy_A}' vs Comp='{eq_strategy_B}'")
        print(f"       Expected Profit: ${equilibrium_profit:,.2f}")
        
        recommended_strategy = eq_strategy_A
        
        # 5. Check for Prisoner's Dilemma
        # Definition: Equilibrium is suboptimal. (e.g. Both Discount)
        # But if Both Normal or Both Premium, profit would be higher.
        profit_coop = payoff_matrix_A[1, 1] # Normal/Normal
        profit_premium = payoff_matrix_A[2, 2] # Premium/Premium
        
        if equilibrium_profit < profit_coop or equilibrium_profit < profit_premium:
            is_prisoners_dilemma = True
            print("       [ALERT] Prisoner's Dilemma Detected! We are stuck in a suboptimal price war.")
            print(f"       Potential Profit (Cooperation): ${max(profit_coop, profit_premium):,.2f}")
        else:
            print("       [PASS] Optimal Stability: No better cooperative outcome exists.")


    # 6. Save Findings
    result = {
        "recommended_strategy": recommended_strategy,
        "equilibrium_profit": equilibrium_profit,
        "is_prisoners_dilemma": is_prisoners_dilemma,
        "payoff_matrix_us": payoff_matrix_A.tolist(),
        "baseline_metrics": {"price": baseline_price, "cost": baseline_cost}
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"\n[INFO] Strategy Recommendation saved to: {output_path}")

if __name__ == "__main__":
    run_game_theory_simulation()
