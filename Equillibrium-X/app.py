
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys

# Resolve the base directory of this script so all relative paths work
# correctly regardless of the working directory (critical for Streamlit Cloud).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import Supervisor Logic
sys.path.insert(0, BASE_DIR)
from agents.supervisor.brain import SupervisorAgent


st.set_page_config(
    page_title="Equillibrium-X | Strategic War Room",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Marketing-Grade CSS
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #F8F9FA !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    p, li, label {
        color: #B0B3B8;
    }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #1A1C24;
        border: 1px solid #30333F;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #9CA3AF;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF;
        font-size: 1.8rem;
    }
    
    /* Tabs styling */
    button[data-baseweb="tab"] {
        color: #9CA3AF;
        font-weight: 600;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60A5FA; /* Blue Accent */
        background-color: transparent;
        border-bottom: 2px solid #60A5FA;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161920;
        border-right: 1px solid #30333F;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data
def load_resources():
    """Load pre-computed artifacts (elasticity & strategy) from JSON files.
    Falls back to sensible defaults if files are missing (first run safety).
    """
    default_elasticity = -0.0067
    default_price = 74.90
    default_cost = 57.89
    
    try:
        elasticity_path = os.path.join(BASE_DIR, "artifacts", "elasticity.json")
        strategy_path = os.path.join(BASE_DIR, "artifacts", "strategy_recommendation.json")
        
        if os.path.exists(elasticity_path):
            with open(elasticity_path, 'r') as f:
                elas = json.load(f)['elasticity']
        else:
            elas = default_elasticity
            
        if os.path.exists(strategy_path):
            with open(strategy_path, 'r') as f:
                strat = json.load(f)
                price = strat['baseline_metrics']['price']
                cost = strat['baseline_metrics']['cost']
        else:
            price = default_price
            cost = default_cost
            
    except Exception:
        elas, price, cost = default_elasticity, default_price, default_cost
        
    return elas, price, cost

# --- 3. BUSINESS LOGIC (THE BRAIN) ---
# Logic has been moved to agents/supervisor/brain.py

# --- 4. UI ARCHITECTURE ---
def main():
    # Load Intelligence
    elasticity, base_price, base_cost = load_resources()
    
    # Initialize Supervisor Agent (The Brain)
    supervisor = SupervisorAgent(elasticity, base_price, base_cost)
    
    # SIDEBAR: CONTROL CENTER
    st.sidebar.title("🎛️ 2026 Scenario Control")
    st.sidebar.markdown("Simulate **'The 2024 Problem' (Logistics Crisis)**")
    
    cost_surge = st.sidebar.slider(
        "Logistic Cost Surge (%)", 
        min_value=0, 
        max_value=50, 
        value=0,
        help="Simulates increase in freight costs due to driver shortage."
    )
    
    competitor_aggro = st.sidebar.selectbox(
        "Competitor Behavior",
        ["Rational (Nash Equilibrium)", "Aggressive (Price War)"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Base Parameters**\n\nPrice: ${base_price:.2f}\nCost: ${base_cost:.2f}\nElasticity: {elasticity:.4f}")

    # MAIN TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Simulation (Strategy)", 
        "🏛️ Governance (Compliance)", 
        "♟️ War Room (Nash)", 
        "📑 Final Report"
    ])

    # --- TAB 1: THE COUNTER-INTUITIVE SIMULATION ---
    with tab1:
        st.title("Strategic Simulation Engine")
        st.markdown("### Hypothesis: High Prices Protect Profit during Logistic Shocks")
        
        # Generate Data for Curve
        price_points = np.linspace(0.8, 1.5, 50) # 80% to 150% of price
        results = []
        
        for p_mult in price_points:
            m = supervisor.simulate_scenario(p_mult, cost_surge/100)
            results.append(m)
            
        df_sim = pd.DataFrame(results)
        df_sim['price_pct'] = price_points * 100
        
        # Find Optimal Point
        optimal_idx = df_sim['profit'].idxmax()
        optimal_row = df_sim.iloc[optimal_idx]
        optimal_price_change = price_points[optimal_idx]
        
        # Dual Axis Plot
        fig = go.Figure()
        
        # Profit Line (Green)
        fig.add_trace(go.Scatter(
            x=df_sim['price_pct'], y=df_sim['profit'],
            name="Net Profit ($)",
            line=dict(color='#10B981', width=4),
            yaxis='y1'
        ))
        
        # Volume Line (Red Dashed)
        fig.add_trace(go.Scatter(
            x=df_sim['price_pct'], y=df_sim['market_share_idx'],
            name="Market Share (Vol %)",
            line=dict(color='#EF4444', width=2, dash='dot'),
            yaxis='y2'
        ))
        
        # Optimal Marker
        fig.add_vline(x=optimal_price_change*100, line_dash="dash", line_color="white", annotation_text="Optimal Strategy")
        
        fig.update_layout(
            title="Profitability vs. Market Share Trade-off",
            template="plotly_dark",
            height=500,
            xaxis_title="Price Strategy (% of Baseline)",
            yaxis=dict(title="Net Profit ($)", side="left"),
            yaxis2=dict(title="Market Share Volume (%)", side="right", overlaying="y", range=[0, 150]),
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insight Generator
        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal Price Change", f"{optimal_price_change*100:.0f}%", 
                    f"Target: ${optimal_row['price']:.2f}")
        col2.metric("Projected Profit", f"${optimal_row['profit']:,.0f}", 
                    delta=f"Impact of Surge: {cost_surge}%")
        col3.metric("Volume Shedding", f"{optimal_row['volume']:.0f}", 
                    f"{optimal_row['market_share_idx']-100:.1f}% vs Baseline", delta_color="inverse")
        
        st.success(f"**Counter-Intuitive Insight:** Even with a {cost_surge}% cost surge, the optimal move is to set price at **{optimal_price_change*100:.0f}%**. We sacrifice volume to protect margin, proving that 'Market Share' is a vanity metric in a crisis.")

    # --- TAB 2: GOVERNANCE (AB 325) ---
    with tab2:
        st.title("🛡️ Regulatory Compliance (AB 325)")
        st.markdown("Automated Antitrust Risk Detection System")
        
        st.info("Regulation AB 325 prohibits 'Tacit Collusion' where algorithms sustain prices >20% above marginal cost without demand justification.")
        
        # Check Current Optimal Strategy
        is_risk, reason = supervisor.check_compliance(optimal_row['price'], optimal_row['cost'])
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Implied Markup", f"{(optimal_row['price'] - optimal_row['cost']) / optimal_row['cost'] * 100:.1f}%")
        
        with col2:
            if is_risk:
                st.error(f"""
                ### ⚠️ RISK: POTENTIAL TACIT COLLUSION DETECTED
                **Status:** NON-COMPLIANT
                **Reason:** {reason}. The algorithm is sustaining high margins.
                **Action Required:** Log justification document 'Form 8-K' citing 'Logistics Cost Surge ({cost_surge}%)' as the driver.
                """)
            else:
                st.success("""
                ### ✅ COMPLIANT
                **Status:** SAFE
                **Reason:** Margins are within competitive bounds relative to costs.
                """)

    # --- TAB 3: WAR ROOM ---
    with tab3:
        st.title("♟️ Nash Equilibrium Matrix")
        st.markdown("Payoff Analysis: Us vs. Competitor")
        
        # Re-calc matrix based on sidebar cost
        strategies = ["Discount", "Normal", "Premium"]
        multipliers = [0.9, 1.0, 1.1]
        matrix = np.zeros((3, 3))
        
        for i, m_us in enumerate(multipliers):
            for j, m_comp in enumerate(multipliers):
                # Simple logic for visualization
                # If competitor is cheaper, we lose volume
                price_diff = (m_us - m_comp)
                vol_impact = -elasticity * price_diff * 500 # Simplified interaction
                
                # Base performance
                metrics = supervisor.simulate_scenario(m_us, cost_surge/100)
                metric_profit = metrics['profit']
                
                # Competitor Impact
                final_profit = metric_profit + vol_impact
                matrix[i, j] = final_profit
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=matrix,
            x=strategies,
            y=strategies,
            colorscale='Viridis',
            texttemplate="$%{z:,.0f}"
        ))
        fig_heat.update_layout(
            title="Projected Profit Payoff Matrix",
            xaxis_title="Competitor Strategy",
            yaxis_title="Our Strategy",
            template="plotly_dark"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # --- TAB 4: REPORT ---
    with tab4:
        st.title("📑 Final Portfolio Report")
        st.markdown("""
        ### Executive Summary: The 2026 Algorithmic Price War
        The 2026 market is characterized by logistic cost volatility ("The 2024 Problem"). Project Equilibrium-X demonstrates that a **Causal AI** approach outperforms traditional correlation-based predictions.
        
        ### Methodology
        1. **Causal Inference > Traditional ML**: We utilized DoWhy to isolate price causality from seasonal *confounders*.
        2. **Game Theory**: Nashpy was employed to determine strategic stability, ensuring long-term viability rather than just short-term optimization.
        
        ### Robustness Evidence (Placebo Test)
        The model has been validated using the `random_common_cause` refuter.
        - **Original Elasticity**: -0.0067 (Significant)
        - **Placebo Elasticity**: ~0.002 (Insignificant / Noise)
        *Conclusion: The model is robust and free from statistical hallucinations.*
        
        ### Business Outcomes
        The **Nash Equilibrium** strategy indicates that raising prices to the **Premium** level is the optimal defensive move during a logistics crisis. This strategy increases **Net Margin by 43%** while simultaneously reducing operational shipping burdens.
        """)
        
        # Generate a proper text report for download instead of a fake PDF
        report_text = """
=============================================================
  EQUILLIBRIUM-X: FINAL PORTFOLIO REPORT
  Autonomous Strategic Pricing Agent (2026 Edition)
=============================================================

1. EXECUTIVE SUMMARY
-------------------------------------------------------------
The 2026 market is characterized by logistic cost volatility 
("The 2024 Problem"). Project Equilibrium-X demonstrates that 
a Causal AI approach outperforms traditional correlation-based 
predictions.

2. METHODOLOGY
-------------------------------------------------------------
a) Causal Inference > Traditional ML:
   We utilized DoWhy to isolate price causality from seasonal 
   confounders.
   
b) Game Theory:
   Nashpy was employed to determine strategic stability, 
   ensuring long-term viability rather than just short-term 
   optimization.

3. ROBUSTNESS EVIDENCE (PLACEBO TEST)
-------------------------------------------------------------
The model has been validated using the random_common_cause 
refuter:
  - Original Elasticity: -0.0067 (Significant)
  - Placebo Elasticity:  ~0.002  (Insignificant / Noise)

Conclusion: The model is robust and free from statistical 
hallucinations.

4. BUSINESS OUTCOMES
-------------------------------------------------------------
The Nash Equilibrium strategy indicates that raising prices 
to the Premium level is the optimal defensive move during a 
logistics crisis. This strategy increases Net Margin by 43% 
while simultaneously reducing operational shipping burdens.

5. KEY PARAMETERS
-------------------------------------------------------------
  Base Price:    ${base_price:.2f}
  Base Cost:     ${base_cost:.2f}
  Elasticity:    {elasticity:.4f}

=============================================================
  Generated by Equillibrium-X | Strategic War Room
=============================================================
""".format(base_price=base_price, base_cost=base_cost, elasticity=elasticity)
        
        st.download_button(
            "📥 Download Report (.txt)", 
            data=report_text, 
            file_name="Equillibrium_X_Final_Report.txt",
            mime="text/plain"
        )


if __name__ == "__main__":
    main()
