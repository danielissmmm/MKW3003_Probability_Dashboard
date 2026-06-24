import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

st.set_page_config(page_title="MKW3003 Dashboard", layout="wide")

st.title("📊 MKW3003 Probability Dashboard")
st.markdown("---")

# ============================================
# PROBLEM 1: BINOMIAL STOCK PRICE
# ============================================
st.header("Problem 1: Binomial Stock Price & Option Payoff")

col1, col2 = st.columns(2)

with col1:
    S0 = st.number_input("Stock Price (S₀)", value=100.0)
    u = st.number_input("Up Factor (u)", value=1.20)
    d = st.number_input("Down Factor (d)", value=0.85)
    p = st.slider("Probability of Up Move (p)", 0.0, 1.0, 0.6)

with col2:
    option_payoff = st.number_input("Option Payoff if Up (RM)", value=100.0)
    
    # Calculations
    up_price = S0 * u
    down_price = S0 * d
    
    Y_up = np.log(u)
    Y_down = np.log(d)
    E_Y = p * Y_up + (1 - p) * Y_down
    Var_Y = p * (Y_up**2) + (1-p) * (Y_down**2) - E_Y**2
    
    expected_payoff = p * option_payoff
    var_payoff = p * (option_payoff - expected_payoff)**2 + (1-p) * (0 - expected_payoff)**2
    
    st.write("**Results:**")
    st.write(f"Up Price: RM{up_price:.2f}")
    st.write(f"Down Price: RM{down_price:.2f}")
    st.write(f"Expected Log Return: {E_Y:.4f}")
    st.write(f"Variance of Log Return: {Var_Y:.4f}")
    st.write(f"Expected Option Payoff: RM{expected_payoff:.2f}")
    st.write(f"Variance of Option Payoff: RM{var_payoff:.2f}")

# Simple stock tree visualization using dataframe
st.subheader("Stock Price Tree")
tree_data = pd.DataFrame({
    'Period': ['T=0', 'T=1', 'T=1'],
    'Price': [S0, down_price, up_price],
    'Level': ['Start', 'Down', 'Up']
})
st.dataframe(tree_data, hide_index=True)

# Option payoff bar chart using st.bar_chart
st.subheader("Option Payoff Distribution")
payoff_data = pd.DataFrame({
    'Outcome': ['Up', 'Down'],
    'Payoff': [option_payoff, 0],
    'Probability': [p, 1-p]
})
st.bar_chart(payoff_data.set_index('Outcome')['Payoff'])
st.write(f"Expected Payoff: RM{expected_payoff:.2f}")
st.write(f"Variance: RM{var_payoff:.2f}")

st.markdown("---")

# ============================================
# PROBLEM 2: NORMAL DISTRIBUTION
# ============================================
st.header("Problem 2: Portfolio Risk - Normal Distribution")

col3, col4 = st.columns(2)

with col3:
    m = st.slider("Expected Return (m)", -0.20, 0.30, 0.08, 0.01, format="%.2f")
    sigma = st.slider("Volatility (σ)", 0.05, 0.40, 0.15, 0.01, format="%.2f")

with col4:
    interval_68 = (m - sigma, m + sigma)
    interval_95 = (m - 2*sigma, m + 2*sigma)
    
    st.write("**Empirical Rule Intervals:**")
    st.write(f"67% (1σ): [{interval_68[0]:.2%}, {interval_68[1]:.2%}]")
    st.write(f"95% (2σ): [{interval_95[0]:.2%}, {interval_95[1]:.2%}]")
    
    var_95 = norm.ppf(0.05, m, sigma)
    st.write(f"95% VaR: {var_95:.2%}")

# Normal distribution using st.line_chart
st.subheader("Normal Distribution of Returns")
x = np.linspace(m - 4*sigma, m + 4*sigma, 1000)
y = norm.pdf(x, m, sigma)
norm_data = pd.DataFrame({
    'Return': x,
    'Probability': y
})
st.line_chart(norm_data.set_index('Return'))

# Risk Critique
with st.expander("⚠️ Click to read: Why Gaussian Models Underestimate Risk"):
    st.markdown("""
    **Limitations of Normal Distribution in Finance:**
    
    1. **Fat Tails** - Real returns have more extreme events than normal distribution predicts
    2. **Volatility Clustering** - Volatility changes over time, especially during crises
    3. **Liquidity Risk** - Markets can become illiquid during crashes
    4. **2008 Crisis Example** - 6-sigma events occurred that should be virtually impossible
    
    **Conclusion:** Use fat-tailed distributions for better risk management.
    """)

st.markdown("---")

# ============================================
# PROBLEM 3: POISSON DISTRIBUTION
# ============================================
st.header("Problem 3: High-Frequency Trading - Poisson Distribution")

col5, col6 = st.columns(2)

with col5:
    lambda_param = st.number_input("Arrival Rate (λ per ms)", value=4.0, step=0.5, min_value=0.1)
    k = st.number_input("Number of Orders (k)", value=2, step=1, min_value=0)

with col6:
    prob_exact = poisson.pmf(k, lambda_param)
    prob_leq = poisson.cdf(k, lambda_param)
    
    st.write("**Poisson Properties:**")
    st.write(f"Expected Value E[X] = λ = {lambda_param:.2f}")
    st.write(f"Variance Var(X) = λ = {lambda_param:.2f}")
    st.write(f"P(X = {k}) = {prob_exact:.6f} ({prob_exact:.2%})")
    st.write(f"P(X ≤ {k}) = {prob_leq:.6f} ({prob_leq:.2%})")

# Poisson distribution using st.bar_chart
st.subheader("Poisson Distribution of Order Arrivals")
max_x = int(max(20, lambda_param * 3))
x_poisson = np.arange(0, max_x + 1)
y_poisson = poisson.pmf(x_poisson, lambda_param)
poisson_data = pd.DataFrame({
    'Orders': x_poisson,
    'Probability': y_poisson
})
st.bar_chart(poisson_data.set_index('Orders'))

# Highlight the specific k value
st.write(f"P(X = {k}) = {prob_exact:.6f} ({prob_exact:.2%})")

# Simulation
with st.expander("📊 Click to see arrival simulation"):
    n_sim = st.slider("Number of milliseconds to simulate", 50, 500, 200)
    
    np.random.seed(42)
    simulated = np.random.poisson(lambda_param, n_sim)
    
    # Show simulation data
    sim_data = pd.DataFrame({
        'Millisecond': range(n_sim),
        'Orders': simulated
    })
    st.line_chart(sim_data.set_index('Millisecond'))
    
    # Show statistics
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Simulated Mean", f"{np.mean(simulated):.2f}")
    col_s2.metric("Simulated Variance", f"{np.var(simulated):.2f}")
    col_s3.metric("Max Arrivals", f"{max(simulated)}")
    
    # Show histogram data
    hist_values = np.bincount(simulated) / n_sim
    hist_df = pd.DataFrame({
        'Orders': range(len(hist_values)),
        'Probability': hist_values
    })
    st.bar_chart(hist_df.set_index('Orders'))

st.markdown("---")

# Summary
st.subheader("📋 Summary of Key Results")

summary_data = {
    "Problem": [
        "Binomial Stock Price - Expected Log Return",
        "Binomial Stock Price - Variance of Log Return",
        "Binomial Option - Expected Payoff",
        "Normal Distribution - 67% Interval",
        "Normal Distribution - 95% Interval",
        "Poisson Distribution - Expected Value",
        "Poisson Distribution - Variance"
    ],
    "Value": [
        f"{E_Y:.6f}",
        f"{Var_Y:.6f}",
        f"RM{expected_payoff:.2f}",
        f"[{interval_68[0]:.2%}, {interval_68[1]:.2%}]",
        f"[{interval_95[0]:.2%}, {interval_95[1]:.2%}]",
        f"{lambda_param:.2f}",
        f"{lambda_param:.2f}"
    ]
}

st.dataframe(pd.DataFrame(summary_data), hide_index=True)

st.markdown("---")
st.caption("MKW3003 Probability Theory in Finance | Group Assignment")