import streamlit as st
import numpy as np
import pandas as pd
import math

st.set_page_config(page_title="MKW3003 Dashboard", layout="wide")

st.title("📊 MKW3003 Probability Dashboard")
st.markdown("---")

# ============================================
# HELPER FUNCTIONS (No scipy needed)
# ============================================

def normal_pdf(x, mean, std):
    """Normal distribution probability density function"""
    return (1 / (std * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mean) / std) ** 2)

def normal_cdf(x, mean, std):
    """Normal distribution cumulative distribution function (approximation)"""
    # Using error function approximation
    z = (x - mean) / std
    # Approximation for CDF
    if z < -6:
        return 0
    if z > 6:
        return 1
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1 if z >= 0 else -1
    z = abs(z)
    t = 1 / (1 + p * z)
    y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z / 2)
    return 0.5 * (1 + sign * (y - 0.5))

def poisson_pmf(k, lam):
    """Poisson distribution probability mass function"""
    if k < 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def poisson_cdf(k, lam):
    """Poisson distribution cumulative distribution function"""
    return sum(poisson_pmf(i, lam) for i in range(k + 1))

def norm_ppf(p, mean, std):
    """Normal distribution percent point function (approximation)"""
    # Approximation for inverse CDF
    if p <= 0:
        return -float('inf')
    if p >= 1:
        return float('inf')
    
    # Using rational approximation
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 
         0.0276438810333863, 0.0038405729373609, 0.0003951896511919, 
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
    
    x = p - 0.5
    if abs(x) < 0.42:
        r = x * x
        z = x * (((a[3] * r + a[2]) * r + a[1]) * r + a[0]) / ((((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1)
    else:
        if p < 0.5:
            r = math.sqrt(-2 * math.log(p))
        else:
            r = math.sqrt(-2 * math.log(1 - p))
        z = (((((c[8] * r + c[7]) * r + c[6]) * r + c[5]) * r + c[4]) * r + c[3]) * r + c[2]
        z = z * r + c[1]
        z = z * r + c[0]
        if p < 0.5:
            z = -z
    
    return mean + std * z

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
    
    Y_up = math.log(u)
    Y_down = math.log(d)
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

# Stock tree visualization
st.subheader("Stock Price Tree")
tree_data = pd.DataFrame({
    'Period': ['T=0', 'T=1', 'T=1'],
    'Price': [S0, down_price, up_price],
    'Level': ['Start', 'Down', 'Up']
})
st.dataframe(tree_data, hide_index=True)

# Option payoff
st.subheader("Option Payoff Distribution")
payoff_data = pd.DataFrame({
    'Outcome': ['Up', 'Down'],
    'Payoff': [option_payoff, 0],
    'Probability': [p, 1-p]
})
st.bar_chart(payoff_data.set_index('Outcome')['Payoff'])
st.write(f"Expected Payoff: RM{expected_payoff:.2f}")

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
    
    var_95 = norm_ppf(0.05, m, sigma)
    st.write(f"95% VaR: {var_95:.2%}")

# Normal distribution
st.subheader("Normal Distribution of Returns")
x_vals = np.linspace(m - 4*sigma, m + 4*sigma, 1000)
y_vals = [normal_pdf(x, m, sigma) for x in x_vals]
norm_data = pd.DataFrame({
    'Return': x_vals,
    'Probability': y_vals
})
st.line_chart(norm_data.set_index('Return'))

# Risk Critique
with st.expander("⚠️ Click to read: Why Gaussian Models Underestimate Risk"):
    st.markdown("""
    **Limitations of Normal Distribution in Finance:**
    
    1. **Fat Tails** - Real returns have more extreme events than normal predicts
    2. **Volatility Clustering** - Volatility changes during crises
    3. **Liquidity Risk** - Markets can become illiquid
    4. **2008 Crisis Example** - 6-sigma events occurred
    
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
    prob_exact = poisson_pmf(k, lambda_param)
    prob_leq = poisson_cdf(k, lambda_param)
    
    st.write("**Poisson Properties:**")
    st.write(f"Expected Value E[X] = λ = {lambda_param:.2f}")
    st.write(f"Variance Var(X) = λ = {lambda_param:.2f}")
    st.write(f"P(X = {k}) = {prob_exact:.6f} ({prob_exact:.2%})")
    st.write(f"P(X ≤ {k}) = {prob_leq:.6f} ({prob_leq:.2%})")

# Poisson distribution
st.subheader("Poisson Distribution of Order Arrivals")
max_x = int(max(20, lambda_param * 3))
x_poisson = range(max_x + 1)
y_poisson = [poisson_pmf(i, lambda_param) for i in x_poisson]
poisson_data = pd.DataFrame({
    'Orders': x_poisson,
    'Probability': y_poisson
})
st.bar_chart(poisson_data.set_index('Orders'))

# Highlight specific k
st.write(f"P(X = {k}) = {prob_exact:.6f} ({prob_exact:.2%})")

# Simulation
with st.expander("📊 Click to see arrival simulation"):
    n_sim = st.slider("Number of milliseconds to simulate", 50, 500, 200)
    
    np.random.seed(42)
    simulated = np.random.poisson(lambda_param, n_sim)
    
    sim_data = pd.DataFrame({
        'Millisecond': range(n_sim),
        'Orders': simulated
    })
    st.line_chart(sim_data.set_index('Millisecond'))
    
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Simulated Mean", f"{np.mean(simulated):.2f}")
    col_s2.metric("Simulated Variance", f"{np.var(simulated):.2f}")
    col_s3.metric("Max Arrivals", f"{max(simulated)}")

st.markdown("---")

# Summary
st.subheader("📋 Summary of Key Results")

summary_data = {
    "Problem": [
        "Binomial Stock - Expected Log Return",
        "Binomial Stock - Variance of Log Return",
        "Binomial Option - Expected Payoff",
        "Normal - 67% Interval",
        "Normal - 95% Interval",
        "Poisson - Expected Value",
        "Poisson - Variance"
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