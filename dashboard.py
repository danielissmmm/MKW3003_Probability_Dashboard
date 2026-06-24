import streamlit as st
import sys
import os

# Force matplotlib to use Agg backend (no GUI)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    
    # Log return
    Y_up = np.log(u)
    Y_down = np.log(d)
    E_Y = p * Y_up + (1 - p) * Y_down
    Var_Y = p * (Y_up**2) + (1-p) * (Y_down**2) - E_Y**2
    
    # Option payoff
    expected_payoff = p * option_payoff
    var_payoff = p * (option_payoff - expected_payoff)**2 + (1-p) * (0 - expected_payoff)**2
    
    st.write("**Results:**")
    st.write(f"Up Price: RM{up_price:.2f}")
    st.write(f"Down Price: RM{down_price:.2f}")
    st.write(f"Expected Log Return: {E_Y:.4f}")
    st.write(f"Variance of Log Return: {Var_Y:.4f}")
    st.write(f"Expected Option Payoff: RM{expected_payoff:.2f}")
    st.write(f"Variance of Option Payoff: RM{var_payoff:.2f}")

# Visualizations for Problem 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Stock tree
nodes = [(0, 0, S0), (1, 0, down_price), (1, 1, up_price)]
x_pos = [0, 1, 1]
y_pos = [0, -1, 1]

for i, (x, y, price) in enumerate(zip(x_pos, y_pos, [S0, down_price, up_price])):
    ax1.scatter(x, y, s=300, c='lightblue', edgecolors='black')
    ax1.annotate(f'RM{price:.2f}', (x, y), ha='center', va='center')

ax1.plot([0,1], [0,-1], 'k-')
ax1.plot([0,1], [0,1], 'k-')
ax1.set_title("Stock Price Tree")
ax1.set_xticks([0,1])
ax1.set_xticklabels(['T=0', 'T=1'])
ax1.set_yticks([])

# Option payoff
ax2.bar(['Up', 'Down'], [option_payoff, 0], color=['green', 'red'])
ax2.axhline(y=expected_payoff, color='orange', linestyle='--', label=f'Expected: RM{expected_payoff:.2f}')
ax2.set_title("Option Payoff")
ax2.legend()
ax2.set_ylabel("Payoff (RM)")

st.pyplot(fig)

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
    # Calculate intervals
    interval_68 = (m - sigma, m + sigma)
    interval_95 = (m - 2*sigma, m + 2*sigma)
    
    st.write("**Empirical Rule Intervals:**")
    st.write(f"67% (1σ): [{interval_68[0]:.2%}, {interval_68[1]:.2%}]")
    st.write(f"95% (2σ): [{interval_95[0]:.2%}, {interval_95[1]:.2%}]")
    
    # VaR
    var_95 = norm.ppf(0.05, m, sigma)
    st.write(f"95% VaR: {var_95:.2%}")

# Plot normal distribution
fig, ax = plt.subplots(figsize=(10, 5))
x = np.linspace(m - 4*sigma, m + 4*sigma, 1000)
y = norm.pdf(x, m, sigma)

ax.plot(x, y, 'b-', linewidth=2)
ax.axvline(x=m, color='red', linestyle='--', label=f'Mean: {m:.2%}')
ax.axvline(x=interval_68[0], color='green', linestyle=':', alpha=0.7)
ax.axvline(x=interval_68[1], color='green', linestyle=':', alpha=0.7)
ax.axvline(x=interval_95[0], color='orange', linestyle=':', alpha=0.7)
ax.axvline(x=interval_95[1], color='orange', linestyle=':', alpha=0.7)

ax.fill_between(x, y, where=(x >= interval_68[0]) & (x <= interval_68[1]), alpha=0.3, color='green', label='67%')
ax.fill_between(x, y, where=(x >= interval_95[0]) & (x <= interval_95[1]), alpha=0.2, color='yellow', label='95%')

ax.set_xlabel("Return")
ax.set_ylabel("Probability Density")
ax.set_title("Normal Distribution of Returns")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))

st.pyplot(fig)

# Risk Critique
with st.expander("⚠️ Click to read: Why Gaussian Models Underestimate Risk"):
    st.markdown("""
    **Limitations of Normal Distribution in Finance:**
    
    1. **Fat Tails** - Real returns have more extreme events than normal distribution predicts
    2. **Volatility Clustering** - Volatility changes over time, especially during crises
    3. **Liquidity Risk** - Markets can become illiquid during crashes
    4. **2008 Crisis Example** - 6-sigma events occurred that should be virtually impossible
    
    **Conclusion:** Use fat-tailed distributions (Student's t, Extreme Value Theory) for better risk management.
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
    # Calculations
    prob_exact = poisson.pmf(k, lambda_param)
    prob_leq = poisson.cdf(k, lambda_param)
    
    st.write("**Poisson Properties:**")
    st.write(f"Expected Value E[X] = λ = {lambda_param:.2f}")
    st.write(f"Variance Var(X) = λ = {lambda_param:.2f}")
    st.write(f"P(X = {k}) = {prob_exact:.6f} ({prob_exact:.2%})")
    st.write(f"P(X ≤ {k}) = {prob_leq:.6f} ({prob_leq:.2%})")

# Plot Poisson distribution
fig, ax = plt.subplots(figsize=(10, 5))
max_x = int(max(20, lambda_param * 3))
x_poisson = np.arange(0, max_x + 1)
y_poisson = poisson.pmf(x_poisson, lambda_param)

bars = ax.bar(x_poisson, y_poisson, alpha=0.7, color='steelblue')
if k <= max_x:
    bars[k].set_color('red')
    bars[k].set_edgecolor('darkred')
    bars[k].set_linewidth(2)

ax.axvline(x=lambda_param, color='red', linestyle='--', label=f'Mean = {lambda_param:.2f}')
ax.set_xlabel("Number of Orders per Millisecond")
ax.set_ylabel("Probability")
ax.set_title("Poisson Distribution of Order Arrivals")
ax.legend()
ax.grid(True, alpha=0.3)

if k <= max_x:
    ax.text(k, y_poisson[k] + 0.01, f'P(X={k}) = {y_poisson[k]:.4f}', 
            ha='center', fontweight='bold', color='red')

st.pyplot(fig)

# Simulation
with st.expander("📊 Click to see arrival simulation"):
    n_sim = st.slider("Number of milliseconds to simulate", 50, 500, 200)
    
    np.random.seed(42)
    simulated = np.random.poisson(lambda_param, n_sim)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    ax1.plot(simulated, 'b-', alpha=0.7)
    ax1.axhline(y=lambda_param, color='red', linestyle='--', label=f'Mean = {lambda_param:.2f}')
    ax1.set_xlabel("Millisecond")
    ax1.set_ylabel("Orders")
    ax1.set_title("Simulated Arrivals Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(simulated, bins=range(max(simulated)+2), density=True, alpha=0.7, color='steelblue')
    x_hist = np.arange(0, max(simulated)+1)
    y_hist = poisson.pmf(x_hist, lambda_param)
    ax2.plot(x_hist, y_hist, 'r-', linewidth=2, label='Theoretical')
    ax2.set_xlabel("Orders")
    ax2.set_ylabel("Probability")
    ax2.set_title("Empirical vs Theoretical")
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Simulated Mean", f"{np.mean(simulated):.2f}")
    col_s2.metric("Simulated Variance", f"{np.var(simulated):.2f}")
    col_s3.metric("Max Arrivals", f"{max(simulated)}")

st.markdown("---")
st.caption("MKW3003 Probability Theory in Finance | Group Assignment")