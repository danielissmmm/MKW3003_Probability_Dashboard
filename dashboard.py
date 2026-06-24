import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

# Stock tree visualization using plotly
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[0, 1, 1],
    y=[0, -1, 1],
    mode='markers+text',
    marker=dict(size=50, color='lightblue', line=dict(color='black', width=2)),
    text=[f'RM{S0:.2f}', f'RM{down_price:.2f}', f'RM{up_price:.2f}'],
    textposition='middle center',
    hoverinfo='none'
))
fig.update_layout(
    title="Stock Price Tree",
    xaxis=dict(tickvals=[0, 1], ticktext=['T=0', 'T=1']),
    yaxis=dict(showticklabels=False),
    showlegend=False,
    height=400
)
st.plotly_chart(fig, use_container_width=True)

# Option payoff chart
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=['Up', 'Down'],
    y=[option_payoff, 0],
    marker_color=['green', 'red'],
    text=[f'P={p:.2f}', f'P={1-p:.2f}'],
    textposition='outside'
))
fig2.add_hline(y=expected_payoff, line_dash="dash", line_color="orange", 
               annotation_text=f'Expected: RM{expected_payoff:.2f}')
fig2.update_layout(
    title="Option Payoff Distribution",
    yaxis_title="Payoff Amount (RM)",
    height=400
)
st.plotly_chart(fig2, use_container_width=True)

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

# Normal distribution plot using plotly
x = np.linspace(m - 4*sigma, m + 4*sigma, 1000)
y = norm.pdf(x, m, sigma)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Normal Distribution', line=dict(color='blue')))
fig3.add_vline(x=m, line_dash="dash", line_color="red", annotation_text=f'Mean: {m:.2%}')
fig3.add_vline(x=interval_68[0], line_dash="dot", line_color="green", opacity=0.7)
fig3.add_vline(x=interval_68[1], line_dash="dot", line_color="green", opacity=0.7)
fig3.add_vline(x=interval_95[0], line_dash="dot", line_color="orange", opacity=0.7)
fig3.add_vline(x=interval_95[1], line_dash="dot", line_color="orange", opacity=0.7)

fig3.add_vrect(x0=interval_68[0], x1=interval_68[1], fillcolor="green", opacity=0.2, annotation_text="67%")
fig3.add_vrect(x0=interval_95[0], x1=interval_95[1], fillcolor="yellow", opacity=0.1, annotation_text="95%")

fig3.update_layout(
    title="Normal Distribution of Returns",
    xaxis_title="Return",
    yaxis_title="Probability Density",
    xaxis_tickformat='.0%',
    height=400
)
st.plotly_chart(fig3, use_container_width=True)

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

# Poisson distribution plot
max_x = int(max(20, lambda_param * 3))
x_poisson = np.arange(0, max_x + 1)
y_poisson = poisson.pmf(x_poisson, lambda_param)

fig4 = go.Figure()
colors = ['steelblue'] * len(x_poisson)
if k <= max_x:
    colors[k] = 'red'

fig4.add_trace(go.Bar(
    x=x_poisson, 
    y=y_poisson,
    marker_color=colors,
    text=[f'{v:.4f}' if v > 0.01 else '' for v in y_poisson],
    textposition='outside'
))
fig4.add_vline(x=lambda_param, line_dash="dash", line_color="red", annotation_text=f'Mean = {lambda_param:.2f}')
fig4.update_layout(
    title="Poisson Distribution of Order Arrivals",
    xaxis_title="Number of Orders per Millisecond",
    yaxis_title="Probability",
    height=400
)
st.plotly_chart(fig4, use_container_width=True)

# Simulation
with st.expander("📊 Click to see arrival simulation"):
    n_sim = st.slider("Number of milliseconds to simulate", 50, 500, 200)
    
    np.random.seed(42)
    simulated = np.random.poisson(lambda_param, n_sim)
    
    # Time series
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=list(range(n_sim)),
        y=simulated,
        mode='lines',
        name='Simulated',
        line=dict(color='blue')
    ))
    fig5.add_hline(y=lambda_param, line_dash="dash", line_color="red", annotation_text=f'Mean = {lambda_param:.2f}')
    fig5.update_layout(
        title="Simulated Arrivals Over Time",
        xaxis_title="Millisecond",
        yaxis_title="Number of Orders",
        height=300
    )
    st.plotly_chart(fig5, use_container_width=True)
    
    # Histogram
    hist_values, bin_edges = np.histogram(simulated, bins=range(max(simulated)+2), density=True)
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(
        x=bin_edges[:-1],
        y=hist_values,
        name='Empirical',
        marker_color='steelblue',
        opacity=0.7
    ))
    fig6.add_trace(go.Scatter(
        x=x_poisson,
        y=poisson.pmf(x_poisson, lambda_param),
        mode='lines+markers',
        name='Theoretical',
        line=dict(color='red', width=2)
    ))
    fig6.update_layout(
        title="Empirical vs Theoretical Distribution",
        xaxis_title="Number of Orders",
        yaxis_title="Probability",
        height=300
    )
    st.plotly_chart(fig6, use_container_width=True)
    
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Simulated Mean", f"{np.mean(simulated):.2f}")
    col_s2.metric("Simulated Variance", f"{np.var(simulated):.2f}")
    col_s3.metric("Max Arrivals", f"{max(simulated)}")

st.markdown("---")
st.caption("MKW3003 Probability Theory in Finance | Group Assignment")