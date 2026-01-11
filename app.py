import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 1. SETTINGS & MODERN UI ---
st.set_page_config(page_title="Alpha-Quant Intelligence", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    div[data-testid="stMetric"] { background-color: #161b22; border-radius: 10px; border-left: 5px solid #00d4ff; padding: 15px; }
    .report-card { background-color: #1c2128; padding: 25px; border-radius: 15px; border: 1px solid #30363d; }
    .stButton>button { 
        background: linear-gradient(90deg, #00d4ff 0%, #0055ff 100%); 
        color: white; border-radius: 8px; font-weight: bold; border: none; height: 3.5em; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HEADER ---
st.title("⚖️ Alpha-Quant Portfolio Advisor")
st.markdown("Institutional Grade Risk Management & Historical Data Analysis")
st.divider()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Quant Controls")
    tickers_input = st.text_input("Enter Tickers (Separated by comma)", "RELIANCE.NS, TCS.NS, NVDA, AAPL")
    investment = st.number_input("Investment Capital", value=100000)
    period = st.selectbox("Lookback Period", ["1y", "2y", "5y", "max"])
    analyze_btn = st.button("RUN ENGINE")

if analyze_btn:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    try:
        with st.spinner('Accessing Global Market Data...'):
            # Download with auto_adjust to avoid 'Adj Close' issues
            data_raw = yf.download(tickers, period=period, auto_adjust=True)
            
            if data_raw.empty:
                st.error("Data fetch nahi hua. Symbols check karein.")
                st.stop()

            # Handle Multi-stock vs Single-stock
            df = data_raw['Close'] if len(tickers) > 1 else pd.DataFrame(data_raw['Close'], columns=tickers)
            df = df.ffill().dropna()
            returns = df.pct_change().dropna()

        # --- 4. CORE CALCULATIONS ---
        weights = np.array([1/len(tickers)] * len(tickers))
        port_ret = np.sum(returns.mean() * weights) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe = (port_ret - 0.05) / port_vol if port_vol != 0 else 0
        var_95 = np.percentile(returns.dot(weights), 5) # Value at Risk

        # --- 5. TOP METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Annual Return", f"{port_ret:.2%}")
        m2.metric("Portfolio Risk", f"{port_vol:.2%}")
        m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        m4.metric("Max 1-Day Loss (VaR)", f"₹{abs(var_95 * investment):,.0f}")

        # --- 6. AI VERDICT & GAUGE ---
        st.divider()
        v_col1, v_col2 = st.columns([1, 2])
        
        with v_col1:
            score = int(min(max((sharpe * 30) + 40, 0), 100))
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = score, title = {'text': "AI Health Score"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00d4ff"},
                         'steps' : [{'range': [0, 40], 'color': "#3e1c1c"},
                                    {'range': [40, 75], 'color': "#3e3e1c"},
                                    {'range': [75, 100], 'color': "#1c3e1c"}]}))
            fig_gauge.update_layout(height=280, template="plotly_dark", margin=dict(t=50, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with v_col2:
            st.subheader("🤖 Investment Recommendation")
            if score > 65:
                st.success("### ✅ VERDICT: STRONG BUY\nPortfolio is highly efficient. Diversification is optimal and returns are stable.")
            elif score > 40:
                st.warning("### ⚠️ VERDICT: NEUTRAL / HOLD\nModerate risk detected. AI suggests rebalancing after next quarterly results.")
            else:
                st.error("### ❌ VERDICT: AVOID / RESTRUCTURE\nHigh asset correlation found. This portfolio is vulnerable to market crashes.")

        # --- 7. TABS (GRAPHS & DIVERSIFICATION) ---
        t1, t2, t3 = st.tabs(["Performance Graph", "Risk Correlation", "Future Prediction"])
        
        with t1:
            norm_df = (df / df.iloc[0]) * 100
            fig_line = px.line(norm_df, title="Cumulative Performance (Normalized)", template="plotly_dark")
            st.plotly_chart(fig_line, use_container_width=True)

        with t2:
            if len(tickers) > 1:
                fig_corr = px.imshow(returns.corr(), text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark")
                st.plotly_chart(fig_corr, use_container_width=True)
                
            else:
                st.info("Correlation matrix dekhne ke liye 2+ stocks dalein.")

        with t3:
            st.subheader("Monte Carlo Simulation (Potential Future Paths)")
            mu, sigma = returns.dot(weights).mean(), returns.dot(weights).std()
            sim_results = []
            for _ in range(50): # 50 scenarios
                prices = [investment]
                for _ in range(252):
                    prices.append(prices[-1] * (1 + np.random.normal(mu, sigma)))
                sim_results.append(prices)
            
            fig_sim = go.Figure()
            for s in sim_results:
                fig_sim.add_trace(go.Scatter(y=s, mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
            fig_sim.update_layout(template="plotly_dark", xaxis_title="Trading Days", yaxis_title="Portfolio Value")
            st.plotly_chart(fig_sim, use_container_width=True)
            

        # --- 8. NEW: INTERACTIVE DATE LOOKUP ---
        st.divider()
        st.subheader("📅 Historical Price Lookup")
        st.markdown("Graph ke kisi bhi purane din ka data dekhne ke liye date select karein.")
        
        c_date, c_result = st.columns([1, 2])
        with c_date:
            target_date = st.date_input("Select Date", value=df.index[-1], min_value=df.index[0], max_value=df.index[-1])
        
        with c_result:
            # Finding nearest trading day (skips weekends/holidays)
            nearest_idx = df.index.get_indexer([pd.to_datetime(target_date)], method='nearest')[0]
            nearest_date = df.index[nearest_idx]
            prices_at_date = df.loc[nearest_date]
            
            st.write(f"**Data for: {nearest_date.strftime('%A, %d %B %Y')}**")
            cols = st.columns(len(tickers))
            for i, tick in enumerate(tickers):
                cols[i].metric(tick, f"₹{prices_at_date[tick]:,.2f}")

        # --- 9. DATA TABLE ---
        with st.expander("Show Full Price History Table"):
            st.dataframe(df.sort_index(ascending=False), use_container_width=True)

    except Exception as e:
        st.error(f"Quant Error: {e}")

else:
    st.info("👋 Welcome! Side menu mein stock tickers (e.g., RELIANCE.NS, TSLA) dalein aur 'RUN ENGINE' par click karein.")