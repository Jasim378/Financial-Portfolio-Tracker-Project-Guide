import streamlit as st
import streamlit_authenticator as stauth
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 1. USER AUTHENTICATION SETUP ---
# Inhe aap apne hisaab se badal sakte hain
names = ['Jasim', 'Deloitte HR']
usernames = ['jasim378', 'hr_deloitte']
passwords = ['jasim123', 'admin123']

# Passwords ko hash karna security ke liye
hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(
    {'usernames': {
        usernames[0]: {'name': names[0], 'password': hashed_passwords[0]},
        usernames[1]: {'name': names[1], 'password': hashed_passwords[1]}
    }},
    'portfolio_dashboard', 'auth_key', cookie_expiry_days=30
)

# Login Widget dikhana
name, authentication_status, username = authenticator.login('Login', 'main')

# --- 2. AUTHENTICATION LOGIC ---

if authentication_status == False:
    st.error('Username/password galat hai. Kripya sahi details bharein.')

elif authentication_status == None:
    st.info('👋 Alpha-Quant Advisor mein swagat hai. Kripya login karein.')

elif authentication_status:
    # --- SUCCESSFUL LOGIN: AB SAARA MAIN CODE START HOGA ---
    
    # 1. Page Config (Login ke baad wide layout)
    # Note: Page config login ke baad call ho raha hai taaki login screen centered rahe
    st.set_page_config(page_title="Alpha-Quant Intelligence", layout="wide", page_icon="🏦")
    
    # Sidebar mein Logout aur Welcome Message
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.success(f'Logged in as: {name}')

    # Custom CSS for Modern Dark Look
    st.markdown("""
        <style>
        .main { background-color: #0d1117; }
        div[data-testid="stMetric"] { background-color: #161b22; border-radius: 10px; border-left: 5px solid #00d4ff; padding: 15px; }
        .stButton>button { 
            background: linear-gradient(90deg, #00d4ff 0%, #0055ff 100%); 
            color: white; border-radius: 8px; font-weight: bold; border: none; height: 3.5em; width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)

    # --- 3. HEADER ---
    st.title("⚖️ Alpha-Quant Portfolio Advisor")
    st.markdown("Institutional Grade Risk Management & Historical Data Analysis")
    st.divider()

    # --- 4. SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("⚙️ Quant Controls")
        tickers_input = st.text_input("Enter Tickers (Separated by comma)", "RELIANCE.NS, TCS.NS, NVDA, AAPL")
        investment = st.number_input("Investment Capital", value=100000)
        period = st.selectbox("Lookback Period", ["1y", "2y", "5y", "max"])
        analyze_btn = st.button("RUN ENGINE")

    # --- 5. MAIN ANALYSIS ENGINE ---
    if analyze_btn:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        
        try:
            with st.spinner('Accessing Global Market Data...'):
                # Data download logic
                data_raw = yf.download(tickers, period=period, auto_adjust=True)
                
                if data_raw.empty:
                    st.error("Data fetch nahi hua. Symbols check karein.")
                else:
                    # Clean Data
                    df = data_raw['Close'] if len(tickers) > 1 else pd.DataFrame(data_raw['Close'], columns=tickers)
                    df = df.ffill().dropna()
                    returns = df.pct_change().dropna()

                    # --- 6. CORE CALCULATIONS ---
                    weights = np.array([1/len(tickers)] * len(tickers))
                    port_ret = np.sum(returns.mean() * weights) * 252
                    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
                    sharpe = (port_ret - 0.05) / port_vol if port_vol != 0 else 0
                    var_95 = np.percentile(returns.dot(weights), 5) # 95% Confidence VaR

                    # --- 7. TOP METRICS DISPLAY ---
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Annual Return", f"{port_ret:.2%}")
                    m2.metric("Portfolio Risk", f"{port_vol:.2%}")
                    m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    m4.metric("Max 1-Day Loss (VaR)", f"₹{abs(var_95 * investment):,.0f}")

                    # --- 8. AI VERDICT & GAUGE ---
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
                            st.success("### ✅ VERDICT: STRONG BUY\nPortfolio is highly efficient. Returns are stable relative to risk.")
                        elif score > 40:
                            st.warning("### ⚠️ VERDICT: NEUTRAL / HOLD\nModerate risk detected. Monitor asset correlations.")
                        else:
                            st.error("### ❌ VERDICT: AVOID\nPortfolio is vulnerable. High risk with insufficient returns.")

                    # --- 9. TABS (GRAPHS) ---
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
                            st.info("Correlation dikhane ke liye 2 ya usse zyada stocks enter karein.")

                    with t3:
                        st.subheader("Monte Carlo Simulation (Potential Future Paths)")
                        mu, sigma = returns.dot(weights).mean(), returns.dot(weights).std()
                        sim_results = []
                        for _ in range(50): 
                            prices = [investment]
                            for _ in range(252):
                                prices.append(prices[-1] * (1 + np.random.normal(mu, sigma)))
                            sim_results.append(prices)
                        
                        fig_sim = go.Figure()
                        for s in sim_results:
                            fig_sim.add_trace(go.Scatter(y=s, mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
                        fig_sim.update_layout(template="plotly_dark", xaxis_title="Trading Days", yaxis_title="Portfolio Value")
                        st.plotly_chart(fig_sim, use_container_width=True)

                    # --- 10. HISTORICAL LOOKUP ---
                    st.divider()
                    st.subheader("📅 Historical Price Lookup")
                    c_date, c_result = st.columns([1, 2])
                    with c_date:
                        target_date = st.date_input("Select Date", value=df.index[-1], min_value=df.index[0], max_value=df.index[-1])
                    
                    with c_result:
                        nearest_idx = df.index.get_indexer([pd.to_datetime(target_date)], method='nearest')[0]
                        nearest_date = df.index[nearest_idx]
                        prices_at_date = df.loc[nearest_date]
                        st.write(f"**Data for: {nearest_date.strftime('%A, %d %B %Y')}**")
                        st.write(prices_at_date)

        except Exception as e:
            st.error(f"Quant Error: {e}")

    else:
        st.info("👋 Welcome! Use the sidebar to enter tickers and run the AI engine.")