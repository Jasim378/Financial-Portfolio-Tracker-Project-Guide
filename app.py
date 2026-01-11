import streamlit as st
import streamlit_authenticator as stauth
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 1. SETTINGS & AUTH CONFIG ---
# Page config sabse pehle hona chahiye
st.set_page_config(page_title="Alpha-Quant Intelligence", layout="wide", page_icon="🏦")

# User Credentials
names = ['Jasim', 'Deloitte HR', 'Guest User']
usernames = ['jasim378', 'hr_deloitte', 'guest']
passwords = ['jasim123', 'admin123', 'guest123']

# Authenticator setup (Latest Syntax)
authenticator = stauth.Authenticate(
    {'usernames': {
        usernames[0]: {'name': names[0], 'password': passwords[0]},
        usernames[1]: {'name': names[1], 'password': passwords[1]},
        usernames[2]: {'name': names[2], 'password': passwords[2]}
    }},
    'portfolio_dashboard', 
    'auth_key',            
    cookie_expiry_days=30
)

# --- 2. LOGIN / SIGNUP UI ---
tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])

with tab1:
    # Latest version login call
    authenticator.login(location='main')

    # session_state se status check (TypeError Fix)
    authentication_status = st.session_state.get("authentication_status")
    name = st.session_state.get("name")

    st.divider()
    st.markdown("### 🌐 Or Login with (Enterprise SSO):")
    col_g1, col_g2 = st.columns(2)
    col_g1.button("Google", icon="🌐", use_container_width=True, help="Enterprise SSO integration in progress")
    col_g2.button("LinkedIn", icon="🔗", use_container_width=True, help="Enterprise SSO integration in progress")
    st.info("💡 **Demo Access:** User: `guest` | Pass: `guest123` (For HR Review)")

with tab2:
    try:
        # Fixed: Added location='main' for registration tab
        if authenticator.register_user(location='main'):
            st.success('User registered successfully! Now go to Login tab.')
    except Exception as e:
        st.error(f"Registration Error: {e}")

# --- 3. MAIN DASHBOARD LOGIC ---
if authentication_status == False:
    st.error('Username/password galat hai. Kripya dubara koshish karein.')
elif authentication_status == None:
    st.warning('Kripya login karein ya guest credentials use karein.')

elif authentication_status:
    # --- SUCCESSFUL LOGIN ---
    authenticator.logout(location='sidebar')
    st.sidebar.success(f'Authenticated: {name}')

    # Professional Theme CSS
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

    st.title("⚖️ Alpha-Quant Portfolio Advisor")
    st.markdown(f"**Institutional Grade Risk Analysis** | Welcome, {name}")
    st.divider()

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("⚙️ Quant Controls")
        tickers_input = st.text_input("Enter Tickers (Separated by comma)", "RELIANCE.NS, TCS.NS, NVDA, AAPL")
        investment = st.number_input("Investment Capital", value=100000)
        period = st.selectbox("Lookback Period", ["1y", "2y", "5y", "max"])
        analyze_btn = st.button("RUN ENGINE")

    if analyze_btn:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        
        try:
            with st.spinner('Calculating Financial Metrics...'):
                data_raw = yf.download(tickers, period=period, auto_adjust=True)
                
                if data_raw.empty:
                    st.error("Data fetch nahi hua. Symbols check karein.")
                else:
                    df = data_raw['Close'] if len(tickers) > 1 else pd.DataFrame(data_raw['Close'], columns=tickers)
                    df = df.ffill().dropna()
                    returns = df.pct_change().dropna()

                    # Quant Engine Calculations
                    weights = np.array([1/len(tickers)] * len(tickers))
                    port_ret = np.sum(returns.mean() * weights) * 252
                    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
                    sharpe = (port_ret - 0.05) / port_vol if port_vol != 0 else 0
                    var_95 = np.percentile(returns.dot(weights), 5)

                    # --- METRICS DISPLAY ---
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Annual Return", f"{port_ret:.2%}")
                    m2.metric("Portfolio Risk", f"{port_vol:.2%}")
                    m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    m4.metric("Max 1-Day Loss (VaR)", f"₹{abs(var_95 * investment):,.0f}")

                    # --- AI GAUGE ---
                    st.divider()
                    score = int(min(max((sharpe * 30) + 40, 0), 100))
                    c1, col_gauge = st.columns([1, 2])
                    with c1:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number", value=score, title={'text': "AI Health Score"},
                            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#00d4ff"},
                                   'steps': [{'range': [0, 40], 'color': "#3e1c1c"},
                                             {'range': [75, 100], 'color': "#1c3e1c"}]}))
                        fig_gauge.update_layout(height=280, template="plotly_dark", margin=dict(t=50, b=10))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with col_gauge:
                        st.subheader("🤖 AI Verdict")
                        if score > 65: st.success("### ✅ STRONG BUY\nEfficient portfolio with high Sharpe ratio.")
                        elif score > 40: st.warning("### ⚠️ HOLD\nModerate risk detected. Rebalancing suggested.")
                        else: st.error("### ❌ AVOID\nPoor risk-reward ratio. High volatility detected.")

                    # --- CHARTS TABS ---
                    t1, t2, t3 = st.tabs(["📈 Performance", "🔗 Correlation", "🔮 Forecast"])
                    
                    with t1:
                        st.plotly_chart(px.line((df / df.iloc[0]) * 100, template="plotly_dark", title="Cumulative Growth"), use_container_width=True)
                        
                    
                    with t2:
                        if len(tickers) > 1:
                            st.plotly_chart(px.imshow(returns.corr(), text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark"), use_container_width=True)
                            
                        else:
                            st.info("Correlation analysis ke liye 2+ stocks dalein.")
                    
                    with t3:
                        # Monte Carlo Simulation
                        mu, sigma = returns.dot(weights).mean(), returns.dot(weights).std()
                        sims = [investment * np.cumprod(1 + np.random.normal(mu, sigma, 252)) for _ in range(30)]
                        fig_sim = go.Figure()
                        for s in sims: fig_sim.add_trace(go.Scatter(y=s, mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
                        fig_sim.update_layout(template="plotly_dark", title="Monte Carlo 1-Year Forecast")
                        st.plotly_chart(fig_sim, use_container_width=True)

                    # --- HISTORICAL LOOKUP ---
                    st.divider()
                    st.subheader("📅 Historical Price Lookup")
                    target_date = st.date_input("Select Date", value=df.index[-1], min_value=df.index[0], max_value=df.index[-1])
                    nearest_idx = df.index.get_indexer([pd.to_datetime(target_date)], method='nearest')[0]
                    nearest_date = df.index[nearest_idx]
                    st.write(f"**Data for: {nearest_date.strftime('%d %B %Y')}**")
                    st.write(df.loc[nearest_date])

        except Exception as e:
            st.error(f"Quant Engine Error: {e}")
    else:
        st.info("👈 Side menu se stocks select karein aur RUN ENGINE dabayein.")