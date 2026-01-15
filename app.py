import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Initialize NLTK for Sentiment Analysis
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Alpha-Quant Multi-Asset Terminal", layout="wide", page_icon="🌍")

# Professional Theme CSS
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #e6edf3; }
    div[data-testid="stMetric"] { 
        background-color: #161b22; border: 1px solid #30363d; 
        border-radius: 12px; padding: 20px; 
    }
    .stButton>button { 
        background: linear-gradient(90deg, #00d4ff 0%, #0055ff 100%); 
        color: white; border: none; font-weight: bold; width: 100%; height: 3.5em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ANALYTICS ENGINES ---
def get_sentiment(ticker):
    try:
        t = yf.Ticker(ticker)
        news = t.news[:3]
        if not news: return 0, "Neutral"
        sid = SentimentIntensityAnalyzer()
        score = np.mean([sid.polarity_scores(n['title'])['compound'] for n in news])
        label = "BULLISH" if score > 0.05 else "BEARISH" if score < -0.05 else "NEUTRAL"
        return score, label
    except: return 0, "N/A"

def get_tech_signals(data):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    sma_20 = data.rolling(window=20).mean()
    sma_50 = data.rolling(window=50).mean()
    return rsi.iloc[-1], sma_20.iloc[-1], sma_50.iloc[-1]

# --- 3. DASHBOARD UI ---
st.title("⚖️ Alpha-Quant Global Intelligence")
st.markdown("`Multi-Asset Risk Engine | Stocks • Crypto • Commodities • Forex`")
st.divider()

with st.sidebar:
    st.header("🌍 Global Asset Config")
    
    # Preset Selection
    preset = st.selectbox("Select Asset Preset", ["Custom", "Balanced Global", "Inflation Hedge", "High Growth Tech"])
    
    if preset == "Balanced Global":
        default_tickers = "AAPL, MSFT, GC=F, BTC-USD, EURUSD=X"
    elif preset == "Inflation Hedge":
        default_tickers = "GC=F, CL=F, TIP, XOM, IAU"
    elif preset == "High Growth Tech":
        default_tickers = "NVDA, TSLA, AMD, QQQ, ETH-USD"
    else:
        default_tickers = "RELIANCE.NS, TCS.NS, AAPL, GC=F, BTC-USD"

    tickers_input = st.text_area("Asset Symbols (Separate by comma)", default_tickers)
    benchmark_ticker = st.text_input("Benchmark (Alpha Ref)", "^GSPC") 
    capital = st.number_input("Capital Allocation", value=100000)
    history = st.selectbox("Lookback Period", ["1y", "2y", "5y"])
    
    st.divider()
    st.subheader("🛡️ Stress Test")
    crash_val = st.slider("Simulate Market Drop (%)", 0, 50, 15)
    run_engine = st.button("EXECUTE QUANT ENGINE")

# --- 4. ENGINE EXECUTION ---
if run_engine:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    try:
        with st.spinner('Syncing Global Markets...'):
            all_tickers = tickers + [benchmark_ticker]
            raw_data = yf.download(all_tickers, period=history)['Close']
            
            df = raw_data[tickers].ffill().dropna()
            bench_df = raw_data[benchmark_ticker].ffill().dropna()
            returns = df.pct_change().dropna()
            
            # Optimization: Inverse Volatility (Weights adjusted by risk)
            vols = returns.std() * np.sqrt(252)
            weights = (1/vols) / (1/vols).sum()
            
            # Portfolio Metrics
            port_daily_ret = returns.dot(weights)
            ann_ret = np.sum(returns.mean() * weights) * 252
            ann_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            sharpe = (ann_ret - 0.05) / ann_vol
            
            # Metrics Display
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Exp. Annual Return", f"{ann_ret:.2%}")
            m2.metric("Portfolio Volatility", f"{ann_vol:.2%}")
            m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
            m4.metric("Assets Analyzed", len(tickers))

            t1, t2, t3, t4 = st.tabs(["📊 Performance", "🎯 Asset Mix", "🧠 Signals & AI", "🔮 Risk Forecast"])

            with t1:
                combined_growth = pd.DataFrame({
                    "Portfolio": (1 + port_daily_ret).cumprod() * 100,
                    "Benchmark": (bench_df / bench_df.iloc[0]) * 100
                })
                st.plotly_chart(px.line(combined_growth, template="plotly_dark", title="Cumulative Growth (Portfolio vs Benchmark)"), use_container_width=True)
                
                drawdown = (combined_growth['Portfolio'] / combined_growth['Portfolio'].cummax()) - 1
                st.plotly_chart(px.area(drawdown, title="Maximum Drawdown Analysis", color_discrete_sequence=['#ff4b4b'], template="plotly_dark"), use_container_width=True)

            with t2:
                col_l, col_r = st.columns(2)
                with col_l:
                    st.plotly_chart(px.pie(values=weights, names=tickers, hole=0.5, template="plotly_dark", title="Optimized Weight Distribution"))
                with col_r:
                    # Heatmap to show cross-asset relationships
                    st.plotly_chart(px.imshow(returns.corr(), text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark", title="Cross-Asset Correlation Heatmap"), use_container_width=True)
                    

            with t3:
                st.subheader("Institutional Signals & Sentiment")
                sig_data = []
                for t in tickers:
                    score, label = get_sentiment(t)
                    rsi, sma20, sma50 = get_tech_signals(df[t])
                    trend = "🚀 Bullish" if sma20 > sma50 else "📉 Bearish"
                    sig_data.append({"Ticker": t, "Sentiment": label, "RSI (14d)": round(rsi,2), "MA Trend": trend})
                st.table(pd.DataFrame(sig_data))

            with t4:
                st.subheader("Monte Carlo Path Projection")
                mu, sigma = port_daily_ret.mean(), port_daily_ret.std()
                sims = [capital * np.cumprod(1 + np.random.normal(mu, sigma, 252)) for _ in range(30)]
                fig_sim = go.Figure()
                for s in sims: fig_sim.add_trace(go.Scatter(y=s, mode='lines', opacity=0.1, line=dict(color='#00d4ff'), showlegend=False))
                fig_sim.update_layout(template="plotly_dark", title="1-Year Capital Projection (30 Stochastic Paths)")
                st.plotly_chart(fig_sim, use_container_width=True)
                
                loss_val = capital * (crash_val / 100)
                st.error(f"🚨 Crash Scenario: A {crash_val}% drop results in a potential loss of ${loss_val:,.0f}")

    except Exception as e:
        st.error(f"Terminal Error: {e}")
else:
    st.info("👈 Select a preset or enter symbols, then click 'EXECUTE QUANT ENGINE'.")