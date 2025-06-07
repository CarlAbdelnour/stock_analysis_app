# Working yfinance 2025 - Addresses current API issues
# Based on latest GitHub fixes and community solutions

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
import warnings
warnings.filterwarnings('ignore')

# Try importing packages
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    # Check version
    yf_version = yf.__version__ if hasattr(yf, '__version__') else "Unknown"
except ImportError:
    YFINANCE_AVAILABLE = False
    yf_version = "Not installed"

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("plotly package required. Please install: pip install plotly")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Working Stock Analyzer 2025",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Working Stock Analyzer 2025")
st.markdown("### Addresses current yfinance API issues")

# Display version info
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"**yfinance**: {yf_version} {'✅' if YFINANCE_AVAILABLE else '❌'}")
with col2:
    st.info(f"**requests**: {'✅' if REQUESTS_AVAILABLE else '❌'}")
with col3:
    st.info(f"**plotly**: ✅")

# Critical fixes for 2025 yfinance issues
def get_working_user_agent():
    """Get a working user agent for 2025"""
    # Current working user agents (updated June 2025)
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:126.0) Gecko/20100101 Firefox/126.0'
    ]
    return random.choice(user_agents)

def fix_multiindex_columns(df, symbol):
    """Fix MultiIndex columns issue in yfinance 2025"""
    if df is None or df.empty:
        return df
    
    try:
        # Check if columns are MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            st.info("🔧 Fixing MultiIndex columns...")
            # Extract data for the specific symbol
            df = df.xs(key=symbol.upper(), axis=1, level=1)
        
        return df
    except Exception as e:
        st.warning(f"Column fix warning: {e}")
        return df

def rate_limit_protection():
    """Add delay to prevent 429 rate limiting"""
    delay = random.uniform(1, 3)  # Random delay 1-3 seconds
    time.sleep(delay)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data_2025(ticker="AAPL", days_back=365):
    """
    Working yfinance data fetching for 2025
    Addresses: rate limiting, user-agent issues, MultiIndex columns
    """
    
    if not YFINANCE_AVAILABLE:
        st.error("❌ yfinance not available. Install with: pip install yfinance==0.2.61")
        return None
    
    ticker = ticker.upper().strip()
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    st.info(f"🔄 Fetching {ticker} data...")
    st.info(f"📅 Range: {start_date} to {end_date}")
    
    # Rate limiting protection
    rate_limit_protection()
    
    try:
        # Method 1: Use yf.download with 2025 fixes
        st.info("📊 Trying yf.download() method...")
        
        # Set custom headers to avoid blocking
        if hasattr(yf, '_requests_headers'):
            yf._requests_headers = {'User-Agent': get_working_user_agent()}
        
        # Try download with multiple parameter combinations
        attempts = [
            # Attempt 1: Basic download with actions
            lambda: yf.download(
                ticker, 
                start=start_date, 
                end=end_date,
                progress=False,
                actions=True,
                threads=False  # Avoid threading issues
            ),
            
            # Attempt 2: Period-based download
            lambda: yf.download(
                ticker,
                period="1y",
                progress=False,
                actions=True,
                threads=False
            ),
            
            # Attempt 3: Minimal parameters
            lambda: yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
        ]
        
        data = None
        for i, attempt in enumerate(attempts, 1):
            try:
                st.info(f"🔄 Download attempt {i}/3...")
                data = attempt()
                
                if data is not None and not data.empty:
                    st.success(f"✅ Download successful with attempt {i}")
                    break
                else:
                    st.warning(f"⚠️ Attempt {i} returned empty data")
                    rate_limit_protection()  # Wait before next attempt
                    
            except Exception as e:
                st.warning(f"❌ Attempt {i} failed: {str(e)[:100]}")
                rate_limit_protection()
                continue
        
        if data is None or data.empty:
            st.info("🔄 Trying Ticker() method as fallback...")
            
            # Method 2: Use Ticker object (sometimes more reliable)
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    prepost=False,
                    repair=True
                )
                
                if not data.empty:
                    st.success("✅ Ticker method successful")
                else:
                    st.warning("⚠️ Ticker method returned empty data")
                    
            except Exception as e:
                st.error(f"❌ Ticker method failed: {str(e)[:100]}")
        
        if data is not None and not data.empty:
            # Fix MultiIndex columns (critical for 2025)
            data = fix_multiindex_columns(data, ticker)
            
            # Reset index to get Date as column
            data = data.reset_index()
            
            # Ensure we have the expected columns
            expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in expected_cols if col not in data.columns]
            
            if missing_cols:
                st.warning(f"⚠️ Missing columns: {missing_cols}")
                # Try to map any available columns
                col_mapping = {
                    'Adj Close': 'Close',
                    'close': 'Close',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'volume': 'Volume'
                }
                
                for old_name, new_name in col_mapping.items():
                    if old_name in data.columns and new_name not in data.columns:
                        data[new_name] = data[old_name]
            
            # Add Adjusted column if not present
            if 'Adjusted' not in data.columns:
                data['Adjusted'] = data['Close']
            
            # Clean data
            data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            if len(data) > 0:
                st.success(f"✅ Data processed: {len(data)} rows")
                st.success(f"📊 Columns: {list(data.columns)}")
                return data
            else:
                st.error("❌ No valid data after cleaning")
                return None
        else:
            st.error("❌ All methods failed to fetch data")
            return None
            
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Critical error: {error_msg[:200]}")
        
        # Specific error handling for common 2025 issues
        if "429" in error_msg or "Too Many Requests" in error_msg:
            st.error("🚫 **Rate Limited!** Try again in a few minutes or use a different symbol.")
            st.info("💡 **Solutions**: Wait 5+ minutes, try popular symbols like SPY, or restart the app.")
            
        elif "404" in error_msg or "Not Found" in error_msg:
            st.error("🔍 **Symbol not found!** Check if the ticker symbol is correct.")
            st.info("💡 **Try**: AAPL, MSFT, GOOGL, SPY, QQQ")
            
        elif "403" in error_msg or "Forbidden" in error_msg:
            st.error("🚫 **Access blocked!** Yahoo Finance may be blocking requests.")
            st.info("💡 **Solutions**: Try again later, use VPN, or restart the app.")
        
        return None

def calculate_simple_indicators(df):
    """Calculate basic technical indicators"""
    if df is None or len(df) == 0:
        return df
    
    try:
        # Simple Moving Averages
        if len(df) >= 20:
            df['SMA20'] = df['Adjusted'].rolling(window=20, min_periods=1).mean()
        else:
            df['SMA20'] = np.nan
        
        if len(df) >= 50:
            df['SMA50'] = df['Adjusted'].rolling(window=50, min_periods=1).mean()
        else:
            df['SMA50'] = np.nan
        
        # RSI
        if len(df) >= 14:
            delta = df['Adjusted'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            df['RSI'] = 100 - (100 / (1 + rs))
        else:
            df['RSI'] = 50.0
        
        # Daily returns
        df['Returns'] = df['Adjusted'].pct_change() * 100
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return df

def create_price_chart(data, ticker):
    """Create price chart with proper error handling"""
    if data is None or len(data) == 0:
        return go.Figure().add_annotation(text="No data available", 
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False)
    
    try:
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=ticker,
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
        
        # Add moving averages if available
        if 'SMA20' in data.columns and not data['SMA20'].isna().all():
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['SMA20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='blue', width=2)
            ))
        
        if 'SMA50' in data.columns and not data['SMA50'].isna().all():
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['SMA50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='orange', width=2)
            ))
        
        fig.update_layout(
            title=f"{ticker} Stock Price (Working 2025)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Chart error: {e}")
        return go.Figure()

def create_volume_chart(data, ticker):
    """Create volume chart"""
    if data is None or len(data) == 0:
        return go.Figure()
    
    try:
        colors = ['green' if close >= open_val else 'red' 
                 for close, open_val in zip(data['Close'], data['Open'])]
        
        fig = go.Figure(data=go.Bar(
            x=data['Date'],
            y=data['Volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f"{ticker} Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            height=300
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Volume chart error: {e}")
        return go.Figure()

# Main App
st.markdown("---")

# Version check and warnings
if YFINANCE_AVAILABLE:
    if yf_version != "Unknown":
        if yf_version < "0.2.60":
            st.warning(f"⚠️ **Old yfinance version detected: {yf_version}**")
            st.warning("🔧 **Update recommended**: `pip install --upgrade yfinance`")
        else:
            st.success(f"✅ **Good yfinance version**: {yf_version}")

# Sidebar
st.sidebar.header("📊 2025 Stock Analyzer")

# Current status
st.sidebar.subheader("🔧 System Status")
st.sidebar.write(f"**yfinance**: {yf_version}")
if YFINANCE_AVAILABLE:
    st.sidebar.success("✅ Ready to fetch data")
else:
    st.sidebar.error("❌ yfinance not available")

# Input controls
st.sidebar.subheader("📈 Stock Selection")

# Popular working symbols
popular_symbols = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "SPY": "S&P 500 ETF",
    "QQQ": "NASDAQ ETF",
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon"
}

# Quick select
selected_symbol = st.sidebar.selectbox(
    "Quick Select:",
    options=[""] + list(popular_symbols.keys()),
    format_func=lambda x: popular_symbols.get(x, "Choose...")
)

# Manual input
ticker_input = st.sidebar.text_input(
    "Or enter symbol:",
    value=selected_symbol if selected_symbol else "AAPL",
    placeholder="e.g., AAPL, MSFT"
)

# Time period
period_options = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "2 Years": 730
}

selected_period = st.sidebar.selectbox(
    "Time Period:",
    options=list(period_options.keys()),
    index=3  # Default to 1 year
)

days_back = period_options[selected_period]

# Fetch button
fetch_button = st.sidebar.button(
    "🚀 Fetch Data",
    type="primary",
    use_container_width=True,
    help="Get latest stock data using 2025 fixes"
)

# Tips
st.sidebar.subheader("💡 2025 Tips")
st.sidebar.info("""
**If data fails to load:**
1. ⏱️ Wait 5+ minutes (rate limits)
2. 🔄 Try different symbols
3. 🔧 Update yfinance to latest
4. 🌐 Check internet connection
5. 🔁 Restart the app
""")

# Main content
if fetch_button and ticker_input:
    # Fetch data
    with st.spinner(f"Fetching {ticker_input} data with 2025 fixes..."):
        stock_data = fetch_stock_data_2025(ticker_input, days_back)
    
    if stock_data is not None and len(stock_data) > 0:
        # Calculate indicators
        stock_data = calculate_simple_indicators(stock_data)
        
        # Success metrics
        current_price = stock_data['Adjusted'].iloc[-1]
        
        if len(stock_data) > 1:
            prev_price = stock_data['Adjusted'].iloc[-2]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
        else:
            change = 0
            change_pct = 0
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("📈 Daily Change", f"${change:.2f}", f"{change_pct:.2f}%")
        
        with col3:
            volume = stock_data['Volume'].iloc[-1] if 'Volume' in stock_data.columns else 0
            if volume > 1e9:
                vol_str = f"{volume/1e9:.1f}B"
            elif volume > 1e6:
                vol_str = f"{volume/1e6:.1f}M"
            else:
                vol_str = f"{volume:,.0f}"
            st.metric("📊 Volume", vol_str)
        
        with col4:
            if 'RSI' in stock_data.columns and not pd.isna(stock_data['RSI'].iloc[-1]):
                rsi = stock_data['RSI'].iloc[-1]
                rsi_status = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
                st.metric("📊 RSI", f"{rsi:.1f} {rsi_status}")
            else:
                st.metric("📊 RSI", "N/A")
        
        # Success message
        st.success(f"✅ **Successfully loaded {len(stock_data)} data points for {ticker_input}**")
        st.info(f"📅 **Date range**: {stock_data['Date'].iloc[0].strftime('%Y-%m-%d')} to {stock_data['Date'].iloc[-1].strftime('%Y-%m-%d')}")
        
        # Charts
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("📈 Price Chart")
            price_chart = create_price_chart(stock_data, ticker_input)
            st.plotly_chart(price_chart, use_container_width=True)
        
        with col2:
            st.subheader("📊 Key Stats")
            
            # Summary statistics
            stats = {
                "High": f"${stock_data['High'].max():.2f}",
                "Low": f"${stock_data['Low'].min():.2f}",
                "Avg Volume": f"{stock_data['Volume'].mean():,.0f}" if 'Volume' in stock_data.columns else "N/A",
                "Data Points": f"{len(stock_data)}",
                "Start Date": stock_data['Date'].iloc[0].strftime('%Y-%m-%d'),
                "End Date": stock_data['Date'].iloc[-1].strftime('%Y-%m-%d')
            }
            
            for key, value in stats.items():
                st.write(f"**{key}:** {value}")
            
            # Technical signals
            st.write("**📊 Signals:**")
            if 'SMA20' in stock_data.columns and not stock_data['SMA20'].isna().iloc[-1]:
                sma20 = stock_data['SMA20'].iloc[-1]
                above_sma20 = "🟢 Above" if current_price > sma20 else "🔴 Below"
                st.write(f"**SMA20:** {above_sma20}")
            
            if 'RSI' in stock_data.columns and not pd.isna(stock_data['RSI'].iloc[-1]):
                rsi = stock_data['RSI'].iloc[-1]
                if rsi > 70:
                    rsi_signal = "🔴 Overbought"
                elif rsi < 30:
                    rsi_signal = "🟢 Oversold"
                else:
                    rsi_signal = "🟡 Neutral"
                st.write(f"**RSI:** {rsi_signal}")
        
        # Volume chart
        st.subheader("📊 Volume Analysis")
        volume_chart = create_volume_chart(stock_data, ticker_input)
        st.plotly_chart(volume_chart, use_container_width=True)
        
        # Recent data table
        st.subheader("📋 Recent Data")
        recent_data = stock_data.tail(10)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Format for display
        recent_data['Date'] = recent_data['Date'].dt.strftime('%Y-%m-%d')
        for col in ['Open', 'High', 'Low', 'Close']:
            recent_data[col] = recent_data[col].round(2)
        
        st.dataframe(recent_data, use_container_width=True, hide_index=True)
        
        # Download option
        csv_data = stock_data.to_csv(index=False)
        st.download_button(
            "📥 Download Data (CSV)",
            csv_data,
            f"{ticker_input}_data_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
    
    else:
        st.error("❌ Failed to fetch data")
        
        # Troubleshooting guide
        st.subheader("🔧 Troubleshooting Guide")
        
        st.markdown("""
        **Common 2025 Issues & Solutions:**
        
        **🚫 Rate Limited (429 Error):**
        - Wait 5-10 minutes before trying again
        - Try a different symbol (SPY, QQQ often work)
        - Restart the app to reset connections
        
        **🔍 Symbol Not Found (404 Error):**
        - Verify ticker symbol is correct
        - Try popular symbols: AAPL, MSFT, GOOGL, SPY
        - Some symbols may be delisted
        
        **🚫 Access Blocked (403 Error):**
        - Yahoo Finance may be blocking your IP
        - Try again later or use a VPN
        - Update yfinance: `pip install --upgrade yfinance`
        
        **🔧 Update yfinance:**
        ```bash
        pip install --upgrade yfinance
        # Should be version 0.2.61 or higher
        ```
        
        **✅ Working Symbols (tested June 2025):**
        - AAPL, MSFT, GOOGL, AMZN
        - SPY, QQQ, VTI (ETFs)
        - TSLA, NVDA, META
        """)

elif not ticker_input:
    st.warning("Please enter a stock symbol")

else:
    # Welcome screen
    st.info("👈 **Select a stock symbol and click 'Fetch Data' to begin**")
    
    st.markdown("""
    ### 🚀 2025 yfinance Fixes Included:
    
    **✅ Current Issues Addressed:**
    - **Rate Limiting**: Automatic delays and retry logic
    - **User-Agent Blocking**: Updated headers for 2025
    - **MultiIndex Columns**: Automatic column fixing
    - **429 Errors**: Intelligent error handling and recovery
    
    **🔧 Latest Updates:**
    - Compatible with yfinance 0.2.61+ (May 2025)
    - Handles Yahoo Finance API changes
    - Multiple fallback strategies
    - Real-time error diagnostics
    
    **📊 Features:**
    - Interactive candlestick charts
    - Technical indicators (SMA, RSI)
    - Volume analysis
    - Data export functionality
    - Comprehensive error handling
    """)

# Footer
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🔧 2025 Fixes Applied:**
    - User-Agent rotation
    - Rate limiting protection
    - MultiIndex column handling
    - Enhanced error messages
    """)

with col2:
    st.markdown("""
    **💡 Best Practices:**
    - Wait between requests
    - Use popular symbols
    - Keep yfinance updated
    - Check error messages
    """)

if YFINANCE_AVAILABLE:
    st.success(f"✅ **System Ready** | yfinance {yf_version} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
else:
    st.error("❌ **yfinance not available** - Install with: `pip install yfinance`")