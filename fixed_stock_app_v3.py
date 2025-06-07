# Stock Market App - Simplified R-Style Approach
# Based on successful R quantmod approach

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try importing packages
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("plotly package not found. Please install it.")
    st.stop()

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Simple Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Simple Stock Market Analyzer")
st.markdown("### R-quantmod inspired Python approach")

# Simple data fetching function (mimicking R quantmod approach)
@st.cache_data(ttl=600)  # 10 minute cache
def get_stock_data_simple(ticker="AAPL", days_back=365):
    """Simple data fetching mimicking R quantmod approach"""
    
    if not YFINANCE_AVAILABLE:
        st.error("yfinance not available")
        return None
    
    ticker = ticker.upper().strip()
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    st.info(f"Fetching data for: {ticker}")
    st.info(f"Date range: {start_date} to {end_date}")
    
    try:
        # Simple approach like R quantmod - minimal parameters
        stock = yf.Ticker(ticker)
        
        # Try the most basic approach first (like R)
        hist_data = stock.history(
            start=start_date,
            end=end_date,
            auto_adjust=False,  # Keep raw data like R
            actions=False,      # No dividends/splits initially
            interval='1d'
        )
        
        if hist_data.empty:
            st.warning("No data returned, trying period-based approach...")
            # Fallback: period-based like R might do
            hist_data = stock.history(period="1y", auto_adjust=False)
        
        if hist_data.empty:
            st.error("Still no data, trying with auto_adjust=True...")
            # Last resort
            hist_data = stock.history(start=start_date, end=end_date, auto_adjust=True)
        
        if not hist_data.empty:
            st.success(f"✅ Data fetched successfully: {len(hist_data)} rows")
            
            # Convert to DataFrame similar to R structure
            df = hist_data.copy()
            df.reset_index(inplace=True)
            
            # Rename columns to match R style
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # Add Adjusted column (use Close if auto_adjust was False)
            df['Adjusted'] = df['Close']
            
            # Remove any rows with NA (like R complete.cases)
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Adjusted'])
            
            if len(df) == 0:
                st.error("No valid data after cleaning")
                return None
            
            st.info(f"Data processed successfully: {len(df)} rows")
            return df
        else:
            st.error("No data available")
            return None
            
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Calculate technical indicators (like R)
def calculate_indicators_simple(df):
    """Calculate technical indicators similar to R TTR package"""
    if df is None or len(df) == 0:
        return df
    
    try:
        # Simple Moving Averages (like R SMA)
        if len(df) >= 20:
            df['SMA20'] = df['Adjusted'].rolling(window=20, min_periods=1).mean()
        else:
            df['SMA20'] = np.nan
        
        if len(df) >= 50:
            df['SMA50'] = df['Adjusted'].rolling(window=50, min_periods=1).mean()
        else:
            df['SMA50'] = np.nan
        
        # RSI (like R RSI)
        if len(df) >= 14:
            delta = df['Adjusted'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            df['RSI'] = 100 - (100 / (1 + rs))
        else:
            df['RSI'] = np.nan
        
        # Calculate daily returns (like R)
        df['Returns'] = df['Adjusted'].pct_change() * 100
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return df

# Create candlestick chart (like R plotly)
def create_candlestick_chart(data, ticker):
    """Create candlestick chart similar to R plotly"""
    if data is None or len(data) == 0:
        return go.Figure()
    
    fig = go.Figure(data=go.Candlestick(
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
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        showlegend=True,
        xaxis_rangeslider_visible=True
    )
    
    return fig

# Create volume chart (like R)
def create_volume_chart(data, ticker):
    """Create volume chart similar to R"""
    if data is None or len(data) == 0:
        return go.Figure()
    
    # Color based on price movement (like R)
    colors = ['green' if close > open_price else 'red' 
              for close, open_price in zip(data['Close'], data['Open'])]
    
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

# Create technical indicators chart (like R)
def create_tech_chart(data):
    """Create RSI chart similar to R"""
    if data is None or len(data) == 0 or 'RSI' not in data.columns or data['RSI'].isna().all():
        return go.Figure()
    
    fig = go.Figure()
    
    # RSI line
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='blue', width=2)
    ))
    
    # Overbought/Oversold lines (like R)
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    
    fig.update_layout(
        title="RSI (Relative Strength Index)",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        height=300
    )
    
    return fig

# Sidebar inputs (like R dashboard)
st.sidebar.header("📊 Stock Selection")

# Simple inputs like R
ticker_input = st.sidebar.text_input("Stock Ticker:", value="AAPL", placeholder="Enter symbol (e.g., AAPL, MSFT)")

# Date range (like R)
col1, col2 = st.sidebar.columns(2)
with col1:
    days_back = st.selectbox("Period:", [30, 90, 180, 365, 730], index=3)

# Update button (like R)
update_button = st.sidebar.button("Update Data", type="primary", use_container_width=True)

# Status info
st.sidebar.subheader("📡 Status")
if YFINANCE_AVAILABLE:
    st.sidebar.success("✅ yfinance available")
else:
    st.sidebar.error("❌ yfinance not available")

# Main logic
if update_button or ticker_input:
    if ticker_input:
        # Fetch data (like R reactive)
        with st.spinner(f"Fetching {ticker_input} data..."):
            stock_data = get_stock_data_simple(ticker_input, days_back)
        
        if stock_data is not None:
            # Calculate indicators
            stock_data = calculate_indicators_simple(stock_data)
            
            # Quick stats (like R sidebar)
            if len(stock_data) > 0:
                current_price = stock_data['Adjusted'].iloc[-1]
                prev_price = stock_data['Adjusted'].iloc[-2] if len(stock_data) > 1 else current_price
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                
                st.sidebar.subheader("📈 Quick Stats")
                st.sidebar.metric("Price", f"${current_price:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
                st.sidebar.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,.0f}")
                
                if not stock_data['RSI'].isna().iloc[-1]:
                    st.sidebar.metric("RSI", f"{stock_data['RSI'].iloc[-1]:.1f}")
            
            # Main content layout (like R dashboard)
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Price chart
                st.subheader("💹 Price Chart")
                price_chart = create_candlestick_chart(stock_data, ticker_input)
                st.plotly_chart(price_chart, use_container_width=True)
            
            with col2:
                # Summary statistics (like R summary table)
                st.subheader("📊 Summary Statistics")
                
                if len(stock_data) > 0:
                    current_price = stock_data['Adjusted'].iloc[-1]
                    returns = stock_data['Returns'].dropna()
                    
                    summary_stats = {
                        "Current Price": f"${current_price:.2f}",
                        "Day High": f"${stock_data['High'].max():.2f}",
                        "Day Low": f"${stock_data['Low'].min():.2f}",
                        "Avg Volume": f"{stock_data['Volume'].mean():,.0f}",
                        "Volatility": f"{returns.std():.2f}%" if len(returns) > 0 else "N/A",
                        "RSI": f"{stock_data['RSI'].iloc[-1]:.1f}" if not stock_data['RSI'].isna().iloc[-1] else "N/A"
                    }
                    
                    # Display as metrics
                    for metric, value in summary_stats.items():
                        st.write(f"**{metric}:** {value}")
                    
                    # Above/below SMA indicators
                    if not stock_data['SMA20'].isna().iloc[-1]:
                        above_sma20 = "Yes" if current_price > stock_data['SMA20'].iloc[-1] else "No"
                        st.write(f"**Above SMA20:** {above_sma20}")
                    
                    if not stock_data['SMA50'].isna().iloc[-1]:
                        above_sma50 = "Yes" if current_price > stock_data['SMA50'].iloc[-1] else "No"
                        st.write(f"**Above SMA50:** {above_sma50}")
            
            # Second row (like R dashboard)
            col1, col2 = st.columns(2)
            
            with col1:
                # Volume chart
                st.subheader("📊 Volume")
                volume_chart = create_volume_chart(stock_data, ticker_input)
                st.plotly_chart(volume_chart, use_container_width=True)
            
            with col2:
                # Technical indicators
                st.subheader("📈 Technical Indicators")
                tech_chart = create_tech_chart(stock_data)
                st.plotly_chart(tech_chart, use_container_width=True)
            
            # Recent data table (like R)
            st.subheader("📋 Recent Data")
            recent_data = stock_data.tail(10)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Returns']].copy()
            
            # Format data for display
            for col in ['Open', 'High', 'Low', 'Close']:
                recent_data[col] = recent_data[col].round(2)
            recent_data['Returns'] = recent_data['Returns'].round(2)
            recent_data['Volume'] = recent_data['Volume'].astype(int)
            
            st.dataframe(recent_data, use_container_width=True, hide_index=True)
            
            # Data info
            st.info(f"📊 **Data loaded successfully** | **{len(stock_data)} rows** | **Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        else:
            st.error("❌ Unable to fetch data")
            st.write("**Troubleshooting tips:**")
            st.write("• Check if the ticker symbol is correct")
            st.write("• Try popular symbols like AAPL, MSFT, GOOGL")
            st.write("• Check your internet connection")
            st.write("• The symbol might be delisted or unavailable")
    else:
        st.warning("Please enter a stock ticker symbol")
else:
    st.info("👈 Enter a stock symbol and click 'Update Data' to begin")
    
    # Show what this simple approach offers
    st.markdown("""
    ### 🎯 Simple R-Style Approach
    
    This simplified version mimics the successful R quantmod approach:
    
    **✅ Key Features:**
    - **Minimal Parameters**: Like R quantmod's simple approach
    - **Direct API Calls**: No complex retry strategies
    - **Clean Data Structure**: Matches R dataframe style
    - **Simple Error Handling**: Straightforward like R tryCatch
    
    **📊 Available Analysis:**
    - Candlestick charts with moving averages
    - Volume analysis
    - RSI technical indicator
    - Summary statistics
    - Recent data table
    
    **🔧 Why This Might Work Better:**
    - Uses minimal yfinance parameters (like R quantmod)
    - Simpler approach may be more stable
    - Direct translation of your working R logic
    - Less complex = fewer failure points
    """)

# Footer
st.markdown("---")
st.markdown("### 📋 Comparison with R Version")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **✅ R quantmod Success Factors:**
    - Simple `getSymbols()` call
    - Minimal parameters
    - Direct Yahoo Finance access
    - Stable API endpoints
    - Built-in error handling
    """)

with col2:
    st.markdown("""
    **🐍 Python Adaptation:**
    - Simplified `yf.Ticker().history()`
    - Minimal parameter strategy
    - Direct translation of R logic
    - Same data structure approach
    - Clean error handling
    """)

st.info(f"""
**🔧 System Status:** yfinance {'✅ Available' if YFINANCE_AVAILABLE else '❌ Not Available'} | 
plotly {'✅ Available' if PLOTLY_AVAILABLE else '❌ Not Available'} | 
scikit-learn {'✅ Available' if SKLEARN_AVAILABLE else '❌ Not Available'}
""")

st.markdown(f"""
**📊 Simple Stock Analyzer** | R-Style Python Approach  
**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")