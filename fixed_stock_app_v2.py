# Stock Market Data Science Application - Fixed Version
# A comprehensive data science project with improved error handling

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try importing packages with helpful error messages
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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
    page_title="Stock Market Analyzer",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Market Data Science Application")
st.markdown("### Enhanced stock analysis and prediction with improved error handling")

# Show package status
col1, col2, col3 = st.columns(3)
with col1:
    status = "✅ Available" if YFINANCE_AVAILABLE else "❌ Not Available"
    st.info(f"**yfinance**: {status}")
with col2:
    status = "✅ Available" if SKLEARN_AVAILABLE else "❌ Not Available"
    st.info(f"**scikit-learn**: {status}")
with col3:
    st.info(f"**plotly**: ✅ Available")

st.markdown("---")

# Enhanced sample data generation
def generate_enhanced_sample_data(symbol, start_date, end_date):
    """Generate more realistic sample stock data with better patterns"""
    np.random.seed(hash(symbol) % 2**32)
    
    # Create date range (business days only)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    
    if n_days == 0:
        # Fallback to regular date range if no business days
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
    
    # Enhanced base prices and volatilities for different stocks
    stock_params = {
        'AAPL': {'price': 175.0, 'volatility': 0.025, 'trend': 0.0002},
        'GOOGL': {'price': 2800.0, 'volatility': 0.030, 'trend': 0.0001},
        'MSFT': {'price': 350.0, 'volatility': 0.022, 'trend': 0.0003},
        'AMZN': {'price': 3200.0, 'volatility': 0.035, 'trend': 0.0001},
        'TSLA': {'price': 220.0, 'volatility': 0.045, 'trend': 0.0002},
        'META': {'price': 320.0, 'volatility': 0.040, 'trend': 0.0001},
        'NVDA': {'price': 450.0, 'volatility': 0.050, 'trend': 0.0004},
        'SPY': {'price': 420.0, 'volatility': 0.015, 'trend': 0.0002}
    }
    
    params = stock_params.get(symbol.upper(), 
                             {'price': 100.0, 'volatility': 0.025, 'trend': 0.0001})
    
    initial_price = params['price']
    daily_vol = params['volatility']
    trend = params['trend']
    
    # Generate price path with realistic patterns
    prices = [initial_price]
    volumes = []
    
    for i in range(1, n_days):
        # Add trend, mean reversion, and random walk components
        random_shock = np.random.normal(0, daily_vol)
        mean_reversion = -0.05 * (prices[-1] - initial_price) / initial_price
        trend_component = trend
        
        daily_return = trend_component + mean_reversion + random_shock
        new_price = prices[-1] * (1 + daily_return)
        
        # Keep prices within reasonable bounds
        new_price = max(new_price, initial_price * 0.3)
        new_price = min(new_price, initial_price * 5)
        prices.append(new_price)
        
        # Generate realistic volume (inversely correlated with price stability)
        base_volume = 1000000 + hash(symbol) % 5000000
        volume_multiplier = 1 + abs(daily_return) * 20
        volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 2.0))
        volumes.append(volume)
    
    # Add first day volume
    volumes.insert(0, int(base_volume * np.random.uniform(0.8, 1.2)))
    
    # Generate OHLC data
    data = []
    for i, (date, close, volume) in enumerate(zip(dates, prices, volumes)):
        if i == 0:
            open_price = close * np.random.uniform(0.995, 1.005)
        else:
            # Gaps between days
            gap = np.random.normal(0, daily_vol * 0.3)
            open_price = prices[i-1] * (1 + gap)
        
        # Intraday range
        daily_range = abs(np.random.normal(0, daily_vol * 0.8))
        high = max(open_price, close) * (1 + daily_range)
        low = min(open_price, close) * (1 - daily_range)
        
        # Ensure OHLC relationships
        high = max(open_price, high, close, low)
        low = min(open_price, high, close, low)
        
        data.append({
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

# Improved data fetching with better error handling and retry logic
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data_improved(symbol, start, end):
    """Enhanced stock data fetching with multiple fallback strategies"""
    
    if not symbol or len(symbol.strip()) == 0:
        st.error("Please enter a valid stock symbol")
        return None, None
    
    symbol = symbol.upper().strip()
    
    # If yfinance is not available, use sample data immediately
    if not YFINANCE_AVAILABLE:
        st.info(f"📊 Using enhanced sample data for {symbol} (yfinance not available)")
        data = generate_enhanced_sample_data(symbol, start, end)
        info = {"symbol": symbol, "longName": f"{symbol} Corporation (Enhanced Sample Data)"}
        return data, info
    
    # Try to fetch real data with yfinance
    try:
        stock = yf.Ticker(symbol)
        
        # Multiple fetch strategies
        data = None
        fetch_method = None
        
        # Strategy 1: Direct date range
        try:
            data = stock.history(start=start, end=end, period=None, interval='1d', 
                               auto_adjust=True, prepost=False, repair=True)
            if not data.empty:
                fetch_method = "Direct date range"
        except Exception as e:
            st.warning(f"Direct fetch failed: {str(e)[:100]}")
        
        # Strategy 2: Extended date range
        if data is None or data.empty:
            try:
                extended_start = start - timedelta(days=60)
                data = stock.history(start=extended_start, end=end, period=None, 
                                   interval='1d', auto_adjust=True, prepost=False, repair=True)
                if not data.empty:
                    # Filter to requested range
                    data = data[data.index.date >= start]
                    data = data[data.index.date <= end]
                    fetch_method = "Extended date range"
            except Exception as e:
                st.warning(f"Extended fetch failed: {str(e)[:100]}")
        
        # Strategy 3: Use period instead of dates
        if data is None or data.empty:
            try:
                # Calculate period based on date range
                days_diff = (end - start).days
                if days_diff <= 5:
                    period = "5d"
                elif days_diff <= 30:
                    period = "1mo"
                elif days_diff <= 90:
                    period = "3mo"
                elif days_diff <= 180:
                    period = "6mo"
                elif days_diff <= 365:
                    period = "1y"
                elif days_diff <= 730:
                    period = "2y"
                else:
                    period = "5y"
                
                data = stock.history(period=period, interval='1d', 
                                   auto_adjust=True, prepost=False, repair=True)
                if not data.empty:
                    # Filter to requested range
                    data = data[data.index.date >= start]
                    data = data[data.index.date <= end]
                    fetch_method = f"Period-based ({period})"
            except Exception as e:
                st.warning(f"Period-based fetch failed: {str(e)[:100]}")
        
        # Strategy 4: Try without repair
        if data is None or data.empty:
            try:
                data = stock.history(start=start, end=end, auto_adjust=True, repair=False)
                if not data.empty:
                    fetch_method = "Without repair"
            except Exception as e:
                st.warning(f"No-repair fetch failed: {str(e)[:100]}")
        
        # Check if we got valid data
        if data is not None and not data.empty and len(data) > 0:
            # Clean the data
            data = data.dropna(how='all')  # Remove rows where all values are NaN
            
            if len(data) > 0:
                # Get stock info with error handling
                info = {"symbol": symbol, "longName": f"{symbol} Corporation"}
                try:
                    stock_info = stock.info
                    if stock_info and isinstance(stock_info, dict):
                        info.update(stock_info)
                except Exception:
                    pass  # Use default info
                
                st.success(f"✅ Successfully fetched real data for {symbol} using {fetch_method}")
                st.info(f"📊 Retrieved {len(data)} data points from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
                return data, info
        
        # If all strategies failed, fall back to sample data
        raise Exception("All yfinance strategies failed")
        
    except Exception as e:
        error_msg = str(e)[:200]  # Truncate long error messages
        st.warning(f"⚠️ Could not fetch real data for {symbol}: {error_msg}")
        st.info(f"🔄 Using enhanced sample data for demonstration")
        
        # Generate sample data as fallback
        data = generate_enhanced_sample_data(symbol, start, end)
        info = {"symbol": symbol, "longName": f"{symbol} Corporation (Enhanced Sample Data)"}
        return data, info

# Enhanced technical indicators with better error handling
def calculate_technical_indicators(data):
    """Calculate technical indicators with improved error handling"""
    try:
        data = data.copy()
        
        # Ensure we have enough data
        if len(data) < 2:
            st.warning("Not enough data for technical indicators")
            return data
        
        # Moving averages with minimum periods
        data['MA_20'] = data['Close'].rolling(window=min(20, len(data)), min_periods=1).mean()
        data['MA_50'] = data['Close'].rolling(window=min(50, len(data)), min_periods=1).mean()
        
        # RSI with improved calculation
        if len(data) >= 14:
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            
            # Avoid division by zero
            rs = avg_gain / (avg_loss + 1e-10)
            data['RSI'] = 100 - (100 / (1 + rs))
        else:
            data['RSI'] = 50  # Neutral RSI for insufficient data
        
        # Bollinger Bands
        bb_window = min(20, len(data))
        data['BB_Middle'] = data['Close'].rolling(window=bb_window, min_periods=1).mean()
        bb_std = data['Close'].rolling(window=bb_window, min_periods=1).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Daily returns
        data['Daily_Return'] = data['Close'].pct_change()
        
        # Additional indicators
        if len(data) >= 12:
            # MACD
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            data['MACD'] = exp1 - exp2
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        return data
        
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return data

# Enhanced prediction model
def create_enhanced_prediction_model(data):
    """Enhanced prediction model with better feature engineering"""
    if not SKLEARN_AVAILABLE:
        st.warning("scikit-learn not available for prediction modeling")
        return None, None, None, None, None, None, None
    
    try:
        data_clean = data.dropna()
        
        if len(data_clean) < 10:
            raise ValueError("Insufficient data for model training (need at least 10 points)")
        
        # Enhanced feature engineering
        features = pd.DataFrame(index=data_clean.index)
        
        # Price features
        features['Close_lag1'] = data_clean['Close'].shift(1)
        features['Close_lag2'] = data_clean['Close'].shift(2)
        features['Open'] = data_clean['Open']
        features['High'] = data_clean['High']
        features['Low'] = data_clean['Low']
        features['Volume'] = data_clean['Volume']
        
        # Technical indicators
        if 'MA_20' in data_clean.columns:
            features['MA_20'] = data_clean['MA_20']
            features['Price_to_MA20'] = data_clean['Close'] / data_clean['MA_20']
        
        if 'RSI' in data_clean.columns:
            features['RSI'] = data_clean['RSI']
        
        # Additional features
        features['Price_Range'] = (data_clean['High'] - data_clean['Low']) / data_clean['Close']
        features['Open_Close_Ratio'] = data_clean['Open'] / data_clean['Close']
        
        if 'Daily_Return' in data_clean.columns:
            features['Return_lag1'] = data_clean['Daily_Return'].shift(1)
            features['Volatility'] = data_clean['Daily_Return'].rolling(5, min_periods=1).std()
        
        # Target variable
        target = data_clean['Close'].shift(-1)  # Next day's price
        
        # Align features and target
        combined = pd.concat([features, target.rename('Target')], axis=1).dropna()
        
        if len(combined) < 5:
            raise ValueError("Not enough valid data points after feature engineering")
        
        X = combined.drop('Target', axis=1)
        y = combined['Target']
        
        # Split data
        test_size = max(0.2, min(0.4, 5/len(X)))  # Adaptive test size
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return model, mse, r2, X_test, y_test, y_pred, list(X.columns)
        
    except Exception as e:
        st.error(f"Error creating prediction model: {e}")
        return None, None, None, None, None, None, None

# Sidebar with enhanced controls
st.sidebar.header("📊 Stock Selection")

# Popular stocks with descriptions
popular_stocks = {
    "": "Select a stock...",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation", 
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms Inc.",
    "NVDA": "NVIDIA Corporation",
    "SPY": "S&P 500 ETF"
}

selected_stock = st.sidebar.selectbox(
    "Quick Select:", 
    options=list(popular_stocks.keys()),
    format_func=lambda x: popular_stocks[x]
)

stock_symbol = st.sidebar.text_input(
    "Or enter stock symbol:", 
    value=selected_stock if selected_stock else "AAPL",
    placeholder="e.g., AAPL, MSFT, GOOGL"
)

# Enhanced date controls
st.sidebar.subheader("📅 Date Range")

# Preset ranges
preset_ranges = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "2 Years": 730,
    "5 Years": 1825
}

selected_range = st.sidebar.selectbox("Quick Range:", list(preset_ranges.keys()), index=3)

# Calculate dates
end_date = datetime.now().date()
start_date = end_date - timedelta(days=preset_ranges[selected_range])

# Allow custom dates
use_custom = st.sidebar.checkbox("Use custom dates")
if use_custom:
    start_date = st.sidebar.date_input(
        "Start Date:", 
        value=start_date,
        max_value=end_date
    )
    end_date = st.sidebar.date_input(
        "End Date:", 
        value=end_date,
        min_value=start_date,
        max_value=datetime.now().date()
    )

# Main application
if stock_symbol and start_date < end_date:
    # Fetch data with loading indicator
    with st.spinner(f"Fetching data for {stock_symbol}..."):
        stock_data, stock_info = fetch_stock_data_improved(stock_symbol, start_date, end_date)
    
    if stock_data is not None and not stock_data.empty:
        # Calculate indicators
        stock_data = calculate_technical_indicators(stock_data)
        
        # Enhanced metrics display
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            current_price = stock_data['Close'].iloc[-1]
            st.metric("💰 Current Price", f"${current_price:.2f}")
        
        with col2:
            if len(stock_data) > 1:
                prev_price = stock_data['Close'].iloc[-2]
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
                st.metric("📈 Daily Change", f"${change:.2f}", f"{change_pct:.2f}%")
            else:
                st.metric("📈 Daily Change", "N/A")
        
        with col3:
            volume = stock_data['Volume'].iloc[-1]
            if volume > 1e9:
                vol_str = f"{volume/1e9:.1f}B"
            elif volume > 1e6:
                vol_str = f"{volume/1e6:.1f}M"
            else:
                vol_str = f"{volume:,.0f}"
            st.metric("📊 Volume", vol_str)
        
        with col4:
            if len(stock_data) >= 52:
                high_52w = stock_data['High'].tail(252).max()  # ~1 year of trading days
                low_52w = stock_data['Low'].tail(252).min()
                st.metric("📊 52W High", f"${high_52w:.2f}")
            else:
                period_high = stock_data['High'].max()
                st.metric("📊 Period High", f"${period_high:.2f}")
        
        with col5:
            if len(stock_data) >= 52:
                low_52w = stock_data['Low'].tail(252).min()
                st.metric("📊 52W Low", f"${low_52w:.2f}")
            else:
                period_low = stock_data['Low'].min()
                st.metric("📊 Period Low", f"${period_low:.2f}")
        
        # Data source info
        data_source = "Sample Data" if "Sample Data" in stock_info.get('longName', '') else "Real Data"
        st.info(f"📊 **{data_source}** | **{len(stock_data)} data points** | **{stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}**")
        
        # Enhanced tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Price Charts", "📈 Technical Analysis", "📉 Statistics", 
            "🤖 AI Prediction", "📋 Raw Data", "ℹ️ Stock Info"
        ])
        
        with tab1:
            st.subheader(f"📊 {stock_symbol} Price Analysis")
            
            # Main price chart with enhanced features
            fig = go.Figure()
            
            # Candlestick chart option
            chart_type = st.radio("Chart Type:", ["Line", "Candlestick"], horizontal=True)
            
            if chart_type == "Candlestick" and len(stock_data) > 1:
                fig.add_trace(go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name=stock_symbol
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=stock_data.index, 
                    y=stock_data['Close'], 
                    mode='lines', 
                    name='Close Price', 
                    line=dict(color='#1f77b4', width=2)
                ))
            
            # Add moving averages
            show_ma = st.checkbox("Show Moving Averages", value=True)
            if show_ma:
                if 'MA_20' in stock_data.columns:
                    fig.add_trace(go.Scatter(
                        x=stock_data.index, 
                        y=stock_data['MA_20'], 
                        mode='lines', 
                        name='MA 20', 
                        line=dict(color='orange', width=1.5)
                    ))
                
                if 'MA_50' in stock_data.columns:
                    fig.add_trace(go.Scatter(
                        x=stock_data.index, 
                        y=stock_data['MA_50'], 
                        mode='lines', 
                        name='MA 50', 
                        line=dict(color='red', width=1.5)
                    ))
            
            fig.update_layout(
                title=f"{stock_symbol} Stock Price Analysis",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=600,
                hovermode='x unified',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume chart
            fig_volume = go.Figure()
            colors = ['red' if row['Close'] < row['Open'] else 'green' 
                     for _, row in stock_data.iterrows()]
            
            fig_volume.add_trace(go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                name='Volume',
                marker_color=colors
            ))
            
            fig_volume.update_layout(
                title=f"{stock_symbol} Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=300
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with tab2:
            st.subheader("📈 Technical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # RSI Chart
                if 'RSI' in stock_data.columns:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ))
                    
                    # RSI zones
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                                     annotation_text="Overbought (70)")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", 
                                     annotation_text="Oversold (30)")
                    fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", 
                                     annotation_text="Neutral (50)")
                    
                    fig_rsi.update_layout(
                        title="RSI (Relative Strength Index)",
                        yaxis_title="RSI",
                        height=400,
                        yaxis=dict(range=[0, 100])
                    )
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    # RSI interpretation
                    current_rsi = stock_data['RSI'].iloc[-1]
                    if current_rsi > 70:
                        st.warning(f"🔴 RSI: {current_rsi:.1f} - Potentially Overbought")
                    elif current_rsi < 30:
                        st.success(f"🟢 RSI: {current_rsi:.1f} - Potentially Oversold")
                    else:
                        st.info(f"🟡 RSI: {current_rsi:.1f} - Neutral Zone")
            
            with col2:
                # Bollinger Bands
                if all(col in stock_data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                    fig_bb = go.Figure()
                    
                    # Fill between bands
                    fig_bb.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['BB_Upper'],
                        mode='lines',
                        name='Upper Band',
                        line=dict(color='red', width=1),
                        showlegend=False
                    ))
                    
                    fig_bb.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['BB_Lower'],
                        mode='lines',
                        name='Lower Band',
                        line=dict(color='green', width=1),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        showlegend=False
                    ))
                    
                    fig_bb.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['BB_Middle'],
                        mode='lines',
                        name='Middle Band (MA20)',
                        line=dict(color='blue', width=1)
                    ))
                    
                    fig_bb.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='black', width=2)
                    ))
                    
                    fig_bb.update_layout(
                        title="Bollinger Bands",
                        yaxis_title="Price ($)",
                        height=400
                    )
                    st.plotly_chart(fig_bb, use_container_width=True)
            
            # MACD if available
            if 'MACD' in stock_data.columns and 'MACD_Signal' in stock_data.columns:
                st.subheader("MACD (Moving Average Convergence Divergence)")
                fig_macd = go.Figure()
                
                fig_macd.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue')
                ))
                
                fig_macd.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['MACD_Signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='red')
                ))
                
                fig_macd.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_macd.update_layout(
                    title="MACD Analysis", 
                    height=300,
                    yaxis_title="MACD Value"
                )
                st.plotly_chart(fig_macd, use_container_width=True)
        
        with tab3:
            st.subheader("📉 Statistical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced price statistics
                st.write("**📊 Price Statistics:**")
                close_data = stock_data['Close']
                
                # Calculate additional statistics
                price_change = close_data.iloc[-1] - close_data.iloc[0]
                price_change_pct = (price_change / close_data.iloc[0]) * 100
                
                stats_df = pd.DataFrame({
                    'Metric': [
                        'Current Price', 'Starting Price', 'Total Change', 'Total Change %',
                        'Average Price', 'Median Price', 'Price Std Dev', 
                        'Minimum Price', 'Maximum Price', 'Price Range'
                    ],
                    'Value': [
                        f"${close_data.iloc[-1]:.2f}",
                        f"${close_data.iloc[0]:.2f}",
                        f"${price_change:.2f}",
                        f"{price_change_pct:.2f}%",
                        f"${close_data.mean():.2f}",
                        f"${close_data.median():.2f}",
                        f"${close_data.std():.2f}",
                        f"${close_data.min():.2f}",
                        f"${close_data.max():.2f}",
                        f"${close_data.max() - close_data.min():.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Volume statistics
                st.write("**📊 Volume Statistics:**")
                volume_data = stock_data['Volume']
                volume_stats = pd.DataFrame({
                    'Metric': ['Average Volume', 'Median Volume', 'Max Volume', 'Min Volume'],
                    'Value': [
                        f"{volume_data.mean():,.0f}",
                        f"{volume_data.median():,.0f}",
                        f"{volume_data.max():,.0f}",
                        f"{volume_data.min():,.0f}"
                    ]
                })
                st.dataframe(volume_stats, use_container_width=True, hide_index=True)
            
            with col2:
                # Returns analysis
                if 'Daily_Return' in stock_data.columns:
                    returns = stock_data['Daily_Return'].dropna()
                    
                    if len(returns) > 0:
                        st.write("**📈 Returns Analysis:**")
                        
                        # Annualized statistics
                        annual_return = returns.mean() * 252 * 100  # 252 trading days
                        annual_volatility = returns.std() * np.sqrt(252) * 100
                        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
                        
                        returns_stats = pd.DataFrame({
                            'Metric': [
                                'Daily Return (Avg)', 'Annual Return (Est)', 'Daily Volatility',
                                'Annual Volatility (Est)', 'Sharpe Ratio (Est)', 'Best Day',
                                'Worst Day', 'Positive Days', 'Negative Days'
                            ],
                            'Value': [
                                f"{returns.mean()*100:.3f}%",
                                f"{annual_return:.2f}%",
                                f"{returns.std()*100:.2f}%",
                                f"{annual_volatility:.2f}%",
                                f"{sharpe_ratio:.2f}",
                                f"{returns.max()*100:.2f}%",
                                f"{returns.min()*100:.2f}%",
                                f"{(returns > 0).sum()} ({(returns > 0).mean()*100:.1f}%)",
                                f"{(returns < 0).sum()} ({(returns < 0).mean()*100:.1f}%)"
                            ]
                        })
                        st.dataframe(returns_stats, use_container_width=True, hide_index=True)
                        
                        # Returns distribution
                        fig_hist = px.histogram(
                            x=returns,
                            nbins=30,
                            title="Daily Returns Distribution",
                            labels={'x': 'Daily Return', 'y': 'Frequency'}
                        )
                        fig_hist.add_vline(
                            x=returns.mean(),
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Mean: {returns.mean()*100:.3f}%"
                        )
                        fig_hist.update_layout(height=350)
                        st.plotly_chart(fig_hist, use_container_width=True)
            
            # Risk metrics
            if 'Daily_Return' in stock_data.columns and len(stock_data) > 20:
                st.write("**⚠️ Risk Metrics:**")
                returns = stock_data['Daily_Return'].dropna()
                
                # VaR (Value at Risk) - 5% and 1%
                var_5 = np.percentile(returns, 5) * 100
                var_1 = np.percentile(returns, 1) * 100
                
                # Maximum drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min() * 100
                
                risk_metrics = pd.DataFrame({
                    'Risk Metric': ['Value at Risk (5%)', 'Value at Risk (1%)', 'Maximum Drawdown'],
                    'Value': [f"{var_5:.2f}%", f"{var_1:.2f}%", f"{max_drawdown:.2f}%"],
                    'Interpretation': [
                        '5% chance of losing more than this in a day',
                        '1% chance of losing more than this in a day',
                        'Largest peak-to-trough decline'
                    ]
                })
                st.dataframe(risk_metrics, use_container_width=True, hide_index=True)
        
        with tab4:
            st.subheader("🤖 AI-Powered Price Prediction")
            
            if not SKLEARN_AVAILABLE:
                st.error("❌ Scikit-learn is required for prediction modeling. Please install it to use this feature.")
            elif len(stock_data) < 10:
                st.warning(f"⚠️ Need at least 10 data points for prediction. Current: {len(stock_data)}")
            else:
                with st.spinner("🔄 Training AI prediction model..."):
                    model_results = create_enhanced_prediction_model(stock_data)
                
                if model_results[0] is not None:
                    model, mse, r2, X_test, y_test, y_pred, feature_names = model_results
                    
                    # Model performance metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🎯 Model Performance:**")
                        rmse = np.sqrt(mse)
                        mae = np.mean(np.abs(y_test - y_pred))
                        
                        # Performance assessment
                        if r2 > 0.8:
                            performance = "🟢 Excellent"
                            performance_color = "success"
                        elif r2 > 0.6:
                            performance = "🟡 Good"
                            performance_color = "info"
                        elif r2 > 0.3:
                            performance = "🟠 Moderate"
                            performance_color = "warning"
                        else:
                            performance = "🔴 Poor"
                            performance_color = "error"
                        
                        metrics_df = pd.DataFrame({
                            'Metric': [
                                'R² Score', 'RMSE', 'MAE', 'Training Points', 
                                'Test Points', 'Performance'
                            ],
                            'Value': [
                                f"{r2:.4f}",
                                f"${rmse:.2f}",
                                f"${mae:.2f}",
                                f"{len(X_test)}",
                                f"{len(y_test)}",
                                performance
                            ]
                        })
                        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                        
                        # Model explanation
                        st.info(f"""
                        **Model Explanation:**
                        - **R² Score**: {r2:.2%} of price variation explained
                        - **RMSE**: Average prediction error of ${rmse:.2f}
                        - **Features**: {len(feature_names)} technical indicators
                        """)
                    
                    with col2:
                        # Prediction accuracy visualization
                        fig_pred = go.Figure()
                        
                        # Actual vs Predicted scatter
                        fig_pred.add_trace(go.Scatter(
                            x=y_test,
                            y=y_pred,
                            mode='markers',
                            name='Predictions',
                            marker=dict(
                                color='blue',
                                size=8,
                                opacity=0.6
                            ),
                            hovertemplate='Actual: $%{x:.2f}<br>Predicted: $%{y:.2f}<extra></extra>'
                        ))
                        
                        # Perfect prediction line
                        min_val = min(y_test.min(), y_pred.min())
                        max_val = max(y_test.max(), y_pred.max())
                        fig_pred.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='red', dash='dash', width=2)
                        ))
                        
                        fig_pred.update_layout(
                            title="Predicted vs Actual Prices",
                            xaxis_title="Actual Price ($)",
                            yaxis_title="Predicted Price ($)",
                            height=400
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Feature importance
                    if len(feature_names) > 1:
                        st.write("**🔍 Feature Importance Analysis:**")
                        importance = np.abs(model.coef_)
                        importance_normalized = importance / importance.sum() * 100
                        
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importance,
                            'Importance (%)': importance_normalized
                        }).sort_values('Importance', ascending=False)
                        
                        fig_importance = px.bar(
                            importance_df.head(10),  # Top 10 features
                            x='Importance (%)',
                            y='Feature',
                            orientation='h',
                            title="Top 10 Most Important Features",
                            color='Importance (%)',
                            color_continuous_scale='viridis'
                        )
                        fig_importance.update_layout(height=400)
                        st.plotly_chart(fig_importance, use_container_width=True)
                        
                        # Show feature importance table
                        with st.expander("📊 Detailed Feature Importance"):
                            st.dataframe(
                                importance_df.round(4),
                                use_container_width=True,
                                hide_index=True
                            )
                    
                    # Prediction timeline
                    st.write("**📈 Prediction Timeline:**")
                    
                    # Create timeline showing actual vs predicted
                    timeline_data = pd.DataFrame({
                        'Date': X_test.index,
                        'Actual': y_test.values,
                        'Predicted': y_pred
                    })
                    
                    fig_timeline = go.Figure()
                    
                    fig_timeline.add_trace(go.Scatter(
                        x=timeline_data['Date'],
                        y=timeline_data['Actual'],
                        mode='lines+markers',
                        name='Actual Price',
                        line=dict(color='blue', width=2),
                        marker=dict(size=6)
                    ))
                    
                    fig_timeline.add_trace(go.Scatter(
                        x=timeline_data['Date'],
                        y=timeline_data['Predicted'],
                        mode='lines+markers',
                        name='Predicted Price',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=6, symbol='x')
                    ))
                    
                    fig_timeline.update_layout(
                        title="Actual vs Predicted Prices Over Time",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Disclaimer
                    st.warning("""
                    ⚠️ **Important Disclaimer:**
                    This prediction model is for educational purposes only. Stock prices are influenced by many 
                    unpredictable factors. Never make investment decisions based solely on these predictions.
                    Always consult with financial professionals and do your own research.
                    """)
                
                else:
                    st.error("❌ Unable to create prediction model with current data.")
        
        with tab5:
            st.subheader("📋 Raw Data Explorer")
            
            # Data display options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                data_view = st.selectbox(
                    "📊 Select View:",
                    ["Recent (Last 20)", "Full Dataset", "Date Range", "Summary Statistics"]
                )
            
            with col2:
                if data_view == "Date Range":
                    view_start = st.date_input("Start:", value=stock_data.index[-30].date())
                    view_end = st.date_input("End:", value=stock_data.index[-1].date())
            
            with col3:
                show_indicators = st.checkbox("Show Technical Indicators", value=False)
            
            # Display data based on selection
            if data_view == "Recent (Last 20)":
                display_data = stock_data.tail(20)
            elif data_view == "Full Dataset":
                display_data = stock_data
            elif data_view == "Date Range":
                mask = (stock_data.index.date >= view_start) & (stock_data.index.date <= view_end)
                display_data = stock_data.loc[mask]
            else:  # Summary Statistics
                display_data = stock_data.describe()
            
            # Filter columns
            if not show_indicators and data_view != "Summary Statistics":
                basic_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                available_basic = [col for col in basic_columns if col in display_data.columns]
                display_data = display_data[available_basic]
            
            # Format data for display
            if data_view != "Summary Statistics":
                # Round numeric columns
                numeric_columns = display_data.select_dtypes(include=[np.number]).columns
                display_data[numeric_columns] = display_data[numeric_columns].round(2)
            
            st.dataframe(
                display_data,
                use_container_width=True,
                height=400
            )
            
            # Data download options
            st.write("**💾 Download Options:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Basic data CSV
                basic_csv = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].to_csv()
                st.download_button(
                    "📥 Download Basic Data (CSV)",
                    basic_csv,
                    f"{stock_symbol}_basic_{start_date}_{end_date}.csv",
                    "text/csv"
                )
            
            with col2:
                # Full data with indicators
                full_csv = stock_data.to_csv()
                st.download_button(
                    "📥 Download Full Data (CSV)",
                    full_csv,
                    f"{stock_symbol}_full_{start_date}_{end_date}.csv",
                    "text/csv"
                )
            
            with col3:
                # Summary statistics
                summary_csv = stock_data.describe().to_csv()
                st.download_button(
                    "📥 Download Summary (CSV)",
                    summary_csv,
                    f"{stock_symbol}_summary_{start_date}_{end_date}.csv",
                    "text/csv"
                )
            
            # Data information
            st.write("**📊 Dataset Information:**")
            info_df = pd.DataFrame({
                'Property': [
                    'Total Records', 'Date Range', 'Trading Days', 'Columns',
                    'Data Source', 'Last Updated'
                ],
                'Value': [
                    f"{len(stock_data):,}",
                    f"{stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}",
                    f"{len(stock_data)} days",
                    f"{len(stock_data.columns)} ({', '.join(stock_data.columns)})",
                    data_source,
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                ]
            })
            st.dataframe(info_df, use_container_width=True, hide_index=True)
        
        with tab6:
            st.subheader(f"ℹ️ Stock Information - {stock_symbol}")
            
            # Stock info display
            if stock_info:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📋 Basic Information:**")
                    
                    info_items = [
                        ('Symbol', stock_info.get('symbol', 'N/A')),
                        ('Company Name', stock_info.get('longName', stock_info.get('shortName', 'N/A'))),
                        ('Sector', stock_info.get('sector', 'N/A')),
                        ('Industry', stock_info.get('industry', 'N/A')),
                        ('Exchange', stock_info.get('exchange', 'N/A')),
                        ('Currency', stock_info.get('currency', 'USD'))
                    ]
                    
                    for label, value in info_items:
                        st.write(f"**{label}:** {value}")
                
                with col2:
                    st.write("**💰 Financial Metrics:**")
                    
                    # Format large numbers
                    def format_large_number(num):
                        if num is None or num == 'N/A':
                            return 'N/A'
                        try:
                            num = float(num)
                            if num >= 1e12:
                                return f"${num/1e12:.2f}T"
                            elif num >= 1e9:
                                return f"${num/1e9:.2f}B"
                            elif num >= 1e6:
                                return f"${num/1e6:.2f}M"
                            else:
                                return f"${num:,.0f}"
                        except:
                            return 'N/A'
                    
                    financial_items = [
                        ('Market Cap', format_large_number(stock_info.get('marketCap'))),
                        ('Enterprise Value', format_large_number(stock_info.get('enterpriseValue'))),
                        ('P/E Ratio', stock_info.get('trailingPE', 'N/A')),
                        ('EPS', f"${stock_info.get('trailingEps', 'N/A')}" if stock_info.get('trailingEps') else 'N/A'),
                        ('Dividend Yield', f"{stock_info.get('dividendYield', 0)*100:.2f}%" if stock_info.get('dividendYield') else 'N/A'),
                        ('Beta', stock_info.get('beta', 'N/A'))
                    ]
                    
                    for label, value in financial_items:
                        st.write(f"**{label}:** {value}")
                
                # Company description
                if stock_info.get('longBusinessSummary'):
                    st.write("**📝 Company Description:**")
                    st.write(stock_info['longBusinessSummary'][:500] + "..." if len(stock_info['longBusinessSummary']) > 500 else stock_info['longBusinessSummary'])
                
                # Additional metrics in expandable section
                with st.expander("📊 Additional Financial Metrics"):
                    additional_metrics = {
                        'Price Metrics': [
                            ('52 Week High', f"${stock_info.get('fiftyTwoWeekHigh', 'N/A')}" if stock_info.get('fiftyTwoWeekHigh') else 'N/A'),
                            ('52 Week Low', f"${stock_info.get('fiftyTwoWeekLow', 'N/A')}" if stock_info.get('fiftyTwoWeekLow') else 'N/A'),
                            ('50 Day Average', f"${stock_info.get('fiftyDayAverage', 'N/A')}" if stock_info.get('fiftyDayAverage') else 'N/A'),
                            ('200 Day Average', f"${stock_info.get('twoHundredDayAverage', 'N/A')}" if stock_info.get('twoHundredDayAverage') else 'N/A')
                        ],
                        'Valuation Metrics': [
                            ('P/B Ratio', stock_info.get('priceToBook', 'N/A')),
                            ('P/S Ratio', stock_info.get('priceToSalesTrailing12Months', 'N/A')),
                            ('PEG Ratio', stock_info.get('pegRatio', 'N/A')),
                            ('Book Value', f"${stock_info.get('bookValue', 'N/A')}" if stock_info.get('bookValue') else 'N/A')
                        ],
                        'Profitability': [
                            ('Profit Margin', f"{stock_info.get('profitMargins', 0)*100:.2f}%" if stock_info.get('profitMargins') else 'N/A'),
                            ('Operating Margin', f"{stock_info.get('operatingMargins', 0)*100:.2f}%" if stock_info.get('operatingMargins') else 'N/A'),
                            ('ROE', f"{stock_info.get('returnOnEquity', 0)*100:.2f}%" if stock_info.get('returnOnEquity') else 'N/A'),
                            ('ROA', f"{stock_info.get('returnOnAssets', 0)*100:.2f}%" if stock_info.get('returnOnAssets') else 'N/A')
                        ]
                    }
                    
                    for category, metrics in additional_metrics.items():
                        st.write(f"**{category}:**")
                        for label, value in metrics:
                            st.write(f"  • **{label}:** {value}")
                        st.write("")
            
            else:
                st.info("Detailed stock information not available for this symbol.")
                
            # Quick analysis based on available data
            st.write("**🎯 Quick Technical Analysis:**")
            
            current_price = stock_data['Close'].iloc[-1]
            analysis_points = []
            
            # Price trend analysis
            if len(stock_data) >= 20:
                ma_20 = stock_data['MA_20'].iloc[-1] if 'MA_20' in stock_data.columns else None
                if ma_20:
                    if current_price > ma_20:
                        analysis_points.append("🟢 Price is above 20-day moving average (bullish signal)")
                    else:
                        analysis_points.append("🔴 Price is below 20-day moving average (bearish signal)")
            
            # Volume analysis
            if len(stock_data) >= 10:
                avg_volume = stock_data['Volume'].tail(10).mean()
                current_volume = stock_data['Volume'].iloc[-1]
                if current_volume > avg_volume * 1.5:
                    analysis_points.append("📊 High trading volume detected (increased interest)")
                elif current_volume < avg_volume * 0.5:
                    analysis_points.append("📊 Low trading volume detected (decreased interest)")
            
            # RSI analysis
            if 'RSI' in stock_data.columns:
                current_rsi = stock_data['RSI'].iloc[-1]
                if current_rsi > 70:
                    analysis_points.append("⚠️ RSI indicates potentially overbought conditions")
                elif current_rsi < 30:
                    analysis_points.append("💡 RSI indicates potentially oversold conditions")
            
            # Display analysis points
            for point in analysis_points:
                st.write(f"• {point}")
            
            if not analysis_points:
                st.write("• Insufficient data for detailed technical analysis")
    
    else:
        st.error("❌ Unable to fetch any data for this symbol.")
        st.write("**Possible reasons:**")
        st.write("• Invalid or non-existent stock symbol")
        st.write("• Network connectivity issues")
        st.write("• API limitations or rate limits")
        st.write("• Date range issues")
        
        st.info("**Try the following:**")
        st.write("• Verify the stock symbol is correct")
        st.write("• Try popular symbols like AAPL, MSFT, GOOGL")
        st.write("• Adjust the date range")
        st.write("• Refresh the page and try again")

elif start_date >= end_date:
    st.error("⚠️ Start date must be before end date")
else:
    st.info("👈 Please enter a stock symbol in the sidebar to begin analysis.")

# Enhanced footer with troubleshooting
st.markdown("---")
st.markdown("### 🚀 Application Information")

# System status
with st.expander("🔧 System Status & Troubleshooting"):
    st.write("**📦 Package Status:**")
    
    packages = {
        "Core Packages": {
            "streamlit": "✅ Available",
            "pandas": "✅ Available", 
            "numpy": "✅ Available",
            "plotly": "✅ Available" if PLOTLY_AVAILABLE else "❌ Not Available"
        },
        "Data Source": {
            "yfinance": "✅ Available" if YFINANCE_AVAILABLE else "❌ Not Available"
        },
        "Analytics": {
            "scikit-learn": "✅ Available" if SKLEARN_AVAILABLE else "❌ Not Available",
            "matplotlib": "✅ Available" if MATPLOTLIB_AVAILABLE else "❌ Not Available"
        }
    }
    
    for category, items in packages.items():
        st.write(f"**{category}:**")
        for package, status in items.items():
            st.write(f"  {status} {package}")
    
    st.write("\n**🔧 Troubleshooting Tips:**")
    st.write("• If yfinance is unavailable, the app uses realistic sample data")
    st.write("• For real data, ensure yfinance is installed: `pip install yfinance`")
    st.write("• Some features require scikit-learn: `pip install scikit-learn`")
    st.write("• Clear cache (Ctrl+F5) if you encounter data issues")
    st.write("• Check internet connection for real-time data")

# Feature overview
with st.expander("✨ Features Overview"):
    st.markdown("""
    **🎯 Core Features:**
    - 📊 **Real-time Data**: Live stock prices via yfinance API
    - 📈 **Technical Analysis**: RSI, Bollinger Bands, MACD, Moving Averages
    - 📉 **Statistical Analysis**: Returns, volatility, risk metrics
    - 🤖 **AI Prediction**: Machine learning price forecasting
    - 📋 **Data Export**: Download data in CSV format
    - 🎨 **Interactive Charts**: Dynamic visualizations with Plotly
    
    **🔍 Advanced Analytics:**
    - Value at Risk (VaR) calculations
    - Maximum drawdown analysis
    - Sharpe ratio estimation
    - Feature importance analysis
    - Comprehensive risk metrics
    
    **🛡️ Robust Design:**
    - Automatic fallback to sample data
    - Enhanced error handling
    - Multiple data fetch strategies
    - Responsive user interface
    - Real-time performance monitoring
    """)

# Usage guide
with st.expander("📖 Usage Guide"):
    st.markdown("""
    **Getting Started:**
    1. 📝 Enter a stock symbol (e.g., AAPL, MSFT, GOOGL) in the sidebar
    2. 📅 Select your preferred date range
    3. 🔍 Explore different analysis tabs
    4. 💾 Download data for further analysis
    
    **Understanding the Analysis:**
    - **Price Charts**: Candlestick and line charts with moving averages
    - **Technical Analysis**: RSI, Bollinger Bands, and MACD indicators
    - **Statistics**: Comprehensive financial and risk metrics
    - **AI Prediction**: Machine learning model for price forecasting
    - **Raw Data**: Access to underlying data with export options
    - **Stock Info**: Company fundamentals and financial metrics
    
    **Best Practices:**
    - Use 1-2 year date ranges for optimal performance
    - Popular symbols typically have better data availability
    - The prediction model works best with 100+ data points
    - Always verify important information from official sources
    """)

# Disclaimer
st.warning("""
⚠️ **Important Disclaimer:**
This application is for educational and demonstration purposes only. The information provided should not be considered as financial advice. 
Stock markets are inherently risky and unpredictable. Always conduct your own research and consult with qualified financial professionals 
before making any investment decisions. The creators of this application are not responsible for any financial losses incurred.
""")

st.markdown(f"""
---
**📊 Stock Market Data Science Application** | Enhanced Version  
**Technologies**: Python, Streamlit, Pandas, NumPy, Plotly, Scikit-learn{', yfinance' if YFINANCE_AVAILABLE else ''}  