# Stock Market Data Science Application - Python 3.10.10 Version
# Enhanced with virtual environment setup and Python version checking

import streamlit as st
import pandas as pd
import numpy as np
import sys
import platform
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config first
st.set_page_config(
    page_title="Stock Market Analyzer - Python 3.10.10",
    page_icon="📈",
    layout="wide"
)

# Python version check and display
python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
python_full = sys.version

# Title with Python version
st.title("📈 Stock Market Data Science Application")
st.markdown(f"### Python {python_version} | Enhanced Virtual Environment Setup")

# Prominent Python version display
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info(f"🐍 **Python**: {python_version}")
with col2:
    venv_status = "Yes" if (hasattr(sys, 'real_prefix') or 
                           (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)) else "No"
    st.info(f"🌐 **Virtual Env**: {venv_status}")
with col3:
    st.info(f"💻 **Platform**: {platform.system()}")
with col4:
    st.info(f"📊 **Streamlit**: {st.__version__}")

# Python 3.10.10 compatibility check
if sys.version_info.major == 3 and sys.version_info.minor == 10 and sys.version_info.micro == 10:
    st.success("✅ **Perfect!** Running on recommended Python 3.10.10")
elif sys.version_info.major == 3 and sys.version_info.minor == 10:
    st.success(f"✅ **Good!** Running on Python 3.10.{sys.version_info.micro} (close to 3.10.10)")
elif sys.version_info.major == 3 and 8 <= sys.version_info.minor <= 11:
    st.warning(f"⚠️ **Different version detected:** Python {python_version}. Recommended: 3.10.10")
else:
    st.error(f"❌ **Incompatible version:** Python {python_version}. Please use Python 3.10.10")

# Virtual environment setup instructions
with st.expander("🔧 **Setup Instructions for Python 3.10.10 Virtual Environment**"):
    st.markdown("""
    ### 📋 Complete Setup Guide
    
    **Step 1: Check if Python 3.10 is installed**
    ```bash
    # Check available Python versions
    python3.10 --version
    # or
    py -3.10 --version  # Windows
    ```
    
    **Step 2: Create Virtual Environment with Python 3.10**
    ```bash
    # Create virtual environment
    python3.10 -m venv stock_app_env
    # or on Windows:
    py -3.10 -m venv stock_app_env
    ```
    
    **Step 3: Activate Virtual Environment**
    ```bash
    # Windows:
    stock_app_env\\Scripts\\activate
    
    # Linux/Mac:
    source stock_app_env/bin/activate
    ```
    
    **Step 4: Install Required Packages**
    ```bash
    # Upgrade pip first
    python -m pip install --upgrade pip
    
    # Install all required packages
    pip install streamlit==1.28.0
    pip install yfinance==0.2.61
    pip install pandas==2.1.0
    pip install numpy==1.24.3
    pip install plotly==5.17.0
    pip install scikit-learn==1.3.0
    pip install matplotlib==3.7.2
    pip install seaborn==0.12.2
    ```
    
    **Step 5: Verify Installation**
    ```bash
    # Check Python version in virtual environment
    python --version
    
    # List installed packages
    pip list
    ```
    
    **Step 6: Run the Application**
    ```bash
    # Save this file as stock_app_py310.py
    streamlit run stock_app_py310.py
    ```
    
    **Step 7: Deactivate when done**
    ```bash
    deactivate
    ```
    """)
    
    # Show current environment details
    st.markdown("### 🔍 Current Environment Details")
    st.code(f"""
Python Version: {python_full}
Python Executable: {sys.executable}
Virtual Environment: {venv_status}
Platform: {platform.platform()}
Working Directory: {os.getcwd()}
    """)

# Try importing packages with helpful error messages
package_status = {}

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    yf_version = getattr(yf, '__version__', 'Unknown')
    package_status['yfinance'] = f"✅ {yf_version}"
except ImportError as e:
    YFINANCE_AVAILABLE = False
    yf_version = "Not installed"
    package_status['yfinance'] = f"❌ {str(e)}"

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    package_status['matplotlib'] = f"✅ {plt.matplotlib.__version__}"
    package_status['seaborn'] = f"✅ {sns.__version__}"
except ImportError as e:
    st.error(f"matplotlib/seaborn packages not found: {e}")
    st.code("pip install matplotlib seaborn")
    st.stop()

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly
    package_status['plotly'] = f"✅ {plotly.__version__}"
except ImportError as e:
    st.error(f"plotly package not found: {e}")
    st.code("pip install plotly")
    st.stop()

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    import sklearn
    package_status['scikit-learn'] = f"✅ {sklearn.__version__}"
except ImportError as e:
    st.error(f"scikit-learn package not found: {e}")
    st.code("pip install scikit-learn")
    st.stop()

# Package status display
with st.expander("📦 Package Status in Current Environment"):
    st.write("**Core Data Science Packages:**")
    st.write(f"- pandas: ✅ {pd.__version__}")
    st.write(f"- numpy: ✅ {np.__version__}")
    st.write(f"- streamlit: ✅ {st.__version__}")
    
    st.write("**Financial & ML Packages:**")
    for package, status in package_status.items():
        st.write(f"- {package}: {status}")
    
    st.write(f"**Python Environment:**")
    st.write(f"- Python: {python_version}")
    st.write(f"- Virtual Env: {venv_status}")
    st.write(f"- Platform: {platform.system()} {platform.release()}")

# Show yfinance status
if YFINANCE_AVAILABLE:
    st.success(f"✅ Real-time data available via yfinance {yf_version}")
else:
    st.warning("⚠️ yfinance not available - using sample data for demonstration")
    st.info("💡 Install with: `pip install yfinance==0.2.61`")

st.markdown("---")

# Function to generate sample data
def generate_sample_data(symbol, start_date, end_date):
    """Generate realistic sample stock data"""
    np.random.seed(hash(symbol) % 2**32)  # Seed based on symbol for consistency
    
    # Create date range (business days only)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Base prices for different stocks
    base_prices = {
        'AAPL': 150.0, 'GOOGL': 2500.0, 'MSFT': 300.0, 'AMZN': 3000.0,
        'TSLA': 800.0, 'META': 200.0, 'NVDA': 400.0
    }
    
    initial_price = base_prices.get(symbol.upper(), 100.0)
    
    # Generate realistic price movements with trends
    n_days = len(dates)
    trend = np.linspace(0, 0.2, n_days)  # Slight upward trend
    noise = np.random.normal(0, 0.02, n_days)  # Daily volatility
    returns = trend/n_days + noise
    
    prices = [initial_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        # Add some bounds to keep prices realistic
        new_price = max(new_price, initial_price * 0.5)  # Don't go below 50% of initial
        new_price = min(new_price, initial_price * 3)    # Don't go above 300% of initial
        prices.append(new_price)
    
    # Generate OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC values
        daily_vol = abs(np.random.normal(0, 0.02))
        high = close * (1 + daily_vol)
        low = close * (1 - daily_vol)
        
        if i > 0:
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.005))  # Gap up/down
        else:
            open_price = close
        
        # Ensure OHLC relationships are correct
        high = max(open_price, high, close, low)
        low = min(open_price, high, close, low)
        
        # Generate volume (higher volume on bigger price moves)
        base_volume = 5000000
        volume_multiplier = 1 + abs(ret) * 10 if i > 0 else 1
        volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 1.5))
        
        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

# Enhanced function to fetch stock data with Python 3.10.10 optimizations
@st.cache_data
def fetch_stock_data(symbol, start, end):
    """Fetch stock data using yfinance with fallback to sample data - Python 3.10.10 optimized"""
    
    if not YFINANCE_AVAILABLE:
        st.info("🔄 Using sample data (yfinance not available)")
        data = generate_sample_data(symbol, start, end)
        info = {"symbol": symbol, "longName": f"{symbol} Corporation (Sample Data)"}
        return data, info
    
    # Try multiple strategies with yfinance (optimized for Python 3.10.10)
    try:
        # Validate inputs
        if not symbol or len(symbol.strip()) == 0:
            raise ValueError("Stock symbol cannot be empty")
        
        symbol = symbol.upper().strip()
        
        # Strategy 1: Try with standard parameters (Python 3.10.10 optimized)
        stock = yf.Ticker(symbol)
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Fetching data with Python 3.10.10 optimizations...")
            progress_bar.progress(25)
            
            # First try: exact date range with Python 3.10.10 compatible parameters
            data = stock.history(
                start=start, 
                end=end, 
                auto_adjust=True, 
                prepost=False,
                repair=True,  # Use repair for Python 3.10.10
                keepna=False
            )
            progress_bar.progress(50)
            
            if data.empty:
                status_text.text("⚠️ No data for exact range, trying extended period...")
                progress_bar.progress(60)
                
                # Second try: extend the date range
                extended_start = start - timedelta(days=30)
                data = stock.history(
                    start=extended_start, 
                    end=end, 
                    auto_adjust=True, 
                    prepost=False,
                    repair=True
                )
                progress_bar.progress(75)
                
                if data.empty:
                    status_text.text("🔄 Trying period-based approach...")
                    progress_bar.progress(85)
                    
                    # Third try: use period instead of dates
                    data = stock.history(
                        period="1y", 
                        auto_adjust=True, 
                        prepost=False,
                        repair=True
                    )
                    
                    if not data.empty:
                        # Filter to requested date range
                        data = data[data.index >= pd.to_datetime(start)]
                        data = data[data.index <= pd.to_datetime(end)]
            
            progress_bar.progress(100)
            status_text.text("✅ Data fetch completed!")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
        
        except Exception as fetch_error:
            progress_bar.empty()
            status_text.empty()
            st.warning(f"yfinance fetch error: {fetch_error}")
            data = pd.DataFrame()
        
        # If we still don't have data, fall back to sample data
        if data.empty:
            st.warning(f"⚠️ No real data available for {symbol}, using sample data for demonstration")
            data = generate_sample_data(symbol, start, end)
            info = {"symbol": symbol, "longName": f"{symbol} Corporation (Sample Data)"}
            return data, info
        
        # Try to get stock info
        try:
            info = stock.info
            if not info or 'symbol' not in info:
                info = {"symbol": symbol, "longName": f"{symbol} Corporation"}
        except Exception:
            info = {"symbol": symbol, "longName": f"{symbol} Corporation"}
        
        st.success(f"✅ Successfully fetched real data for {symbol} using Python {python_version}")
        return data, info
        
    except Exception as e:
        error_msg = str(e)
        st.warning(f"Error with yfinance: {error_msg}")
        st.info("🔄 Falling back to sample data for demonstration")
        
        # Generate sample data as fallback
        data = generate_sample_data(symbol, start, end)
        info = {"symbol": symbol, "longName": f"{symbol} Corporation (Sample Data)"}
        return data, info

# Function to calculate technical indicators
def calculate_indicators(data):
    """Calculate technical indicators - Python 3.10.10 optimized"""
    try:
        # Create a copy to avoid modifying the original
        data = data.copy()
        
        # Moving averages
        data['MA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
        data['MA_50'] = data['Close'].rolling(window=50, min_periods=1).mean()
        
        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        
        # Avoid division by zero
        rs = gain / (loss + 1e-10)
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20, min_periods=1).mean()
        bb_std = data['Close'].rolling(window=20, min_periods=1).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Daily returns
        data['Daily_Return'] = data['Close'].pct_change()
        
        return data
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return data

# Function to create prediction model
def create_prediction_model(data):
    """Simple linear regression model for price prediction - Python 3.10.10 optimized"""
    try:
        # Prepare features
        data_clean = data.dropna()
        
        if len(data_clean) < 20:
            raise ValueError("Insufficient data for model training")
        
        # Features: using previous days' prices, volume, and technical indicators
        available_features = []
        feature_columns = ['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI']
        
        for col in feature_columns:
            if col in data_clean.columns and not data_clean[col].isna().all():
                available_features.append(col)
        
        if len(available_features) < 3:
            raise ValueError("Not enough features available for modeling")
        
        feature_data = data_clean[available_features].dropna()
        
        # Target: next day's closing price
        target = data_clean['Close'].shift(-1).dropna()
        
        # Align features and target
        min_length = min(len(feature_data), len(target))
        if min_length < 10:
            raise ValueError("Not enough data points for reliable prediction")
            
        X = feature_data.iloc[:min_length]
        y = target.iloc[:min_length]
        
        # Split data (use newer random_state for Python 3.10.10)
        test_size = min(0.3, max(0.1, len(X) * 0.2 / len(X)))  # Adaptive test size
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return model, mse, r2, X_test, y_test, y_pred, available_features
        
    except Exception as e:
        st.error(f"Error creating prediction model: {e}")
        return None, None, None, None, None, None, None

# Sidebar for user inputs
st.sidebar.header("📊 Stock Selection")
st.sidebar.markdown(f"**Environment**: Python {python_version}")
st.sidebar.markdown(f"**Virtual Env**: {venv_status}")

stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL, MSFT):", "AAPL")

# Date inputs with validation
max_date = datetime.now().date()
min_date = datetime.now().date() - timedelta(days=5*365)  # 5 years ago

start_date = st.sidebar.date_input(
    "Start Date", 
    value=datetime.now().date() - timedelta(days=365),
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "End Date", 
    value=datetime.now().date(),
    min_value=start_date,
    max_value=max_date
)

# Popular stock suggestions
st.sidebar.markdown("**Popular Stocks:**")
popular_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"]
selected_popular = st.sidebar.selectbox("Or select from popular stocks:", [""] + popular_stocks)

if selected_popular:
    stock_symbol = selected_popular

# Environment status in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Environment Status")
if sys.version_info.major == 3 and sys.version_info.minor == 10:
    st.sidebar.success("✅ Python 3.10.x")
else:
    st.sidebar.warning(f"⚠️ Python {python_version}")

if YFINANCE_AVAILABLE:
    st.sidebar.success(f"✅ yfinance {yf_version}")
else:
    st.sidebar.error("❌ yfinance missing")

if venv_status == "Yes":
    st.sidebar.success("✅ Virtual Environment")
else:
    st.sidebar.warning("⚠️ No Virtual Environment")

# Main application logic
if stock_symbol and start_date < end_date:
    # Fetch data
    with st.spinner(f"Fetching data for {stock_symbol} using Python {python_version}..."):
        stock_data, stock_info = fetch_stock_data(stock_symbol, start_date, end_date)
    
    if stock_data is not None and not stock_data.empty:
        # Calculate indicators
        stock_data = calculate_indicators(stock_data)
        
        # Display stock info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = stock_data['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            if len(stock_data) > 1:
                daily_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
                change_pct = (daily_change/stock_data['Close'].iloc[-2]*100)
                st.metric("Daily Change", f"${daily_change:.2f}", f"{change_pct:.2f}%")
            else:
                st.metric("Daily Change", "N/A")
        
        with col3:
            volume = stock_data['Volume'].iloc[-1]
            st.metric("Volume", f"{volume:,.0f}")
        
        with col4:
            if stock_info and 'marketCap' in stock_info and stock_info['marketCap']:
                market_cap = stock_info['marketCap']
                if market_cap > 1e12:
                    st.metric("Market Cap", f"${market_cap/1e12:.1f}T")
                elif market_cap > 1e9:
                    st.metric("Market Cap", f"${market_cap/1e9:.1f}B")
                else:
                    st.metric("Market Cap", f"${market_cap/1e6:.0f}M")
            else:
                st.metric("Market Cap", "N/A")
        
        # Show data source info
        data_source = "Sample Data" if "Sample Data" in stock_info.get('longName', '') else "Real Data"
        st.info(f"📊 Data Source: {data_source} | Python: {python_version} | Data Points: {len(stock_data)} | Date Range: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Price Analysis", "📈 Technical Indicators", "📉 Statistics", "🤖 Prediction", "📋 Data"])
        
        with tab1:
            st.subheader("Stock Price Analysis")
            
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stock_data.index, 
                y=stock_data['Close'], 
                mode='lines', 
                name='Close Price', 
                line=dict(color='blue', width=2)
            ))
            
            # Add moving averages if available
            if 'MA_20' in stock_data.columns and not stock_data['MA_20'].isna().all():
                fig.add_trace(go.Scatter(
                    x=stock_data.index, 
                    y=stock_data['MA_20'], 
                    mode='lines', 
                    name='MA 20', 
                    line=dict(color='orange', width=1)
                ))
            
            if 'MA_50' in stock_data.columns and not stock_data['MA_50'].isna().all():
                fig.add_trace(go.Scatter(
                    x=stock_data.index, 
                    y=stock_data['MA_50'], 
                    mode='lines', 
                    name='MA 50', 
                    line=dict(color='red', width=1)
                ))
            
            fig.update_layout(
                title=f"{stock_symbol} Stock Price with Moving Averages (Python {python_version})",
                xaxis_title="Date", 
                yaxis_title="Price ($)",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume chart
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=stock_data.index, 
                y=stock_data['Volume'],
                name='Volume',
                marker_color='lightblue'
            ))
            fig_volume.update_layout(
                title=f"{stock_symbol} Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=300
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with tab2:
            st.subheader("Technical Indicators")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # RSI Chart
                if 'RSI' in stock_data.columns and not stock_data['RSI'].isna().all():
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=stock_data.index, 
                        y=stock_data['RSI'],
                        mode='lines', 
                        name='RSI',
                        line=dict(color='purple')
                    ))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                    fig_rsi.update_layout(
                        title="RSI (Relative Strength Index)",
                        yaxis_title="RSI", 
                        height=400,
                        yaxis=dict(range=[0, 100])
                    )
                    st.plotly_chart(fig_rsi, use_container_width=True)
                else:
                    st.warning("Not enough data to calculate RSI")
            
            with col2:
                # Bollinger Bands
                bb_cols = ['BB_Upper', 'BB_Middle', 'BB_Lower']
                if all(col in stock_data.columns for col in bb_cols):
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
                        fillcolor='rgba(255,0,0,0.1)',
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
                        height=400,
                        yaxis_title="Price ($)"
                    )
                    st.plotly_chart(fig_bb, use_container_width=True)
                else:
                    st.warning("Not enough data to calculate Bollinger Bands")
        
        with tab3:
            st.subheader("Statistical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Basic statistics
                st.write("**Price Statistics:**")
                close_data = stock_data['Close']
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
                    'Value': [
                        f"${close_data.mean():.2f}",
                        f"${close_data.median():.2f}",
                        f"${close_data.std():.2f}",
                        f"${close_data.min():.2f}",
                        f"${close_data.max():.2f}",
                        f"${close_data.max() - close_data.min():.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Daily returns statistics
                if 'Daily_Return' in stock_data.columns and not stock_data['Daily_Return'].isna().all():
                    st.write("**Daily Returns Statistics:**")
                    returns = stock_data['Daily_Return'].dropna()
                    returns_stats = pd.DataFrame({
                        'Metric': ['Mean Return', 'Volatility', 'Min Return', 'Max Return', 'Skewness'],
                        'Value': [
                            f"{returns.mean()*100:.3f}%",
                            f"{returns.std()*100:.2f}%",
                            f"{returns.min()*100:.2f}%",
                            f"{returns.max()*100:.2f}%",
                            f"{returns.skew():.2f}"
                        ]
                    })
                    st.dataframe(returns_stats, use_container_width=True, hide_index=True)
            
            with col2:
                # Returns distribution
                if 'Daily_Return' in stock_data.columns and not stock_data['Daily_Return'].isna().all():
                    returns_clean = stock_data['Daily_Return'].dropna()
                    
                    fig_hist = px.histogram(
                        x=returns_clean, 
                        nbins=30,
                        title="Daily Returns Distribution",
                        labels={'x': 'Daily Return', 'y': 'Frequency'}
                    )
                    fig_hist.add_vline(
                        x=returns_clean.mean(), 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=f"Mean: {returns_clean.mean()*100:.3f}%"
                    )
                    fig_hist.update_layout(height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.info("Daily returns data not available")
        
        with tab4:
            st.subheader(f"Price Prediction Model (Python {python_version})")
            
            if len(stock_data) > 30:  # Reduced minimum requirement
                with st.spinner("Training prediction model with Python 3.10.10 optimizations..."):
                    model_results = create_prediction_model(stock_data)
                    
                    if model_results[0] is not None:
                        model, mse, r2, X_test, y_test, y_pred, feature_names = model_results
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Model Performance:**")
                            rmse = np.sqrt(mse)
                            metrics_df = pd.DataFrame({
                                'Metric': ['R² Score', 'RMSE', 'Mean Squared Error', 'Data Points', 'Python Version'],
                                'Value': [
                                    f"{r2:.4f}",
                                    f"${rmse:.2f}",
                                    f"{mse:.4f}",
                                    f"{len(X_test)}",
                                    f"{python_version}"
                                ]
                            })
                            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                            
                            # Prediction accuracy assessment
                            st.write("**Model Assessment:**")
                            if r2 > 0.8:
                                st.success("🟢 Excellent model performance!")
                            elif r2 > 0.6:
                                st.success("🟡 Good model performance!")
                            elif r2 > 0.3:
                                st.warning("🟠 Moderate model performance")
                            else:
                                st.info("🔴 Model performance could be improved")
                            
                            # Feature list
                            st.write("**Features Used:**")
                            for i, feature in enumerate(feature_names, 1):
                                st.write(f"{i}. {feature}")
                        
                        with col2:
                            # Prediction vs Actual scatter plot
                            fig_pred = go.Figure()
                            
                            # Actual vs Predicted points
                            fig_pred.add_trace(go.Scatter(
                                x=y_test, 
                                y=y_pred,
                                mode='markers',
                                name='Predictions',
                                marker=dict(color='blue', size=8, opacity=0.6)
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
                                title=f"Predicted vs Actual Prices (Python {python_version})",
                                xaxis_title="Actual Price ($)",
                                yaxis_title="Predicted Price ($)",
                                height=400
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # Feature importance
                        if len(feature_names) > 1:
                            importance = abs(model.coef_)
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': importance
                            }).sort_values('Importance', ascending=True)  # Sort for horizontal bar
                            
                            fig_importance = px.bar(
                                importance_df, 
                                y='Feature', 
                                x='Importance',
                                orientation='h',
                                title="Feature Importance in Prediction Model"
                            )
                            fig_importance.update_layout(height=300)
                            st.plotly_chart(fig_importance, use_container_width=True)
                    else:
                        st.info("Unable to create prediction model with current data.")
            else:
                st.warning(f"Need more data points for reliable prediction. Current: {len(stock_data)}, Required: 30+")
        
        with tab5:
            st.subheader("Raw Data")
            
            # Display data options
            data_option = st.selectbox(
                "Select data to display:", 
                ["Recent Data (Last 10 days)", "Full Dataset", "Summary Statistics"]
            )
            
            if data_option == "Recent Data (Last 10 days)":
                recent_data = stock_data.tail(10).round(2)
                st.dataframe(recent_data, use_container_width=True)
            elif data_option == "Full Dataset":
                display_data = stock_data.round(2)
                st.dataframe(display_data, use_container_width=True)
            else:
                summary_data = stock_data.describe().round(2)
                st.dataframe(summary_data, use_container_width=True)
            
            # Download button
            csv = stock_data.to_csv()
            st.download_button(
                label="📥 Download data as CSV",
                data=csv,
                file_name=f"{stock_symbol}_{start_date}_{end_date}_python{python_version}_stock_data.csv",
                mime="text/csv"
            )
            
            # Data info
            st.write("**Dataset Info:**")
            st.write(f"- Total rows: {len(stock_data)}")
            st.write(f"- Columns: {', '.join(stock_data.columns)}")
            st.write(f"- Date range: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
            st.write(f"- Python version: {python_version}")
            st.write(f"- Virtual environment: {venv_status}")
    
    else:
        st.error("Could not fetch any data. This might be due to:")
        st.write("- Invalid stock symbol")
        st.write("- Network connectivity issues")
        st.write("- yfinance API limitations")
        st.write("- Date range issues")
        st.write("- Python environment issues")
        
        st.info("Try:")
        st.write("- A different stock symbol (e.g., AAPL, MSFT, GOOGL)")
        st.write("- A different date range")
        st.write("- Checking your Python 3.10.10 virtual environment")
        st.write("- Refreshing the page")

elif start_date >= end_date:
    st.error("⚠️ Start date must be before end date")

else:
    st.info("👈 Please enter a stock symbol to begin analysis.")
    
    # Quick environment check
    st.markdown("### 🔍 Environment Quick Check")
    if sys.version_info.major == 3 and sys.version_info.minor == 10 and sys.version_info.micro == 10:
        st.success("✅ **Perfect Setup!** Running on Python 3.10.10 as recommended")
    else:
        st.warning(f"⚠️ **Different Python version detected:** {python_version}")
        st.info("💡 **To use Python 3.10.10, follow the setup instructions in the expandable section above.**")

# Footer
st.markdown("---")
st.markdown("### About This Application")

# Enhanced package status for Python 3.10.10
with st.expander("📦 Complete System Information"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🐍 Python Environment:**")
        st.write(f"- **Version**: {python_full}")
        st.write(f"- **Executable**: {sys.executable}")
        st.write(f"- **Platform**: {platform.platform()}")
        st.write(f"- **Architecture**: {platform.architecture()[0]}")
        st.write(f"- **Virtual Env**: {venv_status}")
        
        if venv_status == "Yes":
            st.write(f"- **Base Prefix**: {sys.base_prefix}")
            st.write(f"- **Prefix**: {sys.prefix}")
    
    with col2:
        st.write("**📦 Package Status:**")
        packages_status = {
            "streamlit": st.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "yfinance": yf_version if YFINANCE_AVAILABLE else "❌ Not installed",
            "plotly": plotly.__version__,
            "scikit-learn": sklearn.__version__,
            "matplotlib": plt.matplotlib.__version__,
            "seaborn": sns.__version__
        }
        
        for package, version in packages_status.items():
            status = "✅" if not version.startswith("❌") else "❌"
            st.write(f"{status} **{package}**: {version}")

# Python 3.10.10 specific information
st.markdown(f"""
### 🚀 **Python 3.10.10 Stock Market Analyzer**

**Current Environment**: Python {python_version} | Virtual Environment: {venv_status}

**Why Python 3.10.10?**
- **Stable**: Well-tested and reliable for financial applications
- **Compatible**: Perfect compatibility with yfinance and financial packages  
- **Performance**: Optimized for data science workloads
- **Long-term Support**: Supported until October 2026

**Technologies Used**: Python {python_version}, Pandas, NumPy, Scikit-learn, Plotly, Streamlit{', yfinance' if YFINANCE_AVAILABLE else ''}

**Features Included**:
- 📊 Real-time/sample stock price data
- 📈 Technical indicators (MA, RSI, Bollinger Bands)
- 📉 Statistical analysis and distribution plots
- 🤖 Machine learning price prediction
- 📋 Data export functionality
- 🎨 Interactive, responsive visualizations
- 🌐 Virtual environment support
- 🔧 Python 3.10.10 optimizations

**Note**: This is for educational and demonstration purposes only. Not for actual trading decisions.
""")

# Show current limitations if any
if not YFINANCE_AVAILABLE:
    st.warning("""
    ⚠️ **yfinance Package Not Available**
    
    The app is currently running with sample data. To enable real-time data in your Python 3.10.10 environment:
    
    ```bash
    # Activate your Python 3.10.10 virtual environment
    source stock_app_env/bin/activate  # Linux/Mac
    # or
    stock_app_env\\Scripts\\activate  # Windows
    
    # Install yfinance
    pip install yfinance==0.2.61
    
    # Restart the application
    streamlit run stock_app_py310.py
    ```
    """)

# Python 3.10.10 performance tips
with st.expander("💡 Python 3.10.10 Usage Tips"):
    st.markdown("""
    **For Best Performance with Python 3.10.10:**
    - Always use a virtual environment for isolation
    - Keep packages updated to their Python 3.10.10 compatible versions
    - Choose date ranges of 1-2 years for optimal loading
    - Popular symbols (AAPL, MSFT, GOOGL) typically have better data availability
    - The prediction model works best with 100+ data points
    
    **Virtual Environment Management:**
    ```bash
    # Activate environment
    source stock_app_env/bin/activate
    
    # Check packages
    pip list
    
    # Update packages
    pip install --upgrade streamlit yfinance pandas numpy plotly scikit-learn
    
    # Deactivate when done
    deactivate
    ```
    
    **Understanding the Analysis:**
    - **Moving Averages**: Trend indicators (20-day and 50-day)
    - **RSI**: Momentum indicator (>70 overbought, <30 oversold)
    - **Bollinger Bands**: Volatility indicator showing price channels
    - **R² Score**: Model accuracy (>0.7 is considered good)
    
    **Troubleshooting in Python 3.10.10:**
    - If no data loads, check your virtual environment activation
    - Ensure all packages are installed in the correct environment
    - Try different stock symbols if API issues occur
    - Check internet connectivity for real-time data
    """)

# Final status summary
st.markdown("---")
final_status = []

if sys.version_info.major == 3 and sys.version_info.minor == 10:
    final_status.append("✅ Python 3.10.x")
else:
    final_status.append(f"⚠️ Python {python_version}")

if YFINANCE_AVAILABLE:
    final_status.append(f"✅ yfinance {yf_version}")
else:
    final_status.append("❌ yfinance missing")

if venv_status == "Yes":
    final_status.append("✅ Virtual Environment")
else:
    final_status.append("⚠️ No Virtual Environment")

final_status.append(f"📊 Streamlit {st.__version__}")

st.info(f"**System Status**: {' | '.join(final_status)} | **Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Requirements.txt generator
with st.expander("📄 Generate requirements.txt for Python 3.10.10"):
    st.markdown("**Copy this requirements.txt for your Python 3.10.10 environment:**")
    requirements = f"""streamlit=={st.__version__}
pandas=={pd.__version__}
numpy=={np.__version__}
plotly=={plotly.__version__}
scikit-learn=={sklearn.__version__}
matplotlib=={plt.matplotlib.__version__}
seaborn=={sns.__version__}
yfinance==0.2.61
requests>=2.28.0
"""
    st.code(requirements, language="text")
    
    st.download_button(
        label="📥 Download requirements.txt",
        data=requirements,
        file_name="requirements_python310.txt",
        mime="text/plain"
    )