# Stock Market Data Science Application
# A simple yet comprehensive data science project for portfolio

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
except ImportError:
    st.error("matplotlib and seaborn packages not found. Please install them.")
    st.stop()

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    st.error("plotly package not found. Please install it.")
    st.stop()

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError:
    st.error("scikit-learn package not found. Please install it.")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Stock Market Analyzer",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Market Data Science Application")
st.markdown("### A comprehensive tool for stock analysis and prediction")

# Show yfinance status
if YFINANCE_AVAILABLE:
    st.success("✅ Real-time data available via yfinance")
else:
    st.warning("⚠️ yfinance not available - using sample data for demonstration")

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

# Function to fetch stock data with multiple fallback strategies
@st.cache_data
def fetch_stock_data(symbol, start, end):
    """Fetch stock data using yfinance with fallback to sample data"""
    
    if not YFINANCE_AVAILABLE:
        st.info("🔄 Using sample data (yfinance not available)")
        data = generate_sample_data(symbol, start, end)
        info = {"symbol": symbol, "longName": f"{symbol} Corporation (Sample Data)"}
        return data, info
    
    # Try multiple strategies with yfinance
    try:
        # Validate inputs
        if not symbol or len(symbol.strip()) == 0:
            raise ValueError("Stock symbol cannot be empty")
        
        symbol = symbol.upper().strip()
        
        # Strategy 1: Try with standard parameters
        stock = yf.Ticker(symbol)
        
        # Try different date formats and periods
        try:
            # First try: exact date range
            data = stock.history(start=start, end=end, auto_adjust=True, prepost=True)
            
            if data.empty:
                st.warning(f"No data for exact date range, trying extended period...")
                # Second try: extend the date range
                extended_start = start - timedelta(days=30)
                data = stock.history(start=extended_start, end=end, auto_adjust=True, prepost=True)
                
                if data.empty:
                    st.warning(f"No data for extended range, trying different method...")
                    # Third try: use period instead of dates
                    data = stock.history(period="1y", auto_adjust=True, prepost=True)
                    
                    if not data.empty:
                        # Filter to requested date range
                        data = data[data.index >= pd.to_datetime(start)]
                        data = data[data.index <= pd.to_datetime(end)]
        
        except Exception as fetch_error:
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
        
        st.success(f"✅ Successfully fetched real data for {symbol}")
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
    """Calculate technical indicators"""
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
    """Simple linear regression model for price prediction"""
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
        
        # Split data
        test_size = min(0.3, max(0.1, len(X) * 0.2 / len(X)))  # Adaptive test size
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
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
st.sidebar.header("Stock Selection")
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

# Main application logic
if stock_symbol and start_date < end_date:
    # Fetch data
    with st.spinner(f"Fetching data for {stock_symbol}..."):
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
        st.info(f"📊 Data Source: {data_source} | Data Points: {len(stock_data)} | Date Range: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
        
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
                title=f"{stock_symbol} Stock Price with Moving Averages",
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
            st.subheader("Price Prediction Model")
            
            if len(stock_data) > 30:  # Reduced minimum requirement
                with st.spinner("Training prediction model..."):
                    model_results = create_prediction_model(stock_data)
                    
                    if model_results[0] is not None:
                        model, mse, r2, X_test, y_test, y_pred, feature_names = model_results
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Model Performance:**")
                            rmse = np.sqrt(mse)
                            metrics_df = pd.DataFrame({
                                'Metric': ['R² Score', 'RMSE', 'Mean Squared Error', 'Data Points'],
                                'Value': [
                                    f"{r2:.4f}",
                                    f"${rmse:.2f}",
                                    f"{mse:.4f}",
                                    f"{len(X_test)}"
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
                                title="Predicted vs Actual Prices",
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
                file_name=f"{stock_symbol}_{start_date}_{end_date}_stock_data.csv",
                mime="text/csv"
            )
            
            # Data info
            st.write("**Dataset Info:**")
            st.write(f"- Total rows: {len(stock_data)}")
            st.write(f"- Columns: {', '.join(stock_data.columns)}")
            st.write(f"- Date range: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
    
    else:
        st.error("Could not fetch any data. This might be due to:")
        st.write("- Invalid stock symbol")
        st.write("- Network connectivity issues")
        st.write("- yfinance API limitations")
        st.write("- Date range issues")
        
        st.info("Try:")
        st.write("- A different stock symbol (e.g., AAPL, MSFT, GOOGL)")
        st.write("- A different date range")
        st.write("- Refreshing the page")

elif start_date >= end_date:
    st.error("⚠️ Start date must be before end date")

else:
    st.info("👈 Please enter a stock symbol to begin analysis.")

# Footer
st.markdown("---")
st.markdown("### About This Application")

# Package status
with st.expander("📦 System Information"):
    st.write("**Package Status:**")
    packages_status = {
        "yfinance": YFINANCE_AVAILABLE,
        "pandas": True,
        "numpy": True,
        "plotly": True,
        "scikit-learn": True,
        "streamlit": True
    }
    
    for package, available in packages_status.items():
        status = "✅" if available else "❌"
        st.write(f"{status} {package}")

st.markdown(f"""