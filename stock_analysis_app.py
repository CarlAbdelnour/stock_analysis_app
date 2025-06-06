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
except ImportError:
    st.error("""
    ❌ **yfinance package not found!**
    
    Please ensure your requirements.txt file contains:
    ```
    yfinance
    ```
    
    If you're running this locally, install it with:
    ```bash
    pip install yfinance
    ```
    """)
    st.stop()

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
st.markdown("---")

# Sidebar for user inputs
st.sidebar.header("Stock Selection")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL, MSFT):", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Popular stock suggestions
st.sidebar.markdown("**Popular Stocks:**")
popular_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"]
selected_popular = st.sidebar.selectbox("Or select from popular stocks:", [""] + popular_stocks)

if selected_popular:
    stock_symbol = selected_popular

# Function to fetch stock data with improved error handling
@st.cache_data
def fetch_stock_data(symbol, start, end):
    """Fetch stock data using yfinance with improved error handling"""
    try:
        # Validate inputs
        if not symbol or len(symbol.strip()) == 0:
            raise ValueError("Stock symbol cannot be empty")
        
        symbol = symbol.upper().strip()
        
        # Create ticker object
        stock = yf.Ticker(symbol)
        
        # Fetch data with error handling
        data = stock.history(start=start, end=end)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Try to get info, but don't fail if it's not available
        try:
            info = stock.info
        except Exception:
            info = {"symbol": symbol}  # Fallback info
        
        return data, info
        
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg.lower():
            st.error(f"Stock symbol '{symbol}' not found. Please check the symbol and try again.")
        elif "timeout" in error_msg.lower():
            st.error("Request timed out. Please try again.")
        elif "connection" in error_msg.lower():
            st.error("Connection error. Please check your internet connection and try again.")
        else:
            st.error(f"Error fetching data for {symbol}: {error_msg}")
        return None, None

# Function to calculate technical indicators
def calculate_indicators(data):
    """Calculate technical indicators"""
    try:
        # Moving averages
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
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
        
        if len(data_clean) < 50:
            raise ValueError("Insufficient data for model training")
        
        # Features: using previous days' prices, volume, and technical indicators
        features = ['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI']
        feature_data = data_clean[features].dropna()
        
        # Target: next day's closing price
        target = data_clean['Close'].shift(-1).dropna()
        
        # Align features and target
        min_length = min(len(feature_data), len(target))
        if min_length < 20:
            raise ValueError("Not enough data points for reliable prediction")
            
        X = feature_data.iloc[:min_length]
        y = target.iloc[:min_length]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return model, mse, r2, X_test, y_test, y_pred
        
    except Exception as e:
        st.error(f"Error creating prediction model: {e}")
        return None, None, None, None, None, None

# Main application logic
if stock_symbol:
    # Fetch data
    with st.spinner(f"Fetching data for {stock_symbol}..."):
        stock_data, stock_info = fetch_stock_data(stock_symbol, start_date, end_date)
    
    if stock_data is not None and not stock_data.empty:
        # Calculate indicators
        stock_data = calculate_indicators(stock_data)
        
        # Display stock info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}")
        with col2:
            if len(stock_data) > 1:
                daily_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
                change_pct = (daily_change/stock_data['Close'].iloc[-2]*100)
                st.metric("Daily Change", f"${daily_change:.2f}", f"{change_pct:.2f}%")
            else:
                st.metric("Daily Change", "N/A")
        with col3:
            st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,.0f}")
        with col4:
            if stock_info and 'marketCap' in stock_info and stock_info['marketCap']:
                st.metric("Market Cap", f"${stock_info['marketCap']:,}")
            else:
                st.metric("Market Cap", "N/A")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Price Analysis", "📈 Technical Indicators", "📉 Statistics", "🤖 Prediction", "📋 Data"])
        
        with tab1:
            st.subheader("Stock Price Analysis")
            
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], 
                                   mode='lines', name='Close Price', line=dict(color='blue')))
            
            # Add moving averages if available
            if 'MA_20' in stock_data.columns and not stock_data['MA_20'].isna().all():
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA_20'], 
                                       mode='lines', name='MA 20', line=dict(color='orange')))
            if 'MA_50' in stock_data.columns and not stock_data['MA_50'].isna().all():
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA_50'], 
                                       mode='lines', name='MA 50', line=dict(color='red')))
            
            fig.update_layout(title=f"{stock_symbol} Stock Price with Moving Averages",
                            xaxis_title="Date", yaxis_title="Price ($)",
                            height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume chart
            fig_volume = px.bar(x=stock_data.index, y=stock_data['Volume'],
                              title=f"{stock_symbol} Trading Volume")
            fig_volume.update_layout(height=300)
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with tab2:
            st.subheader("Technical Indicators")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # RSI Chart
                if 'RSI' in stock_data.columns and not stock_data['RSI'].isna().all():
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'],
                                               mode='lines', name='RSI'))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig_rsi.update_layout(title="RSI (Relative Strength Index)",
                                        yaxis_title="RSI", height=400)
                    st.plotly_chart(fig_rsi, use_container_width=True)
                else:
                    st.warning("Not enough data to calculate RSI")
            
            with col2:
                # Bollinger Bands
                if all(col in stock_data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                    fig_bb = go.Figure()
                    fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['BB_Upper'],
                                              mode='lines', name='Upper Band', line=dict(color='red')))
                    fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['BB_Middle'],
                                              mode='lines', name='Middle Band', line=dict(color='blue')))
                    fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['BB_Lower'],
                                              mode='lines', name='Lower Band', line=dict(color='green')))
                    fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'],
                                              mode='lines', name='Close Price', line=dict(color='black')))
                    fig_bb.update_layout(title="Bollinger Bands", height=400)
                    st.plotly_chart(fig_bb, use_container_width=True)
                else:
                    st.warning("Not enough data to calculate Bollinger Bands")
        
        with tab3:
            st.subheader("Statistical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Basic statistics
                st.write("**Price Statistics:**")
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"${stock_data['Close'].mean():.2f}",
                        f"${stock_data['Close'].median():.2f}",
                        f"${stock_data['Close'].std():.2f}",
                        f"${stock_data['Close'].min():.2f}",
                        f"${stock_data['Close'].max():.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Daily returns statistics
                if 'Daily_Return' in stock_data.columns:
                    st.write("**Daily Returns Statistics:**")
                    returns_stats = pd.DataFrame({
                        'Metric': ['Mean Return', 'Volatility', 'Min Return', 'Max Return'],
                        'Value': [
                            f"{stock_data['Daily_Return'].mean()*100:.2f}%",
                            f"{stock_data['Daily_Return'].std()*100:.2f}%",
                            f"{stock_data['Daily_Return'].min()*100:.2f}%",
                            f"{stock_data['Daily_Return'].max()*100:.2f}%"
                        ]
                    })
                    st.dataframe(returns_stats, use_container_width=True, hide_index=True)
            
            with col2:
                # Returns distribution
                if 'Daily_Return' in stock_data.columns:
                    fig_hist = px.histogram(stock_data, x='Daily_Return', nbins=50,
                                          title="Daily Returns Distribution")
                    fig_hist.update_layout(height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab4:
            st.subheader("Price Prediction Model")
            
            if len(stock_data) > 100:  # Ensure enough data for modeling
                with st.spinner("Training prediction model..."):
                    model_results = create_prediction_model(stock_data)
                    
                    if model_results[0] is not None:
                        model, mse, r2, X_test, y_test, y_pred = model_results
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Model Performance:**")
                            metrics_df = pd.DataFrame({
                                'Metric': ['R² Score', 'Mean Squared Error', 'RMSE'],
                                'Value': [f"{r2:.4f}", f"{mse:.4f}", f"{np.sqrt(mse):.4f}"]
                            })
                            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                            
                            # Prediction vs Actual
                            st.write("**Prediction Accuracy:**")
                            if r2 > 0.7:
                                st.success("Good model performance!")
                            elif r2 > 0.5:
                                st.warning("Moderate model performance")
                            else:
                                st.info("Model performance could be improved")
                        
                        with col2:
                            # Scatter plot of predictions vs actual
                            fig_pred = px.scatter(x=y_test, y=y_pred,
                                                title="Predicted vs Actual Prices")
                            fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                                        y=[y_test.min(), y_test.max()],
                                                        mode='lines', name='Perfect Prediction',
                                                        line=dict(color='red', dash='dash')))
                            fig_pred.update_layout(height=400)
                            st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # Feature importance
                        feature_names = ['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI']
                        importance = abs(model.coef_)
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False)
                        
                        fig_importance = px.bar(importance_df, x='Feature', y='Importance',
                                              title="Feature Importance in Prediction Model")
                        st.plotly_chart(fig_importance, use_container_width=True)
                    else:
                        st.info("Unable to create prediction model with current data.")
            else:
                st.warning("Need more data points for reliable prediction. Please select a longer date range.")
        
        with tab5:
            st.subheader("Raw Data")
            
            # Display data options
            data_option = st.selectbox("Select data to display:", 
                                     ["Recent Data (Last 10 days)", "Full Dataset", "Summary Statistics"])
            
            if data_option == "Recent Data (Last 10 days)":
                st.dataframe(stock_data.tail(10), use_container_width=True)
            elif data_option == "Full Dataset":
                st.dataframe(stock_data, use_container_width=True)
            else:
                st.dataframe(stock_data.describe(), use_container_width=True)
            
            # Download button
            csv = stock_data.to_csv()
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"{stock_symbol}_stock_data.csv",
                mime="text/csv"
            )
    
    else:
        st.error("Could not fetch data. Please check the stock symbol and try again.")

else:
    st.info("Please enter a stock symbol to begin analysis.")

# Footer
st.markdown("---")
st.markdown("### About This Application")
st.markdown("""
This stock market data science application demonstrates:
- **Data Collection**: Using yfinance API to fetch real-time stock data
- **Data Processing**: Calculating technical indicators and cleaning data
- **Exploratory Data Analysis**: Statistical analysis and visualization
- **Machine Learning**: Simple linear regression for price prediction
- **Data Visualization**: Interactive charts using Plotly
- **User Interface**: Streamlit dashboard for easy interaction

**Technologies Used**: Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit, yfinance

**Note**: This is for educational purposes only and should not be used for actual trading decisions.
""")