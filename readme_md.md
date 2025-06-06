# Stock Market Data Science Application

A comprehensive stock market analysis and prediction tool built with Streamlit, featuring real-time data fetching, technical indicators, statistical analysis, and machine learning predictions.

## Features

- **Real-time Stock Data**: Fetch current and historical stock data using yfinance
- **Technical Analysis**: Moving averages, RSI, Bollinger Bands
- **Statistical Analysis**: Comprehensive price and returns statistics
- **Machine Learning**: Linear regression model for price prediction
- **Interactive Visualizations**: Plotly charts for data exploration
- **User-friendly Interface**: Streamlit dashboard with multiple analysis tabs

## Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **yfinance**: Stock data API
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models
- **Plotly**: Interactive visualizations
- **Matplotlib & Seaborn**: Additional plotting capabilities

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/stock-market-analyzer.git
cd stock-market-analyzer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT) in the sidebar
2. Select date range for analysis
3. Explore different tabs:
   - **Price Analysis**: Stock price charts with moving averages
   - **Technical Indicators**: RSI and Bollinger Bands
   - **Statistics**: Comprehensive statistical analysis
   - **Prediction**: Machine learning price prediction model
   - **Data**: Raw data view and download options

## Popular Stock Symbols

- AAPL (Apple Inc.)
- GOOGL (Alphabet Inc.)
- MSFT (Microsoft Corporation)
- AMZN (Amazon.com Inc.)
- TSLA (Tesla Inc.)
- META (Meta Platforms Inc.)
- NVDA (NVIDIA Corporation)

## Model Performance

The application uses a linear regression model with the following features:
- Open, High, Low prices
- Trading volume
- 20-day and 50-day moving averages
- RSI (Relative Strength Index)

Model performance is evaluated using R² score and RMSE metrics.

## Disclaimer

⚠️ **Important**: This application is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with financial advisors and conduct thorough research before making investment choices.

## Contributing

Feel free to fork this repository and submit pull requests for improvements or bug fixes.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

For questions or suggestions, please open an issue in this repository.
