import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Initialize Streamlit
st.title('Stock Trend Prediction - Multi-Stock Support')

# Predefined list of stocks (Indian and Foreign Markets)
stocks_csv_path = 'StockStreamTickersData.csv'  # Replace with your CSV file path
stocks_df = pd.read_csv(stocks_csv_path)

# Check if required columns exist in the CSV
if 'Company Name' not in stocks_df.columns or 'Symbol' not in stocks_df.columns:
    st.error("The CSV file must have 'Stock Name' and 'Stock Code' columns.")
else:
    # Map stock names to their codes
    stock_dict = dict(zip(stocks_df['Company Name'], stocks_df['Symbol']))

    # Add a dropdown to select a stock
    selected_stock_name = st.selectbox(
        "Select a stock to analyze",
        options=list(stock_dict.keys()),
        format_func=lambda x: f"{x} ({stock_dict[x]})"
    )

    # Retrieve the selected stock code
    stock_code = stock_dict[selected_stock_name]
    st.write(f"Selected Stock: {selected_stock_name} ({stock_code})")
# Date range for fetching data
start = '2010-01-01'
end = '2023-12-31'

# Load pre-trained model
model = load_model('keras_model.h5')

# Fetch stock data
st.subheader(f"Analysis for {selected_stock_name} ({stock_code})")
df = yf.download(stock_code, start=start, end=end)

if df.empty:
    st.write(f"No data found for {stock_code}. Please check the ticker symbol.")
else:
    # Display basic stats
    st.write(df.describe())

    # Plot Closing Prices
    st.subheader('Closing Price Vs Time')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Closing Price')
    plt.legend()
    st.pyplot(fig)

    # Add Moving Averages
    st.subheader('Closing Price Vs Time with 100MA & 200MA')
    ma100 = df['Close'].rolling(100).mean()
    ma200 = df['Close'].rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Closing Price')
    plt.plot(ma100, label='100-Day MA', color='red')
    plt.plot(ma200, label='200-Day MA', color='green')
    plt.legend()
    st.pyplot(fig)

    # Prepare data for prediction
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

    # Scale the data (Use training scaler fitted on AAPL)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_training = scaler.fit_transform(data_training)
    scaled_testing = scaler.transform(pd.concat([data_training.tail(100), data_testing]))

    x_test = []
    y_test = []
    for i in range(100, scaled_testing.shape[0]):
        x_test.append(scaled_testing[i-100:i])
        y_test.append(scaled_testing[i, 0])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Make predictions
    y_predicted = model.predict(x_test)

    # Rescale predictions to original values
    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Plot Original vs Predicted
    st.subheader('Original Vs Prediction')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Original Price', color='blue')
    plt.plot(y_predicted, label='Predicted Price', color='orange')
    plt.legend()
    st.pyplot(fig)
    dates = data_testing.index[100:]
    min_length = min(len(dates), len(y_test), len(y_predicted))
    dates = dates[:min_length]
    y_test = y_test[:min_length]
    y_predicted = y_predicted[:min_length]

    # Create DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Date': dates,
        'Actual Price': y_test.flatten(),
        'Predicted Price': y_predicted.flatten(),
    })
    comparison_df['Percentage Error'] = abs(
        (comparison_df['Actual Price'] - comparison_df['Predicted Price']) / comparison_df['Actual Price'] * 100
    )

    # Display the comparison table
    st.subheader('Prediction vs Actual Prices')
    st.write(comparison_df)
    st.subheader('Actual vs Predicted Price Chart')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(comparison_df['Date'], comparison_df['Actual Price'], label='Actual Price', color='blue')
    ax.plot(comparison_df['Date'], comparison_df['Predicted Price'], label='Predicted Price', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Actual vs Predicted Price Over Time')
    ax.legend()

    # Rotate dates for better readability
    plt.xticks(rotation=45)
    st.pyplot(fig)
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
    mape = mean_absolute_percentage_error(y_test, y_predicted) * 100
    mse = mean_squared_error(y_test, y_predicted)
    rmse = np.sqrt(mse)

    #Dividend Analysis 
    stock = yf.Ticker(stock_code)
    fundamentals = {
    "Previous Close": stock.info.get("previousClose"),
    "Open": stock.info.get("open"),
    "52-Week High": stock.info.get("fiftyTwoWeekHigh"),
    "52-Week Low": stock.info.get("fiftyTwoWeekLow"),
    "Market Cap": stock.info.get("marketCap"),
    "Dividend Yield (%)": stock.info.get("dividendYield") * 100 if stock.info.get("dividendYield") else "N/A",
    "P/E Ratio (TTM)": stock.info.get("trailingPE"),
    "P/B Ratio": stock.info.get("priceToBook"),
    "EPS (TTM)": stock.info.get("trailingEps"),
    "Beta": stock.info.get("beta"),
    "Profit Margins (%)": stock.info.get("profitMargins") * 100 if stock.info.get("profitMargins") else "N/A",}

    fundamentals_df = pd.DataFrame(fundamentals.items(), columns=["Parameter", "Value"])
    fundamentals_df = pd.DataFrame(fundamentals.items(), columns=["Parameter", "Value"])

# Display fundamentals table
    st.subheader(f"Fundamental Parameters of {stock_code}")
    st.write(fundamentals_df)

    st.subheader('Model Accuracy Metrics')
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")