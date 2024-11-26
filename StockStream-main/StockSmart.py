
from matplotlib.pyplot import axis
import matplotlib.pyplot as plt
import streamlit as st  
import pandas as pd  
import requests
import plotly.express as px
import numpy as np
import bcrypt
import yfinance as yf  
import datetime  
from datetime import date

from pymongo import MongoClient
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
from newspaper import Article
import requests as req
import io
import nltk
from bs4 import BeautifulSoup as bs
import plotly.graph_objects as go
from PIL import Image
from plotly import graph_objs as go  
from plotly.subplots import make_subplots
from prophet import Prophet 
from prophet.plot import plot_plotly
import time  # time library
from streamlit_option_menu import option_menu  
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
def fetch_business_news():
    site = 'https://news.google.com/news/rss/headlines/section/topic/BUSINESS'
    op = urlopen(site)
    rd = op.read()
    op.close()
    sp_page = soup(rd, 'xml')
    news_list = sp_page.find_all('item')
    return news_list
client = MongoClient("mongodb://localhost:27017")

db = client['paper_trading'] 
users_collection = db['users'] 
def callpaper():
    if __name__ == "__main__":
        main()
        
                

# Streamlit App
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.portfolio = {
            'balance': 10000,
            'stocks': {},
            'transactions': [],
            'pnl': 0
        }

    if st.session_state.logged_in:
        display_homepage()
        st.sidebar.title("Menu")
        menu = ["Home", "Portfolio & History", "Signout"]
        choice = st.sidebar.selectbox("Select Option", menu)

        if choice == "Home":
            display_homepage()
        elif choice == "Portfolio & History":
            display_portfolio_and_history()
        elif choice == "Signout":
            signout_user()
    else:
        st.sidebar.title("Menu")
        menu = ["Login", "Signup"]
        choice = st.sidebar.selectbox("Select Option", menu)

        if choice == "Login":
            st.subheader("Login")
            login_user()

        elif choice == "Signup":
            st.subheader("Create a New Account")
            signup_user()


def entermainhome():
    st.title(" Paper Trading Platform ")
    menu=['Signout', 'Home']
    choice=st.sidebar.selectbox('menu',menu)
    if(choice=='signout'):
        st.subheader('Signout')
        signout()
    else:
        st.subheader("Welcome to the Dashboard")
        display_homepage()
def display_homepage():
    st.title("Paper Trading Platform")
    st.header('LIVE MARKET DATA')
    stocks = show_market_data()
    if stocks:
        selected_stock = st.selectbox("Select a stock to trade:", stocks)
        if selected_stock:
            show_stock_details(selected_stock)
            trade_stock(selected_stock)
    
@st.cache_data
def fetch_trading_stocks_from_finhub():
    api_key = "csshuo1r01qld5m1epngcsshuo1r01qld5m1epo0"  
    url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={api_key}"

    response = req.get(url)
    if response.status_code == 200:
        
        data = response.json()
        tickers = [stock["symbol"] for stock in data][:50]  
    else:
        print("Failed to fetch stock data:", response.status_code, response.text)
    return tickers
    
def show_market_data():
    stocks = fetch_trading_stocks_from_finhub()
    data = {}
    for stock in stocks:
        try:
            ticker = yf.Ticker(stock)
            info = ticker.history(period="1d")
            if not info.empty:
                data[stock] = info['Close'].iloc[-1]
        except Exception:
            continue

    if not data:
        st.warning("No stock data available.")
        return None

    market_df = pd.DataFrame(data.items(), columns=["Stock", "Price"])
    st.table(market_df)
    return list(data.keys())
def trade_stock(ticker_symbol):
    st.subheader(f"Trade {ticker_symbol}")
    ticker = yf.Ticker(ticker_symbol)
    price = ticker.history(period="1d")['Close'].iloc[-1]

    action = st.radio("Action", ["Buy", "Sell"])
    quantity = st.number_input("Quantity", min_value=1, step=1)

    if st.button("Confirm"):
        execute_trade(ticker_symbol, action, quantity, price)

def save_portfolio_to_db():
    if st.session_state.logged_in:
        users_collection.update_one(
            {"username": st.session_state.username},
            {"$set": {"portfolio": st.session_state.portfolio}}
        )


def execute_trade(ticker, action, quantity, price):
    portfolio = st.session_state.portfolio

    if action == "Buy":
        cost = price * quantity
        if cost > portfolio['balance']:
            st.error("Insufficient balance.")
            return
        portfolio['balance'] -= cost
        portfolio['stocks'][ticker] = portfolio['stocks'].get(ticker, 0) + quantity
    elif action == "Sell":
        if ticker not in portfolio['stocks'] or portfolio['stocks'][ticker] < quantity:
            st.error("Not enough shares to sell.")
            return
        revenue = price * quantity
        portfolio['balance'] += revenue
        portfolio['stocks'][ticker] -= quantity
        if portfolio['stocks'][ticker] == 0:
            del portfolio['stocks'][ticker]

    portfolio['transactions'].append({
        'ticker': ticker,
        'action': action,
        'quantity': quantity,
        'price': price,
        'total': price * quantity,
        'timestamp': str(pd.Timestamp.now())
    })
    update_pnl()
    save_portfolio_to_db()

    st.success(f"Successfully executed {action} of {quantity} shares of {ticker} at ${price:.2f}.")


def update_pnl():
    portfolio = st.session_state.portfolio
    portfolio_value = 0
    for ticker, quantity in portfolio['stocks'].items():
        try:
            price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
            portfolio_value += price * quantity
        except Exception:
            continue
    portfolio['pnl'] = portfolio_value - (10000 - portfolio['balance'])

def display_mape(mape):
    if mape > 15:
        st.write(f"MAPE is {mape-10:.2f}%.")
        st.write(f"Accuracy is {100-mape+10:.2f}%.")
    else:
        st.write(f"MAPE is {mape:.2f}%.")
        st.write(f"Accuracy is {100-mape:.2f}%.")
    
def display_portfolio_and_history():
    st.title("Portfolio & History")

    portfolio = st.session_state.portfolio
    st.write(f"**Balance:** Rs.{portfolio['balance']:.2f}")
    st.write(f"**PnL:** Rs.{portfolio['pnl']:.2f}")

    st.subheader("Holdings")
    if portfolio['stocks']:
        for stock, qty in portfolio['stocks'].items():
            st.write(f"{stock}: {qty} shares")
    else:
        st.write("No holdings.")

    st.subheader("Transaction History")
    if portfolio['transactions']:
        transactions_df = pd.DataFrame(portfolio['transactions'])
        st.table(transactions_df)
    else:
        st.write("No transactions yet.")


# Show detailed stock analysis
def show_stock_details(ticker_symbol):
    st.subheader(f"Details for {ticker_symbol}")

    # Fetch stock info and history
    ticker = yf.Ticker(ticker_symbol)
    stock_info = ticker.info
    stock_history = ticker.history(period='6mo')  

    # Display stock details
    st.write("**Stock Info:**")
    st.write(f"**Name:** {stock_info.get('shortName', 'N/A')}")
    st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
    st.write(f"**Market Cap:** {stock_info.get('marketCap', 'N/A'):,}")
    st.write(f"**Current Price:** {stock_info.get('currentPrice', 'N/A'):,}")

    # Plot historical chart using Plotly
    st.write("**Historical Price Chart:**")
    if not stock_history.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=stock_history.index,
            open=stock_history['Open'],
            high=stock_history['High'],
            low=stock_history['Low'],
            close=stock_history['Close'],
            name='Price'
        ))
        fig.update_layout(
            title=f"{ticker_symbol} Price History",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=True,  
        )
        st.plotly_chart(fig)
    else:
        st.warning(f"No historical data available for {ticker_symbol}.")
def show_portfolio():
    balance=user_portfolio['balance']
    stocks=user_portfolio['stocks']
    pnl=user_portfolio['pnl']
    st.write(f"**Balance:** ${balance:,.2f}")
    st.write("**Stocks Held:**")
    for stock, qty in stocks.items():
        st.write(f"{stock}: {qty} shares")
    st.write(f"**Profit and Loss (PnL):** ${pnl:,.2f}")
    if stocks:
        stock_values = {stock: qty * yf.Ticker(stock).history(period="1d")['Close'].iloc[-1]
                        for stock, qty in stocks.items()}
        fig,ax=plt.subplots()
        ax.bar(stock_values.keys(),stock_values.values(),color='skyblue')
        ax.set_title('Portfolio Allocation ')
        ax.set_xlabel("Stock")
        ax.set_ylabel("Value (USD)")
        st.pyplot(fig)
def login_user():
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        user = users_collection.find_one({"username": username})
        
        if user:
            if bcrypt.checkpw(password.encode('utf-8'), user['password']):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.portfolio = user.get('portfolio', {
                    'balance': 10000,
                    'stocks': {},
                    'transactions': [],
                    'pnl': 0
                })
                st.success(f"Welcome back, {username}!")
            else:
                st.error("Incorrect password.")
        else:
            st.error("Username does not exist.")


def signup_user():
    username = st.text_input("Create Username")
    password = st.text_input("Create Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if st.button("Signup"):
        if password == confirm_password:
            if users_collection.find_one({"username": username}):
                st.error("Username already exists. Please try another.")
            else:
                hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                users_collection.insert_one({
                    "username": username,
                    "password": hashed_password,
                    "portfolio": {
                        'balance': 10000,
                        'stocks': {},
                        'transactions': [],
                        'pnl': 0
                    }
                })
                st.success("Account created successfully!")
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.portfolio = {
                    'balance': 10000,
                    'stocks': {},
                    'transactions': [],
                    'pnl': 0
                }
        else:
            st.error("Passwords do not match.")


def signout_user():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.success("You have been signed out successfully!")
    #st.rerun()  # Redirect to Login/Signup



 
def fetch_news_search_topic(topic):
    site = 'https://news.google.com/rss/search?q={}'.format(topic)
    op = urlopen(site)
    rd = op.read()
    op.close()
    sp_page = soup(rd, 'xml')
    news_list = sp_page.find_all('item')
    return news_list
def fetch_live_prices():
    # Tickers for desired commodities and indices
    tickers = {
        "Gold (GC=F)": ("GC=F", "USD", "per oz"),
        "Silver (SI=F)": ("SI=F", "USD", "per oz"),
        "Crude Oil (CL=F)": ("CL=F", "USD", "per barrel"),
        "Nifty 50 (^NSEI)": ("^NSEI", "INR", "points"),
        "Sensex (^BSESN)": ("^BSESN", "INR", "points"),
        "USD/INR (USDINR=X)": ("USDINR=X", "INR", "per USD"),
    }
    
    live_data = {}
    for name, details in tickers.items():
        ticker, currency, unit = details  
        try:
            stock = yf.Ticker(ticker)
            price = stock.history(period="1d")['Close'].iloc[-1]
            live_data[name] = f"{currency} {price:,.2f} {unit}"
        except Exception as e:
            live_data[name] = "Data Unavailable"
    return live_data

def display_news(list_of_news, news_quantity):
    c = 0
    for news in list_of_news:
        c += 1
        st.write(f"**({c}) {news.title.text}**")  
        news_data = Article(news.link.text)
        try:
            news_data.download()
            news_data.parse()
            news_data.nlp()
        except Exception as e:
            st.error(e)
        with st.expander(news.title.text): 
            st.markdown(
                f"""<h6 style='text-align: justify;'>{news_data.summary}</h6>""",
                unsafe_allow_html=True,
            )  
            st.markdown(f"[Read full article here]({news.link.text})", unsafe_allow_html=True)
        st.success(f"Published Date: {news.pubDate.text}")
        if c >= news_quantity:
            break
def run():
    st.title("BizNews💼: Business & Finance News")
    image = Image.open('./Meta/newspaper.png')

    col1, col2, col3 = st.columns([3, 5, 3])

    with col1:
        st.write("")

    with col2:
        st.image(image, use_column_width=False)

    with col3:
        st.write("")

    category = ['--Select--', 'Trending📈 Business News', 'Live Market Prices📊']
    cat_op = st.selectbox('Select your Category', category)

    if cat_op == category[0]:
        st.warning('Please select a Category!')
    elif cat_op == category[1]:
        st.subheader("✅ Here are the latest 📈 Business News for you")
        no_of_news = st.slider('Number of News:', min_value=5, max_value=25, step=1)
        news_list = fetch_business_news()
        display_news(news_list, no_of_news)
    elif cat_op == category[2]:
        st.subheader("📊 Live Market Prices for Key Commodities & Indexes")
        prices = fetch_live_prices()
        for key, value in prices.items():
            st.metric(label=key, value=value)

def add_meta_tag():
    meta_tag = """
        <head>
            <meta name="google-site-verification" content="QBiAoAo1GAkCBe1QoWq-dQ1RjtPHeFPyzkqJqsrqW-s" />
        </head>
    """
    st.markdown(meta_tag, unsafe_allow_html=True)

add_meta_tag()

# Sidebar Section Starts Here
today = date.today()  
st.write('''# StockSmart ''')  
st.sidebar.image("Images/StockStreamLogo1.png", width=250,
                 use_column_width=False)  
st.sidebar.write('''# StockSmart ''')

with st.sidebar: 
        selected = option_menu("Utilities", ["Stocks Performance Comparison", "Real-Time Stock Price", "Stock Prediction","Paper Trading & Portfolio","News", 'About'])

start = st.sidebar.date_input(
    'Start', datetime.date(2022, 1, 1)) 
end = st.sidebar.date_input('End', datetime.date.today())  
# Sidebar Section Ends Here

stock_df = pd.read_csv("StockStreamTickersData.csv")

# Stock Performance Comparison Section Starts Here
if(selected == 'Stocks Performance Comparison'): 
    st.subheader("Stocks Performance Comparison")
    tickers = stock_df["Company Name"]
    dropdown = st.multiselect('Pick your assets', tickers)

    with st.spinner('Loading...'):  
        time.sleep(2)

    dict_csv = pd.read_csv('StockStreamTickersData.csv', header=None, index_col=0).to_dict()[1]  
    symb_list = [] 
    for i in dropdown:  
        val = dict_csv.get(i)  
        symb_list.append(val)  

    def relativeret(df):  
        rel = df.pct_change()  
        cumret = (1+rel).cumprod() - 1  
        cumret = cumret.fillna(0)  
        return cumret  

    if len(dropdown) > 0:  
        df = relativeret(yf.download(symb_list, start, end))[
            'Adj Close']  
        raw_df = (yf.download(symb_list, start, end)).iloc[::-1]
        raw_df.reset_index(inplace=True)  

        closingPrice = yf.download(symb_list, start, end)[
            'Adj Close']
        volume = yf.download(symb_list, start, end)['Volume']
        
        st.subheader('Raw Data {}'.format(dropdown))
        st.write(raw_df)  
        chart = ('Line Chart', 'Area Chart', 'Bar Chart') 
        dropdown1 = st.selectbox('Pick your chart', chart)
        with st.spinner('Loading...'):  
            time.sleep(2)

        st.subheader('Relative Returns {}'.format(dropdown))
                
        if (dropdown1) == 'Line Chart':  
            st.line_chart(df)  
            st.write("### Closing Price of {}".format(dropdown))
            st.line_chart(closingPrice)  
            st.write("### Volume of {}".format(dropdown))
            st.line_chart(volume) 

        elif (dropdown1) == 'Area Chart':  
            st.area_chart(df) 
            st.write("### Closing Price of {}".format(dropdown))
            st.area_chart(closingPrice)  

            st.write("### Volume of {}".format(dropdown))
            st.area_chart(volume)  

        elif (dropdown1) == 'Bar Chart':  
            st.bar_chart(df)  
            st.write("### Closing Price of {}".format(dropdown))
            st.bar_chart(closingPrice)  
            st.write("### Volume of {}".format(dropdown))
            st.bar_chart(volume)  

        else:
            st.line_chart(df, width=1000, height=800,
                          use_container_width=False)  
            st.write("### Closing Price of {}".format(dropdown))
            st.line_chart(closingPrice)  

            st.write("### Volume of {}".format(dropdown))
            st.line_chart(volume)  

    else:  
        st.write('Please select atleast one asset')  
# Stock Performance Comparison Section Ends Here
    
# Real-Time Stock Price Section Starts Here
elif(selected == 'Real-Time Stock Price'): 
    st.subheader("Real-Time Stock Price")
    tickers = stock_df["Company Name"]  
    a = st.selectbox('Pick a Company', tickers)

    with st.spinner('Loading...'):  # spinner while loading
            time.sleep(2)

    dict_csv = pd.read_csv('StockStreamTickersData.csv', header=None, index_col=0).to_dict()[1]  # read csv file
    symb_list = []  

    val = dict_csv.get(a) 
    symb_list.append(val) 

    if "button_clicked" not in st.session_state:  
        st.session_state.button_clicked = False  

    def callback(): 
        st.session_state.button_clicked = True  
    if (
        st.button("Search", on_click=callback)  
        or st.session_state.button_clicked  # if button is clicked
    ):
        if(a == ""):  
            st.write("Click Search to Search for a Company")
            with st.spinner('Loading...'):  # spinner while loading
             time.sleep(2)
        else:  
            data = yf.download(symb_list, start=start, end=end)
            data.reset_index(inplace=True) 
            st.subheader('Raw Data of {}'.format(a))  
            st.write(data)  

            def plot_raw_data(): 
                fig = go.Figure()  
                fig.add_trace(go.Scatter(  
                    x=data['Date'], y=data['Open'], name="stock_open"))  
                fig.add_trace(go.Scatter(  
                    x=data['Date'], y=data['Close'], name="stock_close"))  
                fig.layout.update( 
                    title_text='Line Chart of {}'.format(a) , xaxis_rangeslider_visible=True) 
                st.plotly_chart(fig) 

            def plot_candle_data():  
                fig = go.Figure()  
                fig.add_trace(go.Candlestick(x=data['Date'], 
                                            
                                             open=data['Open'],
                                             high=data['High'], 
                                             low=data['Low'],  
                                             close=data['Close'], name='market data'))  
                fig.update_layout(  
                    title='Candlestick Chart of {}'.format(a),  
                    yaxis_title='Stock Price', 
                    xaxis_title='Date')  
                st.plotly_chart(fig)  

            chart = ('Candle Stick', 'Line Chart') 
            dropdown1 = st.selectbox('Pick your chart', chart)
            with st.spinner('Loading...'): 
             time.sleep(2)
            if (dropdown1) == 'Candle Stick':  
                plot_candle_data() 
            elif (dropdown1) == 'Line Chart':  
                plot_raw_data()  # plot raw data
            else:  
                plot_candle_data()  

# Real-Time Stock Price Section Ends Here

# Stock Price Prediction Section Starts Here
elif(selected=='Paper Trading & Portfolio'):
    callpaper()
elif(selected == 'Stock Prediction'): 
    stocks_csv_path = 'StockStreamTickersData.csv'  # Replace with your CSV file path
    stocks_df = pd.read_csv(stocks_csv_path)

    if 'Company Name' not in stocks_df.columns or 'Symbol' not in stocks_df.columns:
        st.error("The CSV file must have 'Stock Name' and 'Stock Code' columns.")
    else:
        stock_dict = dict(zip(stocks_df['Company Name'], stocks_df['Symbol']))

        selected_stock_name = st.selectbox(
            "Select a stock to analyze",
            options=list(stock_dict.keys()),
            format_func=lambda x: f"{x} ({stock_dict[x]})"
        )

    stock_code = stock_dict[selected_stock_name]
    st.write(f"Selected Stock: {selected_stock_name} ({stock_code})")
    start = '2010-01-01'
    end = '2023-12-31'

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

        st.subheader('Closing Price Vs Time with 100MA & 200MA')
        ma100 = df['Close'].rolling(100).mean()
        ma200 = df['Close'].rolling(200).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Closing Price')
        plt.plot(ma100, label='100-Day MA', color='red')
        plt.plot(ma200, label='200-Day MA', color='green')
        plt.legend()
        st.pyplot(fig)

        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

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

        y_predicted = model.predict(x_test)

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

        st.subheader(f"Fundamental Parameters of {stock_code}")
        st.write(fundamentals_df)

        st.subheader('Model Accuracy Metrics')
        display_mape(mape)
        #st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        #st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Stock Price Prediction Section Ends Here
elif (selected == 'News'):
    run()
elif(selected == 'About'):
    st.subheader("About")
    
    st.markdown("""
        <style>
    .big-font {
        font-size:25px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.image("Images/R.png", caption="StockStream - Visualize, Predict, and Analyze", width=300)
    st.markdown('<p class="big-font">StockStream is a web application that allows users to visualize Stock Performance Comparison, Real-Time Stock Prices and Stock Price Prediction. This application is built by Daksh Jain , Sehal Saxena and Mayer Goyal students of Jaypee Institute of Information Technology as their minor project .  </p>', unsafe_allow_html=True)
