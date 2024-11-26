import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup as soup
import yfinance as yf
from urllib.request import urlopen
from newspaper import Article
import io
import nltk
from PIL import Image

nltk.download('punkt')

st.set_page_config(page_title='BizNews💼: Business & Finance News', page_icon='./Meta/newspaper.ico')

# Fetch real-time data
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
        ticker, currency, unit = details  # Explicit unpacking
        try:
            stock = yf.Ticker(ticker)
            price = stock.history(period="1d")['Close'].iloc[-1]
            live_data[name] = f"{currency} {price:,.2f} {unit}"
        except Exception as e:
            live_data[name] = "Data Unavailable"
    return live_data

def fetch_news_poster(poster_link):
    try:
        u = urlopen(poster_link)
        raw_data = u.read()
        image = Image.open(io.BytesIO(raw_data))
        st.image(image, use_column_width=True)
    except:
        image = Image.open('./Meta/no_image.jpg')
        st.image(image, use_column_width=True)


def display_news(list_of_news, news_quantity):
    c = 0
    for news in list_of_news:
        c += 1
        st.write('**({}) {}**'.format(c, news.title.text))
        news_data = Article(news.link.text)
        try:
            news_data.download()
            news_data.parse()
            news_data.nlp()
        except Exception as e:
            st.error(e)
        fetch_news_poster(news_data.top_image)
        with st.expander(news.title.text):
            st.markdown(
                '''<h6 style='text-align: justify;'>{}"</h6>'''.format(news_data.summary),
                unsafe_allow_html=True)
            st.markdown("[Read more at {}...]({})".format(news.source.text, news.link.text))
        st.success("Published Date: " + news.pubDate.text)
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

    category = ['--Select--', 'Trending📈 Business News', 'Search🔍 Topic', 'Live Market Prices📊']
    cat_op = st.selectbox('Select your Category', category)

    if cat_op == category[0]:
        st.warning('Please select a Category!')
    elif cat_op == category[1]:
        st.subheader("✅ Here are the latest 📈 Business News for you")
        no_of_news = st.slider('Number of News:', min_value=5, max_value=25, step=1)
        news_list = fetch_business_news()
        display_news(news_list, no_of_news)
    elif cat_op == category[2]:
        user_topic = st.text_input("Enter your Business Topic🔍")
        no_of_news = st.slider('Number of News:', min_value=5, max_value=15, step=1)

        if st.button("Search") and user_topic != '':
            user_topic_pr = user_topic.replace(' ', '')
            news_list = fetch_news_search_topic(topic=f'{user_topic_pr} business')
            if news_list:
                st.subheader("✅ Here are some {} News for you".format(user_topic.capitalize()))
                display_news(news_list, no_of_news)
            else:
                st.error("No News found for {}".format(user_topic))
        else:
            st.warning("Please write a Topic Name to Search🔍")
    elif cat_op == category[3]:
        st.subheader("📊 Live Market Prices for Key Commodities & Indexes")
        prices = fetch_live_prices()
        for key, value in prices.items():
            st.metric(label=key, value=value)


run()
