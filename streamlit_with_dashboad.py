import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import googleapiclient.discovery

# Page configuration
st.set_page_config(page_title="Equity Research Agent", layout="wide")

# Styling
st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
    }
    .button-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .custom-button {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        flex-grow: 1;
        margin: 0 5px;
        text-align: center;
    }
    .news-item {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
        border-left: 4px solid #1f77b4;
    }
    .news-item h4 {
        margin: 0;
        color: #1f77b4;
    }
    .news-item p {
        margin: 5px 0;
        color: #666;
        font-size: 0.9em;
    }
    .news-item a {
        color: #1f77b4;
        text-decoration: none;
    }
    .news-item a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
NIFTY50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
    "HCLTECH.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "NTPC.NS",
    "NESTLEIND.NS", "TATAMOTORS.NS", "SBIN.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "POWERGRID.NS", "TITAN.NS", "ADANIPORTS.NS", "TECHM.NS", "SUNPHARMA.NS",
    "BAJFINANCE.NS", "BRITANNIA.NS", "HINDALCO.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "M&M.NS", "CIPLA.NS", "EICHERMOT.NS", "INDUSINDBK.NS", "APOLLOHOSP.NS",
    "COALINDIA.NS", "BAJAJFINSV.NS", "GRASIM.NS", "HEROMOTOCO.NS", "SBILIFE.NS",
    "UPL.NS", "ADANIENT.NS", "TATASTEEL.NS", "HDFCLIFE.NS", "JSWSTEEL.NS",
    "ONGC.NS", "BPCL.NS", "TATACONSUM.NS", "LTIM.NS", "VEDL.NS"
]

# Initialize session states
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"
if 'chat_model' not in st.session_state:
    st.session_state.chat_model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def get_moneycontrol_news():
    try:
        url = "https://www.moneycontrol.com/news/business/markets/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        news_list = []
        news_items = soup.find_all('li', class_='clearfix')
        
        for item in news_items[:5]:
            title_element = item.find('h2')
            if title_element:
                title = title_element.text.strip()
                link = title_element.find('a')['href'] if title_element.find('a') else ""
                time_element = item.find('span', class_='ago')
                time = time_element.text.strip() if time_element else ""
                
                news_list.append({
                    'title': title,
                    'link': link,
                    'time': time
                })
        
        return news_list
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return get_fake_news()

def get_youtube_videos():
    try:
        youtube = googleapiclient.discovery.build(
            'youtube', 'v3',
            developerKey=st.secrets["YOUTUBE_API_KEY"]
        )

        search_response = youtube.search().list(
            q="indian economy market analysis",
            part="snippet",
            maxResults=4,
            type="video",
            relevanceLanguage="en",
            order="date"
        ).execute()

        videos = []
        for item in search_response['items']:
            video = {
                'title': item['snippet']['title'],
                'video_id': item['id']['videoId'],
                'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                'channel': item['snippet']['channelTitle']
            }
            videos.append(video)
        
        return videos
    except Exception as e:
        st.error(f"Error fetching YouTube videos: {str(e)}")
        return []

def get_fake_news():
    return [
        {
            'title': "Markets Rally on Strong Earnings Reports",
            'summary': "Major indices surge as tech giants exceed expectations",
            'link': "https://example.com/news1",
            'time': "1 hour ago"
        },
        {
            'title': "Central Bank Maintains Interest Rates",
            'summary': "Policy remains unchanged amid economic stability",
            'link': "https://example.com/news2",
            'time': "2 hours ago"
        },
        {
            'title': "New Tech IPO Sees Strong Market Debut",
            'summary': "Shares jump 50% on first day of trading",
            'link': "https://example.com/news3",
            'time': "3 hours ago"
        }
    ]

def get_stock_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        
        if hist.empty:
            return None
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change = ((current_price - prev_close) / prev_close) * 100
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'change_percent': round(change, 2)
        }
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def get_balance_sheet(symbol):
    try:
        stock = yf.Ticker(symbol)
        balance_sheet = stock.quarterly_balance_sheet
        
        if balance_sheet.empty:
            return None
        
        balance_sheet = balance_sheet.fillna(0)
        balance_sheet = balance_sheet.applymap(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
        
        return balance_sheet
    except Exception as e:
        st.error(f"Error fetching balance sheet for {symbol}: {str(e)}")
        return None

def init_chat_model(model_name):
    if model_name == "OpenAI":
        return ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"])
    elif model_name == "Anthropic":
        return ChatAnthropic(anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"])
    elif model_name == "Groq":
        return ChatGroq(api_key=st.secrets["GROQ_API_KEY"])
    return None

def show_dashboard():
    st.markdown('<p class="big-font">Equity Research Agent</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="button-container">
        <div class="custom-button" onclick="window.location.href='#stocks'">View Stocks</div>
        <div class="custom-button" onclick="window.location.href='#equity-report'">Research</div>
        <div class="custom-button" onclick="window.location.href='#portfolio'">Portfolio</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("NIFTY Chart")
        try:
            nifty_data = yf.download("^NSEI", start=datetime.now() - timedelta(days=365), end=datetime.now())
            fig = go.Figure(data=[go.Candlestick(x=nifty_data.index,
                                                 open=nifty_data['Open'],
                                                 high=nifty_data['High'],
                                                 low=nifty_data['Low'],
                                                 close=nifty_data['Close'])])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading NIFTY chart: {str(e)}")
        
        # YouTube Videos Section
        st.subheader("Latest Market Analysis Videos")
        videos = get_youtube_videos()
        video_cols = st.columns(2)
        for idx, video in enumerate(videos):
            with video_cols[idx % 2]:
                st.markdown(f"""
                <div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px;'>
                    <img src="{video['thumbnail']}" style='width:100%;border-radius:5px;'>
                    <h4>{video['title']}</h4>
                    <p>{video['channel']}</p>
                    <a href="https://www.youtube.com/watch?v={video['video_id']}" target="_blank">Watch Video</a>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Latest Market News")
        news_items = get_moneycontrol_news()
        for item in news_items:
            st.markdown(f"""
            <div class="news-item">
                <h4>{item['title']}</h4>
                <p>{item['time']}</p>
                <a href="{item['link']}" target="_blank">Read more</a>
            </div>
            """, unsafe_allow_html=True)

def show_stocks():
    st.title("Stocks Analysis")
    
    search_term = st.text_input("Search for a stock", "").upper()
    
    st.subheader("NIFTY 50 Stocks")
    
    cols = st.columns(5)
    for i, symbol in enumerate(NIFTY50_SYMBOLS):
        info = get_stock_info(symbol)
        if info:
            with cols[i % 5]:
                color = "green" if info['change_percent'] >= 0 else "red"
                st.markdown(f"""
                <div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px;'>
                    <h4>{info['symbol'].replace('.NS', '')}</h4>
                    <p>₹{info['current_price']}</p>
                    <p style='color:{color};'>{info['change_percent']}%</p>
                </div>
                """, unsafe_allow_html=True)

def show_equity_report():
    st.title("Equity Research Report")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.selectbox("Select Stock", NIFTY50_SYMBOLS)
        if symbol:
            try:
                stock = yf.Ticker(symbol)
                
                st.subheader(f"{symbol} Stock Price")
                hist = stock.history(period="1y")
                fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                                     open=hist['Open'],
                                                     high=hist['High'],
                                                     low=hist['Low'],
                                                     close=hist['Close'])])
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Balance Sheet")
                balance_sheet = get_balance_sheet(symbol)
                if balance_sheet is not None and not balance_sheet.empty:
                    st.dataframe(balance_sheet)
                else:
                    st.warning("Balance sheet data not available for this stock.")
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    with col2:
        st.subheader("AI Research Assistant")
        
        model_options = {
            "OpenAI": "GPT-4 - Fast, reliable, good for financial analysis",
            "Anthropic": "Claude - Excellent for detailed research and analysis",
            "Groq": "LLaMA 2 - Fast inference, good for quick insights"
        }
        
        selected_model = st.selectbox(
            "Choose AI Model",
            list(model_options.keys()),
            format_func=lambda x: f"{x}: {model_options[x]}"
        )
        
        if st.button("Initialize Chat"):
            st.session_state.chat_model = init_chat_model(selected_model)
            st.success(f"Initialized {selected_model} model!")
        
        if st.session_state.chat_model:
            user_input = st.text_input("Ask about the selected stock:")
            if user_input and st.button("Send"):
                with st.spinner("Generating response..."):
                    response = st.session_state.chat_model.invoke(user_input)
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("Assistant", response))
            
            for role, message in st.session_state.chat_history:
                if role == "You":
                    st.markdown(f"**You:** {message}")
                else:
                    st.markdown(f"**Assistant:** {message}")

def show_portfolio_analysis():
    st.title("Portfolio Analysis")
    
    # Portfolio input
    st.subheader("Enter Your Portfolio")
    
    portfolio_data = []
    for i in range(5):
        cols = st.columns([2, 1, 1])
        with cols[0]:
            symbol = st.selectbox(f"Stock {i+1}", [""] + NIFTY50_SYMBOLS, key=f"symbol_{i}")
        with cols[1]:
            quantity = st.number_input(f"Quantity", key=f"quantity_{i}", min_value=0)
        with cols[2]:
            buy_price = st.number_input(f"Buy Price", key=f"price_{i}", min_value=0.0)
        
        if symbol and quantity and buy_price:
            portfolio_data.append({
                'symbol': symbol,
                'quantity': quantity,
                'buy_price': buy_price
            })
    
    if st.button("Analyze Portfolio"):
        if portfolio_data:
            st.subheader("Portfolio Analysis")
            total_value = 0
            for item in portfolio_data:
                info = get_stock_info(item['symbol'])
                if info:
                    current_value = info['current_price'] * item['quantity']
                    investment_value = item['buy_price'] * item['quantity']
                    profit_loss = current_value - investment_value
                    total_value += current_value
                    
                    st.markdown(f"""
                    <div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px;'>
                        <h4>{item['symbol']}</h4>
                        <p>Quantity: {item['quantity']}</p>
                        <p>Current Value: ₹{current_value:,.2f}</p>
                        <p>Profit/Loss: ₹{profit_loss:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown(f"### Total Portfolio Value: ₹{total_value:,.2f}")
        else:
            st.warning("Please enter at least one stock in your portfolio.")

# 7. Sidebar Navigation (ADD THIS BEFORE MAIN CONTENT)
with st.sidebar:
    st.markdown('<p class="big-font">Navigation</p>', unsafe_allow_html=True)
    
    if st.button("Dashboard", key="dashboard_btn"):
        st.session_state.page = "Dashboard"
    if st.button("Stocks", key="stocks_btn"):
        st.session_state.page = "Stocks"
    if st.button("Equity Report", key="report_btn"):
        st.session_state.page = "Equity Report"
    if st.button("Portfolio Analysis", key="portfolio_btn"):
        st.session_state.page = "Portfolio Analysis"

# Main content based on selected page
if st.session_state.page == "Dashboard":
    show_dashboard()
elif st.session_state.page == "Stocks":
    show_stocks()
elif st.session_state.page == "Equity Report":
    show_equity_report()
elif st.session_state.page == "Portfolio Analysis":
    show_portfolio_analysis()