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
from PortfolioAnalyser import PortfolioAnalyser, Engine
import io
import sys
import base64

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

def get_fake_news():
    # Simulated news data since web scraping might be unreliable
    return [
        {
            'title': "Markets Rally on Strong Earnings Reports",
            'summary': "Major indices surge as tech giants exceed expectations",
            'link': "https://example.com/news1"
        },
        {
            'title': "Central Bank Maintains Interest Rates",
            'summary': "Policy remains unchanged amid economic stability",
            'link': "https://example.com/news2"
        },
        {
            'title': "New Tech IPO Sees Strong Market Debut",
            'summary': "Shares jump 50% on first day of trading",
            'link': "https://example.com/news3"
        },
        {
            'title': "Global Markets React to Economic Data",
            'summary': "Asian markets lead gains after positive US jobs report",
            'link': "https://example.com/news4"
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
        # Get quarterly balance sheet for more data points
        balance_sheet = stock.quarterly_balance_sheet
        
        if balance_sheet.empty:
            return None
        
        # Format the balance sheet for better display
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

# Sidebar
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

def show_dashboard():
    st.markdown('<p class="big-font">Equity Research Agent</p>', unsafe_allow_html=True)
    
    # Interactive buttons within the main window
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
    
    with col2:
        st.subheader("Latest News")
        news_items = get_fake_news()
        for item in news_items:
            st.markdown(f"""
            <div class="news-item">
                <h4>{item['title']}</h4>
                <p>{item['summary']}</p>
                <a href="{item['link']}" target="_blank">Read more</a>
            </div>
            """, unsafe_allow_html=True)

def show_stocks():
    st.title("Stocks Analysis")
    
    # Search functionality
    search_term = st.text_input("Search for a stock", "").upper()
    
    # Display Nifty 50 stocks
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
                
                # Stock Chart
                st.subheader(f"{symbol} Stock Price")
                hist = stock.history(period="1y")
                fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                                     open=hist['Open'],
                                                     high=hist['High'],
                                                     low=hist['Low'],
                                                     close=hist['Close'])])
                st.plotly_chart(fig, use_container_width=True)
                
                # Balance Sheet
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
        
        # Model selection
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
        
        # Chat interface
        if st.session_state.chat_model:
            user_input = st.text_input("Ask about the selected stock:")
            if user_input and st.button("Send"):
                with st.spinner("Generating response..."):
                    response = st.session_state.chat_model.invoke(user_input)
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("Assistant", response))
            
            # Display chat history
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

            # Portfolio analysis
            st.markdown("### Portfolio Analysis")
            stocks = list(portfolio.keys())
            weights = list(portfolio.values())

            st.markdown("Start date: 2023-04-01")
            portfolio = Engine(
                start_date="2024-04-01",
                portfolio=stocks,
                weights=weights
            )

            buffer = io.StringIO()
            sys.stdout = buffer
            PortfolioAnalyser(portfolio, report=True)
            sys.stdout = sys.__stdout__

            analyser_output = buffer.getvalue()
            st.markdown(f"```{analyser_output}```")

            # View report PDF
            # Opening file from file path
            with open("report.pdf", "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')

            # Embedding PDF in HTML
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
            
            # Displaying File
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.warning("Please enter at least one stock in your portfolio.")


# Main content based on selected page
if st.session_state.page == "Dashboard":
    show_dashboard()
elif st.session_state.page == "Stocks":
    show_stocks()
elif st.session_state.page == "Equity Report":
    show_equity_report()
elif st.session_state.page == "Portfolio Analysis":
    show_portfolio_analysis()
