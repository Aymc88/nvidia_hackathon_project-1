import os
import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_tavily import TavilySearch
from langchain import hub

# --- 自定义工具 ---
@tool
def get_stock_data(ticker: str) -> dict:
    """
    获取指定股票代码的最新关键数据，包括盘前价格、市值、市盈率以及最新的RSI值。
    Use this tool to get quantitative financial data for a stock, including the latest RSI.
    """
    try:
        stock = yf.Ticker(ticker)
        # 获取足够的数据来计算RSI
        hist_for_rsi = stock.history(period="1mo")
        if hist_for_rsi.empty:
            return {"error": "无法获取足够的历史数据来计算RSI。"}

        # 计算 RSI
        delta = hist_for_rsi['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = rsi.iloc[-1]
        
        # 获取当天的分钟级数据
        data = stock.history(period="1d", interval="1m")
        if data.empty:
            return {"error": f"无法获取股票 {ticker} 的当天数据，请检查代码是否正确。"}
        
        pre_market_price = data['Close'].iloc[-1]
        info = stock.info
        
        return {
            "盘前价格 (Pre-Market Price)": f"${pre_market_price:.2f}",
            "前日收盘价 (Previous Close)": info.get("previousClose"),
            "14日RSI (14-Day RSI)": f"{latest_rsi:.2f}",
            "市值 (Market Cap)": f"{info.get('marketCap', 0) / 1_000_000_000:.2f}B",
            "市盈率 (PE Ratio)": info.get("trailingPE"),
        }
    except Exception as e:
        return {"error": f"获取股票数据时发生错误: {e}"}

@tool
def generate_stock_chart(ticker: str) -> str:
    """
    获取股票最近90天的数据，生成一张专业的蜡烛图（K线图），包含布林带、成交量和RSI附图，并保存为图片。返回图片路径。
    Use this tool to visualize the recent price trend of a stock with a candlestick chart, Bollinger Bands, and RSI.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="90d")
        
        if hist.empty:
            return f"无法为 {ticker} 生成图表，因为找不到历史数据。"

        # 计算布林带
        hist['MiddleBand'] = hist['Close'].rolling(window=20).mean()
        hist['UpperBand'] = hist['MiddleBand'] + 2 * hist['Close'].rolling(window=20).std()
        hist['LowerBand'] = hist['MiddleBand'] - 2 * hist['Close'].rolling(window=20).std()

        # 计算RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # 设置图表样式
        mc = mpf.make_marketcolors(up='cyan', down='magenta', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style='nightclouds')

        # 添加布林带和RSI到图表
        add_plot = [
            mpf.make_addplot(hist['UpperBand'], color='orange', width=0.7),
            mpf.make_addplot(hist['MiddleBand'], color='yellow', width=1.2), # 中线加粗
            mpf.make_addplot(hist['LowerBand'], color='orange', width=0.7),
            mpf.make_addplot(hist['RSI'], panel=2, color='lime', ylabel='RSI'),
        ]
        
        chart_path = f"{ticker}_chart.png"
        
        # 将图表标签改为英文
        mpf.plot(
            hist,
            type='candle',
            style=s,
            title=f"\n{ticker} 90-Day Technical Analysis",
            ylabel='Price (USD)',
            volume=True,
            volume_panel=1,
            panel_ratios=(3, 1, 1),
            ylabel_lower='Volume',
            addplot=add_plot,
            savefig=dict(fname=chart_path, dpi=300, bbox_inches='tight')
        )
        
        return f"一张专业的技术分析图表已生成于 {chart_path}。该图表包含蜡烛图、布林带和RSI。"
    except Exception as e:
        return f"生成图表时发生错误: {e}"

# --- Streamlit 应用主体 ---

st.set_page_config(page_title="AI股票圣手", layout="wide")
st.title("📈 AI股票圣手")

with st.sidebar:
    with st.expander("🔑 配置 API 密钥", expanded=False):
        nvidia_api_key ="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" 
        tavily_api_key ="vly-dev-JriFeFv705M2cI04fu7kY2Kyr5svZ4RV" 

    keys_provided = nvidia_api_key and tavily_api_key
    if not keys_provided:
        st.warning("请输入所有必需的 API 密钥以启动智能代理。")

# --- 聊天界面 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # 为历史消息也应用分栏布局
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(message["content"])
        if "chart_path" in message and os.path.exists(message["chart_path"]):
            with col2:
                st.image(message["chart_path"], caption="Technical Analysis Chart")

if keys_provided:
    if ticker_prompt := st.chat_input("请输入您想分析的股票代码 (例如: NVDA)"):
        st.session_state.messages.append({"role": "user", "content": ticker_prompt})
        with st.chat_message("user"):
            st.markdown(ticker_prompt)

        with st.chat_message("assistant"):
            col1, col2 = st.columns([2, 1]) 

            with col1:
                with st.spinner("AI 正在进行深度分析... (这可能需要1-2分钟)"):
                    # 创建 LLM 和工具
                    llm = ChatOpenAI(model="meta/llama-3.1-70b-instruct", openai_api_base="https://integrate.api.nvidia.com/v1", openai_api_key=nvidia_api_key, temperature=0.0)
                    tavily_search = TavilySearch(max_results=5, tavily_api_key=tavily_api_key)
                    
                    # 强制按顺序执行工具，收集信息
                    news_summary = tavily_search.invoke(f"为 {ticker_prompt} 搜索最新的新闻和市场情绪")
                    stock_data = get_stock_data(ticker_prompt)
                    chart_result = generate_stock_chart(ticker_prompt)

                    # START OF CHANGE: 将最终的提示词改为中文
                    final_prompt = f"""
                    你是一位专业的股票分析师。请根据以下信息，为股票 {ticker_prompt} 提供一份全面、结构清晰、易于阅读的中文盘前分析报告。

                    这是你已经收集到的信息:
                    
                    1.  **最新新闻与市场情绪 (来自 Tavily 搜索):**
                        {news_summary}

                    2.  **关键财务数据 (来自 Yahoo Finance):**
                        {stock_data}

                    3.  **技术图表分析:**
                        一份专业的技术分析图表已生成: {chart_result}。请分析其关键特征，如布林带和RSI指标。

                    请综合以上所有信息，形成一份最终的、结论性的中文报告。
                    """
                    # END OF CHANGE
                    
                    # 只让 LLM 进行最后一次调用
                    response = llm.invoke(final_prompt)
                    response_content = response.content
                    st.markdown(response_content)
            
            with col2:
                chart_path = f"{ticker_prompt}_chart.png"
                if os.path.exists(chart_path):
                    st.image(chart_path, caption="Technical Analysis Chart")
            
            assistant_message = {"role": "assistant", "content": response_content, "chart_path": chart_path}
            st.session_state.messages.append(assistant_message)
else:
    st.info("💡 请在左侧边栏输入您的 API 密钥以开始分析。")

