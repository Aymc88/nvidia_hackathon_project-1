import streamlit as st
import os

from dotenv import load_dotenv
# 这会自动加载 .env 文件中的变量
load_dotenv()

import sys
import pandas as pd
from PIL import Image
import yfinance as yf
import mplfinance as mpf
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

# --- 关键步骤：添加子项目的路径 ---
sys.path.append(os.path.abspath('Analysis'))
sys.path.append(os.path.abspath('Color'))

# --- 导入我们需要的模块 ---
from analyzer import analyze_image_sentiment, analyze_image_sentiment_v2

# --- 最终版核心：创建一个带缓存的数据获取函数 (不使用Session) ---
@st.cache_data(ttl=600) # 缓存10分钟
def get_stock_data_from_yf(ticker: str):
    """
    一个带缓存的函数，只请求一次90天的所有数据。
    这是我们应用中唯一会产生网络请求的地方。
    """
    print(f"--- YFINANCE API CALL: Fetching 90d data for {ticker}... (This should only appear ONCE per 10 mins per stock)")
    stock = yf.Ticker(ticker)
    hist = stock.history(period="90d", auto_adjust=True)
    info = stock.info
    return hist, info

# --- 重构工具，让它们接收数据，而不是自己去获取 ---
@tool
def get_financial_metrics(ticker: str, hist_90d: pd.DataFrame, info: dict) -> dict:
    """分析预先获取的财务数据和历史价格，并返回关键指标。"""
    try:
        if hist_90d.empty: return {"error": "历史数据为空。"}
        delta = hist_90d['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = rsi.iloc[-1] if not rsi.empty and pd.notna(rsi.iloc[-1]) else 'N/A'
        return {
            "当前价格 (Current Price)": f"${info.get('currentPrice', info.get('previousClose')):.2f}",
            "14日RSI (14-Day RSI)": f"{latest_rsi:.2f}" if isinstance(latest_rsi, float) else latest_rsi,
            "市值 (Market Cap)": f"{info.get('marketCap', 0) / 1_000_000_000:.2f}B",
            "市盈率 (PE Ratio)": info.get("trailingPE"),
        }
    except Exception as e: return {"error": f"分析财务指标时发生错误: {e}"}

@tool
def generate_stock_chart(ticker: str, hist_90d: pd.DataFrame) -> str:
    """根据预先获取的90天数据生成蜡烛图并保存。"""
    try:
        if hist_90d.empty: return f"无法为 {ticker} 生成图表，因为数据为空。"
        mc = mpf.make_marketcolors(up='cyan', down='magenta', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style='nightclouds')
        chart_path = f"{ticker}_chart.png"
        mpf.plot(
            hist_90d, type='candle', style=s, title=f"\n{ticker} 90-Day Price Action",
            ylabel='Price (USD)', volume=True, savefig=dict(fname=chart_path, dpi=300, bbox_inches='tight')
        )
        return f"一张专业的技术分析图表已在右侧生成。路径: {chart_path}。"
    except Exception as e: return f"生成图表时发生错误: {e}"

# --- 统一的UI界面 ---
st.set_page_config(layout="wide", page_title="AI股票圣手 Pro", page_icon="🏆")
st.title("AI股票圣手 Pro版 🏆")
st.header("数据分析 (Analysis) & 视觉洞察 (Color)")
st.caption(f"由父女团队倾力打造，一个融合20年市场经验与新生代AI创意的作品。")

with st.sidebar:
    st.header("🛠️ 配置与工具")
    with st.expander("🔑 配置 API 密钥", expanded=True):
        nvidia_api_key =os.getenv("NVIDIA_API_KEY")
        tavily_api_key =os.getenv("TAVILY_API_KEY")
    keys_provided = nvidia_api_key and tavily_api_key
    if not keys_provided: st.warning("请输入所有必需的 API 密钥以启动智能代理。")
    st.divider()
    st.header("👀 图表视觉洞察引擎")
    uploaded_chart = st.file_uploader("您也可以上传K线图进行独立的视觉情绪分析", type=["jpg", "png"])
    if uploaded_chart:
        st.image(uploaded_chart, caption="已上传图表")
        if st.button("分析这张图表", use_container_width=True):
            with st.spinner("视觉模块 (Color) 正在分析..."):
                v1_result = analyze_image_sentiment(uploaded_chart)
                v2_result = analyze_image_sentiment_v2(uploaded_chart)
            if v1_result and v2_result and v1_result[0] is not None and v2_result[0] is not None:
                st.success("视觉分析完成！")
                v1_score, _, _ = v1_result
                v2_score, _, _ = v2_result
                st.metric("V1 整体情绪", f"{v1_score:.2%}")
                st.metric("V2 近期趋势", f"{v2_score:.2%}", f"{v2_score - v1_score:.2%}")
            else: st.error("图片处理失败。")

# --- 主聊天界面 ---
st.header("🧠 AI数据分析引擎")

# 初始化聊天记录，确保总有欢迎信息
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！我是您的AI股票圣手。请输入股票代码（如NVDA, TSLA），让我为您提供一份盘前分析报告。"}]

# 创建一个容器来展示聊天记录，确保页面不空
chat_container = st.container(height=500) # 您可以调整height参数
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            col1, col2 = st.columns([2, 1])
            with col1: st.markdown(message["content"])
            if "chart_path" in message and message["chart_path"] and os.path.exists(message["chart_path"]):
                with col2: st.image(message["chart_path"], caption="Technical Analysis Chart")

# 将聊天输入框放在主逻辑中
if keys_provided:
    if ticker_prompt := st.chat_input("请输入您想分析的股票代码 (例如: NVDA)"):
        st.session_state.messages.append({"role": "user", "content": ticker_prompt})
        # Rerun to display the new user message immediately
        st.rerun()
else:
    st.info("💡 请在左侧边栏输入您的 API 密钥以开始分析。")

# --- 主分析逻辑 (当有新消息时) ---
# 检查最后一条消息是否是用户的，并且我们还没有为它生成回答
if "messages" in st.session_state and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    ticker_prompt = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        col1, col2 = st.columns([2, 1]) 
        with col1:
            with st.spinner("AI 正在进行深度分析..."):
                try:
                    llm = ChatOpenAI(model="meta/llama-3.1-70b-instruct", openai_api_base="https://integrate.api.nvidia.com/v1", openai_api_key=nvidia_api_key, temperature=0.0)
                    tavily_search = TavilySearch(max_results=5, tavily_api_key=tavily_api_key)
                    
                    # --- 终极数据流 ---
                    # 1. 只调用一次网络获取数据
                    hist_data, info_data = get_stock_data_from_yf(ticker_prompt)
                    
                    # 2. 将获取到的数据传递给工具
                    news_summary = tavily_search.invoke(f"为 {ticker_prompt} 搜索最新的新闻和市场情绪")
                    financial_data = get_financial_metrics.invoke({"ticker": ticker_prompt, "hist_90d": hist_data, "info": info_data})
                    chart_result = generate_stock_chart.invoke({"ticker": ticker_prompt, "hist_90d": hist_data})

                    # 3. 构造最终提示词
                    final_prompt = f"你是一位专业的股票分析师。请根据以下信息，为股票 {ticker_prompt} 提供一份全面、结构清晰、易于阅读的中文盘前分析报告。\n\n信息:\n1. **最新新闻与市场情绪:**\n{news_summary}\n\n2. **关键财务数据:**\n{financial_data}\n\n3. **技术图表分析:**\n{chart_result}\n\n请综合以上所有信息，形成一份最终的、结论性的中文报告。"
                    response = llm.invoke(final_prompt)
                    response_content = response.content
                    st.markdown(response_content)
                except Exception as e:
                    response_content = f"在分析过程中发生严重错误: {e}"
                    st.error(response_content)

        with col2:
            chart_path = f"{ticker_prompt}_chart.png"
            if os.path.exists(chart_path): st.image(chart_path, caption="Technical Analysis Chart")
            else: st.warning("图表文件未找到或生成失败。")
        
        # 只有当这是新生成的消息时，才添加到历史记录中
        if len(st.session_state.messages) > 0 and st.session_state.messages[-1].get("content") != response_content:
            assistant_message = {"role": "assistant", "content": response_content, "chart_path": chart_path}
            st.session_state.messages.append(assistant_message)
            st.rerun()

cd path/to/wii/project