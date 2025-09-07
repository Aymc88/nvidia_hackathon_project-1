import streamlit as st
from analyzer import analyze_image_sentiment, analyze_image_sentiment_v2
from PIL import Image

# 设置页面为宽屏模式，让对比更清晰
st.set_page_config(page_title="图表情绪分析器", page_icon="📈", layout="wide")

# --- UI界面 ---
st.title("图表颜色情绪分析器 📈")
st.header("一个融合20年市场洞察与新生代视觉AI的家庭项目")
st.caption("由一位父亲和他的15岁女儿联手打造")
st.divider()

st.subheader("第一步：请上传您的股票图表 🖼️")
uploaded_file = st.file_uploader(
    "选择一张K线图截图 (JPG, PNG格式)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed" # 让UI更简洁
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="图片上传成功！", width=600)
    
    st.subheader("第二步：启动AI进行分析 🤖")
    if st.button("开始分析", type="primary", use_container_width=True):
        with st.spinner('AI正在全力看图中，请稍候...'):
            # 同时调用两个版本的分析函数
            v1_result = analyze_image_sentiment(uploaded_file)
            v2_result = analyze_image_sentiment_v2(uploaded_file)
        
        if v1_result and v2_result:
            st.success('分析完成！')
            st.subheader("分析结果对比")

            # 解包结果
            v1_score, green_pixels, red_pixels = v1_result
            v2_score, weighted_green, weighted_red = v2_result

            # 使用分栏布局来并排展示结果
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### V1 基础分析 (整体情绪)")
                st.metric(
                    label="整体情绪指数",
                    value=f"{v1_score:.2%}",
                    help="衡量整张图表中红绿K线的面积占比。"
                )
                with st.expander("查看V1详细数据"):
                    st.write(f"绿色像素数: {int(green_pixels)}")
                    st.write(f"红色像素数: {int(red_pixels)}")

            with col2:
                st.markdown("#### V2 智能加权分析 (近期趋势) ✨")
                st.metric(
                    label="近期趋势情绪指数",
                    value=f"{v2_score:.2%}",
                    delta=f"{v2_score - v1_score:.2%}",
                    delta_color="off", # 让delta值颜色保持中性
                    help="对图表右侧（近期）的颜色变化更敏感，更能反映短期趋势。Delta值显示了它与整体指数的差异。"
                )
                with st.expander("查看V2详细数据"):
                    st.write(f"绿色加权分数: {weighted_green:.2f}")
                    st.write(f"红色加权分数: {weighted_red:.2f}")
            
            st.info("💡 **结论**：智能加权分析更能反映市场的近期动向。当V2指数显著高于或低于V1指数时，往往预示着短期趋势的加强或反转。")
        else:
            st.error("图片处理失败，请尝试另一张图片。")