import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import platform
import re
import time
from datetime import datetime
from dotenv import load_dotenv

# 绘图与统计库
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import statsmodels.api as sm
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
# 加载 .env 文件中的环境变量
load_dotenv()


COZE_API_TOKEN = os.getenv("COZE_API_TOKEN", "") 
COZE_BOT_ID = os.getenv("COZE_BOT_ID", "7614406083259482112") 
COZE_API_URL = os.getenv("COZE_API_URL", "https://hfn2tdzvcb.coze.site/stream_run")


# ===================================================================

# ------------------ 1. 辅助函数：字体与文本处理 ------------------

def register_chinese_font():
    """自动注册系统中可用的中文字体"""
    system = platform.system()
    font_path = ""
    font_name = "ChineseFont"
    
    try:
        if system == "Windows":
            candidates = [
                r"C:\Windows\Fonts\simhei.ttf",
                r"C:\Windows\Fonts\msyh.ttc",
                r"C:\Windows\Fonts\simsun.ttc"
            ]
        elif system == "Darwin": # macOS
            candidates = [
                "/System/Library/Fonts/PingFang.ttc",
                "/Library/Fonts/Songti.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc"
            ]
        else: # Linux
            candidates = ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]
        
        for path in candidates:
            if os.path.exists(path):
                font_path = path
                break
        
        if font_path:
            pdfmetrics.registerFont(TTFont(font_name, font_path))
            return font_name
        else:
            st.warning("⚠️ 未找到常用中文字体，PDF 中文可能显示为方块。建议安装 SimHei 或微软雅黑。")
            return "Helvetica"
    except Exception as e:
        st.error(f"字体注册失败: {e}")
        return "Helvetica"

def clean_markdown_text(text):
    """简单清理 Markdown 符号，以便在 PDF 中正常显示"""
    if not text:
        return ""
    text = re.sub(r'#{1,6}\s*', '', text)       # 移除标题
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) # 移除加粗
    text = re.sub(r'\*(.*?)\*', r'\1', text)     # 移除斜体
    text = re.sub(r'^-\s*', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)
    # 移除我们自定义的思维链标记，避免打印在 PDF 里
    text = re.sub(r'\[思考过程\].*?\[正式回答\]', '', text, flags=re.DOTALL)
    return text

# ------------------ 2. 核心逻辑：统计与绘图 ------------------

def run_ols_regression(df, target, features):
    """运行 OLS 回归模型"""
    # 确保只包含数值列且无空值
    df_clean = df[features + [target]].dropna()
    if len(df_clean) < 5:
        return None, "数据量不足，无法进行回归分析。"
    
    X = df_clean[features]
    X = sm.add_constant(X)
    y = df_clean[target]
    
    model = sm.OLS(y, X).fit()
    return model, model.summary().as_text()

def create_scatter_plot(df, model, feature='pm25', target='disease_rate'):
    """生成散点图与回归线"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 散点
    ax.scatter(df[feature], df[target], color='#3498db', alpha=0.6, label='观测数据')
    
    # 回归线
    # 获取模型中该特征的系数
    if feature in model.params.index:
        slope = model.params[feature]
        intercept = model.params['const']
        x_line = np.linspace(df[feature].min(), df[feature].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='#e74c3c', linewidth=2, label=f'回归拟合 (R²={model.rsquared:.2f})')
    
    ax.set_xlabel(feature.upper())
    ax.set_ylabel(target.upper())
    ax.set_title(f'{feature} vs {target} Analysis')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 保存为字节流
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ------------------ 3. AI 交互逻辑 (含模拟模式) ------------------

def get_ai_analysis(df, stats_summary, user_query="", mode="researcher", is_auto_insight=False, scenario_data=None):
    """
    获取 AI 分析结果。
    如果未配置 Token，则使用本地模拟逻辑演示效果。
    """
    
    # 准备数据上下文
    data_sample = df.head().to_markdown(index=False)
    
    # 构建场景描述
    scenario_text = ""
    if scenario_data:
        changes = [f"{k} 变化 {v*100:.1f}%" for k, v in scenario_data.items()]
        scenario_text = f"\n【模拟情景】: 用户假设 {', '.join(changes)}。请基于回归系数推算结果。"

    # 构建 Prompt
    if is_auto_insight:
        sys_prompt = f"""
        你是首席环境数据分析师。
        任务：阅读统计摘要，主动挖掘价值。
        输出格式严格如下：
        [思考过程]
        - 分析数据显著性 (P值, R²)
        - 识别异常点或趋势
        - 构思建议方向
        [正式回答]
        1. **核心发现**: ...
        2. **异常警示**: ...
        3. **行动建议**: ...
        {scenario_text}
        """
        user_msg = f"统计结果:\n{stats_summary}\n数据预览:\n{data_sample}"
    else:
        sys_prompt = f"""
        你是环境健康专家 (模式:{mode})。
        请严格按此格式回答：
        [思考过程]
        - 拆解用户意图
        - 结合统计证据 (P值/R²) 验证
        - 调用领域知识归因
        [正式回答]
        - 针对{mode}语气的详细解答
        {scenario_text}
        """
        user_msg = f"背景:\n{stats_summary}\n问题: {user_query}"

    # --- 真实 API 调用逻辑 (如果配置了 Token) ---
    # --- 真实 API 调用逻辑 (支持流式 SSE) ---
    if COZE_API_TOKEN != "YOUR_COZE_TOKEN_HERE" and COZE_BOT_ID != "YOUR_BOT_ID_HERE":
        try:
            import requests
            import json
            
            headers = {
                "Authorization": f"Bearer {COZE_API_TOKEN}", 
                "Content-Type": "application/json"
            }
            
            payload = {
                "bot_id": COZE_BOT_ID,
                "user": "streamlit_user",
                "query": f"{sys_prompt}\n\n{user_msg}",
                "stream": True  # ⚠️ 关键：必须开启流式模式以匹配 Coze 默认行为
            }
            
            print(f"🚀 正在请求 Coze API (Stream 模式)...")
            
            # 发起流式请求
            resp = requests.post(COZE_API_URL, json=payload, headers=headers, stream=True, timeout=30)
            resp.raise_for_status() # 检查 HTTP 状态码
            
            full_content = ""
            
            # 逐行处理 SSE 数据
            for line in resp.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    
                    # 跳过非 data 行 (如 event: message)
                    if decoded_line.startswith("data:"):
                        json_str = decoded_line[5:].strip() # 去掉 "data:" 前缀
                        
                        # 跳过 [DONE] 标记
                        if json_str == "[DONE]":
                            break
                            
                        try:
                            data = json.loads(json_str)
                            
                            # 提取 content 字段
                            # Coze 的流式结构通常在 data.content.answer 中
                            if 'content' in data and data['content']:
                                answer_part = data['content'].get('answer', '')
                                if answer_part:
                                    full_content += answer_part
                                    
                        except json.JSONDecodeError:
                            continue # 忽略无法解析的行
            
            if not full_content:
                return "AI 返回了空内容，请检查 Bot 配置或提示词。"
                
            return full_content

        except Exception as e:
            st.error(f"🚨 AI 服务连接失败:\n{str(e)}")
            st.info("💡 提示：已切换到模拟模式。")
            # 降级返回模拟数据
            time.sleep(1.5)
            return "[模拟模式] 由于 API 解析错误，暂时展示模拟回答...\n\n根据统计模型，PM2.5 每增加 10 单位，疾病发病率上升约 0.5%。建议加强空气质量监测。"

    # --- 模拟模式 (用于演示 UI 和逻辑，无需 Key) ---
    time.sleep(1.5) # 模拟延迟
    if is_auto_insight:
        return f"""[思考过程]
- 检测到 PM2.5 与 {df.columns[-1]} 的 P 值 < 0.01，呈现极强正相关。
- R² 达到 {np.random.uniform(0.7, 0.9):.2f}，说明模型解释力很强。
- 数据中存在个别离群点，可能是极端天气导致。
- 需要结合季节性因素给出建议。

[正式回答]
1. **核心发现**: 数据显示 PM2.5 浓度每上升 10 单位，疾病发病率平均上升约 2.5%。统计学上极其显著。
2. **异常警示**: 12 月份的数据点明显偏离回归线，建议核查当月是否有特殊污染事件。
3. **行动建议**: 
   - 对政府：建议在 PM2.5 预警阈值上增加动态调整机制。
   - 对公众：高污染日减少户外剧烈运动，特别是老人与儿童。
"""
    else:
        if scenario_data:
            # 模拟预测逻辑
            pm25_change = scenario_data.get('pm25', 0)
            effect = pm25_change * 2.5 # 假设系数
            return f"""[思考过程]
- 用户设定 PM2.5 变化 {pm25_change*100:.1f}%。
- 基于回归系数 (约 2.5)，估算疾病率变化约为 {effect:.2f}%。
- 这是一个显著的改善/恶化趋势。

[正式回答]
根据您的模拟设定，如果 PM2.5 降低 {abs(pm25_change)*100:.1f}%，预计疾病发病率将下降约 {abs(effect):.2f}%。
这意味着每年可能减少数百例相关病例，具有巨大的公共卫生价值。建议将此目标纳入年度环保考核。
"""
        else:
            return f"""[思考过程]
- 用户询问了关于 {user_query[:10]}... 的问题。
- 结合之前的强相关性结论，这主要归因于颗粒物吸入。
- 需要从预防角度回答。

[正式回答]
根据我们的模型分析，{user_query} 确实与空气质量密切相关。
主要机制是细颗粒物 (PM2.5) 可穿透肺泡进入血液循环，引发系统性炎症。
建议您关注每日空气质量指数 (AQI)，并在污染天采取防护措施。
"""

# ------------------ 4. PDF 生成逻辑 ------------------

def generate_pdf_report(model_summary, ai_response, chart_img_bytes, mode):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    chinese_font = register_chinese_font()
    
    # 标题
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "EnvInsight AI Analysis Report")
    
    c.setFont(chinese_font, 10)
    date_str = f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} | 模式：{mode}"
    c.drawString(50, height - 70, date_str)
    c.line(50, height - 80, width - 50, height - 80)
    
    # 1. 统计摘要
    c.setFont(chinese_font, 14)
    c.drawString(50, height - 110, "1. 统计建模结论")
    c.setFont(chinese_font, 9)
    text_object = c.beginText(50, height - 130)
    for line in model_summary.split('\n')[:15]: # 只取前15行避免溢出
        text_object.textLine(line)
    c.drawText(text_object)
    
    # 2. 图表
    current_y = height - 130 - (15 * 10) - 20
    c.setFont(chinese_font, 14)
    c.drawString(50, current_y, "2. 数据可视化")
    if chart_img_bytes:
        img = ImageReader(io.BytesIO(chart_img_bytes))
        c.drawImage(img, 50, current_y - 220, width=450, height=220, preserveAspectRatio=True)
        current_y -= 240
    else:
        current_y -= 240
        
    # 3. AI 解读
    c.setFont(chinese_font, 14)
    c.drawString(50, current_y, "3. AI 专家深度解读")
    
    # 清理 Markdown 和思维链标记
    clean_text = clean_markdown_text(ai_response)
    
    c.setFont(chinese_font, 10)
    lines = clean_text.split('\n')
    text_y = current_y - 20
    
    for line in lines:
        if text_y < 50:
            c.showPage()
            text_y = height - 50
            c.setFont(chinese_font, 10)
            c.drawString(50, height - 30, f"EnvInsight Report - Page {c.getPageNumber()}")
        
        # 简单换行处理
        if c.stringWidth(line, chinese_font, 10) > width - 100:
            max_chars = 35
            for i in range(0, len(line), max_chars):
                if text_y < 50:
                    c.showPage()
                    text_y = height - 50
                c.drawString(50, text_y, line[i:i+max_chars])
                text_y -= 12
        else:
            c.drawString(50, text_y, line)
            text_y -= 12
            
    c.save()
    buffer.seek(0)
    return buffer

# ------------------ 5. Streamlit 主界面 ------------------

st.set_page_config(page_title="EnvInsight AI Pro", layout="wide", page_icon="🌍")

st.title("🌍 EnvInsight AI Pro: 智能环境健康决策系统")
st.markdown("""
> 融合 **严谨统计建模 (OLS)** 与 **大语言模型推理**，提供从数据上传、归因分析、情景模拟到报告生成的全流程 AI 解决方案。
""")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 控制面板")
    uploaded_file = st.file_uploader("上传数据 (.csv)", type=['csv'])
    st.divider()
    mode = st.selectbox("分析模式", ["researcher", "public"], format_func=lambda x: "🔬 科研专家模式" if x == "researcher" else "👥 大众科普模式")
    
    st.divider()
    st.info("💡 **AI 增强功能已启用**:\n- 自动深度洞察\n- 思维链展示 (CoT)\n- What-If 政策模拟")

if uploaded_file:
    # 读取数据
    try:
        df = pd.read_csv(uploaded_file)
        # 简单的列名映射假设 (实际项目中可做更灵活的映射)
        # 假设用户上传的列包含 'pm25', 'temperature', 'humidity', 'disease_rate'
        # 如果列名不同，这里做个简单提示或自动匹配
        required_cols = ['pm25', 'temperature', 'humidity', 'disease_rate']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ 数据缺少必要列。请确保 CSV 包含: {', '.join(required_cols)}")
            st.stop()
            
        df = df[required_cols]
        
        # 展示原始数据
        with st.expander("📊 查看原始数据"):
            st.dataframe(df)
            
        # 1. 运行统计模型
        features = ['temperature', 'pm25', 'humidity']
        target = 'disease_rate'
        model, summary_text = run_ols_regression(df, target, features)
        
        if model:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📈 可视化分析")
                chart_bytes = create_scatter_plot(df, model, feature='pm25', target=target)
                st.image(chart_bytes, caption="PM2.5 与疾病发病率回归分析", use_container_width=True)
            
            with col2:
                st.subheader("📝 统计摘要")
                st.text(summary_text[:500] + "...") # 简略显示
            
            st.divider()
            
            # 2. ✨ AI 自动深度洞察 (页面加载即触发)
            st.subheader("🧠 AI 自动深度洞察")
            with st.spinner("AI 正在分析数据特征、计算归因并生成策略..."):
                auto_insight = get_ai_analysis(df, summary_text, mode=mode, is_auto_insight=True)
                
                # 解析思维链
                thought = ""
                answer = ""
                if "[思考过程]" in auto_insight and "[正式回答]" in auto_insight:
                    parts = auto_insight.split("[正式回答]")
                    thought = parts[0].replace("[思考过程]", "").strip()
                    answer = parts[1].strip()
                    
                    with st.expander("👁️ 点击查看 AI 推理逻辑 (Chain of Thought)"):
                        st.markdown(thought.replace("\n", "\n\n"))
                    st.success("✅ 分析完成")
                    st.markdown(answer)
                else:
                    st.markdown(auto_insight)
            
            st.divider()
            
            # 3. ✨ What-If 政策模拟器
            st.subheader("🔮 政策模拟器 (What-If Analysis)")
            st.caption("调整滑块，模拟环境指标变化对疾病率的潜在影响。")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                pm25_sim = st.slider("PM2.5 变化 (%)", -50, 50, 0, key="s1")
            with c2:
                temp_sim = st.slider("温度变化 (%)", -20, 20, 0, key="s2")
            with c3:
                hum_sim = st.slider("湿度变化 (%)", -30, 30, 0, key="s3")
            
            if pm25_sim != 0 or temp_sim != 0 or hum_sim != 0:
                with st.spinner("AI 正在推演未来情景..."):
                    scenario_data = {
                        'pm25': pm25_sim / 100.0,
                        'temperature': temp_sim / 100.0,
                        'humidity': hum_sim / 100.0
                    }
                    query = f"如果 PM2.5 变化 {pm25_sim}%, 温度 {temp_sim}%, 湿度 {hum_sim}%?"
                    scenario_resp = get_ai_analysis(df, summary_text, user_query=query, mode=mode, scenario_data=scenario_data)
                    
                    if "[思考过程]" in scenario_resp:
                        parts = scenario_resp.split("[正式回答]")
                        with st.expander("🤖 推演逻辑"):
                            st.write(parts[0].replace("[思考过程]", ""))
                        st.info(parts[1])
                    else:
                        st.info(scenario_resp)
            
            st.divider()
            
            # 4. 多轮对话
            st.subheader("💬 与 AI 分析师对话")
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            if prompt := st.chat_input("例如：为什么冬季数据会异常？"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("思考中..."):
                        response = get_ai_analysis(df, summary_text, user_query=prompt, mode=mode)
                        # 前端展示时也可以剥离思维链，只显示正式回答，或者保留
                        if "[正式回答]" in response:
                            final_display = response.split("[正式回答]")[1].strip()
                            # 可选：在 expander 里放思维链
                            with st.expander("查看思考过程"):
                                st.write(response.split("[正式回答]")[0])
                            st.markdown(final_display)
                            st.session_state.messages.append({"role": "assistant", "content": final_display})
                        else:
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # 5. PDF 导出
            st.divider()
            if st.button("📥 下载完整分析报告 (PDF)"):
                # 收集所有 AI 回答用于 PDF (这里简化为只放入最后的自动洞察或对话历史)
                # 实际项目中可以拼接所有对话
                pdf_content = auto_insight 
                pdf_buf = generate_pdf_report(summary_text, pdf_content, chart_bytes, mode)
                st.download_button(
                    label="点击下载 PDF",
                    data=pdf_buf,
                    file_name=f"EnvInsight_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
        else:
            st.error("模型拟合失败，请检查数据质量。")
            
    except Exception as e:
        st.error(f"发生错误: {e}")
else:
    st.info("👈 请在左侧上传 CSV 文件开始分析。示例数据需包含 pm25, temperature, humidity, disease_rate 列。")
    
    # 生成示例数据按钮
    if st.button("生成示例数据"):
        np.random.seed(42)
        n = 100
        demo_df = pd.DataFrame({
            'pm25': np.random.normal(50, 20, n),
            'temperature': np.random.normal(20, 5, n),
            'humidity': np.random.normal(60, 10, n),
            'disease_rate': np.random.normal(10, 2, n) + np.random.normal(0, 0.1, n) * np.random.normal(50, 20, n) # 构造一点相关性
        })
        demo_df['disease_rate'] = demo_df['disease_rate'].clip(0)
        csv = demo_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载示例 CSV", csv, "demo_data.csv", "text/csv")
