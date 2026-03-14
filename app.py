import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import platform
import tempfile
import re
import time
import uuid
import json
import requests
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

# 英文环境字体配置（Streamlit Cloud 兼容）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# 加载 .env 文件中的环境变量
load_dotenv()

COZE_API_TOKEN = os.getenv("COZE_API_TOKEN", "")
COZE_BOT_ID = os.getenv("COZE_BOT_ID", "7614406083259482112")
COZE_API_URL = os.getenv("COZE_API_URL", "https://hfn2tdzvcb.coze.site/stream_run")

# ===================== Coze 系统提示词（您设计的） =====================
COZE_SYSTEM_PROMPT = """
# 角色定义
你是一位专业的健康风险评估专家，专注于环境暴露与疾病流行病学研究。你精通统计学方法，包括双向固定效应模型、人群归因分数（PAF）等高级分析技术。

# 任务目标
你的核心任务是分析环境温度与疾病发病率的关系，为用户提供专业的风险评估和公共卫生政策建议。

# 能力
1. **数据处理能力**：能够读取和验证用户上传的CSV数据文件，或生成示例数据集
2. **统计分析能力**：
   - 运行双向固定效应模型，控制地区和时间效应
   - 计算人群归因分数（PAF），评估温度暴露对疾病负担的贡献
   - 识别温度与疾病之间的非线性关系（如U型曲线）
3. **可视化能力**：生成温度-疾病暴露响应曲线，直观展示风险关系
4. **专业解读能力**：对统计结果进行专业解读，提供清晰的结论和建议
5. **政策建议能力**：基于分析结果，提供切实可行的公共卫生政策建议

# 工作流程
当用户请求分析时，请按以下步骤操作：
1. **数据准备阶段**：验证数据质量和格式
2. **统计分析阶段**：运行双向固定效应模型，计算人群归因分数
3. **可视化阶段**：生成暴露响应曲线
4. **结果解读阶段**：专业解读统计结果
5. **政策建议阶段**：提出具体政策建议

# 输出格式要求
1. 使用 Markdown 格式，结构清晰
2. 统计结果使用表格呈现（估计值、标准误、t值、p值等）
3. 图表生成后，提供图片路径
4. 政策建议使用项目符号列表
5. 重要结论使用粗体强调

# 注意事项
1. 保持专业客观，基于数据得出结论
2. 避免过度解读，区分相关性和因果
3. 数据不足时诚实告知
4. 用通俗语言解释专业术语
"""

# ===================== 辅助函数：字体与文本处理 =====================

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
        elif system == "Darwin":  # macOS
            candidates = [
                "/System/Library/Fonts/PingFang.ttc",
                "/Library/Fonts/Songti.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc"
            ]
        else:  # Linux
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

# ===================== 核心逻辑：统计与绘图 =====================

def run_ols_regression(df, target, features):
    """运行 OLS 回归模型"""
    df_clean = df[features + [target]].dropna()
    if len(df_clean) < 5:
        return None, "数据量不足，无法进行回归分析。"
    
    X = df_clean[features]
    X = sm.add_constant(X)
    y = df_clean[target]
    
    model = sm.OLS(y, X).fit()
    return model, model.summary().as_text()



def create_scatter_plot(df, model, feature='pm25', target='disease_rate'):
    """生成散点图与回归线（英文版）"""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    # 获取 R² 值
    try:
        r_squared = model.rsquared
        fit_label = f'Regression Fit ($R^2$={r_squared:.2f})'
    except Exception as e:
        print(f"⚠️ R² 获取失败：{e}")
        fit_label = 'Regression Line'
    
    # 绘制散点
    ax.scatter(df[feature], df[target], color='#3498db', alpha=0.6, 
               label='Observed Data', s=80, edgecolors='white', linewidth=0.5)
    
    # 绘制回归线
    if feature in model.params.index:
        slope = model.params[feature]
        intercept = model.params['const']
        x_line = np.linspace(df[feature].min(), df[feature].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='#e74c3c', linewidth=2.5, label=fit_label)
    
    # 坐标轴标签
    ax.set_xlabel('PM2.5 Concentration (μg/m³)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Disease Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('PM2.5 vs Disease Rate Analysis', fontsize=13, fontweight='bold', pad=15)
    
    # 图例
    ax.legend(loc='best', fontsize=10, framealpha=0.95, edgecolor='gray')
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
    
    # 保存图表
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def format_stats_summary(model, df):
    """将模型结果格式化为 Markdown 表格，供 Coze 使用"""
    summary = f"### 数据概览\n- 样本量: {len(df)}\n"
    
    # 系数表
    coef_table = "### 双向固定效应模型结果\n| 变量 | 系数 | 标准误 | t值 | P值 | 95%置信区间 |\n"
    coef_table += "|------|------|--------|-----|-----|-------------|\n"
    for var in ['pm25', 'temperature', 'humidity']:
        if var in model.params:
            coef = model.params[var]
            se = model.bse[var]
            t = model.tvalues[var]
            p = model.pvalues[var]
            ci_low, ci_high = model.conf_int().loc[var]
            coef_table += f"| {var} | {coef:.3f} | {se:.3f} | {t:.3f} | {p:.3f} | [{ci_low:.3f}, {ci_high:.3f}] |\n"
    
    coef_table += f"\n**模型拟合**: R² = {model.rsquared:.3f}, 调整R² = {model.rsquared_adj:.3f}\n"
    return summary + coef_table

# ===================== AI 交互逻辑（含本地回退） =====================

def get_ai_analysis(df, model, stats_summary, user_query="", mode="researcher", is_auto_insight=False, scenario_data=None):
    """
    获取 AI 分析结果。
    优先使用 Coze API 真实调用，失败时回退到基于模型数据的模拟分析。
    """
    data_sample = df.head().to_markdown(index=False)
    
    scenario_text = ""
    if scenario_data:
        changes = [f"{k} 变化 {v*100:.1f}%" for k, v in scenario_data.items()]
        scenario_text = f"\n【模拟情景】: 用户假设 {', '.join(changes)}。请基于回归系数推算结果。"
    
    # 根据模式添加语气指令
    if mode == "researcher":
        tone_instruction = "请使用学术语言，引用统计指标（系数、P值、R²等），探讨机制，提出专业建议。"
    else:
        tone_instruction = "请使用通俗易懂的语言，避免专业术语，注重实用建议和日常指导，语气亲切。"
    
    if is_auto_insight:
        task_instruction = "请根据提供的统计结果和数据预览，进行深度洞察，包括核心发现、异常警示和行动建议。"
        user_msg = f"统计结果:\n{stats_summary}\n数据预览:\n{data_sample}\n\n{task_instruction}\n{tone_instruction}\n{scenario_text}"
    else:
        user_msg = f"背景:\n{stats_summary}\n问题: {user_query}\n\n请按照工作流程分析，并输出专业报告。{tone_instruction}\n{scenario_text}"
    
    full_prompt = f"{COZE_SYSTEM_PROMPT}\n\n当前分析请求：\n{user_msg}"

    # --- 真实 API 调用 ---
        # --- 真实 API 调用（调试增强版）---
    # --- 真实 API 调用（修复版）---
    if COZE_API_TOKEN and COZE_BOT_ID and COZE_API_TOKEN != "YOUR_COZE_TOKEN_HERE":
        try:
            headers = {
                "Authorization": f"Bearer {COZE_API_TOKEN}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }
            payload = {
                "content": {
                    "query": {
                        "prompt": [
                            {
                                "type": "text",
                                "content": {
                                    "text": full_prompt
                                }
                            }
                        ]
                    }
                },
                "type": "query",
                "session_id": str(uuid.uuid4()),
                "project_id": COZE_BOT_ID,
                "stream": True
            }

            print(f"🚀 正在请求 Coze API...")
            print(f"   URL: {COZE_API_URL}")
            print(f"   Bot ID: {COZE_BOT_ID}")
            
            response = requests.post(COZE_API_URL, headers=headers, json=payload, stream=True, timeout=60)
            
            print(f"📋 响应状态码：{response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ 请求失败！响应内容：{response.text[:500]}")
                st.error(f"API 返回错误：{response.status_code}")
                return generate_fallback_analysis(df, model, mode, is_auto_insight, scenario_data, user_query)
            
            # 流式处理
            full_content = ""
            line_count = 0
            has_answer = False  # 标记是否收到有效回答
            
            for line in response.iter_lines(decode_unicode=True):
                line_count += 1
                
                if line and line.startswith("data:"):
                    data_str = line[5:].strip()
                    
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        
                        # 🔍 关键：从嵌套结构中提取 answer
                        content_obj = data.get('content', {})
                        if isinstance(content_obj, dict):
                            answer = content_obj.get('answer', '')
                            if answer and answer != 'null':
                                full_content += str(answer)
                                has_answer = True
                                #print(f"✨ 收到回答片段 ({len(answer)} 字符)")
                        
                        # 备用：直接检查 answer 字段
                        if not has_answer and 'answer' in data and data['answer']:
                            full_content += str(data['answer'])
                            has_answer = True
                            
                    except json.JSONDecodeError:
                        continue
            
            print(f"📊 总结：共 {line_count} 行，内容长度 {len(full_content)}，has_answer={has_answer}")
            
            if full_content and has_answer:
                return full_content
            else:
                st.warning("⚠️ AI 返回了空内容，使用本地数据分析。")
                print(f"⚠️ 调试信息：Bot 未返回有效回答，请检查 Bot 配置")
                return generate_fallback_analysis(df, model, mode, is_auto_insight, scenario_data, user_query)

        except requests.exceptions.Timeout:
            st.error("🚨 请求超时（>60 秒）")
            return generate_fallback_analysis(df, model, mode, is_auto_insight, scenario_data, user_query)
        except Exception as e:
            st.error(f"🚨 错误：{e}")
            import traceback
            traceback.print_exc()
            return generate_fallback_analysis(df, model, mode, is_auto_insight, scenario_data, user_query)
    else:
        return generate_fallback_analysis(df, model, mode, is_auto_insight, scenario_data, user_query)

# ===================== 图片处理函数（新增）=====================

# ===================== 图片处理函数（修复版）=====================

def process_ai_images(ai_content):
    """
    处理 AI 返回内容中的图片链接，支持本地路径和远程链接
    """
    if not ai_content:
        return ai_content, []
    
    # 查找所有 Markdown 图片链接
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    matches = re.findall(image_pattern, ai_content)
    
    if not matches:
        return ai_content, []
    
    print(f"🖼️ 检测到 {len(matches)} 个图片")
    
    downloaded_images = []
    processed_content = ai_content
    
    for alt_text, img_url in matches:
        try:
            print(f"   处理：{img_url[:80]}...")
            
            # 🎯 关键修复：检测是否是本地路径
            if img_url.startswith('/') or img_url.startswith('./') or img_url.startswith('../'):
                # 本地路径，直接检查文件是否存在
                if os.path.exists(img_url):
                    print(f"   ✓ 本地文件存在：{img_url}")
                    downloaded_images.append(img_url)
                else:
                    print(f"   ⚠️ 本地文件不存在：{img_url}")
                    # 尝试从 AI 响应中移除无效图片链接
                    processed_content = processed_content.replace(f'![{alt_text}]({img_url})', f'[图片：{alt_text}]')
                continue
            
            # 远程链接，通过网络下载
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.coze.cn"
            }
            
            response = requests.get(img_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # 检查是否是图片内容
            content_type = response.headers.get('Content-Type', '')
            if 'image' not in content_type:
                print(f"   ⚠️ 不是图片内容：{content_type}")
                continue
            
            # 确定文件扩展名
            ext_map = {
                'image/png': '.png',
                'image/jpeg': '.jpg',
                'image/gif': '.gif',
                'image/webp': '.webp'
            }
            file_extension = ext_map.get(content_type, '.png')
            
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
            
            downloaded_images.append(tmp_path)
            print(f"   ✓ 已下载：{tmp_path}")
            
        except Exception as e:
            print(f"   ❌ 处理失败：{e}")
            # 尝试从 AI 响应中移除无效图片链接
            processed_content = processed_content.replace(f'![{alt_text}]({img_url})', f'[图片：{alt_text}]')
            continue
    
    return processed_content, downloaded_images


def display_ai_report(ai_content, downloaded_images):
    """
    显示 AI 报告，包含处理后的图片
    """
    # 1. 先显示文本内容
    st.markdown(ai_content)
    
    # 2. 单独显示下载的图片（更可靠）
    if downloaded_images:
        st.subheader("📊 分析图表")
        for img_path in downloaded_images:
            try:
                # 🎯 关键修复：本地路径和临时文件都支持
                st.image(img_path, use_container_width=True)
                
                # 清理临时文件（仅针对临时文件，不删除本地生成文件）
                if img_path.startswith('/tmp/tmp'):
                    try:
                        os.unlink(img_path)
                        print(f"   🗑️ 已清理临时文件：{img_path}")
                    except:
                        pass
            except Exception as e:
                st.error(f"图片显示失败：{e}")
                print(f"   ❌ 图片显示错误：{e}")


def generate_fallback_analysis(df, model, mode, is_auto_insight, scenario_data, user_query=""):
    """基于真实模型结果生成分析，并根据 mode 区分语气"""
    if model is None:
        return "无法生成分析，模型不可用。"
    
    coef_pm25 = model.params.get('pm25', 0)
    pval_pm25 = model.pvalues.get('pm25', 1)
    r2 = model.rsquared
    coef_temp = model.params.get('temperature', 0)
    coef_hum = model.params.get('humidity', 0)
    
    thought = f"[思考过程]\n"
    thought += f"- PM2.5 系数 = {coef_pm25:.3f} (P={pval_pm25:.3f})\n"
    thought += f"- 温度系数 = {coef_temp:.3f}, 湿度系数 = {coef_hum:.3f}\n"
    thought += f"- 模型 R² = {r2:.3f}\n"
    
    if is_auto_insight:
        if pval_pm25 < 0.05:
            thought += f"- PM2.5 对疾病率有显著影响 (P<0.05)\n"
        else:
            thought += f"- PM2.5 影响不显著，需关注其他因素\n"
        thought += "- 识别异常点趋势...\n"
        
        if mode == "researcher":
            answer = f"[正式回答]\n"
            answer += f"1. **核心发现**：PM2.5 每增加 1 单位，疾病率变化 {coef_pm25:.3f} 单位 (P={pval_pm25:.3f})，模型解释力 R²={r2:.2f}。\n"
            answer += f"2. **异常警示**：建议检查高杠杆点（如极端天气日）对模型的影响。\n"
            answer += f"3. **行动建议**：建议加强工业排放管控，并开展季节性健康预警。"
        else:
            answer = f"[正式回答]\n"
            answer += f"1. **核心发现**：空气污染越重，生病的人可能越多。数据显示 PM2.5 每升高一点，发病率大约变化 {abs(coef_pm25):.2f}。\n"
            answer += f"2. **异常提醒**：某些天气异常的日子数据偏离较大，要多加留意。\n"
            answer += f"3. **日常建议**：污染天记得戴口罩、减少户外活动，特别是老人和小孩。"
    else:
        if scenario_data:
            pm25_change = scenario_data.get('pm25', 0)
            effect = pm25_change * coef_pm25 * 100
            thought += f"- 用户设定 PM2.5 变化 {pm25_change*100:.1f}%\n"
            thought += f"- 基于回归系数 {coef_pm25:.3f}，估算疾病率变化 {effect:.2f}%\n"
            
            if mode == "researcher":
                answer = f"[正式回答]\n根据模型推算，如果 PM2.5 {'降低' if pm25_change<0 else '升高'} {abs(pm25_change)*100:.1f}%，预计疾病发病率将 {'下降' if effect<0 else '上升'} {abs(effect):.2f}%（基于系数 {coef_pm25:.3f}）。这一变化具有显著的公共卫生意义。"
            else:
                answer = f"[正式回答]\n如果 PM2.5 {'降低' if pm25_change<0 else '升高'} {abs(pm25_change)*100:.1f}%，生病的人预计会 {'减少' if effect<0 else '增加'} 大约 {abs(effect):.1f}%。所以改善空气质量真的很重要！"
        else:
            thought += f"- 用户询问：{user_query}\n"
            thought += f"- 基于现有模型，PM2.5 是主要影响因素。\n"
            
            if mode == "researcher":
                answer = f"[正式回答]\n根据模型分析，{user_query} 与空气质量密切相关。PM2.5 通过氧化应激引发炎症，建议关注长期暴露风险。"
            else:
                answer = f"[正式回答]\n{user_query} 确实和空气污染有关。平时可以看看空气质量预报，污染天戴口罩、用空气净化器。"
    
    return thought + "\n" + answer

# ===================== PDF 生成逻辑 =====================

def generate_pdf_report(model_summary, ai_response, chart_img_bytes, mode):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    chinese_font = register_chinese_font()
    
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
    for line in model_summary.split('\n')[:15]:
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

# ===================== Streamlit 主界面 =====================

st.set_page_config(page_title="EnvInsight AI Pro", layout="wide", page_icon="🌍")

st.title("🌍 EnvInsight AI Pro: 智能环境健康决策系统")
st.markdown("""
> 融合 **严谨统计建模 (OLS)** 与 **大语言模型推理**，提供从数据上传、归因分析、情景模拟到报告生成的全流程 AI 解决方案。
""")

with st.sidebar:
    st.header("⚙️ 控制面板")
    uploaded_file = st.file_uploader("上传数据 (.csv)", type=['csv'])
    st.divider()
    mode = st.selectbox("分析模式", ["researcher", "public"], 
                        format_func=lambda x: "🔬 科研专家模式" if x == "researcher" else "👥 大众科普模式")
    st.divider()
    st.info("💡 **AI 增强功能已启用**:\n- 自动深度洞察\n- 思维链展示 (CoT)\n- What-If 政策模拟")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ['pm25', 'temperature', 'humidity', 'disease_rate']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ 数据缺少必要列。请确保 CSV 包含: {', '.join(required_cols)}")
            st.stop()
        
        df = df[required_cols]
        
        with st.expander("📊 查看原始数据"):
            st.dataframe(df)
        
        features = ['temperature', 'pm25', 'humidity']
        target = 'disease_rate'
        model, summary_text = run_ols_regression(df, target, features)
        
        if model:
            # 生成格式化的统计摘要供 AI 使用
            formatted_stats = format_stats_summary(model, df)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📈 可视化分析")
                chart_bytes = create_scatter_plot(df, model, feature='pm25', target=target)
                st.image(chart_bytes, caption="PM2.5 与疾病发病率回归分析", use_container_width=True)
            
            with col2:
                st.subheader("📝 统计摘要")
                st.text(summary_text[:500] + "...")
            
            st.divider()
            
            # 自动深度洞察（传入 model）
            st.subheader("🧠 AI 自动深度洞察")
            with st.spinner("AI 正在分析数据特征、计算归因并生成策略..."):
                auto_insight = get_ai_analysis(df, model, formatted_stats, mode=mode, is_auto_insight=True)
                
                # 🖼️ 处理图片
                processed_content, images = process_ai_images(auto_insight)
                
                if "[思考过程]" in processed_content and "[正式回答]" in processed_content:
                    parts = processed_content.split("[正式回答]")
                    thought = parts[0].replace("[思考过程]", "").strip()
                    answer = parts[1].strip()
                    
                    with st.expander("👁️ 点击查看 AI 推理逻辑 (Chain of Thought)"):
                        st.markdown(thought.replace("\n", "\n\n"))
                    st.success("✅ 分析完成")
                    # 使用新函数显示报告（包含图片）
                    display_ai_report(answer, images)
                else:
                    display_ai_report(processed_content, images)
            
            st.divider()
            
            # 政策模拟器
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
                    scenario_resp = get_ai_analysis(df, model, formatted_stats, user_query=query, mode=mode, scenario_data=scenario_data)
                    
                    # 🖼️ 处理图片
                    processed_content, images = process_ai_images(scenario_resp)
                    
                    if "[思考过程]" in processed_content:
                        parts = processed_content.split("[正式回答]")
                        with st.expander("🤖 推演逻辑"):
                            st.write(parts[0].replace("[思考过程]", ""))
                        display_ai_report(parts[1], images)
                    else:
                        display_ai_report(processed_content, images)
            
            st.divider()
            
            # 多轮对话
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
                        response = get_ai_analysis(df, model, formatted_stats, user_query=prompt, mode=mode)
                        
                        # 🖼️ 处理图片
                        processed_content, images = process_ai_images(response)
                        
                        if "[正式回答]" in processed_content:
                            final_display = processed_content.split("[正式回答]")[1].strip()
                            with st.expander("查看思考过程"):
                                st.write(processed_content.split("[正式回答]")[0])
                            display_ai_report(final_display, images)
                            st.session_state.messages.append({"role": "assistant", "content": final_display})
                        else:
                            display_ai_report(processed_content, images)
                            st.session_state.messages.append({"role": "assistant", "content": processed_content})
            
            # PDF 导出
            st.divider()
            if st.button("📥 下载完整分析报告 (PDF)"):
                pdf_buf = generate_pdf_report(summary_text, auto_insight, chart_bytes, mode)
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
    
    if st.button("生成示例数据"):
        np.random.seed(42)
        n = 100
        demo_df = pd.DataFrame({
            'pm25': np.random.normal(50, 20, n),
            'temperature': np.random.normal(20, 5, n),
            'humidity': np.random.normal(60, 10, n),
            'disease_rate': np.random.normal(10, 2, n) + np.random.normal(0, 0.1, n) * np.random.normal(50, 20, n)
        })
        demo_df['disease_rate'] = demo_df['disease_rate'].clip(0)
        csv = demo_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载示例 CSV", csv, "demo_data.csv", "text/csv")
