import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import re
from scipy.stats import poisson

# ==============================================================================
# 核心常量与配置
# ==============================================================================
BOOKMAKER_COLORS = {'Bet365': '#1f77b4', '利记': '#ff7f0e', '威廉希尔': '#2ca02c', '平博': '#d62728', '澳门': '#9467bd', '皇冠': '#8c564b'}
DEFAULT_COLOR = '#7f7f7f'
LEAD_BOOKMAKER_WEIGHTS = {'平博': 1.5, '利记': 1.2, 'Bet365': 1.1}
MAX_SCORE = 6

# ==============================================================================
# 阶段一: 基础重构与核心逻辑修正
# ==============================================================================

def _determine_ah_sides(row):
    """
    公理化实现：根据盘口值和水位，权威判定亚洲让球的上盘和下盘方。
    """
    try:
        h_raw = float(row['亚洲让球_盘口'])
        h_water = float(row['亚洲让球_主'])
        a_water = float(row['亚洲让球_客'])

        if h_raw < 0:
            return '主队', '客队'
        elif h_raw > 0:
            return '客队', '主队'
        else: # h_raw == 0 (平手盘)
            return ('主队', '客队') if h_water < a_water else ('客队', '主队')
    except (ValueError, TypeError):
        return None, None

@st.cache_data
def parse_and_build_unified_df(file_content: str):
    """
    重构的数据解析器，执行以下操作：
    1. 解析比赛基本信息，并健壮地处理跨年时间戳。
    2. 遍历所有机构和玩法，提取赔率数据。
    3. 构建一个统一、标准化的DataFrame，包含公平概率等核心分析字段。
    """
    lines = file_content.split('\n')
    match_info = {}
    for line in lines[:5]:
        if 'vs' in line and '#' in line:
            match_info['title'] = line.strip('# ').strip()
        match = re.search(r'开赛时间：\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})', line)
        if match:
            match_info['match_time'] = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M")
    
    if 'match_time' not in match_info:
        raise ValueError("无法解析到开赛时间")
    
    match_time = match_info['match_time']

    sections = re.split(r'##\s*(.*?)\s*-\s*全场赔率表', file_content)[1:]
    all_data_rows = []

    for i in range(0, len(sections), 2):
        bookmaker = sections[i].strip()
        table_str = sections[i+1]
        table_lines = [line.strip() for line in table_str.split('\n') if line.strip().startswith('|')]
        if len(table_lines) < 3: continue
        
        header = [h.strip() for h in table_lines[0].strip('|').split('|')]
        data_rows = [[c.strip() for c in r.strip('|').split('|')] for r in table_lines[2:]]
        
        df = pd.DataFrame(data_rows, columns=header).replace('', np.nan).dropna(how='all')

        # --- 时间戳解析 (健壮版) ---
        def parse_timestamp(s):
            try:
                dt = datetime.strptime(str(s).strip(), '%m-%d %H:%M')
                dt = dt.replace(year=match_time.year)
                # 如果解析出的时间比开赛时间还晚，说明是跨年数据，年份减一
                if dt > match_time:
                    dt = dt.replace(year=match_time.year - 1)
                return dt
            except:
                return pd.NaT

        # --- 1X2 胜平负 ---
        if all(c in df.columns for c in ['胜平负_胜赔率', '胜平负_平赔率', '胜平负_负赔率', '胜平负_变化时间']):
            df_1x2 = df[['胜平负_胜赔率', '胜平负_平赔率', '胜平负_负赔率', '胜平负_返还率', '胜平负_变化时间']].copy().dropna()
            df_1x2['timestamp'] = df_1x2['胜平负_变化时间'].apply(parse_timestamp)
            for col in ['胜平负_胜赔率', '胜平负_平赔率', '胜平负_负赔率']:
                df_1x2[col] = pd.to_numeric(df_1x2[col], errors='coerce')
            df_1x2['payout_rate'] = 1 / (1/df_1x2['胜平负_胜赔率'] + 1/df_1x2['胜平负_平赔率'] + 1/df_1x2['胜平负_负赔率'])
            
            for _, row in df_1x2.iterrows():
                if pd.isna(row['timestamp']): continue
                payout = row['payout_rate']
                all_data_rows.append({'timestamp': row['timestamp'], 'bookmaker': bookmaker, 'market': '1X2', 'handicap': 0, 'outcome': '主胜', 'price': row['胜平负_胜赔率'], 'prob_fair': 1/row['胜平负_胜赔率'] * payout, 'payout_rate': payout})
                all_data_rows.append({'timestamp': row['timestamp'], 'bookmaker': bookmaker, 'market': '1X2', 'handicap': 0, 'outcome': '平局', 'price': row['胜平负_平赔率'], 'prob_fair': 1/row['胜平负_平赔率'] * payout, 'payout_rate': payout})
                all_data_rows.append({'timestamp': row['timestamp'], 'bookmaker': bookmaker, 'market': '1X2', 'handicap': 0, 'outcome': '客胜', 'price': row['胜平负_负赔率'], 'prob_fair': 1/row['胜平负_负赔率'] * payout, 'payout_rate': payout})

        # --- 亚洲让球 ---
        if all(c in df.columns for c in ['亚洲让球_主', '亚洲让球_盘口', '亚洲让球_客', '亚洲让球_变化时间']):
            df_ah = df[['亚洲让球_主', '亚洲让球_盘口', '亚洲让球_客', '亚洲让球_变化时间']].copy().dropna()
            df_ah['timestamp'] = df_ah['亚洲让球_变化时间'].apply(parse_timestamp)
            for col in ['亚洲让球_主', '亚洲让球_盘口', '亚洲让球_客']:
                df_ah[col] = pd.to_numeric(df_ah[col], errors='coerce')
            df_ah['payout_rate'] = 1 / (1/(df_ah['亚洲让球_主']+1) + 1/(df_ah['亚洲让球_客']+1))
            
            for _, row in df_ah.iterrows():
                if pd.isna(row['timestamp']): continue
                upper_team, lower_team = _determine_ah_sides(row)
                if not upper_team: continue
                
                payout = row['payout_rate']
                prob_upper = (1 / (row['亚洲让球_主'] + 1)) if upper_team == '主队' else (1 / (row['亚洲让球_客'] + 1))
                prob_lower = 1 - prob_upper
                
                all_data_rows.append({'timestamp': row['timestamp'], 'bookmaker': bookmaker, 'market': 'AH', 'handicap': row['亚洲让球_盘口'], 'outcome': '上盘', 'price': row['亚洲让球_主'] if upper_team == '主队' else row['亚洲让球_客'], 'prob_fair': prob_upper * payout, 'payout_rate': payout})
                all_data_rows.append({'timestamp': row['timestamp'], 'bookmaker': bookmaker, 'market': 'AH', 'handicap': row['亚洲让球_盘口'], 'outcome': '下盘', 'price': row['亚洲让球_客'] if upper_team == '主队' else row['亚洲让球_主'], 'prob_fair': prob_lower * payout, 'payout_rate': payout})

        # --- 大小球 ---
        if all(c in df.columns for c in ['大小球_大于', '大小球_盘口', '大小球_小于', '大小球_变化时间']):
            df_ou = df[['大小球_大于', '大小球_盘口', '大小球_小于', '大小球_变化时间']].copy().dropna()
            df_ou['timestamp'] = df_ou['大小球_变化时间'].apply(parse_timestamp)
            for col in ['大小球_大于', '大小球_盘口', '大小球_小于']:
                df_ou[col] = pd.to_numeric(df_ou[col], errors='coerce')
            df_ou['payout_rate'] = 1 / (1/(df_ou['大小球_大于']+1) + 1/(df_ou['大小球_小于']+1))
            
            for _, row in df_ou.iterrows():
                if pd.isna(row['timestamp']): continue
                payout = row['payout_rate']
                prob_over = 1 / (row['大小球_大于'] + 1)
                prob_under = 1 / (row['大小球_小于'] + 1)
                
                all_data_rows.append({'timestamp': row['timestamp'], 'bookmaker': bookmaker, 'market': 'O/U', 'handicap': row['大小球_盘口'], 'outcome': '大球', 'price': row['大小球_大于'], 'prob_fair': prob_over * payout, 'payout_rate': payout})
                all_data_rows.append({'timestamp': row['timestamp'], 'bookmaker': bookmaker, 'market': 'O/U', 'handicap': row['大小球_盘口'], 'outcome': '小球', 'price': row['大小球_小于'], 'prob_fair': prob_under * payout, 'payout_rate': payout})

    if not all_data_rows:
        return match_info, pd.DataFrame()
        
    unified_df = pd.DataFrame(all_data_rows).sort_values('timestamp').reset_index(drop=True)
    return match_info, unified_df

# ==============================================================================
# 阶段二: 高级分析模型引入
# ==============================================================================

def dixon_coles_tau(lambda_h, lambda_a, rho, h, a):
    """ Dixon-Coles模型的tau调整因子，修正低比分概率 """
    if h == 0 and a == 0: return 1 - lambda_h * lambda_a * rho
    if h == 1 and a == 0: return 1 + lambda_a * rho
    if h == 0 and a == 1: return 1 + lambda_h * rho
    if h == 1 and a == 1: return 1 - rho
    return 1.0

def generate_score_matrix(p_h, p_d, p_a, ou_line, p_over, rho= -0.1):
    """
    升级版比分推演引擎：
    1. 基于1X2和O/U市场共识，估算期望进球数。
    2. 使用Dixon-Coles tau因子修正的二元泊松分布生成比分矩阵。
    """
    if p_h + p_d + p_a == 0: return np.zeros((MAX_SCORE, MAX_SCORE))
    
    # 启发式估算总期望进球数
    total_lambda = ou_line * (p_over / (1 - p_over))**0.5 if p_over < 1 else ou_line * 2

    # 启发式估算主客队期望进球数
    ratio = p_h / p_a if p_a > 0 else 100
    lambda_h = total_lambda * np.sqrt(ratio) / (1 + np.sqrt(ratio))
    lambda_a = total_lambda - lambda_h
    
    matrix = np.zeros((MAX_SCORE, MAX_SCORE))
    for h in range(MAX_SCORE):
        for a in range(MAX_SCORE):
            prob = poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
            tau = dixon_coles_tau(lambda_h, lambda_a, rho, h, a)
            matrix[h, a] = prob * tau
    
    return matrix / matrix.sum()

def get_market_prob_from_matrix(matrix, handicap):
    """ 
    从比分矩阵中计算指定玩法的"主队赢盘"概率。
    注意：此函数逻辑始终返回主队在盘口上"获胜"(Cover)的概率。
    """
    prob = 0.0
    for h in range(MAX_SCORE):
        for a in range(MAX_SCORE):
            d = h - a
            # 简化结算逻辑：判定主队是否赢盘
            # 主队赢盘条件：d > -handicap
            if d > -handicap: 
                prob += matrix[h, a]
    return prob

@st.cache_data
def run_advanced_analysis(_df, match_time):
    """
    执行所有高级分析的核心函数
    """
    # 1. 计算加权市场共识
    df_copy = _df.copy()
    df_copy['time_delta_mins'] = (match_time - df_copy['timestamp']).dt.total_seconds() / 60
    # 时间权重：指数衰减，半衰期约为3小时
    df_copy['time_weight'] = np.exp(-df_copy['time_delta_mins'] / (3 * 60)) 
    # 返还率权重
    df_copy['payout_weight'] = df_copy['payout_rate'] ** 2
    # 机构权重
    df_copy['bookmaker_weight'] = df_copy['bookmaker'].map(LEAD_BOOKMAKER_WEIGHTS).fillna(1.0)
    df_copy['total_weight'] = df_copy['time_weight'] * df_copy['payout_weight'] * df_copy['bookmaker_weight']
    
    consensus_list = []
    tension_list = []
    coherence_list = []
    
    timestamps = sorted(df_copy['timestamp'].unique())
    
    for ts in timestamps:
        df_ts = df_copy[df_copy['timestamp'] == ts]
        
        # --- 市场共识 ---
        consensus = {}
        for market in ['1X2', 'AH', 'O/U']:
            market_df = df_ts[df_ts['market'] == market]
            if market_df.empty: continue
            
            # 找到该时间点的主流盘口
            main_handicap = market_df['handicap'].mode()[0]
            main_df = market_df[market_df['handicap'] == main_handicap]
            
            probs = main_df.groupby('outcome').apply(lambda x: np.average(x['prob_fair'], weights=x['total_weight']))
            consensus[market] = {'handicap': main_handicap, **probs.to_dict()}
        
        if '1X2' not in consensus or 'O/U' not in consensus: continue
        
        # --- 市场张力 ---
        tension_df = df_ts[(df_ts['market'] == '1X2') & (df_ts['outcome'] == '主胜')]
        tension = tension_df['prob_fair'].std() * 100 if len(tension_df) > 1 else 0
        
        # --- 跨市场一致性 ---
        # 矩阵A: 基于1X2和O/U
        matrix_a = generate_score_matrix(
            consensus['1X2']['主胜'], consensus['1X2']['平局'], consensus['1X2']['客胜'],
            consensus['O/U']['handicap'], consensus['O/U']['大球']
        )
        
        coherence = np.nan
        if 'AH' in consensus:
            ah_handicap = consensus['AH']['handicap']
            
            # 核心修正：矩阵计算的是"主队赢盘"概率
            prob_home_covers = get_market_prob_from_matrix(matrix_a, ah_handicap)
            
            # 市场实际的亚盘上盘概率
            prob_ah_from_market = consensus['AH']['上盘']
            
            # 逻辑转换：
            # 如果盘口 < 0 (主队是上盘)，则市场概率 = 主队赢盘概率 -> 直接比较
            # 如果盘口 > 0 (客队是上盘)，则市场概率 = 客队赢盘概率 = 1 - 主队赢盘概率 -> 反向比较
            target_prob = prob_home_covers if ah_handicap < 0 else (1 - prob_home_covers)
            
            # 一致性计算
            coherence = 1 - abs(prob_ah_from_market - target_prob)

        consensus_list.append({'timestamp': ts, **consensus})
        tension_list.append({'timestamp': ts, 'tension': tension})
        coherence_list.append({'timestamp': ts, 'coherence': coherence})

    consensus_df = pd.DataFrame(consensus_list)
    tension_df = pd.DataFrame(tension_list)
    coherence_df = pd.DataFrame(coherence_list)
    
    return consensus_df, tension_df, coherence_df, matrix_a # 返回最后一个时间点的矩阵作为最终预测

# ==============================================================================
# 阶段三: 交互与解读层增强
# ==============================================================================
def display_analysis_summary(consensus, tension, coherence):
    st.subheader("📊 决策智能分析摘要")
    if consensus.empty:
        st.warning("数据不足，无法生成分析摘要。")
        return

    final = consensus.iloc[-1]
    final_tension = tension.iloc[-1]['tension']
    final_coherence = coherence.iloc[-1]['coherence']

    # --- 健壮性修正: 区分 '0.0' (平手盘) 和 NaN (数据缺失) ---
    ah_data = final.get('AH')
    
    # 如果 ah_data 是字典（即使是 handicap=0.0 的平手盘），则正常处理
    if isinstance(ah_data, dict):
        ah_h = ah_data.get('handicap', 0.0)
        ah_p = ah_data.get('上盘', 0.0) * 100
        # 注意：即使是 0.0 (平手盘) 也是有效数据，不需要显示警告
    else:
        # 只有当 AH 列本身就是 float (即 NaN) 时，才视为数据缺失
        ah_h = "N/A"
        ah_p = 0.0
        final_coherence = 0.0 # 数据缺失时一致性默认为 0
        st.warning("⚠️ 当前筛选的时间范围内缺少亚洲让球(AH)数据，分析摘要中的亚盘部分不可用。")
    # ---------------------------------------------------------------------

    # 1. 市场共识
    try:
        p_h = final['1X2']['主胜']*100
        p_d = final['1X2']['平局']*100
        p_a = final['1X2']['客胜']*100
        outcome = '主胜' if p_h >= p_a else '客胜'
        summary_1x2 = f"市场共识倾向于 **{outcome}** (胜/平/负: {p_h:.1f}% / {p_d:.1f}% / {p_a:.1f}%)。"
    except (KeyError, TypeError):
        summary_1x2 = "数据解析异常。"

    # 2. 核心盘口
    try:
        ou_h = final['O/U']['handicap']
        ou_p = final['O/U']['大球']*100
        # 格式化盘口显示：如果是0显示0.0，如果是整数也显示.0
        ah_h_str = f"{float(ah_h):.2f}" if ah_h != "N/A" else "N/A"
        summary_markets = f"亚盘: {ah_h_str} (上盘 {ah_p:.1f}%) | 大小球: {float(ou_h):.2f} (大球 {ou_p:.1f}%)。"
    except (KeyError, TypeError):
        summary_markets = f"亚盘: {ah_h} | 大小球信息缺失。"

    # 3. 一致性
    coh_text = f"**{final_coherence:.2f}/1.00**" if isinstance(final_coherence, float) else "**N/A**"
    summary_coherence = f"市场一致性: {coh_text}。"

    # 4. 市场张力
    ten_text = f"**{final_tension:.2f}**"
    summary_tension = f"市场分歧度: {ten_text}。"

    st.markdown(f"""
    - **市场偏向:** {summary_1x2}
    - **盘口分布:** {summary_markets}
    - **内在逻辑:** {summary_coherence}
    - **资金分歧:** {summary_tension}
    """)

def get_odds_str(p_percent):
    if p_percent == 0: return ""
    p_val = p_percent / 100.0
    fair_odds = 1 / p_val
    return f"[@{fair_odds:.2f} | @{fair_odds*0.92:.2f} | @{fair_odds*0.85:.2f}]"

def display_score_prediction_ui(final_matrix):
    col1, col2 = st.columns([1, 1])
    
    scores = {}
    for h in range(MAX_SCORE):
        for a in range(MAX_SCORE):
            scores[f"{h}-{a}"] = final_matrix[h, a] * 100
            
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    with col1:
        st.subheader("🎯 最终赛果概率推演 (Top 10)")
        hot_set = sorted_scores[:5]
        cold_set = [s for s in sorted_scores if s[1] < 2.0][:5]
        
        st.write("**🔥 高概率常规集**")
        for s, p in hot_set:
            st.write(f"- **{s}**: {p:.2f}% {get_odds_str(p)}")
        
        st.write("**❄️ 低概率潜在集**")
        for s, p in cold_set:
            st.write(f"- **{s}**: {p:.2f}% {get_odds_str(p)}")

    with col2:
        st.subheader("🌡️ 比分概率热力图")
        z_data = final_matrix * 100
        fig_hm = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[f"客{i}" for i in range(MAX_SCORE)],
            y=[f"主{i}" for i in range(MAX_SCORE)],
            colorscale='YlOrRd',
            hovertemplate='比分 %{y}-%{x}<br>最终概率: %{z:.2f}%<extra></extra>'
        ))
        fig_hm.update_layout(height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig_hm, use_container_width=True)

def create_main_plot(df, consensus_df, tension_df, coherence_df, title):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
                        row_heights=[0.7, 0.3])

    # --- 上部图：赔率/概率图 ---
    for bk, group in df.groupby('bookmaker'):
        color = BOOKMAKER_COLORS.get(bk, DEFAULT_COLOR)
        # 只绘制主胜的公平概率作为代表
        group_h = group[(group['market'] == '1X2') & (group['outcome'] == '主胜')]
        if not group_h.empty:
            fig.add_trace(go.Scatter(
                x=group_h['timestamp'], y=group_h['prob_fair']*100, name=f"{bk} 主胜公平概率",
                line=dict(color=color, width=1.5), mode='lines+markers', marker=dict(size=4),
                hovertemplate="<b>%{fullData.name}</b><br>时间: %{x}<br>公平概率: %{y:.2f}%<extra></extra>"
            ), secondary_y=False, row=1, col=1)

    # 绘制共识曲线
    if not consensus_df.empty:
        consensus_1x2 = pd.json_normalize(consensus_df['1X2'])
        if '主胜' in consensus_1x2.columns:
            fig.add_trace(go.Scatter(
                x=consensus_df['timestamp'], y=consensus_1x2['主胜']*100, name="市场共识: 主胜",
                line=dict(color='black', width=4, dash='solid'), mode='lines',
                hovertemplate="<b>市场共识: 主胜</b><br>时间: %{x}<br>加权公平概率: %{y:.2f}%<extra></extra>"
            ), secondary_y=False, row=1, col=1)

    # --- 下部图：分析指标图 ---
    if not tension_df.empty:
        fig.add_trace(go.Scatter(
            x=tension_df['timestamp'], y=tension_df['tension'], name="市场张力指数",
            line=dict(color='purple', width=2), fill='tozeroy',
            hovertemplate="<b>市场张力</b><br>时间: %{x}<br>分歧度: %{y:.2f}<extra></extra>"
        ), row=2, col=1)

    if not coherence_df.empty:
        fig.add_trace(go.Scatter(
            x=coherence_df['timestamp'], y=coherence_df['coherence'], name="跨市场一致性",
            line=dict(color='green', width=2),
            hovertemplate="<b>跨市场一致性</b><br>时间: %{x}<br>自洽度: %{y:.2f}/1.0<extra></extra>"
        ), row=2, col=1)

    fig.update_layout(height=800, title=dict(text=title, x=0.5), hovermode='x unified', template='plotly_white',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="概率 (%)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="分析指数", row=2, col=1)
    fig.update_xaxes(showticklabels=True, row=2, col=1)

    return fig

# ==============================================================================
# Streamlit 主应用
# ==============================================================================
st.set_page_config(layout="wide", page_title="足球量化决策系统")
st.title("⚽ 足球量化决策系统 v2.0")

with st.sidebar:
    st.header("⚙️ 控制面板")
    uploaded_file = st.file_uploader("上传赛事文档 (.md)", type=['md', 'txt'])
    analysis_mode = st.radio("分析模式", ["决策智能分析", "原始数据探索"], index=0)
    
    if uploaded_file:
        try:
            content = uploaded_file.getvalue().decode('utf-8')
            match_info, master_df = parse_and_build_unified_df(content)

            time_opts = {"全部": 9999, "24h": 24, "12h": 12, "6h": 6, "3h": 3, "1h": 1}
            selected_time = st.radio("时间范围", list(time_opts.keys()), horizontal=True, index=3)
            
            all_bks = sorted(master_df['bookmaker'].unique())
            default_bks = [bk for bk in all_bks if bk in LEAD_BOOKMAKER_WEIGHTS] or all_bks
            selected_bks = st.multiselect("选择机构", all_bks, default=default_bks)
            
            time_limit = match_info['match_time'] - timedelta(hours=time_opts[selected_time])
            df_filtered = master_df[(master_df['timestamp'] >= time_limit) & (master_df['bookmaker'].isin(selected_bks))]
        except Exception as e:
            st.error(f"文件解析或处理失败: {e}")
            st.stop()

if 'df_filtered' in locals() and not df_filtered.empty:
    st.header(match_info.get('title', '比赛详情'))
    st.caption(f"开赛时间: {match_info.get('match_time', '').strftime('%Y-%m-%d %H:%M')}")

    if analysis_mode == "决策智能分析":
        if len(df_filtered['bookmaker'].unique()) < 2:
            st.warning("决策智能分析至少需要选择2家机构的数据以计算市场张力。")
        else:
            consensus_df, tension_df, coherence_df, final_matrix = run_advanced_analysis(df_filtered, match_info['match_time'])
            
            display_analysis_summary(consensus_df, tension_df, coherence_df)
            st.markdown("---")
            display_score_prediction_ui(final_matrix)
            st.markdown("---")
            st.subheader("📈 市场演变与分析指标")
            st.plotly_chart(create_main_plot(df_filtered, consensus_df, tension_df, coherence_df, "市场共识与分析指标演变图"), use_container_width=True)

    elif analysis_mode == "原始数据探索":
        st.info("在此模式下，您可以探索各机构、各玩法的原始公平概率走势。")
        market_select = st.selectbox("选择玩法", df_filtered['market'].unique())
        available_outcomes = df_filtered[df_filtered['market']==market_select]['outcome'].unique()
        outcome_select = st.multiselect("选择投注项", available_outcomes)
        
        plot_df = df_filtered[(df_filtered['market'] == market_select) & (df_filtered['outcome'].isin(outcome_select))]
        
        if not plot_df.empty:
            fig = go.Figure()
            for name, group in plot_df.groupby(['bookmaker', 'outcome']):
                bk, outcome = name
                fig.add_trace(go.Scatter(
                    x=group['timestamp'], y=group['prob_fair']*100, name=f"{bk} - {outcome}",
                    line=dict(color=BOOKMAKER_COLORS.get(bk, DEFAULT_COLOR)),
                    hovertemplate="<b>%{fullData.name}</b><br>盘口: %{customdata}<br>公平概率: %{y:.2f}%<extra></extra>",
                    customdata=group['handicap']
                ))
            fig.update_layout(title=f"{market_select} 市场公平概率走势", yaxis_title="公平概率 (%)", hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("审查已加载的标准化数据"):
        st.dataframe(df_filtered)
else:
    st.info("👈 请在左侧上传赛事文档并配置分析选项。")
