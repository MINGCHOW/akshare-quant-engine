# -*- coding: utf-8 -*-
"""
V10.0 Quant Logic Module
Technical indicators, signal generation, stock name/ETF detection
"""
import pandas as pd
import akshare as ak
import logging

logger = logging.getLogger(__name__)


# --- Utility ---
def safe_round(val, decimals=2):
    try:
        if pd.isna(val) or val == float('inf') or val == float('-inf'):
            return 0.0
        return round(float(val), decimals)
    except Exception:
        return 0.0


from .fetcher import DataFetcher

# --- Stock Name & ETF Detection ---
def get_stock_name(code: str, market: str = "CN") -> str:
    """获取股票真实名称 (Delegates to DataFetcher Cache)"""
    return DataFetcher.get_stock_name(code, market)


def detect_etf(code: str, market: str = "CN") -> bool:
    """
    V10.0: 精确 ETF 检测
    港股: 特定代码区间 + 名称匹配
    A股: 代码前缀匹配
    """
    clean_code = str(code).strip().upper().replace("HK", "").replace("SH", "").replace("SZ", "")
    
    if market == "HK":
        if clean_code.isdigit():
            num = int(clean_code)
            # 港股 ETF 精确代码区间
            hk_etf_ranges = [
                (2800, 2849),   # 盈富基金、恒生ETF等
                (3000, 3199),   # 南方A50、华夏恒生等
                (7200, 7399),   # 杠杆/反向产品
                (7500, 7599),   # 杠杆/反向产品
                (8200, 8299),   # 人民币计价ETF
                (9000, 9099),   # 人民币计价ETF
                (9800, 9899),   # 人民币计价ETF
            ]
            for low, high in hk_etf_ranges:
                if low <= num <= high:
                    return True
            
            # 额外：通过名称检测
            try:
                name = get_stock_name(code, market)
                if "ETF" in name.upper():
                    return True
            except Exception:
                pass
    else:
        # A股 ETF 代码规则
        a_etf_prefixes = ('51', '15', '16', '58', '56', '52')
        return clean_code.startswith(a_etf_prefixes)
    
    return False


# --- Technical Indicators ---
def calculate_technicals(df: pd.DataFrame):
    """
    V10.0: 技术指标计算
    MA(5/10/20/60), EMA(13/26), RSI(14), ATR(14), MACD, BIAS, 量比
    """
    if df.empty: return {}
    df = df.sort_values('date')
    closes = df['close']
    highs = df['high']
    lows = df['low']
    volumes = df['volume']
    
    # Ensure sufficient data
    if len(df) < 5: return {}

    ma5 = closes.rolling(5).mean().iloc[-1]
    ma10 = closes.rolling(10).mean().iloc[-1]
    ma20 = closes.rolling(20).mean().iloc[-1]
    ma60 = closes.rolling(60).mean().iloc[-1]
    ema13 = closes.ewm(span=13, adjust=False).mean().iloc[-1]
    ema26 = closes.ewm(span=26, adjust=False).mean().iloc[-1]
    
    delta = closes.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    
    # Safe RSI calc
    loss_val = loss.iloc[-1]
    if loss_val == 0:
        rsi14 = 100.0 if gain.iloc[-1] > 0 else 50.0 
    else:
        rs = gain.iloc[-1] / loss_val
        rsi14 = 100 - (100 / (1 + rs))
    
    tr1 = highs - lows
    tr2 = (highs - closes.shift(1)).abs()
    tr3 = (lows - closes.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]

    current_price = closes.iloc[-1]
    # Safe Bias calc
    bias_ma5 = ((current_price - ma5) / ma5) * 100 if (ma5 and ma5 != 0) else 0.0
    
    current_volume = volumes.iloc[-1]
    volume_ma5 = volumes.rolling(5).mean().iloc[-1]
    # Safe Volume Ratio
    volume_ratio = current_volume / volume_ma5 if (volume_ma5 and volume_ma5 > 0) else 1.0
    
    ma_alignment = "趋势不明 ⚖️"
    # Safe MA alignment check (handle NaNs)
    if all(not pd.isna(x) for x in [ma5, ma10, ma20, ma60]):
        if ma5 > ma10 > ma20 > ma60:
            ma_alignment = "多头排列 📈"
        elif ma5 < ma10 < ma20 < ma60:
            ma_alignment = "空头排列 📉"
        elif ma5 > ma10 > ma20:
            ma_alignment = "短期多头 📈"
        elif ma5 < ma10 < ma20:
            ma_alignment = "短期空头 📉"
            
    # V10.0: MACD Calculation
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26_series = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26_series
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    
    macd_val = macd_line.iloc[-1]
    macd_sig_val = macd_signal.iloc[-1]
    macd_hist_val = macd_hist.iloc[-1]
    # MACD cross detection
    macd_cross = "none"
    if len(macd_hist) >= 2:
        prev_hist = macd_hist.iloc[-2]
        if prev_hist <= 0 and macd_hist_val > 0:
            macd_cross = "golden"  # 金叉
        elif prev_hist >= 0 and macd_hist_val < 0:
            macd_cross = "death"   # 死叉
    
    recent_lows = lows.tail(20).min()
    # Safe Support/Resistance
    support_level = min(recent_lows, ma20) if not pd.isna(ma20) else recent_lows
    
    recent_highs = highs.tail(20).max()
    res_list = [x for x in [recent_highs, ma5, ma10] if not pd.isna(x)]
    resistance_level = max(res_list) if res_list else current_price * 1.1
    
    return {
        "current_price": safe_round(current_price),
        "ma5": safe_round(ma5), "ma10": safe_round(ma10), 
        "ma20": safe_round(ma20), "ma60": safe_round(ma60),
        "ema13": safe_round(ema13), "ema26": safe_round(ema26),
        "rsi14": safe_round(rsi14),
        "atr14": safe_round(atr14),
        "bias_ma5": safe_round(bias_ma5),
        "volume_ratio": safe_round(volume_ratio),
        "ma_alignment": ma_alignment,
        "support_level": safe_round(support_level),
        "resistance_level": safe_round(resistance_level),
        # V10.0 MACD
        "macd": safe_round(macd_val, 4),
        "macd_signal": safe_round(macd_sig_val, 4),
        "macd_hist": safe_round(macd_hist_val, 4),
        "macd_cross": macd_cross
    }


# --- Signal Generation ---
def generate_signal(tech, is_hk=False):
    """
    V10.0: 重构信号生成器
    - 对称评分体系 (买卖平衡)
    - MACD 金叉/死叉判断
    - 卖出信号生成
    - 动态 ATR 止损/止盈 (盈亏比 2:1)
    """
    score = 50  # 中性起点
    reasons = []
    
    p = tech.get('current_price', 0)
    ma5 = tech.get('ma5', 0)
    ma10 = tech.get('ma10', 0)
    ma20 = tech.get('ma20', 0)
    rsi = tech.get('rsi14', 50)
    vol_ratio = tech.get('volume_ratio', 1)
    macd_cross = tech.get('macd_cross', 'none')
    macd_hist = tech.get('macd_hist', 0)
    
    # === 均线系统 (对称 ±) ===
    if p > ma5: 
        score += 5
    else: 
        score -= 5
    
    if p > ma20: 
        score += 15
        reasons.append("站上月线")
    else: 
        score -= 15
        reasons.append("跌破月线")
    
    # === MACD (V10.0 新增) ===
    if macd_cross == 'golden':
        score += 15
        reasons.append("MACD金叉 🔥")
    elif macd_cross == 'death':
        score -= 15
        reasons.append("MACD死叉 ⚠️")
    elif macd_hist > 0:
        score += 5
    else:
        score -= 5
    
    # === RSI (对称 ±) ===
    if rsi > 80:
        score -= 15
        reasons.append("RSI严重超买")
    elif rsi > 70:
        score -= 10
        reasons.append("RSI超买")
    elif rsi < 20:
        score += 15
        reasons.append("RSI严重超卖")
    elif rsi < 30:
        score += 10
        reasons.append("RSI超卖")
    
    # === 量比 (对称) ===
    if vol_ratio > 2.0:
        if p > ma5:
            score += 10
            reasons.append("放量突破")
        else:
            score -= 10
            reasons.append("放量下跌")
    elif vol_ratio > 1.5 and p > ma5:
        score += 5
        reasons.append("温和放量")
    elif vol_ratio < 0.5:
        score -= 5
        reasons.append("严重缩量")
    
    # === 均线形态 ===
    alignment = tech.get('ma_alignment', '')
    if '多头' in alignment:
        score += 10
        reasons.append("均线多头排列")
    elif '空头' in alignment:
        score -= 10
        reasons.append("均线空头排列")
    
    # === VCP 粘合突破 ===
    if p > ma20 and ma5 > ma20 and ma20 > 0:
        if abs(ma5 - ma20) / ma20 < 0.03 and vol_ratio > 1.2:
            score += 10
            reasons.append("均线粘合放量突破 (VCP)")
    
    # === 限制分数范围 ===
    score = max(0, min(100, score))
    
    # === 生成信号 (买卖平衡) ===
    if score >= 80:
        signal = "强烈买入 🚀"
    elif score >= 65:
        signal = "买入 🟢"
    elif score >= 45:
        signal = "观望 😶"
    elif score >= 30:
        signal = "减仓 🟡"
    else:
        signal = "卖出 🔴"
    
    # === 动态 ATR 止损/止盈 (V10.0) ===
    atr = tech.get('atr14', 0)
    if not atr or atr <= 0:
        atr = p * 0.03  # Fallback: 3% of price
    
    # 港股波动更大，使用更宽的乘数
    if is_hk:
        stop_multiplier = 3.0
    else:
        stop_multiplier = 2.0
    
    # 根据波动率动态调整: 高波动→宽止损
    volatility_pct = (atr / p * 100) if p > 0 else 3.0
    if volatility_pct > 5:
        stop_multiplier += 0.5  # 高波动多加 0.5 ATR
    
    atr_stop = p - (stop_multiplier * atr)
    supp = tech.get('support_level', 0)
    
    # 止损: ATR止损和支撑位取较近的一个 (保护资金)
    if supp > 0 and supp < p:
        stop_loss = max(atr_stop, supp * 0.98)  # V10.0: 取较近的(max)
    else:
        stop_loss = atr_stop
    
    # 确保止损不超过现价的15%
    max_loss_pct = 0.15 if is_hk else 0.10
    min_stop = p * (1 - max_loss_pct)
    stop_loss = max(stop_loss, min_stop)
    
    risk_per_share = p - stop_loss
    # V10.0: 盈亏比提升至 2:1
    take_profit = p + (2.0 * risk_per_share) if risk_per_share > 0 else p * 1.1
    
    suggested_buy = max(supp, p * 0.98) if supp > 0 else p * 0.98
    
    return {
        "signal": signal,
        "signal_reasons": reasons,
        "trend_score": int(score),
        "stop_loss": safe_round(stop_loss),
        "take_profit": safe_round(take_profit),
        "suggested_buy": safe_round(suggested_buy),
        "support_level": safe_round(supp),
        "resistance_level": safe_round(tech.get('resistance_level', 0))
    }
