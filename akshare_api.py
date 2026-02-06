# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import akshare as ak
import pandas as pd
import requests
import datetime
import traceback
import random
import time
import json
import logging
import math # Added for NaN checks

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing optional libraries
try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from pytdx.hq import TdxHq_API
    tdx_api = TdxHq_API()
except ImportError:
    tdx_api = None

try:
    import baostock as bs
    # Init Baostock on startup
    bs.login()
except ImportError:
    bs = None

try:
    import qstock as qs
except ImportError:
    qs = None

app = FastAPI(title="AkShare Quant API V8.3 (Limitless Edition)", version="8.3")

# ... (Constants Omitted) ...
# --- Constants ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
]

def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "*/*",
        "Connection": "keep-alive"
    }

# --- Models ---
class AnalyzeRequest(BaseModel):
    code: str
    balance: float = 100000.0
    risk: float = 0.01

# --- Data Fetcher with 7-Layer Fallback (Scientific Order) ---
class DataFetcher:
    """
    V8.4 Updates:
    - Added `_clean_data` helper to enforce numeric types (Fixes 500 Error)
    - Local instantiation of Pytdx API (Thread Safety)
    - Scientific Fallback Hierarchy: AkShare->Tencent->Qstock->Pytdx->Baostock->Sina->Yahoo
    """
    @staticmethod
    def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and types"""
        try:
            # Ensure proper columns exist
            required_cols = {'date', 'open', 'close', 'high', 'low', 'volume'}
            if not required_cols.issubset(df.columns):
                return pd.DataFrame() # Missing columns

            # Standardize date
            df['date'] = pd.to_datetime(df['date'])
            
            # Enforce numeric types (Critical for calculation safety)
            for col in ['open', 'close', 'high', 'low', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop NaNs created by coercion
            df.dropna(subset=['open', 'close'], inplace=True)
            return df
        except:
            return pd.DataFrame()

    @staticmethod
    def get_a_share_history(code: str, retries=2):
        time.sleep(random.uniform(0.5, 1.5)) # Anti-Blocking
        
        symbol = code.replace("sh", "").replace("sz", "")
        market_prefix = "sh" if code.startswith("6") else "sz"
        
        # 1. AkShare (EastMoney)
        for attempt in range(retries):
            try:
                logger.info(f"Attempting AkShare (#1) for {code}...")
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
                if not df.empty and len(df) > 30:
                    df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', 
                                       '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
                    return DataFetcher._clean_data(df)
            except Exception as e:
                logger.warning(f"AkShare failed: {e}")
                time.sleep(1.0) 

        # 2. Tencent (HTTP)
        try:
            logger.info(f"Attempting Tencent (#2) Fallback...")
            full_code = f"{market_prefix}{symbol}"
            url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={full_code},day,,,320,qfq" 
            r = requests.get(url, headers=get_headers(), timeout=8)
            data = r.json()
            k_data = data['data'][full_code]['day']
            df = pd.DataFrame(k_data, columns=['date', 'open', 'close', 'high', 'low', 'volume', 'unknown'])
            df = df[['date', 'open', 'close', 'high', 'low', 'volume']]
            print(f"[Fallback] Recovered {code} using Tencent.")
            return DataFetcher._clean_data(df)
        except Exception as e:
            logger.warning(f"Tencent failed: {e}")

        # 3. Qstock (Tonghuashun)
        if qs:
            try:
                logger.info(f"Attempting Qstock-THS (#3) Fallback...")
                df = qs.get_data(code_list=[code], start='20240101', end=datetime.date.today().strftime('%Y%m%d'), freq='d')
                if not df.empty:
                    if 'date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                        df.reset_index(inplace=True)
                        df.rename(columns={'index': 'date'}, inplace=True)
                    
                    # Try renaming standard Chinese columns
                    df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', 
                                       '最高': 'high', '最低': 'low', '成交量': 'volume', '成交': 'volume'}, inplace=True)
                    
                    return DataFetcher._clean_data(df[['date', 'open', 'close', 'high', 'low', 'volume']])
            except Exception as e:
                logger.warning(f"Qstock failed: {e}")

        # 4. Pytdx (TCP) - Thread Safe Version
        # Note: Do not use global tdx_api instance for concurrency safety
        try:
            from pytdx.hq import TdxHq_API
            local_tdx = TdxHq_API()
            logger.info(f"Attempting Pytdx (#4 TCP) Fallback...")
            with local_tdx.connect('119.147.212.81', 7709): 
                market_code = 1 if code.startswith("6") else 0
                data = local_tdx.get_security_bars(9, market_code, symbol, 0, 100)
                if data:
                    df = local_tdx.to_df(data)
                    df.rename(columns={'datetime': 'date', 'vol': 'volume'}, inplace=True)
                    return DataFetcher._clean_data(df[['date', 'open', 'close', 'high', 'low', 'volume']])
        except Exception as e:
             logger.warning(f"Pytdx failed: {e}")

        # 5. Baostock (Official)
        if bs:
            try:
                logger.info(f"Attempting Baostock (#5) Fallback...")
                rs = bs.query_history_k_data_plus(f"{market_prefix}.{symbol}",
                    "date,open,high,low,close,volume",
                    start_date=(datetime.date.today() - datetime.timedelta(days=365)).strftime('%Y-%m-%d'), 
                    end_date=datetime.date.today().strftime('%Y-%m-%d'),
                    frequency="d", adjustflag="1")
                
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                
                if data_list:
                    df = pd.DataFrame(data_list, columns=rs.fields)
                    return DataFetcher._clean_data(df)
            except Exception as e:
                logger.error(f"Baostock failed: {e}")

        # 6. Sina (Legacy)
        try:
            logger.info(f"Attempting Sina (#6) Fallback...")
            sina_symbol = f"{market_prefix}{symbol}"
            df = ak.stock_zh_a_daily(symbol=sina_symbol, adjust="qfq")
            if not df.empty:
                return DataFetcher._clean_data(df)
        except Exception as e:
            logger.error(f"Sina failed: {e}")

        # 7. Yahoo Finance
        if yf:
            try:
                logger.info(f"Attempting Yahoo (#7) Fallback...")
                suffix = ".SS" if code.startswith("6") else ".SZ"
                y_symbol = f"{symbol}{suffix}"
                ticker = yf.Ticker(y_symbol)
                df = ticker.history(period="1y")
                
                if not df.empty:
                    df.reset_index(inplace=True)
                    df.rename(columns={'Date': 'date', 'Open': 'open', 'Close': 'close', 
                                       'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace=True)
                    df['date'] = df['date'].dt.tz_localize(None)
                    print(f"[Fallback] Recovered {code} using Yahoo.")
                    return DataFetcher._clean_data(df)
            except Exception as e:
                logger.warning(f"Yahoo failed: {e}")

        return pd.DataFrame()


    @staticmethod
    def get_hk_share_history(code: str):
        try:
            time.sleep(random.uniform(0.5, 1.5)) # Anti-Blocking
            
            # Safe clean of code
            clean_code = str(code).strip().upper().replace("HK", "")
            if not clean_code.isdigit():
                 # Handle cases like "00700" -> "00700" (fine)
                 # Handle cases like "HK00700" -> "00700" (fine)
                 # Handle garbage -> return empty
                 logger.warning(f"Invalid HK code format: {code}")
                 return pd.DataFrame()

            symbol = f"{int(clean_code):05d}" # Standardize to 5 chars (e.g. 00700)
            
            # 1. Try AkShare
            try:
                df = ak.stock_hk_daily(symbol=symbol, adjust="qfq")
                if not df.empty:
                    df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', 
                                       '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
                    return DataFetcher._clean_data(df)
            except Exception as e:
                logger.warning(f"AkShare HK failed for {code}: {e}")

            # 2. Try Yahoo
            if yf:
                try:
                    # Yahoo needs 4 digits + .HK usually, or 5 if it's 5 digits. 
                    # Most HK stocks on Yahoo are 4 digits .HK (e.g. 0700.HK)
                    # Use safe int conversion
                    y_symbol = f"{int(clean_code):04d}.HK" 
                    ticker = yf.Ticker(y_symbol)
                    df = ticker.history(period="1y")
                    
                    if not df.empty:
                        df.reset_index(inplace=True)
                        df.rename(columns={'Date': 'date', 'Open': 'open', 'Close': 'close', 
                                           'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace=True)
                        df['date'] = df['date'].dt.tz_localize(None)
                        return DataFetcher._clean_data(df)
                except Exception as e:
                    logger.warning(f"Yahoo HK failed for {code}: {e}")
        except Exception as e:
            logger.error(f"Critical crash in HK fetcher for {code}: {e}")
            
        return pd.DataFrame()

# --- Quant Logic (V8.1 with NaN Safety) ---
def calculate_technicals(df: pd.DataFrame):
    if df.empty: return {}
    df = df.sort_values('date')
    closes = df['close']
    highs = df['high']
    lows = df['low']
    volumes = df['volume']
    
    # ... (Calculations identical to V7, omitted for brevity, will fill in full file) ...
    # WRITING FULL LOGIC HERE TO ENSURE INTEGRITY
    
    ma5 = closes.rolling(5).mean().iloc[-1]
    ma10 = closes.rolling(10).mean().iloc[-1]
    ma20 = closes.rolling(20).mean().iloc[-1]
    ma60 = closes.rolling(60).mean().iloc[-1]
    ema13 = closes.ewm(span=13, adjust=False).mean().iloc[-1]
    ema26 = closes.ewm(span=26, adjust=False).mean().iloc[-1]
    
    delta = closes.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi14 = 100 - (100 / (1 + rs)).iloc[-1]
    
    prev_close = closes.shift(1)
    tr1 = highs - lows
    tr2 = (highs - prev_close).abs()
    tr3 = (lows - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]

    current_price = closes.iloc[-1]
    bias_ma5 = ((current_price - ma5) / ma5) * 100 if ma5 else 0
    
    current_volume = volumes.iloc[-1]
    volume_ma5 = volumes.rolling(5).mean().iloc[-1]
    volume_ratio = current_volume / volume_ma5 if volume_ma5 > 0 else 1.0
    
    if ma5 > ma10 > ma20 > ma60:
        ma_alignment = "多头排列 📈"
    elif ma5 < ma10 < ma20 < ma60:
        ma_alignment = "趋势向下 📉"
    else:
        ma_alignment = "趋势不明 ⚖️"
    
    recent_lows = lows.tail(20).min()
    support_level = min(recent_lows, ma20)
    recent_highs = highs.tail(20).max()
    resistance_level = max(recent_highs, ma5, ma10)
    
    return {
        "current_price": round(float(current_price), 2),
        "ma5": round(float(ma5), 2), "ma10": round(float(ma10), 2), 
        "ma20": round(float(ma20), 2), "ma60": round(float(ma60), 2),
        "ema13": round(float(ema13), 2), "ema26": round(float(ema26), 2),
        "rsi14": round(float(rsi14), 2), "atr14": round(float(atr14), 3),
        "bias_ma5": round(float(bias_ma5), 2), "volume_ratio": round(float(volume_ratio), 2),
        "ma_alignment": ma_alignment,
        "support_level": round(float(support_level), 2), 
        "resistance_level": round(float(resistance_level), 2),
        "trend_score": 0
    }

def generate_signal(tech_data, is_hk=False):
    price = tech_data.get('current_price', 0)
    ma5 = tech_data.get('ma5', 0)
    ma10 = tech_data.get('ma10', 0)
    ma20 = tech_data.get('ma20', 0)
    ema13 = tech_data.get('ema13', 0)
    ema26 = tech_data.get('ema26', 0)
    atr = tech_data.get('atr14', 0)
    volume_ratio = tech_data.get('volume_ratio', 1.0)
    support_level = tech_data.get('support_level', 0)
    resistance_level = tech_data.get('resistance_level', 0)
    
    signal = "观望 ⚪"
    reasons = []
    score = 50
    
    if price > ma20: score += 10
    if ma5 > ma10 and ma10 > ma20: score += 20
    if ema13 > ema26: score += 20
    if volume_ratio > 1.5:
        reasons.append("放量上涨")
        score += 10
    elif volume_ratio < 0.8:
        reasons.append("缩量整理")
    
    if price > ma20 and ma5 > ma20:
        if abs(ma5 - ma20)/ma20 < 0.05:
            signal = "强烈买入 🚀"
            reasons.append("均线粘合突破 (VCP特征)")
        else:
            signal = "买入 🟢"
            reasons.append("多头趋势")
            
    multiplier = 2.5 if is_hk else 2.0
    atr_stop = price - (multiplier * atr)
    stop_loss = min(atr_stop, support_level * 0.98)
    risk_per_share = price - stop_loss
    take_profit = price + (1.5 * risk_per_share)
    suggested_buy = max(support_level, price * 0.98)
    
    return {
        "signal": signal,
        "signal_reasons": reasons,
        "trend_score": score,
        "stop_loss": round(float(stop_loss), 2),
        "take_profit": round(float(take_profit), 2),
        "suggested_buy": round(float(suggested_buy), 2),
        "support_level": round(float(support_level), 2),
        "resistance_level": round(float(resistance_level), 2)
    }

# --- Endpoints ---
@app.get("/market")
def get_market_context():
    try:
        time.sleep(random.uniform(0.5, 1.0)) # Anti-bot
        index_df = ak.stock_zh_index_daily(symbol="sh000001")
        if index_df.empty: raise ValueError("Index Data Empty")
        price = float(index_df['close'].iloc[-1])
        ma20 = float(index_df['close'].rolling(20).mean().iloc[-1])
        status = "Bull" if price > ma20 else "Bear"
        return {"market_status": status, "index_price": price, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    except Exception as e:
        return {"market_status": "Correction", "error": str(e)}

@app.post("/analyze_full")
def analyze_full(req: AnalyzeRequest):
    try:
        code = req.code
        is_hk = len(str(code)) == 5
        if is_hk:
            df = DataFetcher.get_hk_share_history(code)
            market = "HK"
        else:
            df = DataFetcher.get_a_share_history(code)
            market = "CN"
            
        if df.empty:
            return {"error": "No Data found", "code": code}
            
        tech = calculate_technicals(df)
        sig = generate_signal(tech, is_hk)
        tech['trend_score'] = sig['trend_score']
        
        # --- V8.1 NaN Safety Fix ---
        def safe_int(val):
            try:
                if pd.isna(val) or val == float('inf') or val == float('-inf'): return 0
                return int(val)
            except:
                return 0

        risk_per_share = tech['current_price'] - sig['stop_loss']
        if risk_per_share <= 0: risk_per_share = tech['atr14']
        
        account_risk_money = req.balance * req.risk
        
        if risk_per_share <= 0.0001 or pd.isna(risk_per_share):
            suggested_shares = 0
        else:
            raw_shares = account_risk_money / risk_per_share / 100
            suggested_shares = safe_int(raw_shares) * 100
            
        if suggested_shares < 100: suggested_shares = 0
        # ---------------------------
        
        return {
            "日期": datetime.datetime.now().strftime("%Y-%m-%d"),
            "市场": market,
            "代码": code,
            "名称": code, # Name logic omitted
            "信号类型": sig['signal'],
            "综合评分": sig['trend_score'],
            "现价": tech['current_price'],
            "ATR": tech['atr14'],
            "乖离率": tech['bias_ma5'],
            "RSI": tech['rsi14'],
            "量比": tech['volume_ratio'],
            "均线形态": tech['ma_alignment'],
            "买入价": sig['suggested_buy'],
            "止损价": sig['stop_loss'],
            "目标价": sig['take_profit'],
            "支撑位": sig['support_level'],
            "压力位": sig['resistance_level'],
            "technical": tech,
            "signal": sig,
            "risk_ctrl": {
                "suggested_position": suggested_shares,
                "risk_money": account_risk_money
            },
            "prompt_data": {
                "price_info": f"现价: {tech['current_price']}, MA20: {tech['ma20']}",
                "risk_info": f"止损: {sig['stop_loss']}, ATR: {tech['atr14']}",
                "volume_info": f"量比: {tech['volume_ratio']}, 均线: {tech['ma_alignment']}",
                "levels_info": f"支撑: {sig['support_level']}, 压力: {sig['resistance_level']}"
            }
        }
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
