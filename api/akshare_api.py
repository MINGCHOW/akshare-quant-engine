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

try:
    import efinance as ef
except ImportError:
    ef = None

from fake_useragent import UserAgent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

app = FastAPI(title="AkShare Quant API V9.4 (Titan + Name/ETF Fix)", version="9.4")

# --- V8.0 Evolution: Circuit Breaker & Error Tracking ---
error_counter = {
    "count": 0, 
    "last_error": None,
    "last_reset": datetime.datetime.now(),
    "circuit_open": False
}

async def send_emergency_alert(error_msg: str):
    """
    V8.0 P0: 发送紧急告警 (飞书加急消息)
    实际部署时替换为真实的飞书 Webhook
    """
    logger.critical(f"🔴 CIRCUIT BREAKER TRIGGERED: {error_msg}")
    # TODO: 实现飞书加急消息推送
    # webhook_url = os.getenv("FEISHU_ALERT_WEBHOOK")
    # if webhook_url:
    #     requests.post(webhook_url, json={
    #         "msg_type": "text",
    #         "content": {"text": f"🚨 API熔断告警: {error_msg}"}
    #     })

def reset_circuit_breaker():
    """重置熔断器"""
    global error_counter
    error_counter["count"] = 0
    error_counter["circuit_open"] = False
    error_counter["last_reset"] = datetime.datetime.now()

def record_error(error_msg: str):
    """记录错误并检查是否需要触发熔断"""
    global error_counter
    error_counter["count"] += 1
    error_counter["last_error"] = error_msg
    
    if error_counter["count"] >= 3 and not error_counter["circuit_open"]:
        error_counter["circuit_open"] = True
        import asyncio
        asyncio.create_task(send_emergency_alert(error_msg))

def record_success():
    """成功时重置错误计数"""
    global error_counter
    if error_counter["count"] > 0:
        error_counter["count"] = 0
        error_counter["circuit_open"] = False

# --- Constants ---
# Dynamic User-Agent Generator
ua = UserAgent()

def get_headers():
    return {
        "User-Agent": ua.random,
        "Accept": "*/*",
        "Connection": "keep-alive"
    }

# --- Models ---
class AnalyzeRequest(BaseModel):
    code: str
    balance: float = 100000.0
    risk: float = 0.01

# --- Data Fetcher with 8-Layer Fallback (V9.0 Titan) ---
class DataFetcher:
    """
    V9.0 Titan Hierarchy:
    0. efinance (EastMoney API) - Priority 0 (Fastest/Stable)
    1. AkShare (EastMoney Scraper) - Priority 1
    2. Tencent (HTTP) - High Availability
    3. Qstock (Tonghuashun) - Independent Source
    4. Pytdx (TCP) - Anti-Blocking
    5. Baostock (Official) - Backup
    6. Sina (Legacy) - Backup
    7. Yahoo (International) - Last Resort
    """
    @staticmethod
    def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and types (Universal Gatekeeper V9.1)"""
        try:
            if df.empty: return pd.DataFrame()

            # 1. Normalize columns to lowercase (Handle 'Date' vs 'date')
            df.columns = [str(c).lower().strip() for c in df.columns]

            # 2. Map Chinese or variant names
            rename_map = {
                '日期': 'date', 'time': 'date', 'datetime': 'date',
                '开盘': 'open', 'open': 'open',
                '收盘': 'close', 'close': 'close',
                '最高': 'high', 'high': 'high',
                '最低': 'low', 'low': 'low',
                '成交量': 'volume', '成交': 'volume', 'volume': 'volume', 'vol': 'volume'
            }
            df.rename(columns=rename_map, inplace=True)

            # 3. Ensure proper columns exist
            required = {'date', 'open', 'close', 'high', 'low', 'volume'}
            if not required.issubset(df.columns):
                # Try to salvage if only volume is missing (set to 0)
                if required - df.columns == {'volume'}:
                    df['volume'] = 0
                else:
                    return pd.DataFrame()

            # 4. Standardize Date (TZ-Naive)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date'], inplace=True)
            # Properly strip timezone if present (Yahoo returns tz-aware)
            if hasattr(df['date'].dt, 'tz') and df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_convert(None)  # Convert to UTC then remove TZ
            
            # 5. Deduplicate and Sort
            df.drop_duplicates(subset=['date'], keep='last', inplace=True)
            df.sort_values('date', inplace=True)
            
            # 6. Enforce numeric types
            cols = ['open', 'close', 'high', 'low', 'volume']
            for col in cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(subset=['open', 'close'], inplace=True)
            
            return df[list(required)] # Return clean order
        except Exception as e:
            logger.warning(f"Data Cleaning Failed: {e}")
            return pd.DataFrame()

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception))
    def fetch_with_retry(func, *args, **kwargs):
        """Generic retry wrapper"""
        return func(*args, **kwargs)

    @staticmethod
    def get_a_share_history(code: str):
        time.sleep(random.uniform(0.5, 1.5)) 
        
        symbol = code.replace("sh", "").replace("sz", "")
        market_prefix = "sh" if code.startswith("6") else "sz"
        
        # 0. efinance (Priority 0)
        if ef:
            try:
                # logger.info(f"Attempting efinance (#0) for {code}...")
                df = ef.stock.get_quote_history(symbol)
                if not df.empty and len(df) > 30:
                    return DataFetcher._clean_data(df)
            except Exception as e:
                logger.warning(f"efinance failed: {e}")

        # 1. AkShare (EastMoney)
        # Using manual try/except here effectively, but we could wrap specific calls if needed.
        # For the fallback chain, we don't want to retry getting a broken source 3 times before moving to the next.
        # We want to fail fast to the next source.
        # So actually, 'tenacity' on the *whole* method isn't right because we have internal fallbacks.
        # We should use tenacity for the *specific high-value* calls if they are flaky, OR just rely on the fallback chain.
        # Given the 8 layers, fast failover is better than retrying one source.
        # HOWEVER, AkShare network errors (timeouts) specifically benefit from a quick retry.
        
        try:
             # logger.info(f"Attempting AkShare (#1) for {code}...")
             df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
             if not df.empty and len(df) > 30:
                 df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', 
                                    '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
                 return DataFetcher._clean_data(df)
        except Exception as e:
             logger.warning(f"AkShare failed: {e}")
             # Optional: minimal manual retry for AkShare specifically
             try:
                 time.sleep(1)
                 df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
                 if not df.empty: return DataFetcher._clean_data(df)
             except: pass

        # 2. Tencent (HTTP)
        try:
            # logger.info(f"Attempting Tencent (#2) Fallback...")
            full_code = f"{market_prefix}{symbol}"
            url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={full_code},day,,,320,qfq" 
            r = requests.get(url, headers=get_headers(), timeout=8)
            data = r.json()
            if data and 'data' in data and full_code in data['data']:
                qt_data = data['data'][full_code]
                if 'day' in qt_data:
                    k_data = qt_data['day']
                    # Dynamic parsing: just take first 6 columns
                    # Date, Open, Close, High, Low, Volume
                    df = pd.DataFrame(k_data)
                    if df.shape[1] >= 6:
                        df = df.iloc[:, :6]
                        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume']
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
                # Lazy login to prevent timeout
                bs.login() 
                logger.info(f"Attempting Baostock (#5) Fallback...")
                rs = bs.query_history_k_data_plus(f"{market_prefix}.{symbol}",
                    "date,open,high,low,close,volume",
                    start_date=(datetime.date.today() - datetime.timedelta(days=365)).strftime('%Y-%m-%d'), 
                    end_date=datetime.date.today().strftime('%Y-%m-%d'),
                    frequency="d", adjustflag="1")
                
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                
                bs.logout() # Cleanup
                
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
                    return DataFetcher._clean_data(df)
            except Exception as e:
                logger.warning(f"Yahoo failed: {e}")

        return pd.DataFrame()


    @staticmethod
    def get_hk_share_history(code: str):
        try:
            time.sleep(random.uniform(0.5, 1.5)) 
            clean_code = str(code).strip().upper().replace("HK", "")
            if not clean_code.isdigit():
                 return pd.DataFrame()

            symbol = f"{int(clean_code):05d}" # Standardize to 5 chars (e.g. 00700)
            
            # 1. Try AkShare (Eastmoney)
            try:
                logger.info(f"Attempting AkShare HK (#1) for {code}...")
                df = ak.stock_hk_daily(symbol=symbol, adjust="qfq")
                if not df.empty:
                    df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', 
                                       '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
                    return DataFetcher._clean_data(df)
            except Exception as e:
                logger.warning(f"AkShare HK failed for {code}: {e}")

            # 2. Try Tencent HK (HTTP) - Very Reliable
            try:
                logger.info(f"Attempting Tencent HK (#2) for {code}...")
                # Tencent format: hk00700
                tencent_code = f"hk{symbol}"
                # URL: http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=hk00700,day,,,320,qfq
                url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={tencent_code},day,,,320,qfq" 
                r = requests.get(url, headers=get_headers(), timeout=8)
                data = r.json()
                if data and 'data' in data and tencent_code in data['data']:
                    qt_data = data['data'][tencent_code]
                    if 'day' in qt_data:
                        k_data = qt_data['day']
                        # Dynamic parsing: HK returns 6 columns (date, open, close, high, low, volume)
                        df = pd.DataFrame(k_data)
                        if df.shape[1] >= 6:
                            df = df.iloc[:, :6]
                            df.columns = ['date', 'open', 'close', 'high', 'low', 'volume']
                            return DataFetcher._clean_data(df)
            except Exception as e:
                logger.warning(f"Tencent HK failed: {e}")

            # 3. Try Sina HK (Legacy)
            try:
                # logger.info(f"Attempting Sina HK (#3) for {code}...")
                pass
            except:
                pass

            # 4. Try Yahoo Finance (International) - Best for HK
            if yf:
                try:
                    logger.info(f"Attempting Yahoo HK (#4) for {code}...")
                    y_symbol = f"{symbol}.HK"
                    ticker = yf.Ticker(y_symbol)
                    df = ticker.history(period="1y")
                    if not df.empty:
                        df.reset_index(inplace=True)
                        df.rename(columns={'Date': 'date', 'Open': 'open', 'Close': 'close', 
                                           'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace=True)
                        df['date'] = df['date'].dt.tz_localize(None)
                        return DataFetcher._clean_data(df)
                except Exception as e:
                    logger.warning(f"Yahoo HK failed: {e}")

            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Critical HK Fetch Error: {e}")
            return pd.DataFrame()

# --- V9.4: Stock Name & ETF Detection ---
def get_stock_name(code: str, market: str = "CN") -> str:
    """
    获取股票真实名称
    """
    try:
        clean_code = str(code).strip().upper().replace("HK", "").replace("SH", "").replace("SZ", "")
        
        if market == "HK":
            # 港股：尝试从 AkShare 获取
            try:
                symbol = f"{int(clean_code):05d}"
                df = ak.stock_hk_spot_em()
                if not df.empty:
                    match = df[df['代码'] == symbol]
                    if not match.empty:
                        return match.iloc[0]['名称']
            except Exception as e:
                logger.debug(f"HK name fetch failed: {e}")
            
            # 备用：返回格式化代码
            return f"{int(clean_code):05d}.HK"
        else:
            # A股：尝试从 AkShare 获取
            try:
                df = ak.stock_zh_a_spot_em()
                if not df.empty:
                    match = df[df['代码'] == clean_code]
                    if not match.empty:
                        return match.iloc[0]['名称']
            except Exception as e:
                logger.debug(f"A-share name fetch failed: {e}")
            
            # 备用：返回代码
            return clean_code
    except Exception as e:
        logger.warning(f"get_stock_name error: {e}")
        return code

def detect_etf(code: str, market: str = "CN") -> bool:
    """
    检测是否为 ETF
    """
    clean_code = str(code).strip().upper().replace("HK", "").replace("SH", "").replace("SZ", "")
    
    if market == "HK":
        # 港股 ETF 代码规则：
        # 0xxxx, 1xxxx, 5xxxx 开头通常是 ETF
        # 补零处理：2824 -> 02824
        if clean_code.isdigit():
            padded = f"{int(clean_code):05d}"
            return padded.startswith('0') or padded.startswith('1') or padded.startswith('5')
    else:
        # A股 ETF 代码规则：
        # 51xxxx (上证 ETF), 15xxxx/16xxxx (深证 ETF)
        return (clean_code.startswith('51') or 
                clean_code.startswith('15') or 
                clean_code.startswith('16'))
    
    return False

# --- Quant Logic (V8.6 Global NaN Protection) ---
def safe_round(val, decimals=2):
    try:
        if pd.isna(val) or val == float('inf') or val == float('-inf'):
            return 0.0
        return round(float(val), decimals)
    except:
        return 0.0

def calculate_technicals(df: pd.DataFrame):
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
            ma_alignment = "趋势向下 📉"
            
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
        "resistance_level": safe_round(resistance_level)
    }

def generate_signal(tech, is_hk=False):
    score = 50
    reasons = []
    signal = "观望 😶"
    
    p = tech.get('current_price', 0)
    ma5 = tech.get('ma5', 0)
    ma20 = tech.get('ma20', 0)
    rsi = tech.get('rsi14', 50)
    vol_ratio = tech.get('volume_ratio', 1)
    
    if p > ma5: score += 10
    if p > ma20: score += 20; reasons.append("站上月线")
    else: score -= 20; reasons.append("跌破月线")
    
    if rsi > 70: score -= 10; reasons.append("RSI超买")
    elif rsi < 30: score += 10; reasons.append("RSI超卖")
    
    if vol_ratio > 1.5:
        reasons.append("放量上涨")
        score += 10
    elif vol_ratio < 0.8:
        reasons.append("缩量整理")
    
    if p > ma20 and ma5 > ma20:
        if ma20 > 0 and abs(ma5 - ma20)/ma20 < 0.05:
            signal = "强烈买入 🚀"
            reasons.append("均线粘合突破 (VCP特征)")
        else:
            signal = "买入 🟢"
            reasons.append("多头趋势")
            
    multiplier = 2.5 if is_hk else 2.0
    atr = tech.get('atr14', 0) if tech.get('atr14') else p * 0.03 # Fallback ATR
    
    atr_stop = p - (multiplier * atr)
    supp = tech.get('support_level', 0)
    stop_loss = min(atr_stop, supp * 0.98) if supp > 0 else atr_stop
    
    risk_per_share = p - stop_loss
    take_profit = p + (1.5 * risk_per_share) if risk_per_share > 0 else p * 1.1
    
    suggested_buy = max(supp, p * 0.98)
    
    return {
        "signal": signal,
        "signal_reasons": reasons,
        "trend_score": int(score), # int is safe
        "stop_loss": safe_round(stop_loss),
        "take_profit": safe_round(take_profit),
        "suggested_buy": safe_round(suggested_buy),
        "support_level": safe_round(supp),
        "resistance_level": safe_round(tech.get('resistance_level', 0))
    }

# --- Endpoints ---

# V8.0 P0: Health Check Endpoint
@app.get("/health")
def health_check():
    """
    V8.0 P0: 系统健康检查
    - 检查数据源可用性
    - 返回熔断器状态
    - 返回系统延迟
    """
    start_time = time.time()
    checks = {}
    overall_status = "healthy"
    
    # 1. 数据源检查 (快速测试)
    try:
        test_df = DataFetcher.get_a_share_history("000001")
        if test_df.empty:
            checks["data_source"] = {"status": "warning", "message": "Empty data returned"}
            overall_status = "degraded"
        else:
            checks["data_source"] = {"status": "ok", "rows": len(test_df)}
            record_success()  # 成功时重置错误计数
    except Exception as e:
        checks["data_source"] = {"status": "error", "message": str(e)}
        overall_status = "degraded"
        record_error(str(e))
    
    # 2. 熔断器状态
    checks["circuit_breaker"] = {
        "error_count": error_counter["count"],
        "is_open": error_counter["circuit_open"],
        "last_error": error_counter["last_error"],
        "last_reset": error_counter["last_reset"].isoformat() if error_counter["last_reset"] else None
    }
    
    if error_counter["circuit_open"]:
        overall_status = "critical"
    
    # 3. 可选库检查
    checks["optional_libs"] = {
        "efinance": ef is not None,
        "yfinance": yf is not None,
        "pytdx": tdx_api is not None,
        "baostock": bs is not None,
        "qstock": qs is not None
    }
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    return {
        "status": overall_status,
        "timestamp": datetime.datetime.now().isoformat(),
        "latency_ms": latency_ms,
        "checks": checks,
        "version": "9.1 + V8.0 Evolution"
    }

# V8.0 P0: Reset Circuit Breaker (Manual)
@app.post("/health/reset")
def reset_health():
    """手动重置熔断器"""
    reset_circuit_breaker()
    return {"status": "ok", "message": "Circuit breaker reset"}

# V8.0 P1: Enhanced Market Status
@app.get("/market")
def get_market_context():
    """
    V8.0 P1: 增强版大盘状态
    - 新增涨跌家数统计
    - 新增市场冰点标记
    """
    try:
        time.sleep(random.uniform(0.5, 1.0))  # Anti-bot
        
        # 指数数据
        index_df = ak.stock_zh_index_daily(symbol="sh000001")
        if index_df.empty:
            raise ValueError("Index Data Empty")
        price = float(index_df['close'].iloc[-1])
        ma20 = float(index_df['close'].rolling(20).mean().iloc[-1])
        status = "Bull" if price > ma20 else "Bear"
        
        # V8.0 P1: 涨跌家数统计
        up_count, down_count, flat_count = 0, 0, 0
        try:
            stats = ak.stock_zh_a_spot_em()
            if not stats.empty and '涨跌幅' in stats.columns:
                up_count = len(stats[stats['涨跌幅'] > 0])
                down_count = len(stats[stats['涨跌幅'] < 0])
                flat_count = len(stats[stats['涨跌幅'] == 0])
        except Exception as e:
            logger.warning(f"Failed to get up/down count: {e}")
        
        # 市场冰点判断 (上涨家数 < 800)
        is_frozen = up_count > 0 and up_count < 800
        
        # 调整市场状态 (极端冰点时强制标记)
        if is_frozen:
            status = "Crash" if up_count < 500 else "Bear"
        
        return {
            "market_status": status,
            "index_price": safe_round(price),
            "ma20": safe_round(ma20),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # V8.0 P1 新增字段
            "up_count": up_count,
            "down_count": down_count,
            "flat_count": flat_count,
            "up_down_ratio": safe_round(up_count / max(down_count, 1)),
            "is_frozen": is_frozen
        }
    except Exception as e:
        record_error(str(e))
        return {"market_status": "Correction", "error": str(e), "is_frozen": False}

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
        
        # Risk Logic
        risk_per_share = tech['current_price'] - sig['stop_loss']
        if risk_per_share <= 0: risk_per_share = tech['atr14']
        
        account_risk_money = req.balance * req.risk
        
        if risk_per_share <= 0.0001:
            suggested_shares = 0
        else:
            raw_shares = account_risk_money / risk_per_share / 100
            suggested_shares = int(raw_shares) * 100
            
        if suggested_shares < 100: suggested_shares = 0
        
        # V9.4: Get real stock name and ETF status
        stock_name = get_stock_name(code, market)
        is_etf = detect_etf(code, market)
        
        return {
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "market": market,
            "code": code,
            "name": stock_name,
            "is_etf": is_etf,
            "data_source": "AkShare",
            "signal_type": sig['signal'],
            "trend_score": sig['trend_score'],
            "current_price": tech['current_price'],
            "atr14": tech['atr14'],
            "bias_ma5": tech['bias_ma5'],
            "rsi14": tech['rsi14'],
            "volume_ratio": tech['volume_ratio'],
            "ma_alignment": tech['ma_alignment'],
            "suggested_buy": sig['suggested_buy'],
            "stop_loss": sig['stop_loss'],
            "take_profit": sig['take_profit'],
            "support_level": sig['support_level'],
            "resistance_level": sig['resistance_level'],
            "technical": tech,
            "signal": sig,
            "risk_ctrl": {
                "risk_per_share": safe_round(risk_per_share),
                "suggested_position": suggested_shares
            },
            "prompt_data": {
                "price_info": f"现价: {tech['current_price']}, MA20: {tech['ma20']}",
                "market_stat": market,
                "volume_info": f"量比: {tech['volume_ratio']}, 均线: {tech['ma_alignment']}",
                "levels_info": f"支撑: {sig['support_level']}, 压力: {sig['resistance_level']}"
            }
        }
    except Exception as e:
        logger.error(traceback.format_exc())
        record_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

# --- V8.0 P2: Portfolio Management Models ---
class PositionItem(BaseModel):
    code: str
    market: str = "CN"
    buy_price: float
    current_stop: float
    target_price: float
    shares: int = 0  # 买入手数，用于计算盈亏金额
    record_id: str = ""  # 飞书记录ID，用于回写更新

class PositionCheckRequest(BaseModel):
    positions: list[PositionItem]

class SignalItem(BaseModel):
    code: str
    signal_date: str  # YYYY-MM-DD
    entry_price: float
    stop_loss: float
    take_profit: float
    signal_result: str = "进行中"

class SignalSettleRequest(BaseModel):
    signals: list[SignalItem]

# V8.0 P2: Check Positions Endpoint
@app.post("/check_positions")
def check_positions(req: PositionCheckRequest):
    """
    V8.0 P2: 批量检查持仓状态
    - 判断是否触发止损/止盈
    - 计算移动止损价
    - 返回操作建议
    """
    results = []
    
    for pos in req.positions:
        try:
            code = pos.code
            is_hk = len(str(code)) == 5 or pos.market == "HK"
            
            # 获取最新价
            if is_hk:
                df = DataFetcher.get_hk_share_history(code)
            else:
                df = DataFetcher.get_a_share_history(code)
            
            if df.empty:
                results.append({
                    "code": code,
                    "action": "ERROR",
                    "reason": "无法获取数据",
                    "current_price": None,
                    "new_stop": None
                })
                continue
            
            current_price = float(df['close'].iloc[-1])
            current_stop = pos.current_stop
            target = pos.target_price
            buy_price = pos.buy_price
            
            # 判断状态 (注意: 0 表示"未设定", 跳过对应检查)
            if current_stop > 0 and current_price <= current_stop:
                action = "SELL_STOP"
                reason = f"🔴 触发止损 (现价 {current_price:.2f} ≤ 止损 {current_stop:.2f})"
                pnl = (current_price - buy_price) / buy_price * 100
                new_stop = None
            elif target > 0 and current_price >= target:
                action = "SELL_TARGET"
                reason = f"🟢 触发止盈 (现价 {current_price:.2f} ≥ 目标 {target:.2f})"
                pnl = (current_price - buy_price) / buy_price * 100
                new_stop = None
            else:
                action = "HOLD"
                # 移动止损: 价格上涨时提高止损 (保护7%利润)
                trailing_stop = current_price * 0.93
                new_stop = max(current_stop, trailing_stop) if current_stop > 0 else trailing_stop
                
                if current_stop > 0 and new_stop > current_stop:
                    reason = f"📈 上调止损 ({current_stop:.2f} → {new_stop:.2f})"
                else:
                    reason = f"继续持有 (现价 {current_price:.2f})"
                
                pnl = (current_price - buy_price) / buy_price * 100

            # 计算盈亏金额 (直接使用输入的股数)
            shares = pos.shares if pos.shares > 0 else 0
            pnl_amount = (current_price - buy_price) * shares if shares > 0 else 0
            
            results.append({
                "code": code,
                "current_price": safe_round(current_price),
                "action": action,
                "reason": reason,
                "pnl_percent": safe_round(pnl),
                "pnl_amount": safe_round(pnl_amount),  # 盈亏金额
                "new_stop": safe_round(new_stop) if new_stop else None,
                "record_id": pos.record_id  # 传递飞书记录ID用于回写
            })
            
        except Exception as e:
            logger.error(f"Position check error for {pos.code}: {e}")
            results.append({
                "code": pos.code,
                "action": "ERROR",
                "reason": str(e),
                "current_price": None,
                "new_stop": None
            })
    
    return {"positions": results, "timestamp": datetime.datetime.now().isoformat()}

# V8.0 P2.5: Settle Signals Endpoint
@app.post("/settle_signals")
def settle_signals(req: SignalSettleRequest):
    """
    V8.0 P2.5: 结算历史信号
    - 判断信号是否成功/失败/超时
    - 计算实际收益率
    - 返回结算结果
    """
    results = []
    
    for sig in req.signals:
        # 跳过已结算的信号
        if sig.signal_result != "进行中":
            results.append({
                "code": sig.code,
                "signal_result": sig.signal_result,
                "action": "SKIP",
                "reason": "已结算"
            })
            continue
        
        try:
            code = sig.code
            is_hk = len(str(code)) == 5
            
            # 获取最新价
            if is_hk:
                df = DataFetcher.get_hk_share_history(code)
            else:
                df = DataFetcher.get_a_share_history(code)
            
            if df.empty:
                results.append({
                    "code": code,
                    "signal_result": "进行中",
                    "action": "ERROR",
                    "reason": "无法获取数据"
                })
                continue
            
            current_price = float(df['close'].iloc[-1])
            entry = sig.entry_price
            stop = sig.stop_loss
            target = sig.take_profit
            
            # 计算持仓天数
            try:
                signal_date = datetime.datetime.strptime(sig.signal_date, "%Y-%m-%d")
                days_held = (datetime.datetime.now() - signal_date).days
            except:
                days_held = 0
            
            # 判断结果
            if current_price >= target:
                result = "成功 ✅"
                pnl = (target - entry) / entry * 100
                action = "SETTLED"
            elif current_price <= stop:
                result = "失败 ❌"
                pnl = (stop - entry) / entry * 100
                action = "SETTLED"
            elif days_held > 20:
                result = "超时 ⏰"
                pnl = (current_price - entry) / entry * 100
                action = "SETTLED"
            else:
                result = "进行中 ⏳"
                pnl = (current_price - entry) / entry * 100
                action = "PENDING"
            
            results.append({
                "code": code,
                "signal_result": result,
                "action": action,
                "current_price": safe_round(current_price),
                "pnl_percent": safe_round(pnl),
                "days_held": days_held
            })
            
        except Exception as e:
            logger.error(f"Signal settle error for {sig.code}: {e}")
            results.append({
                "code": sig.code,
                "signal_result": "进行中",
                "action": "ERROR",
                "reason": str(e)
            })
    
    return {"signals": results, "timestamp": datetime.datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

