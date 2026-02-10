# -*- coding: utf-8 -*-
import os
import time
import datetime
import traceback
import logging
import math
import random
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# V10.0 Modular Imports
from .fetcher import DataFetcher
from .quant import (
    calculate_technicals, 
    generate_signal, 
    detect_etf, 
    safe_round, 
    get_stock_name
)

# Optional libraries for health check only
try:
    import efinance as ef
except ImportError:
    ef = None
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
except ImportError:
    bs = None
try:
    import qstock as qs
except ImportError:
    qs = None

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AkShare Quant API V10.0", version="10.0")

# --- V10.0: API Key Authentication ---
API_KEY = os.environ.get("API_KEY", "aqe-k8x7m2pQ9vR4wL6nJ3sY5tB1")
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/health/reset"}

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    
    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if api_key != API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API Key"})
    
    return await call_next(request)

# --- Circuit Breaker ---
error_counter = {
    "count": 0, 
    "last_error": None,
    "last_reset": datetime.datetime.now(),
    "circuit_open": False
}

def send_emergency_alert(error_msg: str):
    logger.critical(f"🔴 CIRCUIT BREAKER TRIGGERED: {error_msg}")

def reset_circuit_breaker():
    global error_counter
    error_counter["count"] = 0
    error_counter["circuit_open"] = False
    error_counter["last_reset"] = datetime.datetime.now()

def record_error(error_msg: str):
    global error_counter
    error_counter["count"] += 1
    error_counter["last_error"] = error_msg
    
    if error_counter["count"] >= 3 and not error_counter["circuit_open"]:
        error_counter["circuit_open"] = True
        logger.error(f"Circuit breaker opened due to: {error_msg}")

def record_success():
    global error_counter
    if error_counter["count"] > 0:
        error_counter["count"] = 0
        error_counter["circuit_open"] = False

# --- Models ---
class AnalyzeRequest(BaseModel):
    code: str
    balance: float = 100000.0
    risk: float = 0.01

class PositionItem(BaseModel):
    code: str
    market: str = "CN"
    buy_price: float
    current_stop: float
    target_price: float
    shares: int = 0
    record_id: str = ""

class PositionCheckRequest(BaseModel):
    positions: list[PositionItem]

class SignalItem(BaseModel):
    code: str
    signal_date: str
    entry_price: float
    stop_loss: float
    take_profit: float
    signal_result: str = "进行中"

class SignalSettleRequest(BaseModel):
    signals: list[SignalItem]

# --- Endpoints ---

@app.get("/health")
def health_check():
    """
    V10.0: 系统健康检查
    """
    start_time = time.time()
    checks = {}
    overall_status = "healthy"
    
    # 1. 数据源检查
    try:
        test_df = DataFetcher.get_a_share_history("000001")
        if test_df.empty:
            checks["data_source"] = {"status": "warning", "message": "Empty data returned"}
            overall_status = "degraded"
        else:
            checks["data_source"] = {"status": "ok", "rows": len(test_df)}
            record_success()
    except Exception as e:
        checks["data_source"] = {"status": "error", "message": str(e)}
        overall_status = "degraded"
        record_error(str(e))
    
    # 2. 熔断器状态
    checks["circuit_breaker"] = {
        "error_count": error_counter["count"],
        "is_open": error_counter["circuit_open"],
        "last_error": error_counter["last_error"]
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
        "version": "10.0"
    }

@app.post("/health/reset")
def reset_health():
    reset_circuit_breaker()
    return {"status": "ok", "message": "Circuit breaker reset"}

@app.get("/market")
def get_market_context():
    """
    V10.0: 增强版大盘状态 (A股 + 港股 + 涨跌家数)
    """
    try:
        time.sleep(random.uniform(0.5, 1.0))
        import akshare as ak
        
        # A股指数数据
        index_df = ak.stock_zh_index_daily(symbol="sh000001")
        if index_df.empty:
            raise ValueError("Index Data Empty")
        price = float(index_df['close'].iloc[-1])
        ma20 = float(index_df['close'].rolling(20).mean().iloc[-1])
        cn_status = "Bull" if price > ma20 else "Bear"
        
        # V10.0: 港股市场判断 (恒生指数)
        hk_status = "Unknown"
        hk_price = 0
        hk_ma20 = 0
        try:
            hk_df = ak.stock_hk_index_daily_em(symbol="HSI")
            if not hk_df.empty:
                hk_price = float(hk_df['close'].iloc[-1])
                hk_ma20 = float(hk_df['close'].rolling(20).mean().iloc[-1])
                hk_status = "Bull" if hk_price > hk_ma20 else "Bear"
        except Exception as e:
            logger.warning(f"HK index fetch failed: {e}")
            try:
                hk_df = ak.index_zh_a_hist(symbol="HSI", period="daily")
                if not hk_df.empty:
                    hk_price = float(hk_df['收盘'].iloc[-1])
                    hk_ma20 = float(hk_df['收盘'].rolling(20).mean().iloc[-1])
                    hk_status = "Bull" if hk_price > hk_ma20 else "Bear"
            except Exception:
                pass
        
        # 涨跌家数统计
        up_count, down_count, flat_count = 0, 0, 0
        try:
            stats = ak.stock_zh_a_spot_em()
            if not stats.empty and '涨跌幅' in stats.columns:
                up_count = len(stats[stats['涨跌幅'] > 0])
                down_count = len(stats[stats['涨跌幅'] < 0])
                flat_count = len(stats[stats['涨跌幅'] == 0])
        except Exception as e:
            logger.warning(f"Failed to get up/down count: {e}")
        
        # 市场冰点
        is_frozen = up_count > 0 and up_count < 800
        if is_frozen:
            cn_status = "Crash" if up_count < 500 else "Bear"
        
        market_status = cn_status
        
        return {
            "market_status": market_status,
            "cn_status": cn_status,
            "hk_status": hk_status,
            "index_price": safe_round(price),
            "ma20": safe_round(ma20),
            "hk_index_price": safe_round(hk_price),
            "hk_ma20": safe_round(hk_ma20),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
            raise ValueError("No data found")
            
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
            "macd": tech.get('macd', 0),
            "macd_signal": tech.get('macd_signal', 0),
            "macd_hist": tech.get('macd_hist', 0),
            "macd_cross": tech.get('macd_cross', 'none'),
            "signal_reasons": sig.get('signal_reasons', []),
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
                "levels_info": f"支撑: {sig['support_level']}, 压力: {sig['resistance_level']}",
                "macd_info": f"MACD: {tech.get('macd', 0)}, 信号线: {tech.get('macd_signal', 0)}, 柱状: {tech.get('macd_hist', 0)}, 交叉: {tech.get('macd_cross', 'none')}"
            }
        }
    except Exception as e:
        logger.error(traceback.format_exc())
        record_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check_positions")
def check_positions(req: PositionCheckRequest):
    results = []
    for pos in req.positions:
        try:
            code = pos.code
            is_hk = len(str(code)) == 5 or pos.market == "HK"
            
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
            
            # V10.0: 计算 ATR
            tech = calculate_technicals(df)
            atr = tech.get('atr14', 0)
            if not atr or atr <= 0:
                atr = current_price * 0.03
            
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
                # V10.0: ATR 驱动移动止损
                atr_multiplier = 2.5 if is_hk else 2.0
                trailing_stop = current_price - (atr_multiplier * atr)
                min_trailing = buy_price * 0.93
                trailing_stop = max(trailing_stop, min_trailing)
                
                new_stop = max(current_stop, trailing_stop) if current_stop > 0 else trailing_stop
                
                if current_stop > 0 and new_stop > current_stop:
                    reason = f"📈 上调止损 ({current_stop:.2f} → {new_stop:.2f})"
                else:
                    reason = f"继续持有 (现价 {current_price:.2f})"
                
                pnl = (current_price - buy_price) / buy_price * 100

            shares = pos.shares if pos.shares > 0 else 0
            pnl_amount = (current_price - buy_price) * shares if shares > 0 else 0
            
            results.append({
                "code": code,
                "current_price": safe_round(current_price),
                "action": action,
                "reason": reason,
                "pnl_percent": safe_round(pnl),
                "pnl_amount": safe_round(pnl_amount),
                "new_stop": safe_round(new_stop) if new_stop else None,
                "record_id": pos.record_id
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

@app.post("/settle_signals")
def settle_signals(req: SignalSettleRequest):
    results = []
    
    for sig in req.signals:
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
            
            try:
                signal_date = datetime.datetime.strptime(sig.signal_date, "%Y-%m-%d")
                days_held = (datetime.datetime.now() - signal_date).days
            except:
                days_held = 0
            
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
