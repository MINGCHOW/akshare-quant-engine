# -*- coding: utf-8 -*-
"""
V10.0 Data Fetcher Module
8-Layer Fallback for A-Share + 4-Layer for HK
"""
import pandas as pd
import akshare as ak
import requests
import datetime
import random
import time
import logging

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Optional libraries
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

logger = logging.getLogger(__name__)

# V10.0: Static User-Agent Pool (Remove fake_useragent dependency)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36"
]

def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "*/*",
        "Connection": "keep-alive"
    }


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
                df['date'] = df['date'].dt.tz_convert(None)
            
            # 5. Deduplicate and Sort
            df.drop_duplicates(subset=['date'], keep='last', inplace=True)
            df.sort_values('date', inplace=True)
            
            # 6. Enforce numeric types
            cols = ['open', 'close', 'high', 'low', 'volume']
            for col in cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(subset=['open', 'close'], inplace=True)
            
            return df[list(required)]
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
                df = ef.stock.get_quote_history(symbol)
                if not df.empty and len(df) > 30:
                    return DataFetcher._clean_data(df)
            except Exception as e:
                logger.warning(f"efinance failed: {e}")

        # 1. AkShare (EastMoney)
        try:
             df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
             if not df.empty and len(df) > 30:
                 df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', 
                                    '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
                 return DataFetcher._clean_data(df)
        except Exception as e:
             logger.warning(f"AkShare failed: {e}")
             try:
                 time.sleep(1)
                 df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
                 if not df.empty: return DataFetcher._clean_data(df)
             except Exception:
                 pass

        # 2. Tencent (HTTP)
        try:
            full_code = f"{market_prefix}{symbol}"
            url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={full_code},day,,,320,qfq" 
            r = requests.get(url, headers=get_headers(), timeout=8)
            data = r.json()
            if data and 'data' in data and full_code in data['data']:
                qt_data = data['data'][full_code]
                if 'day' in qt_data:
                    k_data = qt_data['day']
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
                    
                    df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', 
                                       '最高': 'high', '最低': 'low', '成交量': 'volume', '成交': 'volume'}, inplace=True)
                    
                    return DataFetcher._clean_data(df[['date', 'open', 'close', 'high', 'low', 'volume']])
            except Exception as e:
                logger.warning(f"Qstock failed: {e}")

        # 4. Pytdx (TCP) - Thread Safe Version
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
                
                bs.logout()
                
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

            symbol = f"{int(clean_code):05d}"
            
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
                tencent_code = f"hk{symbol}"
                url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={tencent_code},day,,,320,qfq" 
                r = requests.get(url, headers=get_headers(), timeout=8)
                data = r.json()
                if data and 'data' in data and tencent_code in data['data']:
                    qt_data = data['data'][tencent_code]
                    if 'day' in qt_data:
                        k_data = qt_data['day']
                        df = pd.DataFrame(k_data)
                        if df.shape[1] >= 6:
                            df = df.iloc[:, :6]
                            df.columns = ['date', 'open', 'close', 'high', 'low', 'volume']
                            return DataFetcher._clean_data(df)
            except Exception as e:
                logger.warning(f"Tencent HK failed: {e}")

            # 3. Try Yahoo Finance (International) - Best for HK
            if yf:
                try:
                    logger.info(f"Attempting Yahoo HK (#3) for {code}...")
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

    # --- V10.0 Real-time Spot Cache ---
    _spot_cache = {
        "CN": {"data": pd.DataFrame(), "time": 0},
        "HK": {"data": pd.DataFrame(), "time": 0}
    }

    @staticmethod
    def get_realtime_price(code: str, market: str = "CN") -> float:
        """
        V10.0: Get real-time spot price efficiently with caching (TTL 30s)
        """
        try:
            now = time.time()
            cache = DataFetcher._spot_cache[market]
            
            # Refresh cache if empty or older than 30s
            if cache["data"].empty or (now - cache["time"] > 30):
                if market == "HK":
                     df = ak.stock_hk_spot_em()
                else:
                     df = ak.stock_zh_a_spot_em()
                
                if not df.empty:
                    # Optimize: Index by code for O(1) lookup
                    if '代码' in df.columns and '最新价' in df.columns:
                         df['代码'] = df['代码'].astype(str)
                         # Standardize codes
                         if market == "HK":
                             df['代码'] = df['代码'].apply(lambda x: f"{int(x):05d}" if x.isdigit() else x)
                         cache["data"] = df.set_index('代码')
                         cache["time"] = now
            
            # Lookup
            df = cache["data"]
            clean_code = str(code).strip().upper().replace("HK","").replace("SH","").replace("SZ","")
            if market == "HK":
                 clean_code = f"{int(clean_code):05d}" if clean_code.isdigit() else clean_code
            
            if clean_code in df.index:
                price = df.loc[clean_code]['最新价']
                return float(price) if price else 0.0
                
        except Exception as e:
            logger.warning(f"AkShare Spot fetch failed for {code}: {e}")
        
        # --- V10.2: Yahoo Finance Fallback (HK Only) ---
        if market == "HK" and yf:
            try:
                # Yahoo requires .HK suffix and 4-digit code usually? No, 0700.HK
                # AkShare uses 00700. Yahoo uses 0700.HK or 00700.HK
                clean_code = str(code).strip()
                if clean_code.isdigit():
                    yf_code = f"{int(clean_code):04d}.HK"
                else:
                    yf_code = clean_code
                
                ticker = yf.Ticker(yf_code)
                # Try fast_info (newer yfinance)
                try:
                    price = ticker.fast_info['last_price']
                    if price and price > 0: return float(price)
                except:
                    # Fallback to history
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        return float(hist['Close'].iloc[-1])
            except Exception as e:
                logger.warning(f"YFinance fallback failed for {code}: {e}")

        return 0.0

    @staticmethod
    def get_stock_name(code: str, market: str = "CN") -> str:
        """
        V10.0: Get stock name from cached spot data
        Same cache as get_realtime_price (TTL 30s works for names too)
        """
        try:
            DataFetcher.get_realtime_price(code, market) # Trigger cache refresh
            cache = DataFetcher._spot_cache[market]["data"]
            
            clean_code = str(code).strip().upper().replace("HK","").replace("SH","").replace("SZ","")
            if market == "HK":
                 clean_code = f"{int(clean_code):05d}" if clean_code.isdigit() else clean_code
            
            if clean_code in cache.index:
                return str(cache.loc[clean_code]['名称'])
                
        except Exception:
            pass
        
        # --- V10.2 YFinance Name Fallback ---
        if market == "HK" and yf:
            try:
                clean_code = str(code).strip()
                yf_code = f"{int(clean_code):04d}.HK" if clean_code.isdigit() else clean_code
                ticker = yf.Ticker(yf_code)
                # Note: Yahoo names are English usually, but better than code
                name = ticker.info.get('shortName') or ticker.info.get('longName')
                if name: return name
            except:
                pass

        return code
