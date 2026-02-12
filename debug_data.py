import akshare as ak
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hk_index():
    print("\n--- Testing HK Index (HSI) ---")
    
    # Method 1: stock_hk_index_daily_em
    try:
        print("Method 1: ak.stock_hk_index_daily_em(symbol='HSI')")
        df = ak.stock_hk_index_daily_em(symbol="HSI")
        if not df.empty:
            print(f"Success! Last close: {df['close'].iloc[-1]}")
        else:
            print("Empty DataFrame returned")
    except Exception as e:
        print(f"Failed: {e}")

    # Method 2: index_zh_a_hist (fallback)
    try:
        print("Method 2: ak.index_zh_a_hist(symbol='HSI')")
        df = ak.index_zh_a_hist(symbol="HSI", period="daily")
        if not df.empty:
            print(f"Success! Last close: {df['收盘'].iloc[-1]}")
        else:
            print("Empty DataFrame returned")
    except Exception as e:
        print(f"Failed: {e}")
        
    # Method 3: stock_hk_index_spot_em (realtime?)
    try:
        print("Method 3: ak.stock_hk_index_spot_em()")
        df = ak.stock_hk_index_spot_em()
        if not df.empty:
            # Look for HSI or 恒生指数
            match = df[df['名称'].str.contains('恒生指数')]
            if not match.empty:
                print(f"Success! Found HSI in spot list: {match.iloc[0].to_dict()}")
            else:
                print("HSI not found in spot list")
        else:
            print("Empty spot list")
    except Exception as e:
        print(f"Failed: {e}")

def test_stock_name():
    print("\n--- Testing Stock Name ---")
    codes = ["00700", "02800", "09988"]
    try:
        df = ak.stock_hk_spot_em()
        for code in codes:
            try:
                symbol = f"{int(code):05d}"
                match = df[df['代码'] == symbol]
                if not match.empty:
                    print(f"Code {code}: {match.iloc[0]['名称']}")
                else:
                    print(f"Code {code}: Not found")
            except Exception as e:
                print(f"Code {code} error: {e}")
    except Exception as e:
        print(f"HK Spot fetch failed: {e}")

if __name__ == "__main__":
    test_hk_index()
    test_stock_name()
