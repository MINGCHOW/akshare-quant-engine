import logging
import time
from api.fetcher import DataFetcher
import akshare as ak

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hk_index_stable():
    print("\n--- Testing HSI Index (Stable Method) ---")
    try:
        df = ak.index_zh_a_hist(symbol="HSI", period="daily")
        if not df.empty:
            price = df['收盘'].iloc[-1]
            print(f"✅ Success! HSI Price: {price}")
        else:
            print("❌ Empty DataFrame")
    except Exception as e:
        print(f"❌ Failed: {e}")

def test_data_fetcher_cache():
    print("\n--- Testing DataFetcher Cache & Realtime ---")
    
    # 1. Warm up cache (CN)
    start = time.time()
    price = DataFetcher.get_realtime_price("000001", "CN")
    print(f"CN 000001 Price: {price}, Time: {time.time() - start:.2f}s")
    
    # 2. Cache Hit (CN)
    start = time.time()
    price = DataFetcher.get_realtime_price("600000", "CN")
    print(f"CN 600000 Price: {price}, Time: {time.time() - start:.2f}s (Should be fast)")
    
    # 3. Code to Name (CN)
    name = DataFetcher.get_stock_name("600000", "CN")
    print(f"CN 600000 Name: {name}")

    # 4. Warm up cache (HK)
    start = time.time()
    price_hk = DataFetcher.get_realtime_price("00700", "HK")
    print(f"HK 00700 Price: {price_hk}, Time: {time.time() - start:.2f}s")
    
    # 5. Cache Hit (HK)
    start = time.time()
    price_hk = DataFetcher.get_realtime_price("09988", "HK")
    print(f"HK 09988 Price: {price_hk}, Time: {time.time() - start:.2f}s (Should be fast)")

    # 6. Code to Name (HK)
    name_hk = DataFetcher.get_stock_name("00700", "HK")
    print(f"HK 00700 Name: {name_hk}")

if __name__ == "__main__":
    test_hk_index_stable()
    test_data_fetcher_cache()
