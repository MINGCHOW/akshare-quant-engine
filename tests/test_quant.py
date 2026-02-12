import sys
import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.quant import detect_etf, get_stock_name, calculate_technicals, generate_signal, safe_round

class TestQuantLogic:
    
    # --- ETF Detection Tests ---
    def test_detect_etf_a_share(self):
        # Known ETFs
        assert detect_etf("510050", "CN") == True
        assert detect_etf("159915", "CN") == True
        # Stocks
        assert detect_etf("600000", "CN") == False
        assert detect_etf("000001", "CN") == False

    @patch('api.quant.get_stock_name')
    def test_detect_etf_hk_share(self, mock_get_name):
        mock_get_name.return_value = "腾讯控股"
        
        # HK ETFs (Code Range)
        assert detect_etf("02800", "HK") == True
        assert detect_etf("03033", "HK") == True
        
        # HK Stocks (Not ETF)
        assert detect_etf("00700", "HK") == False
        assert detect_etf("09988", "HK") == False
        
        # HK ETF by Name
        mock_get_name.return_value = "南方恒生科技ETF"
        assert detect_etf("03033", "HK") == True

    # --- Technical Indicators Tests ---
    def test_calculate_technicals_basic(self):
        # Create dummy dataframe
        dates = pd.date_range(start="2024-01-01", periods=30)
        data = {
            'date': dates,
            'open': [10] * 30,
            'high': [11] * 30,
            'low': [9] * 30,
            'close': [10.0] * 30,
            'volume': [1000] * 30
        }

        df = pd.DataFrame(data)
        
        # Add some trend to create MACD
        for i in range(15, 30):
            df.loc[i, 'close'] = 10 + (i-15)*0.5 # 10, 10.5, 11...
            
        tech = calculate_technicals(df)
        
        assert "macd" in tech
        assert "macd_signal" in tech
        assert "macd_hist" in tech
        assert "atr14" in tech
        assert tech['current_price'] == df['close'].iloc[-1]

    # --- Signal Generation Tests ---
    def test_generate_signal_bull(self):
        # Mock tech data for a strong bull case
        tech = {
            'current_price': 100,
            'ma5': 95,
            'ma10': 90,
            'ma20': 85,  # p > ma5 > ma20
            'ma60': 80,
            'ma_alignment': "多头排列",
            'rsi14': 60, # Neutral
            'macd_cross': 'golden', # Golden cross
            'macd_hist': 1.5,
            'volume_ratio': 2.5, # High volume
            'support_level': 90,
            'resistance_level': 110,
            'atr14': 2.0
        }
        
        sig = generate_signal(tech, is_hk=False)
        
        # Expect High Score
        assert sig['trend_score'] > 60
        assert "买入" in sig['signal']
        assert "MACD金叉 🔥" in sig['signal_reasons']
        
        # Check Stop Loss (ATR driven)
        # Price 100, ATR 2.0. A-share multiplier 2.0
        # ATR Stop = 100 - 4 = 96
        # Support Stop = 90 * 0.98 = 88.2
        # Should take max(96, 88.2) = 96
        assert sig['stop_loss'] == 96.0

    def test_generate_signal_bear(self):
        # Mock tech data for a bear case
        tech = {
            'current_price': 80,
            'ma5': 85,
            'ma20': 90,  # p < ma5 < ma20
            'macd_cross': 'death',
            'macd_hist': -1.5,
            'rsi14': 40,
            'atr14': 2.0,
            'support_level': 70
        }
        
        sig = generate_signal(tech, is_hk=False)
        
        # Expect Low Score
        assert sig['trend_score'] < 50
        assert "MACD死叉 ⚠️" in sig['signal_reasons']

if __name__ == "__main__":
    pytest.main()
