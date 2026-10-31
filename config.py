"""
幣安 AI 趨勢波段自動交易系統 - 設定模組
Binance AI Swing Trader Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 專案路徑
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 載入 .env
load_dotenv(BASE_DIR / ".env")

class Config:
    # 幣安 API 設定
    API_KEY = os.getenv("BINANCE_API_KEY", "")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    
    # 是否使用測試網 (強烈建議初學者預設為 True，零風險測試)
    USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() in ("true", "1", "yes")
    
    # API 端點
    MAINNET_REST_URL = "https://api.binance.com"
    TESTNET_REST_URL = "https://testnet.binance.vision"
    
    @classmethod
    def get_base_url(cls) -> str:
        return cls.TESTNET_REST_URL if cls.USE_TESTNET else cls.MAINNET_REST_URL

    # 交易參數
    DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "BTCUSDT")
    DEFAULT_INTERVAL = os.getenv("DEFAULT_INTERVAL", "1h")   # 趨勢波段建議使用 1h 或 4h
    
    # 資金與風控設定
    INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100.0"))  # 模擬初始資金 (預設 100 USDT)
    RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))       # 每筆交易最大承擔風險 (2%)
    MAX_POSITION_RATIO = float(os.getenv("MAX_POSITION_RATIO", "0.95")) # 最高動用資金比例
    SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0005"))       # 滑點模擬 (0.05%)
    FEE_RATE = float(os.getenv("FEE_RATE", "0.00075"))                # 幣安手續費 (使用 BNB 抵扣約 0.075%)

    # AI 趨勢波段策略參數
    # 策略風格模式: "active" (敏捷波段模式，不空等慢速均線，短線動能確認即進場做波段) 或 "sniper" (穩健狙擊模式，需站上 200 EMA)
    SWING_MODE = os.getenv("SWING_MODE", "active").lower()
    AI_CONFIDENCE_THRESHOLD = float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.65")) # AI 信心閥值 (65%)
    STOP_LOSS_ATR_MULT = float(os.getenv("STOP_LOSS_ATR_MULT", "1.8"))             # 初始保護止損 (1.8x ATR，給予充分波動空間)
    TAKE_PROFIT_ATR_MULT = float(os.getenv("TAKE_PROFIT_ATR_MULT", "3.0"))         # 目標大波段止盈 (3.0x ATR)
    TRAILING_STOP_ATR_MULT = float(os.getenv("TRAILING_STOP_ATR_MULT", "2.0"))     # 移動止盈回撤空間 (2.0x ATR)
    
    # 特徵工程前瞻步數 (向前看 24 根 K 線，確認走出大於 3.0x ATR 的實質大波段)
    TARGET_FORWARD_BARS = int(os.getenv("TARGET_FORWARD_BARS", "24"))
    TARGET_PROFIT_RATIO = float(os.getenv("TARGET_PROFIT_RATIO", "3.0"))

