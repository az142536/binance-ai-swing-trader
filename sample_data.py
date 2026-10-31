"""
歷史行情資料產生器與快取管理 (Sample Data Generator & Local Cache)
提供離線即時模擬資料集 (含波動度叢聚與波段趨勢特徵)，確保離線測試與實盤下載兩者皆能順暢運行。
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict
from config import DATA_DIR

def generate_realistic_crypto_klines(
    num_bars: int = 2400,
    start_price: float = 65000.0,
    interval_seconds: int = 3600
) -> List[Dict[str, float]]:
    """
    使用幾何布朗運動與動態體制轉換 (Regime Switching) 模擬真實加密貨幣 K 線:
    週期性出現強趨勢波段 (Bull Wave)、震盪洗盤 (Chop) 與健康回調 (Pullback)。
    """
    np.random.seed(42)
    bars = []
    
    current_time = int(time.time() * 1000) - (num_bars * interval_seconds * 1000)
    current_price = start_price

    # 體制狀態: 0=震盪, 1=強多頭波段, 2=弱回調
    regime = 0
    regime_duration = 0

    for _ in range(num_bars):
        if regime_duration <= 0:
            regime = np.random.choice([0, 1, 2], p=[0.45, 0.35, 0.20])
            regime_duration = np.random.randint(24, 96) # 每個體制持續 1~4 天
        
        regime_duration -= 1

        if regime == 1:
            drift = 0.0012      # 多頭波段每小時漂移 +0.12%
            volatility = 0.012  # 波動度
        elif regime == 2:
            drift = -0.0008     # 回調每小時漂移 -0.08%
            volatility = 0.010
        else:
            drift = 0.00005     # 橫盤整理
            volatility = 0.006

        shock = np.random.normal(0, 1)
        ret = drift + volatility * shock
        close_p = current_price * (1.0 + ret)
        
        # 產生 High, Low, Open
        intra_vol = volatility * current_price * np.random.uniform(0.6, 1.8)
        high_p = max(current_price, close_p) + abs(np.random.normal(0, intra_vol * 0.5))
        low_p = min(current_price, close_p) - abs(np.random.normal(0, intra_vol * 0.5))
        open_p = current_price

        # 成交量在波段突破時放大
        base_vol = 1500.0
        vol_mult = 2.5 if (regime == 1 and abs(ret) > 0.01) else np.random.uniform(0.7, 1.4)
        volume = base_vol * vol_mult * (1.0 + abs(ret) * 10)

        bars.append({
            "timestamp": current_time,
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
            "volume": round(volume, 2)
        })

        current_price = close_p
        current_time += interval_seconds * 1000

    return bars

def save_klines_to_json(bars: List[Dict[str, float]], file_path: Path):
    """保存 K 線到 JSON 快取"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(bars, f, ensure_ascii=False, indent=2)

def load_klines_from_json(file_path: Path) -> List[Dict[str, float]]:
    """從 JSON 快取載入 K 線"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

