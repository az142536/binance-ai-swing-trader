"""
特徵工程與波段標註模組 (Feature Engineering & Swing Wave Labeling)
將原始 OHLCV 轉換為 AI 模型輸入特徵，並使用三重屏障法 (Triple Barrier) 定義高盈虧比趨勢波段。
"""

import numpy as np
from typing import Dict, List, Tuple
from config import Config
from indicators import (
    calculate_ema,
    calculate_sma,
    calculate_macd,
    calculate_rsi,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_adx
)

FEATURE_NAMES = [
    "ema_9_ratio",
    "ema_21_ratio",
    "ema_50_ratio",
    "ema_200_ratio",
    "trend_alignment_score",
    "ema_50_slope",
    "macd_hist_ratio",
    "macd_hist_diff",
    "rsi_norm",
    "atr_ratio",
    "bb_pct_b",
    "bb_bandwidth",
    "adx_norm",
    "di_diff",
    "volume_ratio",
    "candle_body_ratio",
    "return_1",
    "return_3",
    "return_5"
]

def extract_features(bars: List[Dict[str, float]]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    從 K 線資料中計算所有技術指標與 AI 特徵矩陣
    回傳:
      X: (N, num_features) 特徵矩陣
      indicators_cache: 計算好的原始指標字典 (供策略止損止盈使用)
    """
    n = len(bars)
    if n < 220:
        raise ValueError(f"K 線數量不足 (當前 {n} 根，至少需要 220 根以計算 200 EMA 及 ADX 指標)")

    open_p = np.array([b["open"] for b in bars], dtype=float)
    high_p = np.array([b["high"] for b in bars], dtype=float)
    low_p = np.array([b["low"] for b in bars], dtype=float)
    close_p = np.array([b["close"] for b in bars], dtype=float)
    volume_p = np.array([b["volume"] for b in bars], dtype=float)

    # 1. 均線系統
    ema9 = calculate_ema(close_p, 9)
    ema21 = calculate_ema(close_p, 21)
    ema50 = calculate_ema(close_p, 50)
    ema200 = calculate_ema(close_p, 200)

    # 2. MACD
    macd, macd_sig, macd_hist = calculate_macd(close_p, 12, 26, 9)

    # 3. RSI
    rsi = calculate_rsi(close_p, 14)

    # 4. ATR
    atr = calculate_atr(high_p, low_p, close_p, 14)

    # 5. 布林通道
    bb_upper, bb_mid, bb_lower, bb_bw, bb_pct_b = calculate_bollinger_bands(close_p, 20, 2.0)

    # 6. ADX 趨勢強度
    adx, plus_di, minus_di = calculate_adx(high_p, low_p, close_p, 14)

    # 7. 成交量均線
    vol_sma20 = calculate_sma(volume_p, 20)

    # 構建特徵
    f_ema9 = (close_p - ema9) / (ema9 + 1e-8)
    f_ema21 = (close_p - ema21) / (ema21 + 1e-8)
    f_ema50 = (close_p - ema50) / (ema50 + 1e-8)
    f_ema200 = (close_p - ema200) / (ema200 + 1e-8)

    # 多頭排列評分: 越接近多頭 (9 > 21 > 50 > 200) 分數越高 (+3 到 -3)
    score = np.zeros(n, dtype=float)
    score += np.where(ema9 > ema21, 1.0, -1.0)
    score += np.where(ema21 > ema50, 1.0, -1.0)
    score += np.where(ema50 > ema200, 1.0, -1.0)
    f_trend_score = score / 3.0

    # EMA 50 斜率 (最近 5 根)
    ema50_slope = np.zeros(n, dtype=float)
    for i in range(5, n):
        if not np.isnan(ema50[i - 5]) and ema50[i - 5] > 0:
            ema50_slope[i] = (ema50[i] - ema50[i - 5]) / ema50[i - 5] * 100.0

    # MACD 特徵
    f_macd_hist = macd_hist / (close_p + 1e-8)
    f_macd_diff = np.zeros(n, dtype=float)
    f_macd_diff[1:] = macd_hist[1:] - macd_hist[:-1]

    # RSI 標準化 (-1 到 +1)
    f_rsi = (rsi - 50.0) / 50.0

    # ATR 比率
    f_atr = atr / (close_p + 1e-8)

    # ADX 與 DI 差值
    f_adx = adx / 100.0
    f_di_diff = (plus_di - minus_di) / 100.0

    # 成交量放大倍數
    f_vol_ratio = volume_p / (vol_sma20 + 1e-8)
    f_vol_ratio = np.clip(f_vol_ratio, 0.0, 5.0)

    # K 棒實體比例 (安全除法避免除以零警告)
    candle_range = high_p - low_p
    safe_range = np.where(candle_range > 1e-8, candle_range, 1.0)
    f_body_ratio = np.where(candle_range > 1e-8, np.abs(close_p - open_p) / safe_range, 0.5)

    # 多週期報酬率
    ret1 = np.zeros(n, dtype=float)
    ret3 = np.zeros(n, dtype=float)
    ret5 = np.zeros(n, dtype=float)
    ret1[1:] = (close_p[1:] - close_p[:-1]) / close_p[:-1]
    ret3[3:] = (close_p[3:] - close_p[:-3]) / close_p[:-3]
    ret5[5:] = (close_p[5:] - close_p[:-5]) / close_p[:-5]

    # 組合為 (N, len(FEATURE_NAMES))
    X = np.column_stack([
        f_ema9,
        f_ema21,
        f_ema50,
        f_ema200,
        f_trend_score,
        ema50_slope,
        f_macd_hist,
        f_macd_diff,
        f_rsi,
        f_atr,
        bb_pct_b,
        bb_bw,
        f_adx,
        f_di_diff,
        f_vol_ratio,
        f_body_ratio,
        ret1,
        ret3,
        ret5
    ])

    indicators_cache = {
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close_p,
        "volume": volume_p,
        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,
        "ema200": ema200,
        "macd": macd,
        "macd_sig": macd_sig,
        "macd_hist": macd_hist,
        "rsi": rsi,
        "atr": atr,
        "bb_upper": bb_upper,
        "bb_mid": bb_mid,
        "bb_lower": bb_lower,
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di
    }

    return X, indicators_cache

def create_swing_wave_labels(
    bars: List[Dict[str, float]],
    indicators: Dict[str, np.ndarray],
    forward_bars: int = Config.TARGET_FORWARD_BARS,
    profit_mult: float = Config.TARGET_PROFIT_RATIO,
    stop_mult: float = Config.STOP_LOSS_ATR_MULT
) -> np.ndarray:
    """
    使用「三重屏障法 (Triple-Barrier Method)」標註高勝率趨勢波段：
    向前看 forward_bars 根 K 線 (預設 24 根)：
    - 上邊界: entry_price + profit_mult * ATR (達到目標大波段利潤，預設 3.0x ATR)
    - 下邊界: entry_price - stop_mult * ATR (跌破防守止損位，預設 1.8x ATR)
    若在上邊界先觸發且未打到止損，標註為 1 (高盈虧比優質上漲波段)；否則標註為 0。
    """
    n = len(bars)
    y = np.zeros(n, dtype=int)
    close_p = indicators["close"]
    high_p = indicators["high"]
    low_p = indicators["low"]
    atr = indicators["atr"]

    for i in range(200, n - forward_bars):
        curr_price = close_p[i]
        curr_atr = atr[i]
        if np.isnan(curr_atr) or curr_atr <= 0:
            continue

        take_profit_price = curr_price + profit_mult * curr_atr
        stop_loss_price = curr_price - stop_mult * curr_atr

        triggered_tp = False
        triggered_sl = False

        for k in range(1, forward_bars + 1):
            future_high = high_p[i + k]
            future_low = low_p[i + k]

            # 檢驗是否觸及止損
            if future_low <= stop_loss_price:
                triggered_sl = True
                break

            # 檢驗是否觸及止盈目標
            if future_high >= take_profit_price:
                triggered_tp = True
                break

        if triggered_tp and not triggered_sl:
            y[i] = 1
        else:
            y[i] = 0

    return y

