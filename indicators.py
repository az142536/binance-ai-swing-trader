"""
向量化技術指標計算模組 (純 NumPy 實現，高效且零額外 C 依賴)
包含趨勢指標 (EMA, MACD, ADX)、動能指標 (RSI)、波動度指標 (ATR, 布林通道) 及成交量比率。
"""

import numpy as np
from typing import Tuple

def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """計算指數移動平均線 (EMA)"""
    alpha = 2.0 / (period + 1.0)
    ema = np.empty_like(data, dtype=float)
    ema[:] = np.nan
    if len(data) < period:
        return ema
    
    # 初始值使用前 period 筆的 SMA
    ema[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        ema[i] = alpha * data[i] + (1.0 - alpha) * ema[i - 1]
    return ema

def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """計算簡單移動平均線 (SMA)"""
    sma = np.empty_like(data, dtype=float)
    sma[:] = np.nan
    if len(data) < period:
        return sma
    cumsum = np.cumsum(np.insert(data, 0, 0))
    sma[period - 1:] = (cumsum[period:] - cumsum[:-period]) / float(period)
    return sma

def calculate_macd(
    close: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    計算 MACD 指標
    回傳: (macd_line, signal_line, histogram)
    """
    fast_ema = calculate_ema(close, fast_period)
    slow_ema = calculate_ema(close, slow_period)
    macd_line = fast_ema - slow_ema
    
    # 計算信號線
    valid_idx = np.where(~np.isnan(macd_line))[0]
    signal_line = np.empty_like(close, dtype=float)
    signal_line[:] = np.nan
    
    if len(valid_idx) >= signal_period:
        valid_macd = macd_line[valid_idx]
        sig_valid = calculate_ema(valid_macd, signal_period)
        signal_line[valid_idx] = sig_valid

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """計算相對強弱指標 (RSI)"""
    rsi = np.empty_like(close, dtype=float)
    rsi[:] = np.nan
    n = len(close)
    if n <= period:
        return rsi

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi

def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """計算真實波動區間 (ATR)"""
    n = len(close)
    atr = np.empty(n, dtype=float)
    atr[:] = np.nan
    if n <= period:
        return atr

    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr

def calculate_bollinger_bands(
    close: np.ndarray,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    計算布林通道 (Bollinger Bands)
    回傳: (upper, middle, lower, bandwidth, pct_b)
    """
    n = len(close)
    upper = np.empty(n, dtype=float)
    middle = np.empty(n, dtype=float)
    lower = np.empty(n, dtype=float)
    bandwidth = np.empty(n, dtype=float)
    pct_b = np.empty(n, dtype=float)
    
    upper[:] = np.nan
    middle[:] = np.nan
    lower[:] = np.nan
    bandwidth[:] = np.nan
    pct_b[:] = np.nan

    if n < period:
        return upper, middle, lower, bandwidth, pct_b

    sma = calculate_sma(close, period)
    middle = sma

    for i in range(period - 1, n):
        std = np.std(close[i - period + 1 : i + 1])
        upper[i] = middle[i] + num_std * std
        lower[i] = middle[i] - num_std * std
        if middle[i] != 0:
            bandwidth[i] = (upper[i] - lower[i]) / middle[i]
        diff = upper[i] - lower[i]
        if diff > 0:
            pct_b[i] = (close[i] - lower[i]) / diff
        else:
            pct_b[i] = 0.5

    return upper, middle, lower, bandwidth, pct_b

def calculate_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    計算平均方向性指數 (ADX, +DI, -DI)
    用於精確識別市場是否處於「強趨勢波段」或「盤整無效震盪」。
    """
    n = len(close)
    adx = np.empty(n, dtype=float)
    plus_di = np.empty(n, dtype=float)
    minus_di = np.empty(n, dtype=float)
    adx[:] = np.nan
    plus_di[:] = np.nan
    minus_di[:] = np.nan

    if n < period * 2:
        return adx, plus_di, minus_di

    # 計算 TR, +DM, -DM
    tr = np.empty(n, dtype=float)
    plus_dm = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Wilder 平滑
    tr_smooth = np.empty(n, dtype=float)
    plus_dm_smooth = np.empty(n, dtype=float)
    minus_dm_smooth = np.empty(n, dtype=float)

    tr_smooth[period] = np.sum(tr[1:period + 1])
    plus_dm_smooth[period] = np.sum(plus_dm[1:period + 1])
    minus_dm_smooth[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        tr_smooth[i] = tr_smooth[i - 1] - (tr_smooth[i - 1] / period) + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i - 1] - (plus_dm_smooth[i - 1] / period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i - 1] - (minus_dm_smooth[i - 1] / period) + minus_dm[i]

        if tr_smooth[i] > 0:
            plus_di[i] = 100.0 * (plus_dm_smooth[i] / tr_smooth[i])
            minus_di[i] = 100.0 * (minus_dm_smooth[i] / tr_smooth[i])
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0

    # 計算 DX 與 ADX
    dx = np.empty(n, dtype=float)
    dx[:] = np.nan
    for i in range(period + 1, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0

    adx_start = period * 2
    if n > adx_start:
        adx[adx_start] = np.nanmean(dx[period + 1 : adx_start + 1])
        for i in range(adx_start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx, plus_di, minus_di

