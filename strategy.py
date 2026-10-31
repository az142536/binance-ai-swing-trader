"""
AI 趨勢波段交易策略核心 (AI Swing Master Strategy - High-Profit Version)
整合宏觀大趨勢濾網、動能共振確認、高信度 AI 狙擊濾網與吊燈移動止盈 (Chandelier Trailing Stop)。
"""

from typing import Dict, Any, Optional
from config import Config

class SwingPosition:
    def __init__(self, entry_price: float, quantity: float, entry_time: int, initial_stop: float, take_profit: float):
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.stop_loss = initial_stop
        self.take_profit = take_profit
        self.highest_price = entry_price
        self.is_trailing_active = False

    def update_trailing_stop(self, current_high: float, atr: float, trailing_mult: float):
        """根據新高點更新吊燈動態移動止盈價位 (Chandelier Trailing Stop)"""
        if current_high > self.highest_price:
            self.highest_price = current_high
            
            # 若波段浮盈已超過 2.0x ATR，正式啟動移動止盈機制
            profit_distance = self.highest_price - self.entry_price
            if profit_distance >= 2.0 * atr:
                self.is_trailing_active = True
                trail_stop = self.highest_price - trailing_mult * atr
                if trail_stop > self.stop_loss:
                    self.stop_loss = trail_stop


class AISwingStrategy:
    def __init__(
        self,
        mode: str = Config.SWING_MODE,
        confidence_threshold: float = Config.AI_CONFIDENCE_THRESHOLD,
        stop_loss_atr_mult: float = Config.STOP_LOSS_ATR_MULT,
        take_profit_atr_mult: float = Config.TAKE_PROFIT_ATR_MULT,
        trailing_stop_atr_mult: float = Config.TRAILING_STOP_ATR_MULT,
        risk_per_trade: float = Config.RISK_PER_TRADE
    ):
        self.mode = mode.lower()
        self.confidence_threshold = confidence_threshold
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_atr_mult = take_profit_atr_mult
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.risk_per_trade = risk_per_trade

    def evaluate_entry(
        self,
        ai_prob: float,
        close: float,
        indicators: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        評估是否符合開倉波段條件
        支援兩種風格:
        - active (敏捷波段模式): 只要短線動能轉強即果斷進場捕捉波段，不被慢速均線綁死。
        - sniper (穩健狙擊模式): 嚴格要求宏觀主升浪 (200 EMA)。
        """
        ema21 = indicators.get("ema21", 0.0)
        ema50 = indicators.get("ema50", 0.0)
        ema200 = indicators.get("ema200", 0.0)
        adx = indicators.get("adx", 0.0)
        plus_di = indicators.get("plus_di", 0.0)
        minus_di = indicators.get("minus_di", 0.0)
        rsi = indicators.get("rsi", 50.0)
        atr = indicators.get("atr", 0.0)
        macd_hist = indicators.get("macd_hist", 0.0)
        prev_macd_hist = indicators.get("prev_macd_hist", 0.0)

        # 1. AI 預測勝率閥值 (AI 信心必須達標)
        if ai_prob < self.confidence_threshold:
            return {"should_buy": False, "reason": f"AI 信心不足 ({ai_prob:.1%} < {self.confidence_threshold:.1%})"}

        # 2. 趨勢與波段濾網 (依據模式切換)
        if self.mode == "sniper":
            # 穩健模式：價格與 50 均線皆須在 200 EMA 之上
            if close < ema200 or ema50 < ema200:
                return {"should_buy": False, "reason": "大週期非多頭主升浪 (未在 200 EMA 之上或 50 < 200)"}
            if adx < 20.0 or plus_di <= minus_di:
                return {"should_buy": False, "reason": f"趨勢強度不足或空方佔優 (ADX: {adx:.1f})"}
        else:
            # 敏捷波段模式 (Active Mode)：主動捕捉爆發波段，價格站上 21 EMA 短線動能線即可進場！
            if close < ema21 * 0.990:
                return {"should_buy": False, "reason": f"短線價格尚未站回 21 EMA 支撐 (${ema21:,.4f})"}

        # 3. 動能共振檢查 (依據風格模式)
        if self.mode == "sniper":
            if macd_hist <= prev_macd_hist:
                return {"should_buy": False, "reason": "MACD 動能柱體衰退中"}
        else:
            # 敏捷波段模式：只要 MACD 位於多頭柱狀體 (>0) 或動能正在轉強放大，即認可動能
            if macd_hist <= 0 and macd_hist <= prev_macd_hist:
                return {"should_buy": False, "reason": "MACD 處於空方動能衰退區"}

        # 4. RSI 甜蜜區間 (38 ~ 78)：不過度超賣衰竭、不追最高點超買
        if rsi < 38.0 or rsi > 78.0:
            return {"should_buy": False, "reason": f"RSI 偏離甜蜜區間 (RSI: {rsi:.1f})"}

        # 5. ATR 波動度合理性檢查
        if atr <= 0:
            return {"should_buy": False, "reason": "ATR 數值無效"}

        # 完全符合共振，觸發波段做多訊號！
        stop_price = close - self.stop_loss_atr_mult * atr
        target_price = close + self.take_profit_atr_mult * atr

        return {
            "should_buy": True,
            "reason": f"AI 敏捷波段進場訊號 (AI信心: {ai_prob:.1%}, 短線動能共振開倉)",
            "entry_price": close,
            "stop_loss": stop_price,
            "take_profit": target_price,
            "atr": atr
        }

    def evaluate_exit(
        self,
        position: SwingPosition,
        current_bar: Dict[str, float],
        atr: float,
        ai_prob: float
    ) -> Dict[str, Any]:
        """
        評估持倉是否需止損、止盈或出場
        """
        low = current_bar["low"]
        high = current_bar["high"]
        close = current_bar["close"]

        # 更新吊燈移動止損
        position.update_trailing_stop(high, atr, self.trailing_stop_atr_mult)

        # 1. 優先檢查是否觸發目標大波段止盈 (3.0x ATR)
        if high >= position.take_profit:
            return {
                "should_sell": True,
                "exit_price": position.take_profit,
                "reason": f"達到目標大波段止盈 ({self.take_profit_atr_mult}x ATR)"
            }

        # 2. 檢查是否觸及移動止盈點或防守止損點
        if low <= position.stop_loss:
            exit_price = position.stop_loss
            is_profit = exit_price > position.entry_price
            reason = "移動止盈鎖定利潤" if is_profit else "觸發防守性止損"
            return {"should_sell": True, "exit_price": exit_price, "reason": reason}

        # 3. AI 反轉警報：若 AI 信心暴跌至 20% 以下且價格跌破 21 EMA
        if ai_prob < 0.20 and position.is_trailing_active:
            return {
                "should_sell": True,
                "exit_price": close,
                "reason": "AI 偵測到波段趨勢終結 (動能衰竭預警出場)"
            }

        return {"should_sell": False}

    def calculate_position_size(self, capital: float, entry_price: float, stop_loss: float) -> float:
        """
        以固定風險比例計算動態倉位大小 (Risk Sizing)
        每筆交易最大承擔金額 = capital * risk_per_trade (預設 2%)
        """
        risk_amount = capital * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= 0:
            return 0.0

        quantity = risk_amount / stop_distance
        max_qty = (capital * Config.MAX_POSITION_RATIO) / entry_price
        return min(quantity, max_qty)
