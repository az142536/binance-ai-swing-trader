"""
模擬交易引擎 (Paper Trading Simulator)
即時監控即時行情，執行 AI 訊號推論、虛擬下單、動態移動止盈止損與資產追蹤。
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from tabulate import tabulate
from config import Config, DATA_DIR, MODEL_DIR
from binance_client import BinanceClient
from features import extract_features, create_swing_wave_labels
from ai_model import AISwingModel
from strategy import AISwingStrategy, SwingPosition

class PaperTrader:
    def __init__(self, symbol: str = Config.DEFAULT_SYMBOL, interval: str = Config.DEFAULT_INTERVAL):
        self.symbol = symbol.upper()
        self.interval = interval
        self.client = BinanceClient()
        self.strategy = AISwingStrategy()
        self.model = AISwingModel()
        
        # 模擬帳戶狀態
        self.balance_usdt = Config.INITIAL_CAPITAL
        self.position: Optional[SwingPosition] = None
        self.trade_logs: List[Dict[str, Any]] = []
        self.state_file = DATA_DIR / f"paper_state_{self.symbol}.json"
        
        # 滾動自動適應與持續學習
        self.last_bar_timestamp: Optional[int] = None
        self.new_bars_count: int = 0
        self.auto_retrain_bars: int = 24 # 每累積 24 根新 K 線 (約 1 天) 自動滾動重新學習一次
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.balance_usdt = float(data.get("balance_usdt", Config.INITIAL_CAPITAL))
                pos_data = data.get("position")
                if pos_data:
                    self.position = SwingPosition(
                        entry_price=pos_data["entry_price"],
                        quantity=pos_data["quantity"],
                        entry_time=pos_data["entry_time"],
                        initial_stop=pos_data["stop_loss"],
                        take_profit=pos_data["take_profit"]
                    )
                    self.position.highest_price = pos_data.get("highest_price", pos_data["entry_price"])
                    self.position.is_trailing_active = pos_data.get("is_trailing_active", False)
                self.trade_logs = data.get("trade_logs", [])
            except Exception as e:
                print(f"[提示] 載入模擬狀態失敗，將以全新狀態啟動: {e}")

    def _save_state(self):
        data = {
            "symbol": self.symbol,
            "balance_usdt": self.balance_usdt,
            "position": None if self.position is None else {
                "entry_price": self.position.entry_price,
                "quantity": self.position.quantity,
                "entry_time": self.position.entry_time,
                "stop_loss": self.position.stop_loss,
                "take_profit": self.position.take_profit,
                "highest_price": self.position.highest_price,
                "is_trailing_active": self.position.is_trailing_active
            },
            "trade_logs": self.trade_logs[-50:],
            "last_updated": datetime.now().isoformat()
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_model(self, model_path: Path):
        self.model.load(model_path)

    def run_step(self, bars: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        執行單一輪次的行情檢查與決策
        """
        X, indicators_cache = extract_features(bars)
        last_idx = len(bars) - 1
        curr_bar = bars[last_idx]
        close = curr_bar["close"]
        high = curr_bar["high"]
        low = curr_bar["low"]

        # 滾動自動學習適應 (Walk-Forward Continual Learning)
        current_bar_ts = int(curr_bar.get("timestamp", 0))
        if self.last_bar_timestamp is not None and current_bar_ts != self.last_bar_timestamp:
            self.new_bars_count += 1
            if self.new_bars_count >= self.auto_retrain_bars and len(bars) >= 250:
                self.new_bars_count = 0
                try:
                    y = create_swing_wave_labels(bars, indicators_cache)
                    self.model.train(X, y)
                    model_path = MODEL_DIR / f"model_{self.symbol}_{self.interval}.json"
                    self.model.save(model_path)
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🧠 [AI 自我進化] 累積吸收最新 {self.auto_retrain_bars} 根行情，完成模型權重滾動更新！")
                except Exception as e:
                    pass
        self.last_bar_timestamp = current_bar_ts

        # AI 預測勝率
        ai_prob = float(self.model.predict_bullish_prob(X[last_idx:last_idx+1])[0])

        atr = indicators_cache["atr"][last_idx]
        curr_ind = {
            "ema21": indicators_cache["ema21"][last_idx],
            "ema50": indicators_cache["ema50"][last_idx],
            "ema200": indicators_cache["ema200"][last_idx],
            "adx": indicators_cache["adx"][last_idx],
            "plus_di": indicators_cache["plus_di"][last_idx],
            "minus_di": indicators_cache["minus_di"][last_idx],
            "rsi": indicators_cache["rsi"][last_idx],
            "atr": atr,
            "macd_hist": indicators_cache["macd_hist"][last_idx],
            "prev_macd_hist": indicators_cache["macd_hist"][last_idx - 1] if last_idx > 0 else 0.0
        }

        action_taken = "HOLD"
        action_detail = ""

        # 1. 檢查持倉出場
        if self.position is not None:
            exit_check = self.strategy.evaluate_exit(self.position, curr_bar, atr, ai_prob)
            if exit_check["should_sell"]:
                exit_price = exit_check["exit_price"]
                proceeds = exit_price * self.position.quantity
                fees = proceeds * Config.FEE_RATE
                net_proceeds = proceeds - fees
                pnl = (exit_price - self.position.entry_price) * self.position.quantity - (fees * 2)
                pnl_pct = (exit_price - self.position.entry_price) / self.position.entry_price * 100.0

                self.balance_usdt += net_proceeds
                action_taken = "SELL"
                action_detail = f"{exit_check['reason']} | 賣出價: ${exit_price:,.2f} | 盈虧: {pnl_pct:+.2f}% (${pnl:+,.2f})"

                self.trade_logs.append({
                    "action": "SELL",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "price": exit_price,
                    "quantity": self.position.quantity,
                    "pnl_usd": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "reason": exit_check["reason"],
                    "balance": round(self.balance_usdt, 2)
                })
                self.position = None
                self._save_state()

        # 2. 檢查開倉
        elif self.position is None:
            entry_check = self.strategy.evaluate_entry(ai_prob, close, curr_ind)
            if entry_check["should_buy"]:
                actual_price = close * (1.0 + Config.SLIPPAGE_RATE)
                stop_loss = entry_check["stop_loss"]
                take_profit = entry_check["take_profit"]
                qty = self.strategy.calculate_position_size(self.balance_usdt, actual_price, stop_loss)
                cost = qty * actual_price
                fees = cost * Config.FEE_RATE

                if (cost + fees) <= self.balance_usdt and cost >= 10.0:
                    self.balance_usdt -= (cost + fees)
                    self.position = SwingPosition(
                        entry_price=actual_price,
                        quantity=qty,
                        entry_time=int(time.time()),
                        initial_stop=stop_loss,
                        take_profit=take_profit
                    )
                    action_taken = "BUY"
                    action_detail = f"{entry_check['reason']} | 買入價: ${actual_price:,.2f} | 倉位: {qty:.4f} | 止損: ${stop_loss:,.2f} | 目標: ${take_profit:,.2f}"

                    self.trade_logs.append({
                        "action": "BUY",
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "price": actual_price,
                        "quantity": qty,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "balance": round(self.balance_usdt, 2)
                    })
                    self._save_state()
            else:
                action_detail = entry_check.get("reason", "等待更佳訊號")

        # 計算當前總資產
        total_equity = self.balance_usdt
        floating_pnl_usd = 0.0
        floating_pnl_pct = 0.0
        if self.position is not None:
            pos_val = self.position.quantity * close
            floating_pnl_usd = (close - self.position.entry_price) * self.position.quantity
            floating_pnl_pct = (close - self.position.entry_price) / self.position.entry_price * 100.0
            total_equity += pos_val

        status = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": self.symbol,
            "close_price": close,
            "ai_bullish_prob": ai_prob,
            "adx": curr_ind["adx"],
            "rsi": curr_ind["rsi"],
            "action": action_taken,
            "action_detail": action_detail,
            "has_position": self.position is not None,
            "position_qty": self.position.quantity if self.position else 0.0,
            "position_entry": self.position.entry_price if self.position else 0.0,
            "position_stop": self.position.stop_loss if self.position else 0.0,
            "position_target": self.position.take_profit if self.position else 0.0,
            "floating_pnl_usd": floating_pnl_usd,
            "floating_pnl_pct": floating_pnl_pct,
            "balance_usdt": self.balance_usdt,
            "total_equity": total_equity
        }

        return status

