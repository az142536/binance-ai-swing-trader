"""
幣安實盤/測試網自動交易執行模組 (Binance Live & Testnet Execution Engine)
處理真實訂單委託、動態移動止盈止損、帳戶餘額校驗與異常安全熔斷。
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from config import Config, DATA_DIR, MODEL_DIR
from binance_client import BinanceClient
from features import extract_features, create_swing_wave_labels
from ai_model import AISwingModel
from strategy import AISwingStrategy, SwingPosition

class LiveTrader:
    def __init__(self, symbol: str = Config.DEFAULT_SYMBOL, interval: str = Config.DEFAULT_INTERVAL):
        self.symbol = symbol.upper()
        self.interval = interval
        self.client = BinanceClient()
        self.strategy = AISwingStrategy()
        self.model = AISwingModel()
        
        # 標的基礎幣種 (例如 BTCUSDT -> BTC)
        self.base_asset = self.symbol.replace("USDT", "")
        self.quote_asset = "USDT"

        self.position: Optional[SwingPosition] = None
        self.state_file = DATA_DIR / f"live_state_{self.symbol}.json"
        
        # 滾動自動學習更新
        self.last_bar_timestamp: Optional[int] = None
        self.new_bars_count: int = 0
        self.auto_retrain_bars: int = 24
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
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
            except Exception as e:
                print(f"[警告] 讀取實盤狀態失敗: {e}")

    def _save_state(self):
        data = {
            "symbol": self.symbol,
            "position": None if self.position is None else {
                "entry_price": self.position.entry_price,
                "quantity": self.position.quantity,
                "entry_time": self.position.entry_time,
                "stop_loss": self.position.stop_loss,
                "take_profit": self.position.take_profit,
                "highest_price": self.position.highest_price,
                "is_trailing_active": self.position.is_trailing_active
            },
            "last_updated": datetime.now().isoformat()
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_model(self, model_path: Path):
        self.model.load(model_path)

    def check_connection(self) -> Dict[str, Any]:
        """驗證 API 憑證與獲取帳戶當前資產"""
        if not self.client.api_key or not self.client.api_secret:
            raise ValueError("請先在 .env 檔案中填寫 BINANCE_API_KEY 與 BINANCE_API_SECRET！")
        
        server_time = self.client.get_server_time()
        balances = self.client.get_account_balances()
        usdt_free = balances.get(self.quote_asset, {}).get("free", 0.0)
        base_free = balances.get(self.base_asset, {}).get("free", 0.0)

        return {
            "network": "幣安測試網 (Testnet)" if self.client.use_testnet else "幣安正式實盤 (Mainnet)",
            "server_time": server_time,
            f"{self.quote_asset}_balance": usdt_free,
            f"{self.base_asset}_balance": base_free
        }

    def execute_cycle(self) -> Dict[str, Any]:
        """
        執行單次實盤掃描與下單判斷
        """
        # 1. 獲取最新 K 線
        raw_bars = self.client.get_klines(self.symbol, interval=self.interval, limit=250)
        bars = []
        for b in raw_bars:
            bars.append({
                "timestamp": int(b[0]),
                "open": float(b[1]),
                "high": float(b[2]),
                "low": float(b[3]),
                "close": float(b[4]),
                "volume": float(b[5])
            })

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
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🧠 [AI 自我進化] 累積吸收最新 {self.auto_retrain_bars} 根實盤行情，完成模型權重滾動更新！")
                except Exception as e:
                    pass
        self.last_bar_timestamp = current_bar_ts

        # AI 趨勢預測
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

        balances = self.client.get_account_balances()
        usdt_free = balances.get(self.quote_asset, {}).get("free", 0.0)
        base_free = balances.get(self.base_asset, {}).get("free", 0.0)

        action_result = {"action": "HOLD", "detail": "等待波段訊號"}

        # 2. 持倉出場判斷
        if self.position is not None:
            exit_check = self.strategy.evaluate_exit(self.position, curr_bar, atr, ai_prob)
            if exit_check["should_sell"]:
                # 實盤賣出
                sell_qty = min(base_free, self.position.quantity)
                if sell_qty > 0.0001:
                    order = self.client.create_order(
                        symbol=self.symbol,
                        side="SELL",
                        order_type="MARKET",
                        quantity=round(sell_qty, 5)
                    )
                    action_result = {
                        "action": "SELL",
                        "order": order,
                        "reason": exit_check["reason"]
                    }
                self.position = None
                self._save_state()
            else:
                self._save_state()

        # 3. 開倉判斷
        elif self.position is None:
            entry_check = self.strategy.evaluate_entry(ai_prob, close, curr_ind)
            if entry_check["should_buy"] and usdt_free >= 15.0:
                stop_loss = entry_check["stop_loss"]
                target = entry_check["take_profit"]
                target_qty = self.strategy.calculate_position_size(usdt_free, close, stop_loss)
                quote_order_usd = min(target_qty * close, usdt_free * Config.MAX_POSITION_RATIO)

                if quote_order_usd >= 12.0: # 幣安現貨最小名義價值通常為 5~10 USDT
                    order = self.client.create_order(
                        symbol=self.symbol,
                        side="BUY",
                        order_type="MARKET",
                        quote_order_qty=round(quote_order_usd, 2)
                    )
                    # 依成交量計算持倉
                    executed_qty = float(order.get("executedQty", quote_order_usd / close))
                    self.position = SwingPosition(
                        entry_price=close,
                        quantity=executed_qty,
                        entry_time=int(time.time()),
                        initial_stop=stop_loss,
                        take_profit=target
                    )
                    self._save_state()
                    action_result = {
                        "action": "BUY",
                        "order": order,
                        "reason": entry_check["reason"]
                    }

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": self.symbol,
            "close_price": close,
            "ai_bullish_prob": ai_prob,
            "adx": curr_ind["adx"],
            "rsi": curr_ind["rsi"],
            "has_position": self.position is not None,
            "position_stop": self.position.stop_loss if self.position else None,
            "position_target": self.position.take_profit if self.position else None,
            "action_result": action_result
        }

