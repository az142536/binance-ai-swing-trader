"""
全市場空倉智能波段輪動交易引擎 (Smart Wave Rotation Trader)
核心哲學：【持倉時專注吃滿波段，空倉時自動切換至全市場最強龍頭幣】
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from tabulate import tabulate
import numpy as np

from config import Config, DATA_DIR, MODEL_DIR
from binance_client import BinanceClient
from features import extract_features, create_swing_wave_labels
from ai_model import AISwingModel
from strategy import AISwingStrategy, SwingPosition
from scanner import DEFAULT_WATCHLIST, SymbolScanner

class SmartRotationTrader:
    def __init__(
        self,
        watchlist: Optional[List[str]] = None,
        interval: str = Config.DEFAULT_INTERVAL,
        mode: str = Config.SWING_MODE,
        capital: Optional[float] = None
    ):
        self.watchlist = watchlist or DEFAULT_WATCHLIST
        self.interval = interval
        self.mode = mode
        
        self.client = BinanceClient()
        self.strategy = AISwingStrategy()
        self.strategy.mode = mode
        
        # 輪動資產與持倉狀態 (預設 100 USDT)
        self.default_capital = capital if capital is not None else Config.INITIAL_CAPITAL
        self.balance_usdt: float = self.default_capital
        self.has_position: bool = False
        self.active_symbol: Optional[str] = None
        self.position: Optional[SwingPosition] = None
        self.trade_logs: List[Dict[str, Any]] = []
        
        # 模型快取字典 (symbol -> AISwingModel)
        self.models: Dict[str, AISwingModel] = {}
        self.last_trained_time: Dict[str, float] = {}
        
        self.state_file = DATA_DIR / "rotation_state.json"
        self._load_state()

    def _load_state(self):
        """從硬碟載入輪動交易狀態"""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.balance_usdt = float(data.get("balance_usdt", self.default_capital))
                self.has_position = bool(data.get("has_position", False))
                self.active_symbol = data.get("active_symbol")
                pos_data = data.get("position")
                if pos_data and self.has_position:
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
                print(f"[提示] 載入輪動模擬狀態異常，將以全新狀態啟動: {e}")

    def _save_state(self):
        """將最新輪動資產與持倉持久化至硬碟"""
        data = {
            "balance_usdt": round(self.balance_usdt, 2),
            "has_position": self.has_position,
            "active_symbol": self.active_symbol,
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
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _get_or_train_model(self, symbol: str) -> AISwingModel:
        """獲取或自動訓練指定幣種的專屬 AI 模型 (滿 24 小時自動滾動重訓進化)"""
        now = time.time()
        model_path = MODEL_DIR / f"model_{symbol}_{self.interval}.json"

        # 檢查是否需要滾動重新訓練 (模型不存在，或已超過 24 小時)
        needs_retrain = False
        if symbol not in self.models:
            if not model_path.exists():
                needs_retrain = True
            elif (now - model_path.stat().st_mtime) > 86400:  # 超過 24 小時自動過期更新
                needs_retrain = True
        else:
            if (now - self.last_trained_time.get(symbol, 0)) > 86400:
                needs_retrain = True

        if needs_retrain:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🧠 [AI 自動進化] 正在為 【{symbol}】 抓取最新行情並重訓模型...")
            scanner = SymbolScanner(interval=self.interval)
            bars = scanner.get_symbol_klines(symbol, bars_count=2400)
            X, ind = extract_features(bars)
            y = create_swing_wave_labels(bars, ind)
            model = AISwingModel()
            model.train(X, y)
            model.save(model_path)
            self.models[symbol] = model
            self.last_trained_time[symbol] = now
            return model

        if symbol in self.models:
            return self.models[symbol]

        model = AISwingModel()
        model.load(model_path)
        self.models[symbol] = model
        self.last_trained_time[symbol] = model_path.stat().st_mtime
        return model

    def _get_symbol_bars(self, symbol: str, total_bars: int = 250) -> List[Dict[str, float]]:
        """獲取指定幣種最新 K 線 (優先從幣安下載，異常則從本機快取備援)"""
        try:
            bars = self.client.download_historical_klines(symbol, interval=self.interval, total_bars=total_bars)
            if bars and len(bars) >= 100:
                return bars
        except Exception:
            pass

        data_file = DATA_DIR / f"{symbol}_{self.interval}.json"
        if data_file.exists():
            from sample_data import load_klines_from_json
            return load_klines_from_json(data_file)[-total_bars:]

        scanner = SymbolScanner(interval=self.interval)
        return scanner.get_symbol_klines(symbol, bars_count=total_bars)

    def run_cycle(self) -> Dict[str, Any]:
        """
        執行單一輪次的輪動決策引擎：
        - 若有持倉：全心全意專注於 active_symbol，執行 ATR 移動止盈與風控出場。
        - 若無持倉：全市場掃描強勢幣天梯，挑選出勝率最高且發起突破的第一名強勢幣建倉！
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ========================================================
        # 情境一：持倉中 (專注吃滿當前幣種的大波段)
        # ========================================================
        if self.has_position and self.active_symbol and self.position is not None:
            symbol = self.active_symbol
            bars = self._get_symbol_bars(symbol, total_bars=250)
            X, indicators_cache = extract_features(bars)
            last_idx = len(bars) - 1
            curr_bar = bars[last_idx]
            close = curr_bar["close"]
            atr = indicators_cache["atr"][last_idx]

            model = self._get_or_train_model(symbol)
            ai_prob = float(model.predict_bullish_prob(X[last_idx:last_idx+1])[0])

            # 評估出場
            exit_check = self.strategy.evaluate_exit(self.position, curr_bar, atr, ai_prob)
            floating_pnl = (close - self.position.entry_price) * self.position.quantity
            floating_pnl_pct = (close - self.position.entry_price) / self.position.entry_price * 100.0
            total_equity = self.balance_usdt + (close * self.position.quantity)

            if exit_check["should_sell"]:
                exit_price = exit_check["exit_price"]
                proceeds = exit_price * self.position.quantity
                fees = proceeds * Config.FEE_RATE
                net_proceeds = proceeds - fees
                realized_pnl = (exit_price - self.position.entry_price) * self.position.quantity - (fees * 2)
                realized_pnl_pct = (exit_price - self.position.entry_price) / self.position.entry_price * 100.0

                self.balance_usdt += net_proceeds
                self.has_position = False
                self.active_symbol = None

                self.trade_logs.append({
                    "symbol": symbol,
                    "action": "SELL",
                    "time": now_str,
                    "price": exit_price,
                    "quantity": self.position.quantity,
                    "pnl_usd": round(realized_pnl, 2),
                    "pnl_pct": round(realized_pnl_pct, 2),
                    "reason": exit_check["reason"],
                    "balance": round(self.balance_usdt, 2)
                })
                self.position = None
                self._save_state()

                return {
                    "timestamp": now_str,
                    "phase": "EXIT",
                    "symbol": symbol,
                    "close_price": close,
                    "detail": f"🏆 波段獲利平倉！{exit_check['reason']} | 賣出價: ${exit_price:,.2f} | 盈虧: {realized_pnl_pct:+.2f}% (${realized_pnl:+,.2f} USDT)",
                    "total_equity": self.balance_usdt,
                    "has_position": False
                }
            else:
                self._save_state()
                trail_desc = f"移動止盈線: ${self.position.stop_loss:,.2f}" if self.position.is_trailing_active else f"保護止損線: ${self.position.stop_loss:,.2f}"
                return {
                    "timestamp": now_str,
                    "phase": "HOLD",
                    "symbol": symbol,
                    "close_price": close,
                    "ai_bullish_prob": ai_prob,
                    "detail": f"持倉抱牢中 (進場: ${self.position.entry_price:,.2f}, 浮盈: {floating_pnl_pct:+.2f}%, {trail_desc})",
                    "total_equity": total_equity,
                    "has_position": True
                }

        # ========================================================
        # 情境二：空倉狀態 (全市場天梯掃描，挑選最猛起漲第一名)
        # ========================================================
        else:
            candidates = []
            highest_prob = 0.0
            highest_prob_symbol = ""
            highest_block_reason = ""

            for sym in self.watchlist:
                try:
                    bars = self._get_symbol_bars(sym, total_bars=250)
                    X, indicators_cache = extract_features(bars)
                    last_idx = len(bars) - 1
                    curr_bar = bars[last_idx]
                    close = curr_bar["close"]
                    atr = indicators_cache["atr"][last_idx]
                    
                    model = self._get_or_train_model(sym)
                    ai_prob = float(model.predict_bullish_prob(X[last_idx:last_idx+1])[0])

                    if ai_prob > highest_prob:
                        highest_prob = ai_prob
                        highest_prob_symbol = sym

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

                    entry_check = self.strategy.evaluate_entry(ai_prob, close, curr_ind)
                    if entry_check["should_buy"]:
                        # 計算綜合起漲動能得分
                        adx_val = indicators_cache["adx"][last_idx]
                        macd_val = indicators_cache["macd_hist"][last_idx]
                        momentum_score = (ai_prob * 60.0) + (min(adx_val, 50.0) * 0.5) + (15.0 if macd_val > 0 else 0.0)
                        candidates.append({
                            "symbol": sym,
                            "ai_prob": ai_prob,
                            "score": momentum_score,
                            "close": close,
                            "atr": atr,
                            "reason": entry_check["reason"]
                        })
                    else:
                        if ai_prob >= highest_prob:
                            highest_block_reason = entry_check.get("reason", "風控保護過濾中")
                except Exception:
                    pass

            # 若有符合進場條件的幣種，挑選得分最高的冠軍進場
            if candidates:
                candidates.sort(key=lambda x: x["score"], reverse=True)
                winner = candidates[0]
                win_sym = winner["symbol"]
                win_price = winner["close"] * (1.0 + Config.SLIPPAGE_RATE)
                win_atr = winner["atr"]

                # 依風控計算開倉部位 (支援小資金如 100 USDT，滿足幣安最低 10 USDT 下單要求)
                risk_amount = self.balance_usdt * Config.RISK_PER_TRADE
                stop_dist = win_atr * Config.STOP_LOSS_ATR_MULT
                calculated_qty = risk_amount / max(stop_dist, 1e-6)
                max_qty = (self.balance_usdt * Config.MAX_POSITION_RATIO) / win_price
                min_qty = 10.0 / win_price if self.balance_usdt >= 12.0 else 0.0
                qty = max(min(calculated_qty, max_qty), min_qty)
                qty = min(qty, max_qty)

                cost = win_price * qty
                fee = cost * Config.FEE_RATE
                self.balance_usdt -= (cost + fee)

                initial_stop = win_price - stop_dist
                target_tp = win_price + (win_atr * Config.TAKE_PROFIT_ATR_MULT)

                self.position = SwingPosition(
                    entry_price=win_price,
                    quantity=qty,
                    entry_time=now_str,
                    initial_stop=initial_stop,
                    take_profit=target_tp
                )
                self.has_position = True
                self.active_symbol = win_sym

                self.trade_logs.append({
                    "symbol": win_sym,
                    "action": "BUY",
                    "time": now_str,
                    "price": win_price,
                    "quantity": qty,
                    "stop_loss": initial_stop,
                    "take_profit": target_tp,
                    "reason": winner["reason"],
                    "balance": round(self.balance_usdt, 2)
                })
                self._save_state()

                return {
                    "timestamp": now_str,
                    "phase": "BUY",
                    "symbol": win_sym,
                    "close_price": win_price,
                    "ai_bullish_prob": winner["ai_prob"],
                    "detail": f"🚀 鎖定全市場起漲冠軍！【{win_sym}】買入建倉！(AI信心: {winner['ai_prob']:.1%}, 止損: ${initial_stop:,.2f}, 目標: ${target_tp:,.2f})",
                    "total_equity": self.balance_usdt + cost,
                    "has_position": True
                }
            else:
                if highest_prob >= Config.AI_CONFIDENCE_THRESHOLD:
                    detail = f"全市場掃描完畢：【{highest_prob_symbol}】AI信心高達 {highest_prob:.1%}，但風控保護等待最佳切入點 ({highest_block_reason})，100% USDT 耐心伏擊！"
                else:
                    detail = f"全市場掃描完畢：8大主流幣目前無合適起漲點 (最高信心: {highest_prob_symbol} {highest_prob:.1%} < {Config.AI_CONFIDENCE_THRESHOLD:.0%})，100% USDT 空倉觀望保護本金！"

                return {
                    "timestamp": now_str,
                    "phase": "WAIT",
                    "symbol": highest_prob_symbol or "ALL",
                    "close_price": 0.0,
                    "ai_bullish_prob": highest_prob,
                    "detail": detail,
                    "total_equity": self.balance_usdt,
                    "has_position": False
                }

