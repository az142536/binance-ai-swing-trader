"""
回測引擎模組 (Backtesting Engine)
模擬歷史真實成交環境 (含手續費與滑點)，輸出勝率、盈虧比、最大回撤 (MDD)、夏普比率與交易明細。
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import math
import numpy as np
from typing import List, Dict, Any, Optional
from tabulate import tabulate
from config import Config
from strategy import AISwingStrategy, SwingPosition
from features import extract_features
from ai_model import AISwingModel

class Backtester:
    def __init__(
        self,
        strategy: Optional[AISwingStrategy] = None,
        initial_capital: float = Config.INITIAL_CAPITAL,
        fee_rate: float = Config.FEE_RATE,
        slippage_rate: float = Config.SLIPPAGE_RATE
    ):
        self.strategy = strategy or AISwingStrategy()
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

    def run(
        self,
        bars: List[Dict[str, float]],
        model: AISwingModel,
        start_idx: int = 220
    ) -> Dict[str, Any]:
        """
        執行回測
        start_idx: 開始回測的 K 線索引 (避開前期指標冷啟動階段)
        """
        X, indicators_cache = extract_features(bars)
        ai_probs = model.predict_bullish_prob(X)

        capital = self.initial_capital
        position: Optional[SwingPosition] = None
        trade_history: List[Dict[str, Any]] = []
        equity_curve: List[float] = [capital]

        n_bars = len(bars)

        for i in range(start_idx, n_bars):
            curr_bar = bars[i]
            close = curr_bar["close"]
            atr = indicators_cache["atr"][i]
            ai_p = ai_probs[i]

            # 當前指標切片
            curr_ind = {
                "ema21": indicators_cache["ema21"][i],
                "ema50": indicators_cache["ema50"][i],
                "ema200": indicators_cache["ema200"][i],
                "adx": indicators_cache["adx"][i],
                "plus_di": indicators_cache["plus_di"][i],
                "minus_di": indicators_cache["minus_di"][i],
                "rsi": indicators_cache["rsi"][i],
                "atr": atr,
                "macd_hist": indicators_cache["macd_hist"][i],
                "prev_macd_hist": indicators_cache["macd_hist"][i - 1] if i > 0 else 0.0
            }

            # 1. 若有持倉，檢查出場條件
            if position is not None:
                exit_check = self.strategy.evaluate_exit(position, curr_bar, atr, ai_p)
                if exit_check["should_sell"]:
                    raw_exit_p = exit_check["exit_price"]
                    # 考慮賣出滑點與手續費
                    actual_exit_p = raw_exit_p * (1.0 - self.slippage_rate)
                    gross_pnl = (actual_exit_p - position.entry_price) * position.quantity
                    fees = (position.entry_price + actual_exit_p) * position.quantity * self.fee_rate
                    net_pnl = gross_pnl - fees
                    pnl_pct = (actual_exit_p - position.entry_price) / position.entry_price

                    capital += net_pnl
                    trade_history.append({
                        "trade_no": len(trade_history) + 1,
                        "entry_idx": position.entry_time,
                        "exit_idx": i,
                        "holding_bars": i - position.entry_time,
                        "entry_price": position.entry_price,
                        "exit_price": actual_exit_p,
                        "pnl_usd": net_pnl,
                        "pnl_pct": pnl_pct * 100.0,
                        "reason": exit_check["reason"],
                        "capital_after": capital
                    })
                    position = None

            # 2. 若無持倉，檢查開倉條件
            elif position is None and i < n_bars - 1:
                entry_check = self.strategy.evaluate_entry(ai_p, close, curr_ind)
                if entry_check["should_buy"]:
                    # 考慮買入滑點
                    actual_entry_p = close * (1.0 + self.slippage_rate)
                    stop_loss_p = entry_check["stop_loss"]
                    target_p = entry_check["take_profit"]

                    qty = self.strategy.calculate_position_size(capital, actual_entry_p, stop_loss_p)
                    cost = qty * actual_entry_p
                    if cost > 0 and cost <= capital * Config.MAX_POSITION_RATIO:
                        position = SwingPosition(
                            entry_price=actual_entry_p,
                            quantity=qty,
                            entry_time=i,
                            initial_stop=stop_loss_p,
                            take_profit=target_p
                        )

            # 記錄權益資產
            current_equity = capital
            if position is not None:
                floating_pnl = (close - position.entry_price) * position.quantity
                current_equity += floating_pnl
            equity_curve.append(current_equity)

        # 計算統計指標
        metrics = self._calculate_metrics(trade_history, equity_curve, bars[start_idx]["close"], bars[-1]["close"])
        return {
            "metrics": metrics,
            "trades": trade_history,
            "equity_curve": equity_curve
        }

    def _calculate_metrics(
        self,
        trades: List[Dict[str, Any]],
        equity: List[float],
        start_price: float,
        end_price: float
    ) -> Dict[str, Any]:
        """計算關鍵績效統計指標"""
        final_capital = equity[-1]
        total_pnl = final_capital - self.initial_capital
        total_return_pct = (total_pnl / self.initial_capital) * 100.0
        benchmark_return_pct = ((end_price - start_price) / start_price) * 100.0

        n_trades = len(trades)
        if n_trades == 0:
            return {
                "initial_capital": round(self.initial_capital, 2),
                "final_capital": round(final_capital, 2),
                "net_profit_usd": round(total_pnl, 2),
                "total_return_pct": 0.0,
                "benchmark_return_pct": round(benchmark_return_pct, 2),
                "total_trades": 0,
                "win_trades": 0,
                "loss_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0
            }

        wins = [t for t in trades if t["pnl_usd"] > 0]
        losses = [t for t in trades if t["pnl_usd"] <= 0]
        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = (n_wins / n_trades) * 100.0

        total_gain = sum(t["pnl_usd"] for t in wins)
        total_loss = abs(sum(t["pnl_usd"] for t in losses))
        profit_factor = (total_gain / total_loss) if total_loss > 0 else 999.0

        avg_win_pct = np.mean([t["pnl_pct"] for t in wins]) if wins else 0.0
        avg_loss_pct = np.mean([t["pnl_pct"] for t in losses]) if losses else 0.0

        # 最大回撤 (Maximum Drawdown)
        eq_arr = np.array(equity)
        peaks = np.maximum.accumulate(eq_arr)
        drawdowns = (peaks - eq_arr) / peaks
        max_mdd = float(np.max(drawdowns)) * 100.0

        # 夏普比率 (Sharpe Ratio, 假設無風險年化 2%)
        returns = np.diff(eq_arr) / eq_arr[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            # 依 1h 週期年化約 24 * 365 = 8760
            sharpe = (np.mean(returns) / np.std(returns)) * math.sqrt(8760)
        else:
            sharpe = 0.0

        return {
            "initial_capital": round(self.initial_capital, 2),
            "final_capital": round(final_capital, 2),
            "net_profit_usd": round(total_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "benchmark_return_pct": round(benchmark_return_pct, 2),
            "total_trades": n_trades,
            "win_trades": n_wins,
            "loss_trades": n_losses,
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_win_pct": round(float(avg_win_pct), 2),
            "avg_loss_pct": round(float(avg_loss_pct), 2),
            "max_drawdown_pct": round(max_mdd, 2),
            "sharpe_ratio": round(float(sharpe), 2)
        }

    def print_summary(self, result: Dict[str, Any]):
        """輸出格式化終端報告"""
        m = result["metrics"]
        trades = result["trades"]

        print("\n" + "=" * 65)
        print("          [REPORT] 幣安 AI 趨勢波段策略 - 回測績效報告")
        print("=" * 65)
        
        table_data = [
            ["初始資金 (USDT)", f"${m['initial_capital']:,.2f}", "最終資金 (USDT)", f"${m['final_capital']:,.2f}"],
            ["淨利潤 (USDT)", f"${m['net_profit_usd']:+,.2f}", "總報酬率 (%)", f"{m['total_return_pct']:+.2f}%"],
            ["基準買入持有報酬", f"{m['benchmark_return_pct']:+.2f}%", "總交易次數", f"{m['total_trades']} 筆"],
            ["勝率 (%)", f"{m['win_rate']:.2f}%", "獲利/虧損比 (PF)", f"{m['profit_factor']:.2f}"],
            ["平均獲利 (%)", f"{m['avg_win_pct']:+.2f}%", "平均虧損 (%)", f"{m['avg_loss_pct']:+.2f}%"],
            ["最大回撤 (MDD)", f"{m['max_drawdown_pct']:.2f}%", "年化夏普比率", f"{m['sharpe_ratio']:.2f}"]
        ]
        print(tabulate(table_data, tablefmt="grid"))

        if trades:
            print("\n[*] 最近 10 筆波段交易明細:")
            display_trades = trades[-10:]
            trade_rows = []
            for t in display_trades:
                trade_rows.append([
                    f"#{t['trade_no']}",
                    f"${t['entry_price']:,.2f}",
                    f"${t['exit_price']:,.2f}",
                    f"{t['holding_bars']} 根K",
                    f"{t['pnl_pct']:+.2f}%",
                    f"${t['pnl_usd']:+,.2f}",
                    t['reason']
                ])
            headers = ["編號", "進場價", "出場價", "持倉時間", "盈虧率", "淨利潤(USDT)", "出場原因"]
            print(tabulate(trade_rows, headers=headers, tablefmt="simple"))
        print("=" * 65 + "\n")
