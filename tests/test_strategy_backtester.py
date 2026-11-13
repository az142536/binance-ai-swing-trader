import unittest
from unittest.mock import patch

import numpy as np

from backtester import Backtester
from rotation_trader import _calculate_net_pnl
from strategy import AISwingStrategy, SwingPosition


class RotationAccountingTests(unittest.TestCase):
    def test_net_pnl_accounts_for_each_side_fee_at_its_fill_price(self):
        pnl, pnl_pct = _calculate_net_pnl(100.0, 101.0, 2.0, 0.00075)

        self.assertAlmostEqual(pnl, 1.6985)
        self.assertAlmostEqual(pnl_pct, 0.84925)


class StrategyExitTests(unittest.TestCase):
    def setUp(self):
        self.strategy = AISwingStrategy(trailing_stop_atr_mult=2.0)

    def make_position(self, **kwargs):
        values = {
            "entry_price": 100.0,
            "quantity": 10.0,
            "entry_time": 0,
            "initial_stop": 95.0,
            "take_profit": 120.0,
        }
        values.update(kwargs)
        return SwingPosition(**values)

    def test_historical_stop_precedes_target_when_both_are_touched(self):
        position = self.make_position(take_profit=105.0)
        bar = {"open": 100.0, "high": 110.0, "low": 94.0, "close": 106.0}

        result = self.strategy.evaluate_exit(position, bar, atr=2.0, ai_prob=0.5)

        self.assertTrue(result["should_sell"])
        self.assertEqual(result["exit_price"], 95.0)
        self.assertEqual(result["reason"], "觸發防守性止損")

    def test_historical_gap_below_stop_fills_at_open(self):
        position = self.make_position()
        bar = {"open": 90.0, "high": 101.0, "low": 89.0, "close": 99.0}

        result = self.strategy.evaluate_exit(position, bar, atr=2.0, ai_prob=0.5)

        self.assertTrue(result["should_sell"])
        self.assertEqual(result["exit_price"], 90.0)

    def test_historical_partial_take_profit_uses_threshold_price(self):
        position = self.make_position()
        bar = {"open": 100.0, "high": 106.0, "low": 100.0, "close": 104.0}

        result = self.strategy.evaluate_exit(position, bar, atr=1.0, ai_prob=0.5)

        self.assertTrue(result["is_partial"])
        self.assertAlmostEqual(result["exit_price"], 102.5)

    def test_new_trailing_stop_is_not_applied_to_same_bar_low(self):
        position = self.make_position()
        position.is_partial_tp_taken = True
        bar = {"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.5}

        result = self.strategy.evaluate_exit(position, bar, atr=2.0, ai_prob=0.5)

        self.assertFalse(result["should_sell"])
        self.assertEqual(position.stop_loss, 100.2)


class BacktesterExitTests(unittest.TestCase):
    def test_partial_exit_reduces_position_and_books_each_fill(self):
        class FakeModel:
            def predict_bullish_prob(self, X):
                return np.full(len(X), 0.8)

        class PartialStrategy:
            def __init__(self):
                self.entered = False
                self.exit_quantities = []
                self.exit_count = 0

            def evaluate_entry(self, ai_prob, close, indicators):
                if self.entered:
                    return {"should_buy": False}
                self.entered = True
                return {"should_buy": True, "stop_loss": 90.0, "take_profit": 120.0}

            def calculate_position_size(self, capital, entry_price, stop_loss):
                return 10.0

            def evaluate_exit(self, position, bar, atr, ai_prob):
                self.exit_quantities.append(position.quantity)
                self.exit_count += 1
                if self.exit_count == 1:
                    return {
                        "should_sell": True,
                        "is_partial": True,
                        "sell_ratio": 0.5,
                        "exit_price": 102.5,
                        "reason": "partial",
                    }
                return {
                    "should_sell": True,
                    "is_partial": False,
                    "exit_price": 104.0,
                    "reason": "exit",
                }

        bar = {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}
        bars = [bar.copy() for _ in range(223)]
        indicators = {
            name: np.full(len(bars), value)
            for name, value in {
                "atr": 1.0,
                "ema21": 100.0,
                "ema50": 100.0,
                "ema200": 100.0,
                "adx": 25.0,
                "plus_di": 20.0,
                "minus_di": 10.0,
                "rsi": 50.0,
                "macd_hist": 1.0,
            }.items()
        }
        strategy = PartialStrategy()
        backtester = Backtester(
            strategy=strategy,
            initial_capital=2000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
        )

        with patch("backtester.extract_features", return_value=(np.zeros((len(bars), 1)), indicators)):
            result = backtester.run(bars, FakeModel(), start_idx=220)

        self.assertEqual(strategy.exit_quantities, [10.0, 5.0])
        self.assertEqual(len(result["trades"]), 2)
        self.assertTrue(result["trades"][0]["is_partial"])
        self.assertFalse(result["trades"][1]["is_partial"])
        self.assertEqual(result["metrics"]["final_capital"], 2032.5)

    def test_no_trade_metrics_report_marked_equity_return(self):
        backtester = Backtester(initial_capital=100.0)

        metrics = backtester._calculate_metrics([], [100.0, 90.0], 100.0, 90.0)

        self.assertEqual(metrics["total_return_pct"], -10.0)


if __name__ == "__main__":
    unittest.main()
