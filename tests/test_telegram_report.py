import unittest
from datetime import date

from telegram_report import build_daily_report


class DailyTelegramReportTests(unittest.TestCase):
    def test_summarizes_only_selected_utc_day_and_distinguishes_cash_from_position(self):
        state = {
            "balance_usdt": 42.5,
            "has_position": True,
            "active_symbol": "BTCUSDT",
            "position": {"quantity": 0.001, "entry_price": 60000.0},
            "trade_logs": [
                {
                    "time": "2026-11-15 23:59:00",
                    "symbol": "ETHUSDT",
                    "action": "SELL",
                    "pnl_usd": -2.0,
                },
                {
                    "time": "2026-11-16 00:01:00",
                    "symbol": "BTCUSDT",
                    "action": "BUY",
                },
                {
                    "time": "2026-11-16 00:30:00",
                    "symbol": "BTCUSDT",
                    "action": "SELL (PARTIAL 50%)",
                    "pnl_usd": 1.25,
                },
            ],
        }

        report = build_daily_report(state, date(2026, 11, 16))

        self.assertIn("今日已實現損益：+1.25 USDT", report)
        self.assertIn("今日成交：2 筆（平倉/部分止盈 1 筆）", report)
        self.assertIn("現金餘額不含持倉市值", report)
        self.assertNotIn("-2.0", report)
        self.assertNotIn("2026-11-15", report)

    def test_reports_flat_day_without_transactions(self):
        report = build_daily_report(
            {"balance_usdt": 89.52, "has_position": False, "trade_logs": []},
            date(2026, 11, 16),
        )

        self.assertIn("今日已實現損益：+0.00 USDT", report)
        self.assertIn("今日沒有成交紀錄。", report)
        self.assertIn("目前持倉：空倉", report)

    def test_includes_current_day_risk_pause(self):
        report = build_daily_report(
            {
                "balance_usdt": 96.0,
                "has_position": False,
                "trade_logs": [],
                "daily_risk": {
                    "day": "2026-11-16",
                    "pause_reason": "當日已實現損失達 4.00%",
                    "consecutive_losses": 2,
                },
            },
            date(2026, 11, 16),
        )

        self.assertIn("當日風控：當日已實現損失達 4.00%（連續虧損 2 筆）", report)


if __name__ == "__main__":
    unittest.main()
