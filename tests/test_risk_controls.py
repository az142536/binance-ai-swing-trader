import unittest
from datetime import date

from risk_controls import DailyRiskGuard


class DailyRiskGuardTests(unittest.TestCase):
    def test_daily_loss_limit_trips_and_persists_until_next_day(self):
        guard = DailyRiskGuard(daily_loss_limit=0.04, max_consecutive_losses=3)
        guard.start_day(date(2026, 11, 17), equity=100.0)

        self.assertIsNone(guard.record_exit(-2.0))
        reason = guard.record_exit(-2.0)

        self.assertEqual(reason, "當日已實現損失達 4.00%")
        self.assertEqual(guard.record_exit(-1.0), None)
        self.assertEqual(guard.pause_reason, reason)

        guard.start_day(date(2026, 11, 18), equity=96.0)
        self.assertIsNone(guard.pause_reason)
        self.assertEqual(guard.realized_pnl, 0.0)
        self.assertEqual(guard.consecutive_losses, 0)

    def test_three_full_losses_trip_but_partial_fill_does_not_count_streak(self):
        guard = DailyRiskGuard(daily_loss_limit=0.04, max_consecutive_losses=3)
        guard.start_day(date(2026, 11, 17), equity=100.0)

        self.assertIsNone(guard.record_exit(-0.2, is_partial=True))
        self.assertIsNone(guard.record_exit(-0.2))
        self.assertIsNone(guard.record_exit(-0.2))
        self.assertEqual(guard.record_exit(-0.2), "當日連續虧損 3 筆")

    def test_profitable_full_exit_resets_loss_streak(self):
        guard = DailyRiskGuard(daily_loss_limit=0.04, max_consecutive_losses=3)
        guard.start_day(date(2026, 11, 17), equity=100.0)
        guard.record_exit(-0.2)
        guard.record_exit(-0.2)

        guard.record_exit(0.1)

        self.assertEqual(guard.consecutive_losses, 0)

    def test_state_restores_active_daily_pause(self):
        guard = DailyRiskGuard(state={
            "day": "2026-11-17",
            "start_equity": 100.0,
            "realized_pnl": -4.0,
            "consecutive_losses": 2,
            "pause_reason": "當日已實現損失達 4.00%",
        })

        self.assertEqual(guard.to_dict()["pause_reason"], "當日已實現損失達 4.00%")


if __name__ == "__main__":
    unittest.main()
