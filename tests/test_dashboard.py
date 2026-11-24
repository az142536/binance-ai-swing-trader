import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dashboard import redact_log_line, rotation_command
from dashboard_control import load_control_state, set_new_entries_paused
from rotation_trader import SmartRotationTrader


class DashboardControlTests(unittest.TestCase):
    def test_pause_state_is_persisted_and_can_be_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.json"

            self.assertFalse(load_control_state(path)["pause_new_entries"])
            paused = set_new_entries_paused(True, path)
            self.assertTrue(load_control_state(path)["pause_new_entries"])
            self.assertEqual(paused["updated_by"], "dashboard")

            set_new_entries_paused(False, path)
            self.assertFalse(load_control_state(path)["pause_new_entries"])

    def test_corrupt_pause_state_is_reported_instead_of_allowing_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.json"
            path.write_text("{", encoding="utf-8")

            with self.assertRaises(json.JSONDecodeError):
                load_control_state(path)

    def test_robot_log_redacts_telegram_bot_tokens(self):
        token = "1234567890:" + "A" * 35
        self.assertNotIn(token, redact_log_line(f"request failed for {token}"))
        self.assertIn("[REDACTED_TELEGRAM_TOKEN]", redact_log_line(token))

    def test_start_command_is_fixed_to_paper_rotation_mode(self):
        command = rotation_command()
        self.assertIn("rotate", command)
        self.assertNotIn("live", command)
        self.assertNotIn("--capital", command)

    def test_paused_empty_rotation_does_not_scan_or_enter(self):
        trader = object.__new__(SmartRotationTrader)
        trader.has_position = False
        trader.active_symbol = None
        trader.position = None
        trader.balance_usdt = 50.0
        trader.daily_risk = SimpleNamespace(
            pause_reason=None,
            start_day=lambda *_args: None,
        )
        trader._save_state = lambda: None

        with patch(
            "rotation_trader.load_control_state",
            return_value={"pause_new_entries": True},
        ):
            status = trader.run_cycle()

        self.assertEqual(status["phase"], "WAIT")
        self.assertIn("已暫停新進場", status["detail"])


if __name__ == "__main__":
    unittest.main()
