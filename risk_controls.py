"""Daily loss and consecutive-loss guard for paper rotation trading."""

from datetime import date
from typing import Any, Dict, Optional


class DailyRiskGuard:
    def __init__(
        self,
        daily_loss_limit: float = 0.04,
        max_consecutive_losses: int = 3,
        state: Optional[Dict[str, Any]] = None,
    ):
        self.daily_loss_limit = daily_loss_limit
        self.max_consecutive_losses = max_consecutive_losses
        state = state or {}
        self.day: Optional[str] = state.get("day")
        self.start_equity: Optional[float] = state.get("start_equity")
        self.realized_pnl = float(state.get("realized_pnl", 0.0))
        self.consecutive_losses = int(state.get("consecutive_losses", 0))
        self.pause_reason: Optional[str] = state.get("pause_reason")

    def start_day(self, day: date, equity: float) -> None:
        day_key = day.isoformat()
        if self.day == day_key:
            return
        self.day = day_key
        self.start_equity = max(float(equity), 0.0)
        self.realized_pnl = 0.0
        self.consecutive_losses = 0
        self.pause_reason = None

    def record_exit(self, pnl: float, is_partial: bool = False) -> Optional[str]:
        self.realized_pnl += pnl
        if not is_partial:
            if pnl < 0:
                self.consecutive_losses += 1
            elif pnl > 0:
                self.consecutive_losses = 0
        return self._update_pause_reason()

    def _update_pause_reason(self) -> Optional[str]:
        if self.pause_reason:
            return None
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.pause_reason = f"當日連續虧損 {self.consecutive_losses} 筆"
        elif (
            self.start_equity is not None
            and self.start_equity > 0
            and self.realized_pnl <= -self.start_equity * self.daily_loss_limit
        ):
            loss_pct = abs(self.realized_pnl / self.start_equity) * 100.0
            self.pause_reason = f"當日已實現損失達 {loss_pct:.2f}%"
        else:
            return None
        return self.pause_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day": self.day,
            "start_equity": self.start_equity,
            "realized_pnl": self.realized_pnl,
            "consecutive_losses": self.consecutive_losses,
            "pause_reason": self.pause_reason,
        }
