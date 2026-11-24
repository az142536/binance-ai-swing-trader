"""Shared pause control for the rotation trader and web dashboard."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from config import DATA_DIR

CONTROL_FILE = DATA_DIR / "dashboard_control.json"


def load_control_state(path: Path = CONTROL_FILE) -> Dict[str, Any]:
    if not path.exists():
        return {"pause_new_entries": False}
    with path.open(encoding="utf-8") as control_file:
        data = json.load(control_file)
    if not isinstance(data, dict):
        raise ValueError(f"無效的控制狀態檔：{path}")
    paused = data.get("pause_new_entries", False)
    if not isinstance(paused, bool):
        raise ValueError(f"無效的進場暫停狀態：{path}")
    return {
        "pause_new_entries": paused,
        "updated_at": data.get("updated_at"),
        "updated_by": data.get("updated_by"),
    }


def set_new_entries_paused(paused: bool, path: Path = CONTROL_FILE) -> Dict[str, Any]:
    state = {
        "pause_new_entries": bool(paused),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "updated_by": "dashboard",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temp_path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as control_file:
        json.dump(state, control_file)
        control_file.flush()
        os.fsync(control_file.fileno())
    os.replace(temp_path, path)
    return state
