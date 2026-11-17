"""Send a daily paper-trading summary to a private Telegram chat."""

import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

from config import DATA_DIR

SECRETS_FILE = Path.home() / ".config" / "binance-ai-swing-trader" / "telegram.env"
TELEGRAM_MESSAGE_LIMIT = 4096


def _trade_date(trade: Dict[str, Any]) -> date:
    timestamp = datetime.strptime(trade["time"], "%Y-%m-%d %H:%M:%S")
    return timestamp.date()


def build_daily_report(state: Dict[str, Any], report_date: date) -> str:
    trades: List[Dict[str, Any]] = state.get("trade_logs", [])
    daily_trades = []
    for trade in trades:
        if _trade_date(trade) == report_date:
            daily_trades.append(trade)

    closes = [trade for trade in daily_trades if "pnl_usd" in trade]
    realized_pnl = sum(float(trade["pnl_usd"]) for trade in closes)
    lines = [
        f"📊 模擬交易每日摘要（UTC {report_date.isoformat()}）",
        f"可用現金：{float(state.get('balance_usdt', 0.0)):.2f} USDT",
        f"今日已實現損益：{realized_pnl:+.2f} USDT",
        f"今日成交：{len(daily_trades)} 筆（平倉/部分止盈 {len(closes)} 筆）",
    ]
    risk_state = state.get("daily_risk", {})
    if risk_state.get("day") == report_date.isoformat():
        lines.append(
            f"當日風控：{risk_state.get('pause_reason') or '正常'}"
            f"（連續虧損 {risk_state.get('consecutive_losses', 0)} 筆）"
        )

    position = state.get("position")
    if state.get("has_position") and position:
        lines.append(
            "目前持倉："
            f"{state.get('active_symbol', '未知')}，數量 {float(position.get('quantity', 0.0)):.8g}，"
            f"進場價 {float(position.get('entry_price', 0.0)):.8g}。"
            "現金餘額不含持倉市值。"
        )
    else:
        lines.append("目前持倉：空倉")

    if daily_trades:
        lines.append("今日交易：")
        for trade in daily_trades:
            pnl = f"{float(trade['pnl_usd']):+.2f} USDT" if "pnl_usd" in trade else "進場"
            lines.append(
                f"{trade.get('time', '')} {trade.get('symbol', '')} "
                f"{trade.get('action', '')} {pnl}"
            )
    else:
        lines.append("今日沒有成交紀錄。")

    message = "\n".join(lines)
    if len(message) > TELEGRAM_MESSAGE_LIMIT:
        message = message[: TELEGRAM_MESSAGE_LIMIT - 20] + "\n…（訊息已截短）"
    return message


def send_telegram_message(message: str) -> None:
    load_dotenv(SECRETS_FILE)
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError(f"設定 TELEGRAM_BOT_TOKEN 與 TELEGRAM_CHAT_ID：{SECRETS_FILE}")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        response = requests.post(
            url,
            json={"chat_id": chat_id, "text": message},
            timeout=15,
        )
    except requests.RequestException:
        raise RuntimeError("連線 Telegram Bot API 失敗") from None

    if response.status_code != 200:
        raise RuntimeError(f"Telegram 傳送失敗（HTTP {response.status_code}）")

    result = response.json()
    if not result.get("ok"):
        raise RuntimeError(f"Telegram 傳送失敗：{result.get('description', '未知錯誤')}")


def extract_private_chat_ids(updates: List[Dict[str, Any]]) -> List[int]:
    chat_ids = {
        int(message["chat"]["id"])
        for update in updates
        if (message := update.get("message"))
        and message.get("chat", {}).get("type") == "private"
        and message.get("from", {}).get("id") == message.get("chat", {}).get("id")
    }
    return sorted(chat_ids)


def print_private_chat_ids() -> None:
    load_dotenv(SECRETS_FILE)
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(f"先設定 TELEGRAM_BOT_TOKEN：{SECRETS_FILE}")

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        response = requests.get(url, timeout=15)
    except requests.RequestException:
        raise RuntimeError("連線 Telegram Bot API 失敗") from None

    if response.status_code != 200:
        raise RuntimeError(f"無法讀取 Bot 更新（HTTP {response.status_code}）")
    result = response.json()
    if not result.get("ok"):
        raise RuntimeError(f"無法讀取 Bot 更新：{result.get('description', '未知錯誤')}")

    chat_ids = extract_private_chat_ids(result.get("result", []))
    if not chat_ids:
        raise RuntimeError("尚未收到私人訊息；請先在 Telegram 對 Bot 按 Start 並傳送訊息")
    print("找到私人 Chat ID（只在 VM 終端機顯示）：")
    for chat_id in chat_ids:
        print(chat_id)


def send_daily_report() -> None:
    state_file = DATA_DIR / "rotation_state.json"
    if not state_file.exists():
        raise FileNotFoundError(f"找不到模擬交易帳本：{state_file}")

    import json

    with state_file.open("r", encoding="utf-8") as file:
        state = json.load(file)

    message = build_daily_report(state, datetime.now(timezone.utc).date())
    send_telegram_message(message)
    print("Telegram 每日模擬交易摘要已傳送。")


if __name__ == "__main__":
    if sys.argv[1:] == ["--get-chat-id"]:
        print_private_chat_ids()
    elif not sys.argv[1:]:
        send_daily_report()
    else:
        raise SystemExit("用法：python3 telegram_report.py [--get-chat-id]")
