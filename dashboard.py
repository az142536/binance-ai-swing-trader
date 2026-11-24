"""Private, mobile-friendly control panel for the rotation paper trader."""

import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from dotenv import dotenv_values

from config import BASE_DIR, DATA_DIR
from dashboard_control import load_control_state, set_new_entries_paused

STATE_FILE = DATA_DIR / "rotation_state.json"
PID_FILE = DATA_DIR / "dashboard_rotation.pid"
ROBOT_LOG = DATA_DIR / "dashboard_rotation.log"
ACTION_LOG = DATA_DIR / "dashboard_actions.jsonl"
PASSWORD_FILE = Path.home() / ".config" / "binance-ai-swing-trader" / "dashboard.env"
SESSION_SECONDS = 12 * 60 * 60
TOKEN_PATTERN = re.compile(r"(?<!\w)\d{8,12}:[A-Za-z0-9_-]{30,}")

PAGE = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#101827">
  <title>模擬交易控制台</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, -apple-system, "Noto Sans TC", sans-serif; background:#101827; color:#e7edf5; }
    * { box-sizing:border-box; }
    body { margin:0 auto; max-width:900px; padding:20px 14px calc(36px + env(safe-area-inset-bottom)); }
    h1 { font-size:1.55rem; margin:.2rem 0 .4rem; }
    h2 { font-size:1.08rem; margin:0 0 12px; }
    p { line-height:1.55; }
    .muted { color:#aab8ca; font-size:.9rem; }
    .warning { background:#4a351b; border:1px solid #a47730; padding:12px; border-radius:12px; margin:14px 0; }
    .card { background:#1b2738; border:1px solid #314156; border-radius:14px; padding:16px; margin:12px 0; }
    .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
    .metric { background:#111c2c; padding:12px; border-radius:10px; min-width:0; }
    .label { color:#aab8ca; font-size:.82rem; }
    .value { font-size:1.05rem; font-weight:650; overflow-wrap:anywhere; margin-top:5px; }
    .buttons { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
    .topbar { display:flex; justify-content:space-between; align-items:center; gap:12px; }
    .topbar button { padding:9px 14px; min-height:42px; }
    button { border:0; border-radius:10px; padding:13px 10px; min-height:48px; color:white; background:#2463eb; font:inherit; font-weight:650; }
    button.secondary { background:#34445a; }
    button.danger { background:#a93636; }
    button:disabled { opacity:.45; }
    input { width:100%; min-height:48px; border:1px solid #52657e; border-radius:10px; padding:10px 12px; background:#111c2c; color:white; font:inherit; }
    #login { max-width:460px; margin:12vh auto 0; }
    #login button { width:100%; margin-top:10px; }
    .error { color:#ff9898; min-height:1.4em; }
    .scroll { overflow:auto; max-height:430px; }
    .item { border-top:1px solid #314156; padding:10px 0; white-space:pre-wrap; overflow-wrap:anywhere; font-size:.9rem; }
    .hidden { display:none !important; }
    .ok { color:#74e0a1; } .bad { color:#ff9898; }
    @media (min-width:650px) { .grid { grid-template-columns:repeat(4,minmax(0,1fr)); } .buttons { grid-template-columns:repeat(4,minmax(0,1fr)); } }
  </style>
</head>
<body>
  <section id="login" class="card">
    <h1>模擬交易控制台</h1>
    <p class="muted">請輸入控制台密碼。這個頁面只控制模擬交易，不提供實盤下單。</p>
    <form id="login-form">
      <label for="password">控制台密碼</label>
      <input id="password" type="password" autocomplete="current-password" required>
      <button type="submit">登入</button>
      <div id="login-error" class="error" role="status"></div>
    </form>
  </section>
  <main id="app" class="hidden">
    <div class="topbar"><h1>📊 模擬交易控制台</h1><button id="logout" class="secondary">登出</button></div>
    <div id="updated" class="muted"></div>
    <div class="warning"><strong>注意：</strong>「停止機器人」會中斷持倉監控與出場管理；若要暫停新進場，請用「暫停進場」，現有持倉仍會照常管理。</div>
    <section class="card">
      <h2>狀態</h2>
      <div id="metrics" class="grid"></div>
    </section>
    <section class="card">
      <h2>操作</h2>
      <div class="buttons">
        <button id="pause" class="secondary">暫停新進場</button>
        <button id="resume">恢復新進場</button>
        <button id="start">啟動模擬機器人</button>
        <button id="stop" class="danger">停止整個機器人</button>
      </div>
      <p id="message" class="muted" role="status"></p>
    </section>
    <section class="card">
      <h2>最近交易紀錄</h2>
      <div id="trades" class="scroll muted"></div>
    </section>
    <section class="card">
      <h2>機器人最近動作／執行輸出</h2>
      <div id="events" class="scroll muted"></div>
    </section>
    <section class="card">
      <h2>控制台操作紀錄</h2>
      <div id="actions" class="scroll muted"></div>
    </section>
  </main>
  <script>
    let csrfToken = "";
    const $ = (id) => document.getElementById(id);
    async function request(path, body) {
      const options = { method: body === undefined ? "GET" : "POST", credentials: "same-origin", headers: {} };
      if (body !== undefined) {
        options.headers["Content-Type"] = "application/json";
        options.headers["X-Dashboard-CSRF"] = csrfToken;
        options.body = JSON.stringify(body);
      }
      const response = await fetch(path, options);
      const data = await response.json();
      if (!response.ok) {
        const error = new Error(data.error || `HTTP ${response.status}`);
        error.status = response.status;
        throw error;
      }
      return data;
    }
    function showItems(target, items, formatter) {
      const node = $(target);
      node.replaceChildren();
      if (!items || items.length === 0) {
        node.textContent = "目前沒有紀錄。";
        return;
      }
      items.slice().reverse().forEach((item) => {
        const row = document.createElement("div");
        row.className = "item";
        row.textContent = formatter(item);
        node.appendChild(row);
      });
    }
    async function refresh() {
      try {
        const data = await request("/api/status");
        const s = data.state || {};
        const p = s.position || {};
        const fields = [
          ["機器人", data.process.running ? (data.process.managed ? "執行中（控制台管理）" : "執行中（外部啟動）") : "已停止"],
          ["進場狀態", data.control.pause_new_entries ? "新進場已暫停" : "允許新進場"],
          ["可用現金", s.balance_usdt == null ? "尚無帳本" : `${Number(s.balance_usdt).toFixed(2)} USDT`],
          ["持倉", s.has_position ? `${s.active_symbol || "未知"} × ${p.quantity ?? "?"}` : "空倉"],
          ["進場價", s.has_position ? (p.entry_price ?? "—") : "—"],
          ["止損 / 目標", s.has_position ? `${p.stop_loss ?? "—"} / ${p.take_profit ?? "—"}` : "—"],
          ["當日風控", (s.daily_risk && s.daily_risk.pause_reason) || "正常"],
          ["程序 PID", data.process.pid || "—"]
        ];
        const metrics = $("metrics");
        metrics.replaceChildren();
        fields.forEach(([label, value]) => {
          const box = document.createElement("div");
          box.className = "metric";
          const l = document.createElement("div"); l.className = "label"; l.textContent = label;
          const v = document.createElement("div"); v.className = "value"; v.textContent = String(value);
          box.append(l, v); metrics.appendChild(box);
        });
        $("updated").textContent = `更新時間：${data.generated_at} · 現金不含持倉市值`;
        $("pause").disabled = data.control.pause_new_entries || data.process.external;
        $("resume").disabled = !data.control.pause_new_entries || data.process.external;
        $("start").disabled = data.process.running;
        $("stop").disabled = !data.process.managed;
        $("message").textContent = data.process.external
          ? "偵測到不是由控制台啟動的輪動程序。為避免重複交易，控制台不會操作它；請先安全停止舊程序，再由控制台啟動新版。"
          : (data.process.running ? "控制台可安全管理此模擬程序。" : "目前沒有執行中的模擬程序。");
        showItems("trades", s.trade_logs || [], (t) =>
          `${t.time || ""} | ${t.symbol || ""} | ${t.action || ""} | ` +
          (t.pnl_usd == null ? "進場／紀錄" : `${Number(t.pnl_usd).toFixed(2)} USDT`) +
          (t.reason ? ` | ${t.reason}` : ""));
        showItems("events", data.robot_events || [], (line) => line);
        showItems("actions", data.actions || [], (a) => `${a.time} | ${a.action} | ${a.result}`);
      } catch (error) {
        if (error.status === 401) {
          $("app").classList.add("hidden");
          $("login").classList.remove("hidden");
        } else {
          $("message").textContent = error.message;
        }
      }
    }
    $("login-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      $("login-error").textContent = "";
      try {
        const result = await request("/api/login", { password: $("password").value });
        csrfToken = result.csrf_token;
        $("password").value = "";
        $("login").classList.add("hidden");
        $("app").classList.remove("hidden");
        await refresh();
      } catch (error) {
        $("login-error").textContent = error.message;
      }
    });
    async function act(action, confirmText) {
      if (confirmText && !window.confirm(confirmText)) return;
      $("message").textContent = "正在執行…";
      try {
        await request(`/api/${action}`, {});
        $("message").textContent = "操作已送出。";
        await refresh();
      } catch (error) { $("message").textContent = error.message; }
    }
    $("pause").addEventListener("click", () => act("pause"));
    $("resume").addEventListener("click", () => act("resume"));
    $("start").addEventListener("click", () => act("start"));
    $("stop").addEventListener("click", () => act("stop", "停止後現有持倉將不再被自動監控或管理。確定要停止整個模擬機器人嗎？"));
    $("logout").addEventListener("click", async () => {
      try {
        await request("/api/logout", {});
        csrfToken = "";
        $("app").classList.add("hidden");
        $("login").classList.remove("hidden");
      } catch (error) { $("message").textContent = error.message; }
    });
    request("/api/status").then(() => {
      $("login").classList.add("hidden"); $("app").classList.remove("hidden"); refresh();
    }).catch(() => {});
    setInterval(() => { if (!$("app").classList.contains("hidden")) refresh(); }, 10000);
  </script>
</body>
</html>"""


def _process_matches_rotation(pid: int) -> bool:
    proc_dir = Path("/proc") / str(pid)
    try:
        command = (proc_dir / "cmdline").read_bytes().decode(errors="replace").split("\0")
        cwd = (proc_dir / "cwd").resolve()
    except (OSError, RuntimeError):
        return False
    has_main = any(Path(argument).name == "main.py" for argument in command)
    return cwd == BASE_DIR and has_main and "rotate" in command


def find_rotation_processes() -> List[int]:
    proc_root = Path("/proc")
    if not proc_root.exists():
        return []
    pids = []
    for entry in proc_root.iterdir():
        if entry.name.isdigit() and _process_matches_rotation(int(entry.name)):
            pids.append(int(entry.name))
    return sorted(pids)


def redact_log_line(line: str) -> str:
    return TOKEN_PATTERN.sub("[REDACTED_TELEGRAM_TOKEN]", line.rstrip())


def _read_tail(path: Path, limit: int = 80) -> List[str]:
    if not path.exists():
        return []
    with path.open("rb") as log_file:
        log_file.seek(0, os.SEEK_END)
        size = log_file.tell()
        log_file.seek(max(0, size - 65536), os.SEEK_SET)
        content = log_file.read().decode("utf-8", errors="replace")
    return [redact_log_line(line) for line in deque(content.splitlines(), maxlen=limit)]


def _append_json_line(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_actions(limit: int = 40) -> List[Dict[str, Any]]:
    actions = []
    for line in _read_tail(ACTION_LOG, limit):
        try:
            actions.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return actions


def rotation_command() -> List[str]:
    return [
        sys.executable, "-u", str(BASE_DIR / "main.py"), "rotate",
        "--interval", "1h", "--mode", "active", "--sleep", "60",
    ]


def _read_rotation_state() -> Optional[Dict[str, Any]]:
    if not STATE_FILE.exists():
        return None
    with STATE_FILE.open(encoding="utf-8") as state_file:
        state = json.load(state_file)
    if not isinstance(state, dict):
        raise ValueError("輪動帳本格式無效")
    trade_logs = state.get("trade_logs", [])
    if not isinstance(trade_logs, list):
        raise ValueError("輪動交易紀錄格式無效")
    return {
        "balance_usdt": state.get("balance_usdt"),
        "has_position": bool(state.get("has_position", False)),
        "active_symbol": state.get("active_symbol"),
        "position": state.get("position"),
        "daily_risk": state.get("daily_risk", {}),
        "trade_logs": trade_logs[-30:],
        "last_updated": state.get("last_updated"),
    }


class RotationProcessManager:
    def __init__(self) -> None:
        self.lock = threading.Lock()

    def _managed_pid(self) -> Optional[int]:
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None
        if _process_matches_rotation(pid):
            return pid
        try:
            PID_FILE.unlink()
        except FileNotFoundError:
            pass
        return None

    def status(self) -> Dict[str, Any]:
        with self.lock:
            pids = find_rotation_processes()
            managed_pid = self._managed_pid()
            pid = managed_pid or (pids[0] if pids else None)
            return {
                "running": bool(pids),
                "managed": managed_pid is not None and managed_pid in pids,
                "external": bool(pids) and managed_pid not in pids,
                "pid": pid,
            }

    def start(self) -> Dict[str, Any]:
        with self.lock:
            existing = find_rotation_processes()
            managed_pid = self._managed_pid()
            if existing:
                if managed_pid is not None and managed_pid in existing:
                    return {"started": False, "pid": managed_pid, "already_running": True}
                raise RuntimeError("偵測到外部啟動的輪動程序；為避免重複交易，請先安全遷移。")

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            if ROBOT_LOG.exists() and ROBOT_LOG.stat().st_size > 5_000_000:
                os.replace(ROBOT_LOG, ROBOT_LOG.with_suffix(".log.1"))
            log_descriptor = os.open(
                ROBOT_LOG, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600
            )
            os.fchmod(log_descriptor, 0o600)
            try:
                process = subprocess.Popen(
                    rotation_command(),
                    cwd=BASE_DIR,
                    stdin=subprocess.DEVNULL,
                    stdout=log_descriptor,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            finally:
                os.close(log_descriptor)
            pid_temp = PID_FILE.with_suffix(".tmp")
            pid_descriptor = os.open(
                pid_temp, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600
            )
            try:
                with os.fdopen(pid_descriptor, "w", encoding="utf-8") as pid_file:
                    pid_file.write(f"{process.pid}\n")
                    pid_file.flush()
                    os.fsync(pid_file.fileno())
                os.replace(pid_temp, PID_FILE)
            except OSError:
                os.kill(process.pid, signal.SIGTERM)
                raise
            time.sleep(0.25)
            if process.poll() is not None:
                PID_FILE.unlink(missing_ok=True)
                raise RuntimeError("機器人啟動失敗，請查看最近執行輸出。")
            return {"started": True, "pid": process.pid}

    def stop(self) -> Dict[str, Any]:
        with self.lock:
            pid = self._managed_pid()
            if pid is None:
                if find_rotation_processes():
                    raise RuntimeError("此程序不是由控制台啟動；控制台不會強制終止外部程序。")
                return {"stopped": False, "already_stopped": True}
            os.kill(pid, signal.SIGTERM)
            return {"stopping": True, "pid": pid}


def dashboard_snapshot(manager: RotationProcessManager) -> Dict[str, Any]:
    control = load_control_state()
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "process": manager.status(),
        "control": control,
        "state": _read_rotation_state(),
        "robot_events": _read_tail(ROBOT_LOG),
        "actions": _read_actions(),
    }


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple, password: str, manager: RotationProcessManager):
        super().__init__(address, DashboardRequestHandler)
        self.dashboard_password = password
        self.manager = manager
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.login_failures: Dict[str, tuple] = {}
        self.auth_lock = threading.Lock()
        self.control_lock = threading.Lock()


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server: DashboardHTTPServer
    server_version = "PaperDashboard"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _json(self, status: int, data: Dict[str, Any], cookie: Optional[str] = None) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Dict[str, Any]:
        if self.headers.get_content_type() != "application/json":
            raise ValueError("Content-Type 必須是 application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("無效的請求長度") from exc
        if length < 2 or length > 4096:
            raise ValueError("請求內容大小無效")
        try:
            data = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValueError("JSON 格式無效") from exc
        if not isinstance(data, dict):
            raise ValueError("請求內容必須是 JSON 物件")
        return data

    def _session(self, require_csrf: bool) -> Optional[str]:
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except CookieError:
            return None
        morsel = cookie.get("dashboard_session")
        if morsel is None:
            return None
        session_id = morsel.value
        with self.server.auth_lock:
            session = self.server.sessions.get(session_id)
            if not session or session["expires"] < time.time():
                self.server.sessions.pop(session_id, None)
                return None
            if require_csrf and not secrets.compare_digest(
                self.headers.get("X-Dashboard-CSRF", ""), session["csrf"]
            ):
                return None
        return session_id

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)
            return
        if path != "/api/status":
            self._json(404, {"error": "找不到此路徑"})
            return
        if self._session(require_csrf=False) is None:
            self._json(401, {"error": "Unauthorized"})
            return
        try:
            self._json(200, dashboard_snapshot(self.server.manager))
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            self._json(500, {"error": f"讀取儀表板狀態失敗：{type(exc).__name__}"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        try:
            body = self._body()
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return

        if path == "/api/login":
            self._login(body)
            return

        session_id = self._session(require_csrf=True)
        if session_id is None:
            self._json(401, {"error": "登入已逾期或請求驗證失敗，請重新登入。"})
            return

        try:
            if path == "/api/logout":
                with self.server.auth_lock:
                    self.server.sessions.pop(session_id, None)
                self._json(200, {"ok": True}, "dashboard_session=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Strict")
            elif path == "/api/pause":
                with self.server.control_lock:
                    state = set_new_entries_paused(True)
                self._record_action("暫停新進場", "已設定；持倉管理照常")
                self._json(200, {"ok": True, "control": state})
            elif path == "/api/resume":
                with self.server.control_lock:
                    state = set_new_entries_paused(False)
                self._record_action("恢復新進場", "已設定")
                self._json(200, {"ok": True, "control": state})
            elif path == "/api/start":
                result = self.server.manager.start()
                self._record_action("啟動模擬機器人", f"PID {result.get('pid')}")
                self._json(200, {"ok": True, **result})
            elif path == "/api/stop":
                result = self.server.manager.stop()
                self._record_action("停止模擬機器人", f"停止中 PID {result.get('pid', '—')}")
                self._json(200, {"ok": True, **result})
            else:
                self._json(404, {"error": "找不到此操作"})
        except (OSError, RuntimeError) as exc:
            self._json(409, {"error": str(exc)})

    def _login(self, body: Dict[str, Any]) -> None:
        address = self.client_address[0]
        now = time.time()
        with self.server.auth_lock:
            failures, blocked_until = self.server.login_failures.get(address, (0, 0.0))
            if blocked_until > now:
                self._json(429, {"error": "登入嘗試過多，請稍後再試。"})
                return
            supplied = body.get("password", "")
            if not isinstance(supplied, str) or not secrets.compare_digest(
                supplied, self.server.dashboard_password
            ):
                failures += 1
                blocked_until = now + 900 if failures >= 5 else 0.0
                self.server.login_failures[address] = (failures, blocked_until)
                self._json(401, {"error": "密碼錯誤。"})
                return
            self.server.login_failures.pop(address, None)
            self.server.sessions = {
                key: value for key, value in self.server.sessions.items()
                if value["expires"] >= now
            }
            session_id = secrets.token_urlsafe(32)
            csrf_token = secrets.token_urlsafe(32)
            self.server.sessions[session_id] = {
                "csrf": csrf_token,
                "expires": now + SESSION_SECONDS,
            }
        cookie = (
            f"dashboard_session={session_id}; Path=/; Max-Age={SESSION_SECONDS}; "
            "HttpOnly; Secure; SameSite=Strict"
        )
        self._json(200, {"ok": True, "csrf_token": csrf_token}, cookie)

    def _record_action(self, action: str, result: str) -> None:
        _append_json_line(ACTION_LOG, {
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "action": action,
            "result": result,
        })


def load_dashboard_password() -> str:
    if PASSWORD_FILE.exists():
        if os.name == "posix" and PASSWORD_FILE.stat().st_mode & 0o077:
            raise RuntimeError(f"{PASSWORD_FILE} 權限過寬，請設為 600。")
    values = dotenv_values(PASSWORD_FILE)
    password = values.get("DASHBOARD_PASSWORD") or os.environ.get("DASHBOARD_PASSWORD", "")
    if len(password) < 16:
        raise RuntimeError("請在私密 dashboard.env 設定至少 16 個字元的 DASHBOARD_PASSWORD。")
    return password


def main() -> None:
    parser = argparse.ArgumentParser(description="模擬交易手機控制台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host != "127.0.0.1":
        raise SystemExit("控制台只允許綁定 127.0.0.1；請透過私有 HTTPS 代理存取。")
    password = load_dashboard_password()
    server = DashboardHTTPServer((args.host, args.port), password, RotationProcessManager())
    print(f"模擬交易控制台已啟動於 http://{args.host}:{args.port}（僅本機監聽）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n控制台停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
