"""
幣安 AI 趨勢波段自動交易系統 - 主入口程式 (CLI Main Entry Point)
支援資料下載、AI 模型訓練、策略回測、模擬交易與實盤自動買賣。
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import time
import argparse
from pathlib import Path
from tabulate import tabulate
from config import Config, DATA_DIR, MODEL_DIR
from sample_data import generate_realistic_crypto_klines, save_klines_to_json, load_klines_from_json
from features import extract_features, create_swing_wave_labels
from ai_model import AISwingModel
from backtester import Backtester
from paper_trader import PaperTrader
from trader import LiveTrader
from binance_client import BinanceClient

def cmd_download(args):
    symbol = args.symbol.upper()
    interval = args.interval
    bars_count = args.bars
    file_path = DATA_DIR / f"{symbol}_{interval}.json"

    print(f"\n[1/2] 正在從幣安下載 {symbol} ({interval}) 歷史 K 線 (共 {bars_count} 根)...")
    client = BinanceClient()
    try:
        bars = client.download_historical_klines(symbol=symbol, interval=interval, total_bars=bars_count)
        if not bars:
            raise RuntimeError("下載資料為空，請檢查網路連線或交易對名稱。")
        save_klines_to_json(bars, file_path)
        print(f"[2/2] [OK] 下載完成！共 {len(bars)} 根 K 線，已儲存至: {file_path}")
        print(f"      時間範圍: {bars[0]['timestamp']} ~ {bars[-1]['timestamp']}")
        print(f"      首根價格: ${bars[0]['close']:,.2f} | 最新價格: ${bars[-1]['close']:,.2f}\n")
    except Exception as e:
        print(f"[X] 下載失敗: {e}")
        print("[提示] 若處於離線或沙盒環境，可執行 'python main.py demo' 自動載入真實模擬行情。")

def cmd_train(args):
    symbol = args.symbol.upper()
    interval = args.interval
    data_path = DATA_DIR / f"{symbol}_{interval}.json"
    model_path = MODEL_DIR / f"model_{symbol}_{interval}.json"

    if not data_path.exists():
        print(f"[提示] 未找到 {data_path}，自動生成 2,400 根模擬行情進行示範...")
        bars = generate_realistic_crypto_klines(num_bars=2400)
        save_klines_to_json(bars, data_path)
    else:
        bars = load_klines_from_json(data_path)

    print(f"\n[AI] [1/3] 開始特徵工程與高勝率趨勢波段標註 (K 線總數: {len(bars)})...")
    X, indicators_cache = extract_features(bars)
    y = create_swing_wave_labels(bars, indicators_cache)

    pos_waves = int(sum(y == 1))
    print(f"      識別出波段機會: {pos_waves} 次 (波段佔比: {pos_waves / len(y):.1%})")

    print(f"[AI] [2/3] 正在訓練 AI 趨勢波段模型 (時序防未來函數劃分 80/20)...")
    ai_model = AISwingModel()
    metrics = ai_model.train(X, y)

    ai_model.save(model_path)
    print(f"[AI] [3/3] 模型訓練完成並已儲存至: {model_path}")

    # 輸出驗證評估指標
    print("\n" + "=" * 55)
    print("           [REPORT] AI 模型驗證集表現")
    print("=" * 55)
    metric_table = [
        ["訓練樣本數", f"{metrics['train_samples']} 根", "驗證樣本數", f"{metrics['val_samples']} 根"],
        ["波段預測準確率", f"{metrics['val_accuracy']:.2f}%", "波段精準度 (Precision)", f"{metrics['val_precision']:.2f}%"],
        ["波段召回率 (Recall)", f"{metrics['val_recall']:.2f}%", "F1 分數", f"{metrics['val_f1']:.4f}"]
    ]
    print(tabulate(metric_table, tablefmt="grid"))

    # 輸出重要指標特徵排名
    print("\n[*] 影響波段成形之關鍵技術指標排名 (Top Features):")
    feat_imp = ai_model.get_feature_importances()
    feat_rows = [[name, f"{score:.2f}%"] for name, score in list(feat_imp.items())[:8]]
    print(tabulate(feat_rows, headers=["技術指標特徵", "AI 權重佔比"], tablefmt="simple"))
    print("=" * 55 + "\n")

def cmd_backtest(args):
    symbol = args.symbol.upper()
    interval = args.interval
    mode = getattr(args, "mode", Config.SWING_MODE)
    data_path = DATA_DIR / f"{symbol}_{interval}.json"
    model_path = MODEL_DIR / f"model_{symbol}_{interval}.json"

    if not data_path.exists():
        bars = generate_realistic_crypto_klines(num_bars=2400)
        save_klines_to_json(bars, data_path)
    else:
        bars = load_klines_from_json(data_path)

    ai_model = AISwingModel()
    if not model_path.exists():
        print(f"[提示] 未找到模型 {model_path}，正在自動快速訓練...")
        X, ind = extract_features(bars)
        y = create_swing_wave_labels(bars, ind)
        ai_model.train(X, y)
        ai_model.save(model_path)
    else:
        ai_model.load(model_path)

    print(f"\n[*] 正在對 {symbol} ({interval}) 執行真實條件回測 [模式: {mode.upper()}] (含 0.075% 手續費與 0.05% 滑點)...")
    from strategy import AISwingStrategy
    strat = AISwingStrategy(mode=mode)
    tester = Backtester(strategy=strat)
    res = tester.run(bars, ai_model)
    tester.print_summary(res)

def cmd_demo(args):
    """一鍵完整自動化示範 (資料產生 -> 特徵萃取 -> AI 訓練 -> 策略回測 -> 模擬交易)"""
    print("\n" + "=" * 65)
    print("   [DEMO] 幣安 AI 趨勢波段自動交易系統 - 全流程驗證")
    print("=" * 65)

    symbol = "BTCUSDT"
    interval = "1h"
    data_path = DATA_DIR / f"{symbol}_{interval}.json"
    model_path = MODEL_DIR / f"model_{symbol}_{interval}.json"

    print("\n[步驟 1] 生成 2,400 根具備真實加密貨幣波動特性之 1h K 線資料...")
    bars = generate_realistic_crypto_klines(num_bars=2400, start_price=65000.0)
    save_klines_to_json(bars, data_path)
    print(f"         已保存至 {data_path} (價格範圍: ${min(b['low'] for b in bars):,.2f} ~ ${max(b['high'] for b in bars):,.2f})")

    print("\n[步驟 2] 提取 19 維動量、趨勢與波動指標，標註多頭趨勢波段...")
    X, indicators_cache = extract_features(bars)
    y = create_swing_wave_labels(bars, indicators_cache)
    print(f"         特徵矩陣大小: {X.shape} | 標註高獲利波段數: {int(sum(y == 1))}")

    print("\n[步驟 3] 訓練 AI 趨勢波段 Ensemble 分類器...")
    ai_model = AISwingModel()
    metrics = ai_model.train(X, y)
    ai_model.save(model_path)
    print(f"         驗證集準確率: {metrics['val_accuracy']:.2f}% | 精準度: {metrics['val_precision']:.2f}% | F1: {metrics['val_f1']:.4f}")

    print("\n[步驟 4] 執行真實撮合回測 (帶動態移動止盈止損與風險管理)...")
    tester = Backtester()
    res = tester.run(bars, ai_model)
    tester.print_summary(res)

    print("[步驟 5] 模擬盤即時掃描示範 (最後 5 根 K 線決策):")
    trader = PaperTrader(symbol=symbol, interval=interval)
    trader.load_model(model_path)
    for step in range(len(bars) - 5, len(bars)):
        sub_bars = bars[:step + 1]
        status = trader.run_step(sub_bars)
        act_info = f"[{status['action']}] {status['action_detail']}" if status['action'] != 'HOLD' else "[HOLD] 等待最佳波段進場點"
        print(f"  • K線 #{step+1:04d} | 當前價: ${status['close_price']:,.2f} | AI波段信心: {status['ai_bullish_prob']:.1%} | {act_info}")

    print("\n" + "=" * 65)
    print("[OK] 一鍵全流程驗證圓滿完成！")
    print(f"     帳戶當前資產總值: ${trader.balance_usdt:,.2f} USDT")
    print("     程式已完全就緒，可使用 'python main.py live' 或 'python main.py paper' 進行運作。")
    print("=" * 65 + "\n")

def ensure_model_trained(symbol: str, interval: str) -> AISwingModel:
    """自動確保指定幣種的專屬 AI 模型存在，若不存在則自動訓練"""
    model_path = MODEL_DIR / f"model_{symbol}_{interval}.json"
    ai_model = AISwingModel()
    if not model_path.exists():
        print(f"\n[提示] 尚未建立 {symbol} ({interval}) 的專屬 AI 模型，系統正在為您自動下載數據並快速訓練...")
        data_path = DATA_DIR / f"{symbol}_{interval}.json"
        if not data_path.exists():
            from scanner import SymbolScanner
            scanner = SymbolScanner(interval=interval)
            bars = scanner.get_symbol_klines(symbol, bars_count=2400)
        else:
            bars = load_klines_from_json(data_path)
        
        X, ind = extract_features(bars)
        y = create_swing_wave_labels(bars, ind)
        ai_model.train(X, y)
        ai_model.save(model_path)
        print(f"[OK] {symbol} 專屬 AI 模型已自動訓練完成並就緒！\n")
    else:
        ai_model.load(model_path)
    return ai_model

def cmd_paper(args):
    symbol = args.symbol.upper()
    interval = args.interval
    mode = getattr(args, "mode", Config.SWING_MODE)
    print(f"\n[PAPER] 啟動 {symbol} ({interval}) 模擬盤交易引擎 [風格: {mode.upper()}]...")
    trader = PaperTrader(symbol=symbol, interval=interval)
    trader.strategy.mode = mode
    ai_model = ensure_model_trained(symbol, interval)
    trader.model = ai_model

    client = BinanceClient()
    loops = args.loops
    count = 0
    while True:
        try:
            try:
                raw_bars = client.download_historical_klines(symbol, interval=interval, total_bars=250)
            except Exception:
                # 離線或沙盒環境自動讀取本機快取資料
                data_file = DATA_DIR / f"{symbol}_{interval}.json"
                if data_file.exists():
                    raw_bars = load_klines_from_json(data_file)[-250:]
                else:
                    raise
            status = trader.run_step(raw_bars)

            if status['has_position']:
                pos_desc = f"持倉中 (進場: ${status['position_entry']:,.2f}, 浮盈: {status['floating_pnl_pct']:+.2f}%)"
            else:
                if status['ai_bullish_prob'] >= Config.AI_CONFIDENCE_THRESHOLD and status.get('action_detail'):
                    pos_desc = f"空倉觀望 [風控保護: {status['action_detail']}]"
                else:
                    pos_desc = "空倉觀望"

            print(f"[{status['timestamp']}] {symbol} 價格: ${status['close_price']:,.2f} | AI波段機率: {status['ai_bullish_prob']:.1%} | 狀態: {pos_desc} | 總資產: ${status['total_equity']:,.2f} USDT")
            if status['action'] != "HOLD":
                print(f"   [TRIGGER] 觸發交易訊號: {status['action_detail']}")

            count += 1
            if loops > 0 and count >= loops:
                break
            time.sleep(args.sleep)
        except KeyboardInterrupt:
            print("\n模擬盤已被手動停止。")
            break
        except Exception as e:
            print(f"獲取實時行情發生異常: {e}")
            time.sleep(10)

def cmd_live(args):
    symbol = args.symbol.upper()
    interval = args.interval
    mode = getattr(args, "mode", Config.SWING_MODE)
    ai_model = ensure_model_trained(symbol, interval)

    trader = LiveTrader(symbol=symbol, interval=interval)
    trader.strategy.mode = mode
    try:
        conn = trader.check_connection()
        print("\n" + "=" * 50)
        print(f"         [LIVE] 幣安實盤/測試網自動交易引擎啟動 [風格: {mode.upper()}]")
        print("=" * 50)
        for k, v in conn.items():
            print(f"  {k}: {v}")
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"[X] 連線失敗: {e}")
        return

    trader.model = ai_model
    loops = args.loops
    count = 0

    while True:
        try:
            res = trader.execute_cycle()
            act = res["action_result"]
            print(f"[{res['timestamp']}] {symbol}: ${res['close_price']:,.2f} | AI波段機率: {res['ai_bullish_prob']:.1%} | 行動: {act['action']}")
            if act["action"] != "HOLD":
                print(f"   [ORDER] 訂單執行回報: {act}")

            count += 1
            if loops > 0 and count >= loops:
                break
            time.sleep(args.sleep)
        except KeyboardInterrupt:
            print("\n實盤交易已被手動終止。")
            break
        except Exception as e:
            print(f"實盤循環執行異常: {e}")
            time.sleep(15)

def cmd_scan(args):
    """掃描市場熱門幣種並輸出潛力天梯榜"""
    from scanner import SymbolScanner
    scanner = SymbolScanner(interval=args.interval)
    print(f"\n🔍 正在掃描全市場熱門現貨幣種 ({args.interval})，計算波段爆發潛力天梯榜...")
    ranked = scanner.scan_and_rank()
    scanner.print_leaderboard(ranked)

def cmd_autoscan(args):
    """全自動偵測最佳潛力幣種並自動訓練與回測"""
    from scanner import auto_scan_and_train_best
    auto_scan_and_train_best(interval=args.interval)

def cmd_rotate(args):
    """啟動全市場空倉智能波段輪動交易系統"""
    from rotation_trader import SmartRotationTrader
    interval = args.interval
    mode = getattr(args, "mode", Config.SWING_MODE)
    capital = getattr(args, "capital", Config.INITIAL_CAPITAL)
    sleep_sec = args.sleep
    loops = args.loops

    print("\n" + "=" * 68)
    print("       🚀 幣安全市場空倉智能波段輪動交易系統 (Smart Rotation)")
    print("       策略哲學：【持倉專注吃滿大波段，空倉全網自動切換最強起漲幣】")
    print(f"       週期: {interval} | 風格: {mode.upper()} | 資金規模: ${capital:,.2f} USDT | 間隔: 每 {sleep_sec} 秒")
    print("=" * 68 + "\n")

    trader = SmartRotationTrader(interval=interval, mode=mode, capital=capital)
    count = 0
    while True:
        try:
            status = trader.run_cycle()
            phase = status["phase"]
            ts = status["timestamp"]
            equity = status["total_equity"]
            detail = status["detail"]

            if phase == "BUY":
                print(f"[{ts}] 🟢 [進場買入] {detail} | 總資產: ${equity:,.2f} USDT")
            elif phase == "EXIT":
                print(f"[{ts}] 🔴 [波段平倉] {detail} | 總資產: ${equity:,.2f} USDT")
            elif phase == "HOLD":
                sym = status["symbol"]
                px = status["close_price"]
                prob = status.get("ai_bullish_prob", 0.0)
                print(f"[{ts}] 🔵 [持倉專注] {sym} (${px:,.2f}, AI信心: {prob:.1%}) | {detail} | 總資產: ${equity:,.2f} USDT")
            elif phase == "WAIT":
                print(f"[{ts}] ⚪ [空倉觀望] {detail} | 總資產: ${equity:,.2f} USDT")

            count += 1
            if loops > 0 and count >= loops:
                break
            time.sleep(sleep_sec)
        except KeyboardInterrupt:
            print("\n智能輪動交易已被手動終止。")
            break
        except Exception as e:
            print(f"輪動循環異常: {e}")
            time.sleep(10)

def cmd_report(args):
    """查詢當前帳戶資產與歷史波段獲利戰績"""
    state_file = DATA_DIR / "rotation_state.json"
    if not state_file.exists():
        print("\n[提示] 尚未找到輪動交易帳本，機器人可能剛啟動正在進行第一輪掃描...")
        return

    import json
    with open(state_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    balance = float(data.get("balance_usdt", 100.0))
    has_pos = data.get("has_position", False)
    active_sym = data.get("active_symbol", "無")
    pos_data = data.get("position")
    trade_logs = data.get("trade_logs", [])
    initial_cap = 100.0
    total_pnl = balance - initial_cap
    total_pnl_pct = (total_pnl / initial_cap) * 100.0

    print("\n" + "=" * 65)
    print("         📊 幣安 AI 趨勢波段交易系統 - 最新戰績報表")
    print("=" * 65)
    print(f"  初始本金:       ${initial_cap:,.2f} USDT")
    print(f"  當前可用現金:   ${balance:,.2f} USDT")
    print(f"  累計實現淨收益: {total_pnl_pct:+.2f}% (${total_pnl:+,.2f} USDT)")
    print(f"  當前持倉狀態:   {'【' + str(active_sym) + ' 持倉中】' if has_pos else '【100% USDT 空倉觀望中】'}")
    if has_pos and pos_data:
        print(f"  持倉詳情:       進場價: ${pos_data['entry_price']:,.2f} | 止損: ${pos_data['stop_loss']:,.2f} | 目標: ${pos_data['take_profit']:,.2f}")
    print("=" * 65)

    if trade_logs:
        print("\n📜 最近完成的波段交易紀錄 (Recent Closed Trades):")
        rows = []
        for t in trade_logs[-10:]:
            act = t.get("action", "")
            sym = t.get("symbol", "")
            tm = t.get("time", "")
            px = f"${t.get('price', 0.0):,.2f}"
            pnl_u = f"${t.get('pnl_usd', 0.0):+,.2f}" if "pnl_usd" in t else "-"
            pnl_p = f"{t.get('pnl_pct', 0.0):+.2f}%" if "pnl_pct" in t else "-"
            reason = t.get("reason", "")
            rows.append([tm, sym, act, px, pnl_u, pnl_p, reason])
        headers = ["時間", "幣種", "動作", "價格", "盈虧金額", "盈虧%", "觸發原因"]
        print(tabulate(rows, headers=headers, tablefmt="simple"))
        print()

def cmd_status(args):
    client = BinanceClient()
    print("\n[*] 正在檢查幣安連線狀態與帳戶資產...")
    try:
        ping_ok = client.ping()
        mode_str = "幣安測試網 (Testnet)" if client.use_testnet else "幣安主網 (Mainnet)"
        print(f"  網路模式: {mode_str}")
        print(f"  連線狀態: {'[OK] 正常 (Ping OK)' if ping_ok else '[X] 無法連線'}")
        
        if client.api_key and client.api_secret:
            balances = client.get_account_balances()
            print("\n[*] 當前帳戶非零資產餘額:")
            rows = [[asset, f"{info['free']:.6f}", f"{info['locked']:.6f}", f"{info['total']:.6f}"] for asset, info in balances.items()]
            if rows:
                print(tabulate(rows, headers=["資產 (Asset)", "可用 (Free)", "鎖定 (Locked)", "總計 (Total)"], tablefmt="simple"))
            else:
                print("  (帳戶尚無資產)")
        else:
            print("  [提示] 尚未在 .env 設定 API Key，無法查詢帳戶資產。")
    except Exception as e:
        print(f"[X] 查詢失敗: {e}")
    print()

def main():
    parser = argparse.ArgumentParser(description="幣安 AI 趨勢波段自動交易系統")
    subparsers = parser.add_subparsers(dest="command", help="可用子命令")

    # demo
    subparsers.add_parser("demo", help="一鍵完整流程驗證示範")

    # download
    p_dl = subparsers.add_parser("download", help="從幣安下載歷史行情 K 線")
    p_dl.add_argument("--symbol", default="BTCUSDT", help="交易對 (如 BTCUSDT, ETHUSDT)")
    p_dl.add_argument("--interval", default="1h", help="K 線週期 (如 1h, 4h, 15m)")
    p_dl.add_argument("--bars", type=int, default=2000, help="下載 K 線根數")

    # train
    p_tr = subparsers.add_parser("train", help="提取特徵並訓練 AI 趨勢波段模型")
    p_tr.add_argument("--symbol", default="BTCUSDT", help="交易對")
    p_tr.add_argument("--interval", default="1h", help="K 線週期")

    # backtest
    p_bt = subparsers.add_parser("backtest", help="執行策略回測評估")
    p_bt.add_argument("--symbol", default="BTCUSDT", help="交易對")
    p_bt.add_argument("--interval", default="1h", help="K 線週期")
    p_bt.add_argument("--mode", default=Config.SWING_MODE, choices=["active", "sniper"], help="波段風格: active (敏捷模式，有訊號即開單做波段) 或 sniper (穩健模式)")

    # paper
    p_pp = subparsers.add_parser("paper", help="啟動即時模擬盤交易")
    p_pp.add_argument("--symbol", default="BTCUSDT", help="交易對")
    p_pp.add_argument("--interval", default="1h", help="K 線週期")
    p_pp.add_argument("--mode", default=Config.SWING_MODE, choices=["active", "sniper"], help="波段風格: active 或 sniper")
    p_pp.add_argument("--sleep", type=int, default=60, help="每輪循環間隔秒數")
    p_pp.add_argument("--loops", type=int, default=0, help="執行輪數 (0 代表無限循環)")

    # live
    p_lv = subparsers.add_parser("live", help="啟動幣安實盤/測試網自動交易")
    p_lv.add_argument("--symbol", default="BTCUSDT", help="交易對")
    p_lv.add_argument("--interval", default="1h", help="K 線週期")
    p_lv.add_argument("--mode", default=Config.SWING_MODE, choices=["active", "sniper"], help="波段風格: active 或 sniper")
    p_lv.add_argument("--sleep", type=int, default=60, help="每輪循環間隔秒數")
    p_lv.add_argument("--loops", type=int, default=0, help="執行輪數 (0 代表無限循環)")

    # scan
    p_sc = subparsers.add_parser("scan", help="掃描全市場幣種波段潛力天梯榜")
    p_sc.add_argument("--interval", default="1h", help="K 線週期 (如 1h, 4h)")

    # autoscan
    p_asc = subparsers.add_parser("autoscan", help="自動偵測最具賺錢潛力幣種並自動完成訓練與回測")
    p_asc.add_argument("--interval", default="1h", help="K 線週期 (如 1h, 4h)")

    # rotate (智能輪動)
    p_rt = subparsers.add_parser("rotate", help="啟動全市場空倉智能波段輪動交易 (持倉專注吃波段，空倉全自動挑最強起漲幣)")
    p_rt.add_argument("--interval", default="1h", help="K 線週期 (如 1h, 4h)")
    p_rt.add_argument("--mode", default=Config.SWING_MODE, choices=["active", "sniper"], help="波段風格: active 或 sniper")
    p_rt.add_argument("--capital", type=float, default=Config.INITIAL_CAPITAL, help="初始資金 (USDT，預設 100)")
    p_rt.add_argument("--sleep", type=int, default=60, help="每輪循環間隔秒數")
    p_rt.add_argument("--loops", type=int, default=0, help="執行輪數 (0 代表無限循環)")

    # report (戰績報表)
    subparsers.add_parser("report", help="即時查詢當前資產餘額與歷史波段獲利戰績")

    # status
    subparsers.add_parser("status", help="查詢 API 連線狀態與帳戶資產")

    args = parser.parse_args()

    if not args.command:
        cmd_demo(args)
        return

    commands = {
        "demo": cmd_demo,
        "download": cmd_download,
        "train": cmd_train,
        "backtest": cmd_backtest,
        "paper": cmd_paper,
        "live": cmd_live,
        "scan": cmd_scan,
        "autoscan": cmd_autoscan,
        "rotate": cmd_rotate,
        "report": cmd_report,
        "status": cmd_status
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
