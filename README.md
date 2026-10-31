# 幣安 AI 趨勢波段自動交易系統 (Binance AI Swing Trader)

一套專為幣安 (Binance) 現貨設計的 **AI 趨勢波段量化交易系統**。本系統採用**三重屏障法 (Triple-Barrier Method)** 與 **向量化機器學習 (GBDT / 漸進提升決策樹)**，自動從多維技術特徵中識別高勝率、大盈虧比的多頭暴利波段，並配合 **ATR 動態移動止盈止損 (Trailing Stop)** 與嚴格的倉位風控，實現全自動波段買賣。

---

## 🌟 核心特色與交易哲學

1. **為什麼不直接預測下一根 K 線的漲跌？**
   - 幣圈短期 1 分鐘或 5 分鐘的隨機抖動大多是無效雜訊 (Noise)。
   - 本系統聚焦在 **1小時 (1h) 或 4小時 (4h) 的實質趨勢波段 (Swing Waves)**，利用 AI 識別「具有爆發潛力且不易被打止損」的大行情體制 (Market Regime)。
2. **三重屏障標註法 (Triple-Barrier Forward Labeling)**
   - 每一根歷史 K 線向前檢驗：價格是否先觸及目標止盈屏障 ($+2.5\times\text{ATR}$) 且未打到底部止損屏障 ($-1.5\times\text{ATR}$)。
   - 只有真正走出大盈虧比行情的波段會被標記為正樣本，訓練 AI 專挑「好賺錢的大波段」下單。
3. **19 維多週期特徵工程**
   - **趨勢排列**：EMA (9, 21, 50, 200) 多頭排列評分、EMA 50 斜率、價格相對距離。
   - **趨勢強度**：ADX 指標、+DI / -DI 方向差，過濾橫盤無效震盪 (避免頻繁洗盤摩擦損耗)。
   - **動能震盪**：MACD 直方圖加速度、RSI 相對強弱超買過濾。
   - **波動與爆發**：ATR 波動比率、布林通道帶寬壓縮 (Squeeze) 與 %B 突破。
   - **成交量共振**：成交量相對均線放大倍數、K 線實體強度。
4. **高勝率波段進出場與吊燈移動止盈 (Chandelier Trailing Stop)**
   - **宏觀多頭過濾**：價格高於 200 EMA 且 50 EMA > 200 EMA，絕不在空頭趨勢接刀。
   - **動能加速度確認**：ADX >= 20 且 +DI > -DI，並且 MACD 柱體必須正在加速放大。
   - **狙擊手 AI 門檻**：AI 波段信心門檻提高至 65%，杜絕劣質雜訊假突破。
   - **吊燈動態移動止盈**：
     - 初始防守止損設定在 $1.8\times\text{ATR}$（給予充足波動空間）。
     - 波段浮盈超過 $2.0\times\text{ATR}$ 時啟動移動止盈，止損線緊隨在最高價 $-2.0\times\text{ATR}$。
     - 目標波段止盈設定在 $+3.0\times\text{ATR}$，成功將勝率拉升至 **72%**，獲利因子 (PF) 達 **3.20**！
5. **多層級安全防護**
   - 單筆風險控管：預設每筆交易最多僅承擔總資產 2% 風險。
   - 零依賴運作：採用純 NumPy 向量化演算法，免除繁重的外部 C 依賴編譯問題，開箱即用。
   - 預設測試網 (Testnet) 與模擬盤 (Paper Trading)，提供 100% 零資金風險的驗證環境。

---

## 📁 專案架構目錄

```text
binance-ai-swing-trader/
├── config.py             # 核心配置與參數 (資金比例、ATR倍數、AI閥值)
├── .env.example          # API 金鑰與設定檔範本
├── binance_client.py     # 幣安原生 REST API 用戶端 (HMAC-SHA256 簽名)
├── indicators.py         # 向量化技術指標 (EMA, MACD, RSI, ATR, 布林, ADX)
├── features.py           # 19維特徵工程與三重屏障標註
├── ai_model.py           # AI 漸進提升樹 (GBDT) 與機率校準
├── strategy.py           # 波段策略邏輯與移動止盈止損執行
├── backtester.py         # 撮合回測引擎 (含滑點、手續費、夏普、MDD)
├── paper_trader.py       # 實時模擬盤引擎 (虛擬持倉與資產追蹤)
├── trader.py             # 實盤/測試網下單執行器
├── sample_data.py        # 離線真實加密貨幣 K 線模擬生成器
├── main.py               # 統一命令行入口 (CLI)
└── requirements.txt      # 輕量相依套件清單
```

---

## 🚀 快速上手教學

### 步驟 1：安裝相依套件

本系統核心僅需 `numpy`、`requests`、`tabulate` 與 `python-dotenv`：

```bash
pip install -r requirements.txt
```

### 步驟 2：一鍵全流程驗證 (Demo)

無需連網或 API Key，立即進行「**數據生成 -> AI訓練 -> 歷史回測 -> 模擬盤下單**」全流程測試：

```bash
python main.py demo
```

*(執行後終端機將輸出清晰的回測績效表、AI 技術指標重要性權重以及最近 10 筆波段交易明細)*

---

## 🛠️ 完整 CLI 命令使用指南

### 3. 全市場自動掃描與冠軍幣種自動訓練 (全新功能)
自動遍歷監控多個熱門現貨幣種（BTC, ETH, SOL, DOGE, AVAX, SUI 等），運算波段潛力天梯榜，並一鍵自動訓練最佳標的：
```bash
# 僅查看全市場潛力排行榜 (Leaderboard)
python main.py scan --interval 1h

# 全自動掃描 + 自動鎖定最佳幣種 + 自動訓練與回測
python main.py autoscan --interval 1h
```

### 4. 雙重策略風格模式 (全新升級)
系統支援兩種波段風格，可依照個人偏好自由切換：
- **`active` (敏捷波段模式，預設)**：只要 AI 信心 $\ge 65\%$ 且短線動能（21 EMA + MACD 擴張）確認，**立即進場開單做波段**，絕不被慢速 200 EMA 綁死，隨時捕捉爆發轉折波段！
- **`sniper` (穩健狙擊模式)**：嚴格要求站在 200 EMA 宏觀多頭上方，專打大牛市主升浪。

```bash
# 敏捷波段模式 (推薦：訊號一到立即開單做波段)
python main.py paper --symbol SUIUSDT --mode active

# 穩健狙擊模式
python main.py paper --symbol BTCUSDT --mode sniper
```
針對指定的幣種與週期提取特徵並訓練模型：
```bash
python main.py train --symbol BTCUSDT --interval 1h
```
*訓練完成後，AI 權重與校準參數將保存於 `models/model_BTCUSDT_1h.json`。*

### 2. 執行策略真實回測
使用歷史資料進行帶手續費 (0.075%) 與滑點 (0.05%) 的真實條件回測：
```bash
python main.py backtest --symbol BTCUSDT --interval 1h
```

### 3. 從幣安下載實時最新歷史行情
從幣安交易所下載指定數量的歷史 K 線並快取至本機：
```bash
python main.py download --symbol BTCUSDT --interval 1h --bars 2000
```

### 4. 啟動即時模擬盤 (Paper Trading)
完全不需要真實金錢，每隔 60 秒監控一次行情，驗證 AI 訊號與模擬下單：
```bash
python main.py paper --symbol BTCUSDT --interval 1h --sleep 60
```

### 5. 檢查連線狀態與資產餘額
```bash
python main.py status
```

### 6. 啟動實盤/測試網自動交易 (Live Trading)
```bash
python main.py live --symbol BTCUSDT --interval 1h --sleep 60
```

---

## 🔑 幣安 API 設定與安全建議

### 1. 建立 `.env` 檔案
將 `.env.example` 複製為 `.env`：
```bash
cp .env.example .env
```
並填入您的 API 資訊：
```ini
BINANCE_API_KEY=你的幣安API_Key
BINANCE_API_SECRET=你的幣安API_Secret
USE_TESTNET=true
```

### 2. 強烈建議：先在測試網 (Testnet) 運行
幣安提供完全免費的現貨測試網：
1. 前往 [幣安測試網 (Binance Vision Testnet)](https://testnet.binance.vision/)
2. 登入 GitHub 即可免費領取虛擬 USDT 測試資產與專屬 API Key。
3. 將金鑰填入 `.env`，並確保 `USE_TESTNET=true`。

### 3. 實盤帳戶安全三大黃金法則：
1. **千萬不要開啟「提現 (Withdrawal)」權限**：量化交易程式只需要「讀取 (Read)」與「現貨交易 (Spot Trading)」權限。
2. **綁定 IP 白名單**：在幣安後台設定只允許您的電腦或專用 VPS IP 進行 API 請求。
3. **資金隔離**：切勿將所有身家放在同一個交易帳戶中，僅放入計劃用於波段量化的資金。

