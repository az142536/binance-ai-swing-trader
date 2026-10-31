"""
幣安 REST API 用戶端模組
支援幣安主網與測試網，包含行情獲取、歷史 K 線下載、帳戶查詢與下單執行。
"""

import time
import hmac
import hashlib
import urllib.parse
from typing import Dict, Any, List, Optional
import requests
from config import Config

class BinanceClient:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, use_testnet: Optional[bool] = None):
        self.api_key = api_key if api_key is not None else Config.API_KEY
        self.api_secret = api_secret if api_secret is not None else Config.API_SECRET
        self.use_testnet = use_testnet if use_testnet is not None else Config.USE_TESTNET
        self.base_url = Config.TESTNET_REST_URL if self.use_testnet else Config.MAINNET_REST_URL
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Binance-AI-Swing-Trader/1.0",
            "Content-Type": "application/x-www-form-urlencoded"
        })
        if self.api_key:
            self.session.headers.update({"X-MBX-APIKEY": self.api_key})

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """產生 HMAC-SHA256 簽名與時間戳"""
        if not self.api_secret:
            raise ValueError("未設定 BINANCE_API_SECRET，無法進行私有交易操作。請檢查 .env 檔案。")
        
        params["timestamp"] = int(time.time() * 1000)
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    def ping(self) -> bool:
        """檢查連線狀態"""
        url = f"{self.base_url}/api/v3/ping"
        try:
            res = self.session.get(url, timeout=5)
            return res.status_code == 200
        except Exception as e:
            return False

    def get_server_time(self) -> int:
        """獲取幣安伺服器時間 (毫秒)"""
        url = f"{self.base_url}/api/v3/time"
        res = self.session.get(url, timeout=5)
        res.raise_for_status()
        return res.json()["serverTime"]

    def get_ticker_price(self, symbol: str, use_mainnet: bool = True) -> float:
        """獲取即時成交價 (支援全球多鏡像自動切換，突破美國地區 IP 限制)"""
        endpoints = [
            "https://data-api.binance.vision",
            Config.MAINNET_REST_URL,
            "https://api1.binance.com",
            "https://api2.binance.com",
            "https://api.binance.us"
        ] if use_mainnet else [self.base_url]

        for base in endpoints:
            try:
                url = f"{base}/api/v3/ticker/price"
                res = self.session.get(url, params={"symbol": symbol.upper()}, timeout=6)
                if res.status_code == 200:
                    return float(res.json()["price"])
            except Exception:
                continue
        raise RuntimeError(f"無法從任何幣安端點獲取 {symbol} 即時報價。")

    def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        use_mainnet: bool = True
    ) -> List[List[Any]]:
        """
        獲取歷史 K 線資料 (支援全球無阻擋多鏡像自動切換)
        優先使用 Binance Vision 全球鏡像，突破美國 IP 阻擋限制
        """
        endpoints = [
            "https://data-api.binance.vision",
            Config.MAINNET_REST_URL,
            "https://api1.binance.com",
            "https://api2.binance.com",
            "https://api.binance.us"
        ] if use_mainnet else [self.base_url]

        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        last_err = None
        for base in endpoints:
            try:
                url = f"{base}/api/v3/klines"
                res = self.session.get(url, params=params, timeout=8)
                if res.status_code == 200:
                    data = res.json()
                    if isinstance(data, list) and len(data) > 0:
                        return data
            except Exception as e:
                last_err = e
                continue

        if last_err:
            raise last_err
        raise RuntimeError(f"無法獲取 {symbol} K 線資料")

    def download_historical_klines(
        self,
        symbol: str,
        interval: str = "1h",
        total_bars: int = 1000
    ) -> List[Dict[str, float]]:
        """分頁下載指定數量的歷史 K 線並整理為乾淨的字典清單"""
        all_bars = []
        end_time = None
        limit_per_call = 1000
        remaining = total_bars

        while remaining > 0:
            fetch_count = min(remaining, limit_per_call)
            raw = self.get_klines(symbol, interval=interval, limit=fetch_count, end_time=end_time)
            if not raw:
                break
            
            all_bars = raw + all_bars
            remaining -= len(raw)
            # 將 end_time 設為最舊一根 K 線的時間 - 1 毫秒
            first_bar_time = int(raw[0][0])
            end_time = first_bar_time - 1

            if len(raw) < fetch_count:
                break
            time.sleep(0.1) # 尊重幣安 Rate limit

        # 去重並依時間遞增排序
        seen = set()
        cleaned_bars = []
        for b in all_bars:
            t = int(b[0])
            if t not in seen:
                seen.add(t)
                cleaned_bars.append({
                    "timestamp": t,
                    "open": float(b[1]),
                    "high": float(b[2]),
                    "low": float(b[3]),
                    "close": float(b[4]),
                    "volume": float(b[5]),
                })
        cleaned_bars.sort(key=lambda x: x["timestamp"])
        return cleaned_bars[-total_bars:]

    def get_account_balances(self) -> Dict[str, Dict[str, float]]:
        """獲取帳戶資產餘額 (需 API Key 與 Secret)"""
        url = f"{self.base_url}/api/v3/account"
        params = self._sign({})
        res = self.session.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        balances = {}
        for b in data.get("balances", []):
            free = float(b["free"])
            locked = float(b["locked"])
            if free > 0 or locked > 0:
                balances[b["asset"]] = {"free": free, "locked": locked, "total": free + locked}
        return balances

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "MARKET",
        quantity: Optional[float] = None,
        quote_order_qty: Optional[float] = None,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        建立現貨委託單
        side: BUY 或 SELL
        order_type: MARKET 或 LIMIT
        """
        url = f"{self.base_url}/api/v3/order"
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
        }
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("限價單必須指定 price")
            params["price"] = f"{price:.2f}"
            params["timeInForce"] = "GTC"
            params["quantity"] = f"{quantity:.6f}"
        elif order_type.upper() == "MARKET":
            if quantity is not None:
                params["quantity"] = f"{quantity:.6f}"
            elif quote_order_qty is not None:
                params["quoteOrderQty"] = f"{quote_order_qty:.2f}"
            else:
                raise ValueError("市價單必須指定 quantity 或 quoteOrderQty")

        signed_params = self._sign(params)
        res = self.session.post(url, data=signed_params, timeout=10)
        res.raise_for_status()
        return res.json()

    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """撤銷委託單"""
        url = f"{self.base_url}/api/v3/order"
        params = self._sign({"symbol": symbol.upper(), "orderId": order_id})
        res = self.session.delete(url, params=params, timeout=10)
        res.raise_for_status()
        return res.json()

