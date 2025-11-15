from __future__ import annotations
import hmac, hashlib, time, requests
from urllib.parse import urlencode
from common import *
from typing import Callable, List, Optional


# =======================
# ====== BINANCE UM =====
# =======================

# FULL CLASS REPLACEMENT for BinanceUM
class BinanceUM:
    BASE_URL = "https://fapi.binance.com"

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.key = cfg.BINANCE_KEY
        self.sec = cfg.BINANCE_SEC.encode()
        self.recv_window_ms = cfg.RECV_WINDOW_MS

        self.sess = requests.Session()
        self.sess.headers = {"X-MBX-APIKEY": self.key}
        self._time_offset_ms = 0
        self._sync_time()

    # time
    def _server_time(self):
        r = self.sess.get(f"{self.BASE_URL}/fapi/v1/time", timeout=10); r.raise_for_status()
        return r.json()["serverTime"]

    def now_ms(self) -> int:
        """
        Return the current exchange-synchronized timestamp in milliseconds.

        This method combines the system UTC clock with the last known
        Binance server time offset (`_time_offset_ms`), which is established
        by `_sync_time()` during API initialization or retry.

        Returns
        -------
        int
            Current timestamp in milliseconds, corrected for Binance's
            clock offset.

        Notes
        -----
        - Equivalent to Binance server time (to within a few ms) without
          requiring a live network call.
        - Use this for all local time comparisons that depend on exchange
          timing, such as deciding whether a kline is finalized or signing
          authenticated requests.
        - Relies on `Clock.now_utc_ms()` for local UTC base time.
        """
        return Clock.now_utc_ms() + self._time_offset_ms

    def _sync_time(self):
        st = self._server_time(); lt = Clock.now_utc_ms()
        self._time_offset_ms = int(st) - lt

    # signed
    def _signed_call(self, method:str, path:str, params:dict):
        p = dict(params or {})
        p.setdefault("recvWindow", self.recv_window_ms)
        p["timestamp"] = self.now_ms()
        qs = urlencode(p, doseq=True)
        sig = hmac.new(self.sec, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{self.BASE_URL}{path}?{qs}&signature={sig}"
        r = self.sess.request(method, url, timeout=30)
        if r.status_code == 400 and '"-1021"' in r.text:
            self._sync_time()
            p["timestamp"] = self.now_ms()
            qs = urlencode(p, doseq=True)
            sig = hmac.new(self.sec, qs.encode(), hashlib.sha256).hexdigest()
            url = f"{self.BASE_URL}{path}?{qs}&signature={sig}"
            r = self.sess.request(method, url, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} {r.text}")
        return r.json()
    def _get(self, p, params): return self._signed_call("GET", p, params)

    # endpoints used
    def exchange_info(self): return self.sess.get(f"{self.BASE_URL}/fapi/v1/exchangeInfo", timeout=30).json()
    def agg_trades(self, symbol:str, start_ms:int, end_ms:int, limit:int=1000):
        params = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": min(limit,1000)}
        return self.sess.get(f"{self.BASE_URL}/fapi/v1/aggTrades", params=params, timeout=30).json()

    def klines(self, symbol: str, interval: str,
               start_ms: Optional[int] = None,
               end_ms: Optional[int] = None,
               limit: int = 1000):
        """Fetch klines/candles from Binance Futures API."""
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        r = self.sess.get(f"{self.BASE_URL}/fapi/v1/klines", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def mark_price_klines(self,
                          symbol: str,
                          interval: str,
                          start_ms: Optional[int] = None,
                          end_ms: Optional[int] = None,
                          limit: int = 1000):
        """
        Fetch mark price candles (markPriceKlines) from Binance Futures API.

        Parameters
        ----------
        symbol : str
            Market symbol, e.g. "BTCUSDT".
        interval : str
            Timeframe string, e.g. "1m", "5m", "1h".
        start_ms : int, optional
            Start time in milliseconds.
        end_ms : int, optional
            End time in milliseconds.
        limit : int, default 1000
            Maximum number of candles to return (≤1500 enforced by Binance).

        Returns
        -------
        list
            List of kline arrays: [open_time, open, high, low, close, volume, close_time, ...].

        Notes
        -----
        - Uses the mark price rather than the last trade price.
        - If both start and end are omitted, returns the most recent candles.
        - Automatically caps `limit` to 1500 (Binance API hard limit).
        """
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms

        r = self.sess.get(f"{self.BASE_URL}/fapi/v1/markPriceKlines", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def position_risk(self, symbol:Optional[str]=None):
        params = {"symbol": symbol} if symbol else {}
        return self._get("/fapi/v3/positionRisk", params)
    def account_v2(self): return self._get("/fapi/v2/account", {})
    def income(self, start_ms:int, end_ms:int): return self._get("/fapi/v1/income", {"startTime": start_ms, "endTime": end_ms, "limit": 1000})
