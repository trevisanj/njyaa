# pip install requests python-dateutil
import os, hmac, hashlib, time, requests
from urllib.parse import urlencode
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import sys
sys.path.append("/home/j/yp/saccakeys")
from saccakeys import keys

B = keys.apikeys["binance"]
BINANCE_KEY = B[0]
BINANCE_SEC = B[1]
BASE = "https://fapi.binance.com"   # USDⓈ-M Futures (perpetuals)

class BinanceUM:
    def __init__(self, key, sec, base=BASE, recv_window_ms=60000):
        self.key = key                 # str (header must be string)
        self.sec = sec.encode()        # bytes for HMAC
        self.base = base
        self.recv_window_ms = recv_window_ms
        self.sess = requests.Session()
        self.sess.headers = {"X-MBX-APIKEY": self.key}
        self._time_offset_ms = 0
        self._sync_time()

    # ---------- time sync ----------
    def _server_time(self):
        r = self.sess.get(f"{self.base}/fapi/v1/time", timeout=10)
        r.raise_for_status()
        return r.json()["serverTime"]

    def _now_ms(self):
        return int(time.time() * 1000) + self._time_offset_ms

    def _sync_time(self):
        st = self._server_time()
        lt = int(time.time() * 1000)
        self._time_offset_ms = int(st) - lt

    # ---------- signed request ----------
    def _signed_get(self, path, params: dict):
        return self._signed_call("GET", path, params)

    def _signed_call(self, method, path, params):
        p = dict(params or {})
        p.setdefault("recvWindow", self.recv_window_ms)
        p["timestamp"] = self._now_ms()
        qs = urlencode(p, doseq=True)
        sig = hmac.new(self.sec, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{self.base}{path}?{qs}&signature={sig}"
        r = self.sess.request(method, url, timeout=30)
        if r.status_code == 400 and '"-1021"' in r.text:
            self._sync_time()
            p["timestamp"] = self._now_ms()
            qs = urlencode(p, doseq=True)
            sig = hmac.new(self.sec, qs.encode(), hashlib.sha256).hexdigest()
            url = f"{self.base}{path}?{qs}&signature={sig}"
            r = self.sess.request(method, url, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} {r.text}")
        return r.json()

    # ---------- endpoints ----------
    def income_today(self, start_ms: int, end_ms: int):
        out, params = [], {"startTime": start_ms, "endTime": end_ms, "limit": 1000}
        data = self._signed_get("/fapi/v1/income", params)
        out.extend(data)
        while data and len(data) == 1000:
            params["startTime"] = data[-1]["time"] + 1
            data = self._signed_get("/fapi/v1/income", params)
            out.extend(data)
        return out

    def user_trades_today(self, symbol: str, start_ms: int, end_ms: int):
        out, params = [], {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000}
        data = self._signed_get("/fapi/v1/userTrades", params)
        out.extend(data)
        while data and len(data) == 1000:
            params = {"symbol": symbol, "fromId": data[-1]["id"] + 1, "limit": 1000}
            data = self._signed_get("/fapi/v1/userTrades", params)
            out.extend(data)
        return out

    def _account_positions_map(self):
        """
        Fallback: pull leverage/margin info from /fapi/v2/account.
        Returns {(symbol, positionSide)-> dict(leverage, marginType, isolated)}.
        """
        acc = self._signed_get("/fapi/v2/account", {})
        m = {}
        for p in acc.get("positions", []):
            sym = p.get("symbol")
            side = p.get("positionSide", "BOTH")
            # leverage sometimes "", make it 0 if so
            lev_raw = p.get("leverage")
            try:
                lev = int(float(lev_raw)) if lev_raw not in (None, "", "null") else 0
            except ValueError:
                lev = 0
            iso_flag = p.get("isolated", False)
            iso = True if iso_flag in (True, "true", "TRUE") else False
            m[(sym, side)] = {
                "leverage": lev,
                "marginType": p.get("marginType"),
                "isolated": iso,
            }
        return m

    def positions(self, symbol: str | None = None, include_zero: bool = False):
        """
        Robust UM-Futures positions snapshot.
        - Uses /fapi/v3/positionRisk for PnL/mark/entry.
        - If leverage/margin fields are missing, enriches via /fapi/v2/account.
        """
        params = {"symbol": symbol} if symbol else {}
        data = self._signed_get("/fapi/v3/positionRisk", params)

        def f(x, key, default=0.0):
            try:
                return float(x.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        rows = []
        missing_meta = []  # track which need leverage/margin enrichment
        for p in data:
            amt = f(p, "positionAmt", 0.0)
            if not include_zero and amt == 0.0:
                continue

            mark = f(p, "markPrice", 0.0)
            entry = f(p, "entryPrice", 0.0)
            side = p.get("positionSide", "BOTH")
            sym = p.get("symbol")

            # parse leverage safely (may be missing or empty string)
            lev_raw = p.get("leverage")
            try:
                lev = int(float(lev_raw)) if lev_raw not in (None, "", "null") else 0
            except ValueError:
                lev = 0

            iso_flag = p.get("isolated", None)  # could be missing or "true"/"false"
            iso = None
            if iso_flag is not None:
                iso = True if iso_flag in (True, "true", "TRUE") else False

            marginType = p.get("marginType")  # may be None
            notional = abs(amt) * mark
            row = {
                "symbol": sym,
                "positionSide": side,
                "positionAmt": amt,
                "entryPrice": entry,
                "markPrice": mark,
                "unRealizedProfit": f(p, "unRealizedProfit", 0.0),
                "leverage": lev if lev else None,
                "isolated": iso,  # may be None here
                "marginType": marginType,  # may be None here
                "notional": notional,
                "updateTime": int(p.get("updateTime", 0) or 0),
            }
            if row["leverage"] is None or row["isolated"] is None or row["marginType"] is None:
                missing_meta.append((sym, side))
            rows.append(row)

        # Enrich missing leverage/margin fields (only if needed)
        if missing_meta:
            meta = self._account_positions_map()
            for r in rows:
                key = (r["symbol"], r["positionSide"])
                if key in meta:
                    if r["leverage"] is None:
                        r["leverage"] = meta[key]["leverage"]
                    if r["isolated"] is None:
                        r["isolated"] = meta[key]["isolated"]
                    if r["marginType"] is None:
                        r["marginType"] = meta[key]["marginType"]

                # final safety defaults
                if r["leverage"] is None:
                    r["leverage"] = 0
                if r["isolated"] is None:
                    r["isolated"] = False
                if r["marginType"] is None:
                    r["marginType"] = "UNKNOWN"

        rows.sort(key=lambda r: r["notional"], reverse=True)
        return rows


def fortaleza_midnight_utc_ms():
    tz = ZoneInfo("America/Fortaleza")
    now_local = datetime.now(tz)
    midnight_local = datetime(year=now_local.year, month=now_local.month, day=now_local.day, tzinfo=tz)
    return int(midnight_local.astimezone(timezone.utc).timestamp()*1000)

if __name__ == "__main__":
    start_ms = fortaleza_midnight_utc_ms()
    end_ms = int(time.time()*1000)

    api = BinanceUM(BINANCE_KEY, BINANCE_SEC)

    # 1) Discover symbols you touched today via income (PnL/commission/funding includes a symbol field)
    income = api.income_today(start_ms, end_ms)
    symbols = sorted({row["symbol"] for row in income if row.get("symbol")})

    # Fallback: if income empty but you still want open-position symbols, uncomment:
    # pos = api._signed_get("/fapi/v3/positionRisk", {})  # only symbols with pos/open orders
    # symbols = sorted({p["symbol"] for p in pos if abs(float(p["positionAmt"])) > 0}) or symbols

    # 2) Pull trades per symbol for today
    all_trades = {}
    for sym in symbols:
        all_trades[sym] = api.user_trades_today(sym, start_ms, end_ms)

    # 3) Quick print summary
    print(f"Date window: {start_ms}..{end_ms}")
    total = sum(len(v) for v in all_trades.values())
    print(f"Trades today: {total} across {len(symbols)} symbols -> {symbols}")
    # Example: print first few fills
    for s in symbols:
        for t in all_trades[s][:3]:
            print(s, t["id"], t["side"], t["qty"], t["price"], "maker" if t["maker"] else "taker")


    # ... after fetching income/symbols/trades ...
    positions = api.positions()  # all non-zero
    print(f"Open positions: {len(positions)}")
    for p in positions:
        print(
            f"{p['symbol']:>10} {p['positionSide']:<5} amt={p['positionAmt']:<12} "
            f"entry={p['entryPrice']:<12} mark={p['markPrice']:<12} "
            f"UPnL={p['unRealizedProfit']:<12} x{p['leverage']} "
            f"{'ISO' if p['isolated'] else 'X'} notional={p['notional']:.2f}"
        )

    # If you want only one symbol:
    # print(api.positions(symbol="BTCUSDT"))
