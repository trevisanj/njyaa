# engclasses.py
from __future__ import annotations
import hashlib, math
from typing import Any
from typing import Callable, Dict, Tuple, Optional
from dataclasses import dataclass, field
import sys, json
from common import *
from typing import Callable, List, Optional
from storage import Storage
from datetime import datetime, timezone, timedelta
from binance_um import BinanceUM
import threading, time
from typing import Callable, Dict
from klines_cache import KlinesCache
import tempfile, os, io
from dataclasses import dataclass

if False:
    from commands import CommandRegistry

__all__ = ["MarketCatalog", "PricePoint", "PriceOracle", "Worker", "Position"]

# =======================
# == SYMBOL META/CATALOG
# =======================


@dataclass
class Position:
    position_id: int
    num: str
    den: Optional[str]
    target_usd: float  # signed
    risk: float
    user_ts: Optional[int]
    status: str
    note: Optional[str]
    created_ts: int
    closed_ts: Optional[int] = None

    @classmethod
    def from_row(cls, row: Any) -> "Position":
        return cls(
            position_id=row["position_id"],
            num=row["num"],
            den=row["den"],
            target_usd=row["target_usd"],
            risk=row["risk"],
            user_ts=row["user_ts"],
            status=row["status"],
            note=row["note"] if not isinstance(row, dict) else row.get("note"),
            created_ts=row["created_ts"],
            closed_ts=row["closed_ts"],
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    @property
    def side(self) -> int:
        assert self.target_usd != 0
        return 1 if self.target_usd > 0 else -1

    def get_pair(self, printer_friendly: bool = True) -> str:
        return fmt_pair(self.num, self.den, printer_friendly=printer_friendly)

    def is_open(self) -> bool:
        return self.status == "OPEN"

    @property
    def id(self):
        return self.position_id


class MarketCatalog:
    def __init__(self, api: BinanceUM):
        self.api = api
        self._cache = None
        self._filters: Dict[str, dict] = {}
        self._base_assets: Dict[str, str] = {}  # "STRK"->"STRKUSDT"

    def load(self):
        """
        Populate catalog with **USDT PERPETUAL** only.
        Maps:
          - self._filters[symbol] -> filters dict (LOT_SIZE, etc.)
          - self._base_assets[baseAsset] -> symbol (e.g., 'ETH' -> 'ETHUSDT')
        """
        if self._cache:
            return
        info = self.api.exchange_info()
        self._filters.clear()
        self._base_assets.clear()

        for s in info.get("symbols", []):
            if s.get("status") != "TRADING":
                continue
            if s.get("quoteAsset") != "USDT":
                continue
            if s.get("contractType") != "PERPETUAL":
                continue  # exclude delivery/dated contracts like ETHUSDT_260327

            sym = s["symbol"]  # e.g., ETHUSDT
            # Build filters dict (e.g., LOT_SIZE, PRICE_FILTER, MIN_NOTIONAL)
            fdict = {}
            for f in s.get("filters", []):
                fdict[f["filterType"]] = f
            self._filters[sym] = fdict

            base = s["baseAsset"]  # e.g., ETH
            # Last write wins if multiple listings exist; for PERPETUAL this is unique.
            self._base_assets[base] = sym

        self._cache = info
        log().info("MarketCatalog loaded (USDT PERPETUAL only)",
                 symbols=len(self._filters), bases=len(self._base_assets))

    def normalize(self, token_or_symbol: str) -> str:
        """
        Accepts:
          - base asset (e.g., 'ETH', 'STRK') -> returns PERP symbol ('ETHUSDT', 'STRKUSDT')
          - full symbol ('ETHUSDT') -> validates it’s PERPETUAL
        Rejects anything with '_' (delivery/dated contracts).
        """
        self.load()
        t = token_or_symbol.upper().strip()

        # Reject delivery/dated contracts explicitly
        if "_" in t:
            raise ValueError(f"Delivery/dated contract not supported: {t}")

        # Exact symbol (must be in our PERP filter set)
        if t in self._filters:
            return t

        # Symbol ending with USDT but not in our PERP set -> unknown/not PERP
        if t.endswith("USDT"):
            raise ValueError(f"Unknown/unsupported PERPETUAL symbol: {t}")

        # Base asset -> map to its PERP symbol
        if t in self._base_assets:
            return self._base_assets[t]

        raise ValueError(f"Unknown/unsupported symbol or base asset: {token_or_symbol}")

    def step_round_qty(self, symbol:str, qty:float) -> float:
        f = self._filters[symbol]["LOT_SIZE"]
        step = float(f["stepSize"])
        # round to nearest multiple of step
        return math.floor(qty/step)*step

# =======================
# ===== PRICE ORACLE =====
# =======================

@dataclass
class PricePoint:
    price: float
    price_ts: int
    method: str   # aggTrade|kline|mark_kline


class PriceOracle:
    """
    Backfills a point-in-time price for a futures symbol using multiple sources with
    an expanding time window.

    Strategy (in order):
      1) aggTrades (most precise)
      2) klines(1m) (stable, reproducible proxy)
         proxy when no nearby trade exists or trade APIs return empty.
      3) markPriceKlines(1m) (synthetic fair price)
    """

    def __init__(self, cfg: AppConfig, api: BinanceUM):
        self.cfg = cfg
        self.api = api

    def price_at(self, symbol: str, ts_ms: int) -> PricePoint:
        """
        Backfills a point-in-time price

        Notes:
          - Returned `PricePoint.price_ts` is the chosen series’ native reference time
            (aggTrade time or candle open time), not necessarily equal to `ts_ms`.
        """
        window = max(1, int(self.cfg.PRICE_BACKFILL_WINDOW_SEC))
        max_w = max(window, int(self.cfg.PRICE_BACKFILL_MAX_SEC))
        while window <= max_w:

            # aggTrades
            try:
                start = ts_ms - window * 1000
                end = ts_ms + window * 1000
                log().debug("oracle.try", symbol=symbol, ts=ts_ms, window_sec=window)
                trades = self.api.agg_trades(symbol, start, end)
                if isinstance(trades, list) and trades:
                    closest = min(trades, key=lambda t: abs(int(t["T"]) - ts_ms))
                    price = float(closest["p"]); pts = int(closest["T"])
                    log().debug("oracle.hit", method="aggTrade", price=price, price_ts=pts)
                    return PricePoint(price, pts, "aggTrade")
            except Exception as e:
                log().warn("oracle.fail", method="aggTrades", err=str(e))

            # klines 1m
            try:
                start = ts_ms - window * 1000
                end = ts_ms + window * 1000
                kl = self.api.klines(symbol, "1m", start, end)
                if isinstance(kl, list) and kl:
                    k = min(kl, key=lambda r: abs(int(r[0]) - ts_ms))
                    open_t = int(k[0]);
                    close_t = open_t + 60_000
                    # return `open` if `ts_ms` lies in the first half of the minute, else `close`
                    price = float(k[1] if ts_ms <= (open_t + close_t) // 2 else k[4])
                    log().debug("oracle.hit", method="kline", price=price, candle_open=open_t)
                    return PricePoint(price, open_t, "kline")
            except Exception as e:
                log().warn("oracle.fail", method="klines", err=str(e))
            # markPriceKlines 1m

            try:
                start = ts_ms - window * 1000
                end = ts_ms + window * 1000
                mk = self.api.mark_price_klines(symbol, "1m", start, end)
                if isinstance(mk, list) and mk:
                    k = min(mk, key=lambda r: abs(int(r[0]) - ts_ms))
                    open_t = int(k[0]);
                    close_t = open_t + 60_000
                    price = float(k[1] if ts_ms <= (open_t + close_t) // 2 else k[4])
                    log().debug("oracle.hit", method="mark_kline", price=price, candle_open=open_t)
                    return PricePoint(price, open_t, "mark_kline")
            except Exception as e:
                log().warn("oracle.fail", method="mark_price_klines", err=str(e))

            log().debug("oracle.window.double", window=window*2)
            window *= 2

        ts_ = ts_human(ts_ms)
        log().error("oracle.giveup", symbol=symbol, ts=ts_)
        raise RuntimeError(f"Could not backfill price for {symbol} around {ts_}")


# =======================
# ===== JOB QUEUE/WORKER
# =======================

class Worker:
    def __init__(self, cfg: AppConfig, store: Storage, api: BinanceUM, mc: MarketCatalog, oracle: PriceOracle):
        self.cfg, self.store, self.api, self.mc, self.oracle = cfg, store, api, mc, oracle
        self._stop = threading.Event()

    def stop(self): self._stop.set()

    def run_forever(self, idle_sleep=2):
        log().info("Worker started")

        log().info("Worker started")
        try:
            self.store.recover_stale_jobs()
        except Exception as e:
            log().exc(e, where="worker.recover_stale")

        while not self._stop.is_set():
            job = self.store.fetch_next_job()
            if not job:
                time.sleep(idle_sleep); continue
            jid = job["job_id"]; task = job["task"]
            log().debug("job.start", job_id=jid, task=task)
            try:
                payload = json.loads(job["payload"] or "{}")
                if task == "FETCH_ENTRY_PRICES":
                    self._do_price_backfill_position(jid, payload)
                    self.store.finish_job(jid, ok=True)
                else:
                    raise ValueError(f"Unknown task {task}")
                log().debug("job.done", job_id=jid, task=task)
            except Exception as e:
                log().exc(e, job_id=jid, task=task)
                self.store.finish_job(jid, ok=False, error=str(e))

    def _do_price_backfill_leg(self, job_id: str, payload: dict):
        leg_id = payload["leg_id"]
        user_ts = payload["user_ts"]
        leg = self.store.con.execute("SELECT * FROM legs WHERE leg_id=?", (leg_id,)).fetchone()
        if not leg:
            raise ValueError(f"leg #{leg_id} not found")
        if int(leg["need_backfill"]) == 0:
            log().info("Backfill skip (already filled)", leg_id=leg_id, position_id=leg["position_id"])
            return
        pid = leg["position_id"]
        target_usd = leg["target_usd"]
        if target_usd == 0:
            raise ValueError(f"leg #{leg_id} has zero target_usd")
        sym = leg["symbol"]
        pp = self.oracle.price_at(sym, user_ts)

        def _signed_qty(symbol: str, usd: float, px: float) -> float:
            q_abs = self.mc.step_round_qty(symbol, abs(usd) / px)
            return (1 if usd > 0 else -1) * q_abs

        qty = _signed_qty(sym, target_usd, pp.price)
        self.store.fulfill_leg(leg_id, qty, pp.price, pp.price_ts, pp.method)
        log().info("Backfill complete (leg)", leg_id=leg_id, position_id=pid, symbol=sym, qty=qty)

    def _do_price_backfill_position(self, job_id: str, payload: dict):
        pid = payload["position_id"]
        user_ts = payload["user_ts"]
        legs = self.store.legs_needing_backfill(pid)
        if not legs:
            log().info("Backfill skipped (no pending legs)", position_id=pid)
            return
        for lg in legs:
            lid = int(lg["leg_id"])
            self._do_price_backfill_leg(job_id, {"leg_id": lid, "user_ts": user_ts})
