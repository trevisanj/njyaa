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
from klines_cache import KlinesCache, rows_to_dataframe
import tempfile, os, io

if False:
    from commands import CommandRegistry

__all__ = ["MarketCatalog", "PricePoint", "PriceOracle", "PositionBook", "Reconciler", "Reporter", "Worker",
]

# =======================
# == SYMBOL META/CATALOG
# =======================

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

        ts_ = fmt_ts_ms(ts_ms)
        log().error("oracle.giveup", symbol=symbol, ts=ts_)
        raise RuntimeError(f"Could not backfill price for {symbol} around {ts_}")


# =======================
# ====== PAIR BOOK ======
# =======================

# FULL CLASS REPLACEMENT for PositionBook

class PositionBook:
    def __init__(self, store: Storage, mc: MarketCatalog, oracle: PriceOracle):
        self.store, self.mc, self.oracle = store, mc, oracle

    def open_position(self, num_tok: str, den_tok: Optional[str],
                      usd_notional: int, user_ts: int, note: str = "",
                      risk: Optional[float] = None) -> int:
        num = self.mc.normalize(num_tok)
        den = self.mc.normalize(den_tok) if den_tok else None
        dir_sign = 1 if usd_notional >= 0 else -1
        target = abs(float(usd_notional))
        cfg = self.store.get_config()
        risk_val = cfg["default_risk"] if risk is None else float(risk)
        if risk_val <= 0:
            raise ValueError("risk must be > 0")

        # 1) get/create position (auto-increment id)
        pid = self.store.get_or_create_position(
            num, den, dir_sign, target, risk_val, user_ts, status="OPEN", note=note
        )

        # 2) create leg stubs (single or pair) — qty/price will be filled by backfill job
        self.store.ensure_leg_stub(pid, num)
        if den:
            self.store.ensure_leg_stub(pid, den)

        # 3) enqueue backfill job (will compute qty from USD + price)
        self.store.enqueue_job(f"price:{pid}", "FETCH_ENTRY_PRICES",
                               {"position_id": pid, "user_ts": user_ts}, position_id=pid)

        log().info("Position opened (queued price backfill)", position_id=pid, num=num, den=den,
                 dir=("LONG" if dir_sign > 0 else "SHORT"), usd=target, risk=risk_val)
        return pid

    def size_leg_from_price(self, symbol:str, usd:float, price:float) -> float:
        qty = usd / price
        return self.mc.step_round_qty(symbol, qty)

    def pnl_position(self, pid: str, prices: Dict[str, float]) -> Dict[str, Any]:
        legs = self.store.get_legs(pid)
        if not legs: return {"position_id": pid, "pnl_usd": None, "ok": False}
        pnl = 0.0
        ok = True
        for leg in legs:
            mk = prices.get(leg["symbol"])
            if mk is None: continue
            q = leg["qty"]
            ep = leg["entry_price"]
            if q is None or ep is None:
                pnl = None
                ok = False
                break
            pnl += (mk - ep) * q
        return {"position_id": pid, "pnl_usd": pnl, "ok": ok}


# =======================
# ====== RECONCILER =====
# =======================

class Reconciler:
    def __init__(self, store: Storage, api: BinanceUM, cfg: AppConfig):
        self.store, self.api, self.cfg = store, api, cfg

    def pull_income_today(self):
        tz = self.cfg.TZ_LOCAL
        nowl = datetime.now(tz)
        midnight = datetime(nowl.year, nowl.month, nowl.day, tzinfo=tz).astimezone(timezone.utc)
        start_ms = int(midnight.timestamp() * 1000)
        end_ms = Clock.now_utc_ms()
        rows = self.api.income(start_ms, end_ms)
        if isinstance(rows, list) and rows:
            self.store.insert_income(rows)
            log().info("Income rows inserted", n=len(rows))
        else:
            log().info("Income (none)")

# =======================
# ===== REPORTER =========
# =======================

class Reporter:
    @staticmethod
    def fmt_positions_summary(store, positionbook, rows, prices) -> str:
        total_target = sum(float(r["target_usd"]) for r in rows)
        total_pnl = 0.0
        for r in rows:
            print("BEFORE SHIT")
            res = positionbook.pnl_position(r["position_id"], prices)
            print("AFTER SHIT")
            total_pnl += (res["pnl_usd"] if res["ok"] else 0.0)
        return f"Positions: {len(rows)} | Target ≈ ${total_target:.2f} | PNL ≈ ${total_pnl:.2f}"

    @staticmethod
    def fmt_r_num_den(r):
        return f"{r['num']} / {r['den'] or '-'}"

    @staticmethod
    def fmt_positions_full(store, positionbook, rows, marks, limit:int=100) -> str:
        lines=[]; count=0
        for r in rows:
            if count>=limit: break
            pid=r["position_id"]
            res=positionbook.pnl_position(pid, marks)
            pnl=res["pnl_usd"] if res["ok"] else 0.0
            lines.append(f"{pid} {Reporter.fmt_r_num_den(r)} status={r['status']} target=${r['target_usd']:.2f} PNL=${pnl:.2f}")
            count+=1
        return "\n".join(lines) if lines else "No positions match."

    @staticmethod
    def fmt_positions_legs(store, positionbook, rows, marks, limit:int=100) -> str:
        lines=[]; count=0
        for r in rows:
            if count>=limit: break
            pid=r["position_id"]
            res=positionbook.pnl_position(pid, marks)
            pnl=res["pnl_usd"] if res["ok"] else 0.0
            lines.append(f"{pid} {Reporter.fmt_r_num_den(r)} status={r['status']} target=${r['target_usd']:.2f} PNL=${pnl:.2f}")
            for lg in store.get_legs(pid):
                mk = marks.get(lg["symbol"])
                lines.append(f"  - {lg['symbol']} qty={lg['qty']} entry={lg['entry_price']} (method={lg['price_method']}) mark={mk}")
            count+=1
        return "\n".join(lines) if lines else "No positions match."



# =======================
# ===== JOB QUEUE/WORKER
# =======================

# FULL CLASS REPLACEMENT for Worker
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
                    self._do_price_backfill(jid, payload)
                    self.store.finish_job(jid, ok=True)
                elif task == "PULL_INCOME":
                    rec = Reconciler(self.store, self.api, self.cfg)
                    rec.pull_income_today()
                    self.store.finish_job(jid, ok=True)
                else:
                    raise ValueError(f"Unknown task {task}")
                log().debug("job.done", job_id=jid, task=task)
            except Exception as e:
                log().exc(e, job_id=jid, task=task)
                self.store.finish_job(jid, ok=False, error=str(e))

    def _do_price_backfill(self, job_id: str, payload: dict):
        pid = int(payload["position_id"])
        user_ts = int(payload["user_ts"])
        position = self.store.get_position(pid)
        if not position:
            raise ValueError("position not found")

        intended_usd = float(position["target_usd"])
        num_sym = position["num"]
        den_raw = position["den"]  # may be None
        dir_sign = int(position["dir_sign"])

        den_is_none = (den_raw is None)
        den_str = "" if den_is_none else str(den_raw).strip().upper()
        single_leg = den_is_none or den_str in ("", "1", "UNIT")

        # price for NUM
        pp_num = self.oracle.price_at(num_sym, user_ts)

        def _signed_qty(symbol: str, usd: float, px: float, sign: int) -> float:
            q_abs = self.mc.step_round_qty(symbol, abs(usd) / px)
            return sign * q_abs

        if single_leg:
            q_num = _signed_qty(num_sym, intended_usd, pp_num.price, +1 if dir_sign > 0 else -1)
            self.store.fulfill_leg(pid, num_sym, q_num, pp_num.price, pp_num.price_ts, pp_num.method)
            log().info("Backfill complete (single-leg)", position_id=pid, q_num=q_num)
            return

        # pair
        den_sym = den_str
        pp_den = self.oracle.price_at(den_sym, user_ts)

        q_num = _signed_qty(num_sym, intended_usd, pp_num.price, +1 if dir_sign > 0 else -1)
        q_den = _signed_qty(den_sym, intended_usd, pp_den.price, -1 if dir_sign > 0 else +1)

        self.store.fulfill_leg(pid, num_sym, q_num, pp_num.price, pp_num.price_ts, pp_num.method)
        self.store.fulfill_leg(pid, den_sym, q_den, pp_den.price, pp_den.price_ts, pp_den.method)
        log().info("Backfill complete (pair)", position_id=pid, q_num=q_num, q_den=q_den)
