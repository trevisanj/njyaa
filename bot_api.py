# v03

from __future__ import annotations
import hashlib, math
from typing import Any
from typing import Callable, Dict, Tuple, Optional
from dataclasses import dataclass, field
import sys, json, threading
from common import *
from typing import Callable, List, Optional
from storage import Storage
import re
from datetime import datetime, timezone, timedelta
from binance_um import BinanceUM
import threading, time
from typing import Callable, Dict
from klines_cache import KlinesCache, rows_to_dataframe
# inside bot_api.py (or a separate charts.py helper imported there)
import tempfile, os, io
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

if False:
    from commands import CommandRegistry


class InternalScheduler:
    def __init__(self):
        self._jobs: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    def schedule_every(self, name: str, interval_sec: int, fn: Callable[[], None], run_immediately: bool = False):
        nxt = time.time() if run_immediately else (time.time() + max(1, interval_sec))
        with self._lock:
            self._jobs[name] = {"interval": max(1, int(interval_sec)), "fn": fn, "next": nxt}

    def cancel(self, name: str):
        with self._lock:
            self._jobs.pop(name, None)

    def start(self, thread_name: str = "rv-scheduler"):
        if self._thr and self._thr.is_alive(): return
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, name=thread_name, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr: self._thr.join(timeout=3)

    def _run(self):
        while not self._stop.is_set():
            now = time.time()
            todo: list[tuple[str, Callable[[], None], int]] = []
            with self._lock:
                for name, j in self._jobs.items():
                    if now >= j["next"]:
                        todo.append((name, j["fn"], j["interval"]))
                        j["next"] = now + j["interval"]
            for name, fn, _itv in todo:
                try:
                    fn()
                except Exception as e:
                    log().exc(e, where="scheduler.job", job=name)
            time.sleep(0.5)


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
    def __init__(self, cfg: AppConfig, api: BinanceUM):
        self.cfg = cfg
        self.api = api

    def price_at(self, symbol: str, ts_ms: int) -> PricePoint:
        window = max(1, int(self.cfg.PRICE_BACKFILL_WINDOW_SEC))
        max_w = max(window, int(self.cfg.PRICE_BACKFILL_MAX_SEC))
        while window <= max_w:
            start = ts_ms - window * 1000
            end = ts_ms + window * 1000
            log().debug("oracle.try", symbol=symbol, ts=ts_ms, window_sec=window)

            # aggTrades
            try:
                trades = self.api.agg_trades(symbol, start, end)
                if isinstance(trades, list) and trades:
                    closest = min(trades, key=lambda t: abs(int(t["T"]) - ts_ms))
                    price = float(closest["p"]); pts = int(closest["T"])
                    log().debug("oracle.hit", method="aggTrade", price=price, price_ts=pts)
                    return PricePoint(price, pts, "aggTrade")
            except Exception as e:
                log().debug("oracle.fail", method="aggTrades", err=str(e))

            # klines 1m
            try:
                kl = self.api.klines(symbol, "1m", ts_ms - 60_000, ts_ms + 60_000)
                if isinstance(kl, list) and kl:
                    k = min(kl, key=lambda r: abs(int(r[0]) - ts_ms))
                    open_t = int(k[0]); close_t = open_t + 60_000
                    price = float(k[1] if ts_ms <= (open_t + close_t)//2 else k[4])
                    log().debug("oracle.hit", method="kline", price=price, candle_open=open_t)
                    return PricePoint(price, open_t, "kline")
            except Exception as e:
                log().debug("oracle.fail", method="klines", err=str(e))

            # markPriceKlines 1m
            try:
                mk = self.api.mark_price_klines(symbol, "1m", ts_ms - 60_000, ts_ms + 60_000)
                if isinstance(mk, list) and mk:
                    k = min(mk, key=lambda r: abs(int(r[0]) - ts_ms))
                    open_t = int(k[0]); close_t = open_t + 60_000
                    price = float(k[1] if ts_ms <= (open_t + close_t)//2 else k[4])
                    log().debug("oracle.hit", method="mark_kline", price=price, candle_open=open_t)
                    return PricePoint(price, open_t, "mark_kline")
            except Exception as e:
                log().debug("oracle.fail", method="mark_price_klines", err=str(e))

            window *= 2

        log().error("oracle.giveup", symbol=symbol, ts=ts_ms)
        raise RuntimeError(f"Could not backfill price for {symbol} around {ts_ms}")


# =======================
# ====== PAIR BOOK ======
# =======================

# FULL CLASS REPLACEMENT for PositionBook
def position_id_of(num:str, den:str, dir_sign:int, target_usd:float, user_ts:int)->str:
    raw = f"{num}|{den}|{dir_sign}|{int(target_usd)}|{user_ts}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

class PositionBook:
    def __init__(self, store: Storage, mc: MarketCatalog, oracle: PriceOracle):
        self.store, self.mc, self.oracle = store, mc, oracle

    def open_position(self, num_tok: str, den_tok: Optional[str],
                      usd_notional: int, user_ts: int, note: str = "") -> int:
        num = self.mc.normalize(num_tok)
        den = self.mc.normalize(den_tok) if den_tok else None
        dir_sign = 1 if usd_notional >= 0 else -1
        target = abs(float(usd_notional))

        # 1) get/create position (auto-increment id)
        pid = self.store.get_or_create_position(num, den, dir_sign, target, user_ts, status="OPEN", note=note)

        # 2) create leg stubs (single or pair) — qty/price will be filled by backfill job
        self.store.ensure_leg_stub(pid, num)
        if den:
            self.store.ensure_leg_stub(pid, den)

        # 3) enqueue backfill job (will compute qty from USD + price)
        self.store.enqueue_job(f"price:{pid}", "FETCH_ENTRY_PRICES",
                               {"position_id": pid, "user_ts": user_ts}, position_id=pid)

        log().info("Position opened (queued price backfill)", position_id=pid, num=num, den=den,
                 dir=("LONG" if dir_sign > 0 else "SHORT"), usd=target)
        return pid

    def size_leg_from_price(self, symbol:str, usd:float, price:float) -> float:
        qty = usd / price
        return self.mc.step_round_qty(symbol, qty)

    def pnl_position(self, pid: str, prices: Dict[str, float]) -> Dict[str, Any]:
        legs = self.store.get_legs(pid)
        if not legs: return {"position_id": pid, "pnl_usd": 0.0, "ok": False}
        pnl = 0.0
        for leg in legs:
            mk = prices.get(leg["symbol"])
            if mk is None: continue
            q = float(leg["qty"]); ep = float(leg["entry_price"])
            pnl += (mk - ep) * q
        return {"position_id": pid, "pnl_usd": pnl, "ok": True}


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


# =======================
# ===== ALERTS / RULES ==
# =======================

@dataclass
class AlertRule:
    rule_id: str
    kind: str             # SPREAD_TP, SPREAD_SL, FUNDING_WARN
    position_id: Optional[str]
    threshold: float      # USD for TP/SL; rate for funding
    direction: str        # "ABOVE" or "BELOW"

class AlertsEngine:
    def __init__(self, store:Storage, positionbook:PositionBook):
        self.store, self.positionbook = store, positionbook
        self.rules: Dict[str, AlertRule] = {}

    def add_rule(self, rule: AlertRule):
        self.rules[rule.rule_id] = rule

    def evaluate(self, marks:Dict[str,float]) -> List[str]:
        notes=[]
        for r in self.rules.values():
            if r.kind in ("SPREAD_TP","SPREAD_SL"):
                pnl = self.positionbook.pnl_position(r.position_id, marks)
                if not pnl["ok"]: continue
                v = pnl["pnl_usd"]
                if (r.direction=="ABOVE" and v>=r.threshold) or (r.direction=="BELOW" and v<=r.threshold):
                    notes.append(f"{r.kind} fired for {r.position_id}: PnL {v:.2f} {r.direction} {r.threshold:.2f}")
        return notes

# =======================
# ===== REPORTER =========
# =======================

class Reporter:
    @staticmethod
    def fmt_positions_summary(store, positionbook, rows, marks) -> str:
        total_target = sum(float(r["target_usd"]) for r in rows)
        total_pnl = 0.0
        for r in rows:
            res = positionbook.pnl_position(r["position_id"], marks)
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


# =========================

class TelegramBot:
    """
    Thin Telegram adapter. No internal enable flag.
    """
    def __init__(self, eng: BotEngine):
        self.engine = eng
        self._registry = None

    def set_registry(self, registry: CommandRegistry):
        self._registry = registry

    # ---- generic text handler (still uses registry for non-@ paths) ----
    async def _on_text(self, update, context):
        msg = (update.message.text or "").strip()
        chat_id = update.effective_chat.id
        log().debug("tg.on_text", chat_id=chat_id, text=msg)

        try:
            out = self._registry.dispatch(self.engine, msg)
            await context.bot.send_message(chat_id=chat_id, text=out or "OK")
        except Exception as e:
            log().exc(e, where="tg.on_text.dispatch")
            await context.bot.send_message(chat_id=chat_id, text=f"Error: {e}")


# =========================

class BotEngine:
    """
    Wires Telegram <-> your domain services.
    - Spins the background Worker (price backfills, etc.)
    - Handles '@' commands via TelegramBot parsers
    - Schedules spontaneous heartbeat messages ("are you ok?")
    """
    def __init__(self, cfg: AppConfig):
        from thinkers1 import ThinkerManager
        from commands import CommandRegistry

        self.cfg = cfg

        # lazily built parts
        self.api: Optional[BinanceUM] = None
        self.store: Optional[Storage] = None
        self.mc: Optional[MarketCatalog] = None
        self.oracle: Optional[PriceOracle] = None
        self.positionbook: Optional[PositionBook] = None
        self.alerts: Optional[AlertsEngine] = None
        self.reporter: Optional[Reporter] = None
        self.worker: Optional[Worker] = None
        self.scheduler: Optional[InternalScheduler] = None
        self.tgbot: Optional[TelegramBot] = None
        self._registry: Optional[CommandRegistry] = None
        self.kc: Optional[KlinesCache] = None

        # PTB application (only if Telegram is enabled)
        self._app = None

        # thinkers
        self.tm: Optional[ThinkerManager] = None

    # --------------------------------
    # Building + wiring (single place)
    # --------------------------------
    def _build_parts(self):
        assert self.api is None

        from thinkers1 import ThinkerManager
        from commands import build_registry

        # Clock / TZ once
        tz = self.cfg.TZ_LOCAL
        Clock.set_tz(tz)
        log().info("Clock TZ set", tz=str(tz))

        # Core deps
        self.api    = BinanceUM(self.cfg)
        self.store  = Storage(self.cfg.DB_PATH)
        self.mc     = MarketCatalog(self.api)
        self.oracle = PriceOracle(self.cfg, self.api)
        self.positionbook = PositionBook(self.store, self.mc, self.oracle)
        self.alerts = AlertsEngine(self.store, self.positionbook)
        self.reporter = Reporter()
        self.worker   = Worker(self.cfg, self.store, self.api, self.mc, self.oracle)
        self.scheduler = InternalScheduler()
        self.kc = KlinesCache(self)
        self.tm = ThinkerManager(self)

        # Preload catalog to avoid first-use hiccups
        try:
            self.mc.load()
        except Exception as e:
            log().exc(e, where="engine.build.mc.load")

        # Registry (commands) — single canonical instance
        self._registry = build_registry()


        # Telegram (optional)
        if self.cfg.TELEGRAM_ENABLED:
            try:
                from telegram.ext import Application, MessageHandler, CommandHandler, filters
                self.tgbot = TelegramBot(self)
                self.tgbot.set_registry(self._registry)

                self._app = Application.builder().token(self.cfg.TELEGRAM_TOKEN).build()
                self._app.bot_data["engine"] = self

                # Handlers
                self._app.add_handler(CommandHandler("start", self._cmd_start))
                self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text))

                log().info("Telegram: application prepared")
            except Exception as e:
                log().exc(e, where="engine.telegram.init")
                log().warn("Telegram disabled due to initialization error")

        # Schedule periodic jobs (always via our own scheduler)
        tick_sec = int(self.cfg.THINKING_SEC or 15)
        self.scheduler.schedule_every("thinkers.tick", tick_sec, self._job_thinkers, run_immediately=True)
        self.scheduler.schedule_every("heartbeat", 3600, self._job_heartbeat, run_immediately=False)


    def set_registry(self, registry: CommandRegistry):
        self._registry = registry

    # ---------- interface ----------
    def send_text(self, text: str) -> None:
        if not self.cfg.TELEGRAM_ENABLED:
            log().info("ALERT", text=text);
            return
        assert self._app
        self._app.create_task(self._app.bot.send_message(chat_id=self.cfg.TELEGRAM_CHAT_ID, text=text))

    def send_photo(self, path: str, caption: str | None = None):
        """Send a photo through Telegram if enabled, else open locally."""
        if not os.path.exists(path):
            log().warn("engine.send_photo.missing", path=path)
            return

        if not self.cfg.TELEGRAM_ENABLED:
            log().info("photo.local", path=path)
            os.system(f"xdg-open '{path}' >/dev/null 2>&1 &")
            return

        try:
            assert self._app is not None
            with open(path, "rb") as f:
                self._app.create_task(
                    self._app.bot.send_photo(
                        chat_id=int(self.cfg.TELEGRAM_CHAT_ID),
                        photo=f,
                        caption=caption or "",
                    )
                )
            log().info("photo.sent", path=path, caption=caption)
        except Exception as e:
            log().exc(e, where="engine.send_photo", path=path)

    def refresh_klines_cache(self):
        """Pulls recent klines for all relevant symbols."""

        # Gather all relevant symbols
        open_positions = self.store.list_open_positions()
        syms = {lg["symbol"] for r in open_positions for lg in self.store.get_legs(r["position_id"]) if lg["symbol"]}
        syms.update(self.cfg.AUX_SYMBOLS)
        log().debug(f"refresh_klines_cache(): symbols: {syms}")

        # Refresh all
        for s in syms:
            try:
                self._refresh_symbol(s)  # or your configured timeframes
                log().debug("engine.klines.refreshed", symbol=s)
            except Exception as e:
                log().warn("engine.klines.refresh.fail", symbol=s, err=str(e))

    def _refresh_symbol(self, symbol: str, timeframes=None):
        """Fetch new candles, upsert, then prune."""

        assert not isinstance(timeframes, str)
        timeframes = timeframes or self.cfg.KLINES_TIMEFRAMES

        for tf in timeframes:
            try:
                last_open = self.kc.latest_open_ts(symbol, tf)
                n = self._fetch_and_cache(symbol, tf, since_open_ts=last_open)
                log().debug("engine.klines.refresh.symbol.timeframe", symbol=symbol, tf=tf, added=n, last_open=last_open)
            except Exception as e:
                log().warn("engine.klines.refresh.symbol.timeframe.fail", symbol=symbol, tf=tf, err=str(e))

        self.kc.prune_keep_last_n(keep_n=self.cfg.KLINES_CACHE_KEEP_BARS)

    def _fetch_and_cache(self, symbol: str, timeframe: str, since_open_ts: Optional[int]) -> int:
        """Pull fresh klines from Binance and store them, safely overwriting the live candle."""
        now_ms = self.api.now_ms()
        start = max(0, since_open_ts - tf_ms(timeframe)) if since_open_ts else None
        klines = self.api.klines(symbol, timeframe, start, now_ms, self.cfg.KLINES_CACHE_KEEP_BARS)
        return self.kc.ingest_klines(symbol, timeframe, klines, now_ms=now_ms)

    # --------------------------------
    # Lifecycle
    # --------------------------------

    def start(self):
        # Build everything lazily
        if self.api is None:
            self._build_parts()

        # Start worker + scheduler
        t = threading.Thread(target=self.worker.run_forever, name="rv-worker", daemon=True)
        t.start()
        log().info("Worker thread started")

        self.scheduler.start()

        # If Telegram prepared, run polling (non-blocking loop handled by PTB)
        if self._app:
            log().info("Telegram polling…")
            # Run PTB in a thread so console/other UIs can coexist
            threading.Thread(target=lambda: self._app.run_polling(close_loop=False),
                             name="rv-telegram", daemon=True).start()
        else:
            log().info("Engine running without Telegram")

    def stop(self):
        try:
            if self._app:
                self._app.stop()
        except Exception as e:
            log().exc(e, where="engine.stop.telegram")
        try:
            self.worker.stop()
        finally:
            if self.scheduler:
                self.scheduler.stop()

    # ---------- interface ------------------
    def emit_alert(self, msg: str):
        """Emit alert to Telegram if enabled, else to logs."""
        if self.cfg.TELEGRAM_ENABLED:
            try:
                if self._app:
                    self._app.create_task(
                        self._app.bot.send_message(chat_id=int(self.cfg.TELEGRAM_CHAT_ID), text=msg)
                    )
                else:
                    log().info("ALERT", text=msg)
            except Exception as e:
                log().exc(e, where="emit_alert")
        else:
            log().info("ALERT", text=msg)

    def dispatch_command(self, text: str, chat_id: int = 0) -> str:
        """Build context and route through the registry."""
        return self._registry.dispatch(self, text)

    # ---------- telegram handlers ----------
    async def _cmd_start(self, update, context):
        await update.message.reply_text("RV bot ready. Try: @positions")

    async def _on_text(self, update, context):
        msg = (update.message.text or "").strip()
        chat_id = update.effective_chat.id
        log().debug("engine.on_text", chat_id=chat_id, text=msg)
        try:
            out = self.dispatch_command(msg, chat_id=chat_id)
            self.send_text(out or "OK")
        except Exception as e:
            log().exc(e, where="engine.on_text.dispatch")
            self.send_text(f"Error: {e}")

    # --------------------------------
    # Scheduled jobs (driven by InternalScheduler)
    # --------------------------------
    def _job_heartbeat(self):
        try:
            open_positions = self.store.list_open_positions()
            marks = self._latest_marks_for_open_symbols(open_positions)
            total_pnl = 0.0
            for row in open_positions:
                pid = row["position_id"]
                res = self.positionbook.pnl_position(pid, marks)
                if res["ok"]:
                    total_pnl += res["pnl_usd"]
            txt = f"👋 are you ok?\nOpen positions: {len(open_positions)} | PNL ≈ ${total_pnl:.2f}"
            self.send_text(txt)
        except Exception as e:
            log().exc(e, where="job.heartbeat")

    def _job_thinkers(self):
        try:
            log().debug("Gonna think ...")
            log().info(f"Current log level: {log().level}; log id: {id(log)}")

            self.refresh_klines_cache()
            n = self.tm.run_once()
            if n:
                log().debug("thinkers.cycle", actions=n)
        except Exception as e:
            log().exc(e, where="job.thinkers")

    # ---------- execution helpers ----------
    def _latest_marks_for_open_symbols(self, pair_rows) -> Dict[str, float]:
        """
        Very light mark snapshot: for each symbol involved, take the latest mark close from markPriceKlines.
        Good enough for quick Telegram responses.
        """
        syms = set()
        for r in pair_rows:
            legs = self.store.get_legs(r["position_id"])
            for lg in legs:
                if lg["symbol"]:
                    syms.add(lg["symbol"])
        marks: Dict[str,float] = {}
        if not syms:
            return marks
        now = Clock.now_utc_ms()
        for s in syms:
            try:
                mk = self.api.mark_price_klines(s, "1m", now-60_000, now)
                if isinstance(mk, list) and mk:
                    marks[s] = float(mk[-1][4])  # last close
            except Exception as e:
                log().debug("mark snapshot fail", symbol=s, err=str(e))
        return marks


# TODO:chatgpt use KlinesCache, not directly from Binance API (I don't mind the delay in cached info)
def latest_prices_for_positions(eng: BotEngine, rows) -> Dict[str,float]:
    syms = {leg["symbol"] for r in rows for leg in eng.store.get_legs(r["position_id"]) if leg["symbol"]}
    now = Clock.now_utc_ms(); out={}
    for s in syms:
        try:
            mk = eng.api.mark_price_klines(s, "1m", now-60_000, now)
            if mk: out[s] = float(mk[-1][4])
        except Exception as e:
            log().debug("mark snapshot fail", symbol=s, err=str(e))
    return out


def pnl_for_position(eng: BotEngine, position_id, prices) -> float:
    res = eng.positionbook.pnl_position(position_id, prices)
    return res["pnl_usd"] if res["ok"] else 0.0


def exec_positions(eng: BotEngine, args) -> str:
    """
    Pure function used by @positions.

    args keys: status, what, limit, position_id, pair
    """
    store = eng.store
    reporter = eng.reporter
    positionbook = eng.positionbook

    status = args.get("status", "open")
    what   = args.get("what", "summary")
    limit  = int(args.get("limit", 100))
    pair   = args.get("pair")
    pid    = args.get("position_id")

    # rows by status
    if status == "open":
        rows = store.list_open_positions()
    elif status == "closed":
        rows = store.con.execute("SELECT * FROM positions WHERE status='CLOSED'").fetchall()
    else:
        rows = store.con.execute("SELECT * FROM positions").fetchall()

    # optional filters
    if pid:
        rows = [r for r in rows if r["position_id"] == pid]

    if pair:
        num, den = pair
        def _match(r):
            if den is None:
                legs = store.get_legs(r["position_id"])
                return any((lg["symbol"] or "").startswith(num) for lg in legs)
            return (r["num"].startswith(num) and r["den"].startswith(den))
        rows = [r for r in rows if _match(r)]

    # marks & render
    prices = latest_prices_for_positions(eng, rows)
    if what == "summary":
        return reporter.fmt_positions_summary(store, positionbook, rows, prices)
    if what == "full":
        return reporter.fmt_positions_full(store, positionbook, rows, prices, limit)
    if what == "legs":
        return reporter.fmt_positions_legs(store, positionbook, rows, prices, limit)

    # Signal mis-usage clearly (your earlier convention)
    raise RuntimeError('ChatGPT: unknown "what". Use one of summary|full|legs')


def render_chart(eng: BotEngine, symbol: str, timeframe: str, outdir: str = "/tmp") -> str:
    """Render a candlestick+volume chart from KlinesCache → PNG. Returns file path."""
    kc = eng.kc
    rows = kc.last_n(symbol, timeframe, n=200)
    if not rows:
        raise ValueError(f"No cached klines for {symbol} {timeframe}")

    df = rows_to_dataframe(rows)

    # Simple indicator (20-period MA)
    df["MA20"] = df["Close"].rolling(window=20).mean()

    style = mpf.make_mpf_style(base_mpf_style="binance", y_on_right=True)

    fig, axlist = mpf.plot(
        df,
        type="candle",
        mav=(20,),
        volume=True,
        style=style,
        title=f"{symbol} {timeframe} – last {len(df)} candles",
        returnfig=True,
        figsize=(9, 6),
    )

    out_path = os.path.join(outdir, f"chart_{symbol}_{timeframe}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_pair_or_single(eng: BotEngine, raw: str) -> Tuple[str, Optional[str]]:
    """
    Parse a trading spec into (num_symbol, den_symbol_or_None).

    Accepts:
      - "ETH/STRK"            -> ("ETHUSDT", "STRKUSDT")
      - "ETHUSDT/STRKUSDT"    -> ("ETHUSDT", "STRKUSDT")
      - "ETH" or "ETHUSDT"    -> ("ETHUSDT", None)     # single-leg
      - "ETH/" or "ETH/1"     -> ("ETHUSDT", None)     # explicit single-leg

    Rules:
      - Uses mc.normalize() for both sides.
      - Denominator tokens "1", "UNIT", "" mean single-leg.
      - Plain "USDT" as denominator is **not** allowed (ambiguous). Omit the denominator instead.

    Raises:
      ValueError with a precise message on bad input.
    """
    if not raw or not isinstance(raw, str):
        raise ValueError("Empty pair/symbol.")

    s = raw.strip().upper()

    # Single token (no slash): single-leg
    if "/" not in s:
        try:
            num = eng.mc.normalize(s)
        except Exception as e:
            raise ValueError(f"Unknown/unsupported symbol or base asset: {raw}") from e
        return num, None

    # Pair form
    left, right = (t.strip() for t in s.split("/", 1))

    if not left:
        raise ValueError("Missing numerator before '/'.")

    # Explicit single-leg hints on the right side
    if right in ("", "1", "UNIT", "-"):
        try:
            num = eng.mc.normalize(left)
        except Exception as e:
            raise ValueError(f"Unknown/unsupported numerator: {left}") from e
        return num, None

    if right == "USDT":
        # Denominator must be a PERPETUAL symbol, not the quote token.
        raise ValueError("Denominator 'USDT' is not valid. For single-leg, omit the denominator (e.g., 'ETH' or 'ETHUSDT').")

    # Proper pair: normalize both sides
    try:
        num = eng.mc.normalize(left)
    except Exception as e:
        raise ValueError(f"Unknown/unsupported numerator: {left}") from e

    try:
        den = eng.mc.normalize(right)
    except Exception as e:
        raise ValueError(f"Unknown/unsupported denominator: {right}") from e

    return num, den
