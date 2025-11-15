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

class _SchedTask:
    __slots__ = ("fn", "interval", "next_ts")
    def __init__(self, fn: Callable[[], None], interval: float, first_delay: float):
        self.fn = fn
        self.interval = float(interval)
        self.next_ts = time.time() + float(first_delay)


class InternalScheduler:
    """
    Minimal repeating scheduler:
      - run_repeating(fn, interval, first=0)
      - stop()
    Runs in a single background thread. Functions are executed serially.
    """
    def __init__(self, name: str = "rv-scheduler"):
        self._tasks: List[_SchedTask] = []
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._name = name
        self._lock = threading.Lock()

    def run_repeating(self, fn: Callable[[], None], interval: float, first: float = 0.0):
        with self._lock:
            self._tasks.append(_SchedTask(fn, interval, first))

    def start(self):
        if self._thr and self._thr.is_alive():
            return
        def _loop():
            log.info("scheduler.start", name=self._name, tasks=len(self._tasks))
            while not self._stop.is_set():
                now = time.time()
                fired = False
                with self._lock:
                    for t in self._tasks:
                        if now >= t.next_ts:
                            try:
                                t.fn()
                            except Exception as e:
                                log.exc(e, where="scheduler.fn")
                            t.next_ts = now + t.interval
                            fired = True
                if not fired:
                    time.sleep(0.2)  # light idle
            log.info("scheduler.stop", name=self._name)
        self._thr = threading.Thread(target=_loop, name=self._name, daemon=True)
        self._thr.start()

    def stop(self, timeout: float = 2.0):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=timeout)


def parse_when(s: str) -> int:
    """
    Return UTC epoch milliseconds parsed from:
      - "now" or "now-<n>[s|min|h|d]"   (e.g., now-30min, now-5h, now-2d, now-45s)
      - YYYYMMDD                        (local midnight)
      - YYYYMMDDHHMM or YYYYMMDDHHMMSS  (local time)
      - ISO 8601 (e.g., 2025-11-10T13:44:05[+TZ])
      - epoch seconds (10 digits) or milliseconds (13 digits)

    Local tz for naive dates = system local tz (so it works even without Clock.set_tz()).
    """
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty timestamp")

    # now / now-*
    m = re.fullmatch(r"now(?:-(\d+)(s|min|h|d))?", s.lower())
    if m:
        n, unit = m.groups()
        dt = datetime.now(timezone.utc)
        if n:
            n = int(n)
            delta = {"s": "seconds", "min": "minutes", "h": "hours", "d": "days"}[unit]
            dt = dt - timedelta(**{delta: n})
        return int(dt.timestamp() * 1000)

    # pure digits
    if s.isdigit():
        if len(s) == 13:
            return int(s)  # epoch ms
        if len(s) == 10:
            return int(s) * 1000  # epoch sec
        # YYYYMMDD / YYYYMMDDHHMM / YYYYMMDDHHMMSS
        if len(s) in (8, 12, 14):
            year = int(s[0:4]); month = int(s[4:6]); day = int(s[6:8])
            hh = mm = ss = 0
            if len(s) >= 12:
                hh = int(s[8:10]); mm = int(s[10:12])
            if len(s) == 14:
                ss = int(s[12:14])
            # interpret as *local* time then convert to UTC
            lt = datetime(year, month, day, hh, mm, ss).astimezone()  # system local tz
            return int(lt.astimezone(timezone.utc).timestamp() * 1000)

    # ISO8601 (naive => local)
    try:
        tz_local = Clock.get_tz()
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz_local)
        return int(dt.astimezone(timezone.utc).timestamp() * 1000)

    except Exception:
        pass

    raise ValueError(f"Unrecognized timestamp format: {s}")


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
        log.info("MarketCatalog loaded (USDT PERPETUAL only)",
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

# FULL CLASS REPLACEMENT for PriceOracle
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
            log.debug("oracle.try", symbol=symbol, ts=ts_ms, window_sec=window)

            # aggTrades
            try:
                trades = self.api.agg_trades(symbol, start, end)
                if isinstance(trades, list) and trades:
                    closest = min(trades, key=lambda t: abs(int(t["T"]) - ts_ms))
                    price = float(closest["p"]); pts = int(closest["T"])
                    log.debug("oracle.hit", method="aggTrade", price=price, price_ts=pts)
                    return PricePoint(price, pts, "aggTrade")
            except Exception as e:
                log.debug("oracle.fail", method="aggTrades", err=str(e))

            # klines 1m
            try:
                kl = self.api.klines(symbol, "1m", ts_ms - 60_000, ts_ms + 60_000)
                if isinstance(kl, list) and kl:
                    k = min(kl, key=lambda r: abs(int(r[0]) - ts_ms))
                    open_t = int(k[0]); close_t = open_t + 60_000
                    price = float(k[1] if ts_ms <= (open_t + close_t)//2 else k[4])
                    log.debug("oracle.hit", method="kline", price=price, candle_open=open_t)
                    return PricePoint(price, open_t, "kline")
            except Exception as e:
                log.debug("oracle.fail", method="klines", err=str(e))

            # markPriceKlines 1m
            try:
                mk = self.api.mark_price_klines(symbol, "1m", ts_ms - 60_000, ts_ms + 60_000)
                if isinstance(mk, list) and mk:
                    k = min(mk, key=lambda r: abs(int(r[0]) - ts_ms))
                    open_t = int(k[0]); close_t = open_t + 60_000
                    price = float(k[1] if ts_ms <= (open_t + close_t)//2 else k[4])
                    log.debug("oracle.hit", method="mark_kline", price=price, candle_open=open_t)
                    return PricePoint(price, open_t, "mark_kline")
            except Exception as e:
                log.debug("oracle.fail", method="mark_price_klines", err=str(e))

            window *= 2

        log.error("oracle.giveup", symbol=symbol, ts=ts_ms)
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

    def open_position(self, num_tok: str, den_tok: str, usd_notional: int, user_ts: int, note: str = "") -> str:
        num = self.mc.normalize(num_tok)
        den = self.mc.normalize(den_tok) if den_tok else None
        dir_sign = 1 if usd_notional >= 0 else -1
        target = abs(float(usd_notional))
        pid = position_id_of(num, den, dir_sign, target, user_ts)
        self.store.upsert_position({
            "position_id": pid, "num": num, "den": den,
            "dir_sign": dir_sign, "target_usd": target,
            "user_ts": user_ts, "status": "OPEN", "note": note,
            "created_ts": Clock.now_utc_ms()
        })
        self.store.enqueue_job(f"price:{pid}", "FETCH_ENTRY_PRICES",
                               {"position_id": pid, "user_ts": user_ts}, position_id=pid)
        log.info("Position opened (queued price backfill)", position_id=pid, num=num, den=den,
                 dir=("LONG num/SHORT den" if dir_sign > 0 else "SHORT num/LONG den"), usd=target)
        return pid

    def size_leg_from_price(self, symbol:str, usd:float, price:float) -> float:
        qty = usd / price
        return self.mc.step_round_qty(symbol, qty)

    def pnl_position(self, pid: str, marks: Dict[str, float]) -> Dict[str, Any]:
        legs = self.store.get_legs(pid)
        if not legs: return {"position_id": pid, "pnl_usd": 0.0, "ok": False}
        pnl = 0.0
        for lg in legs:
            mk = marks.get(lg["symbol"])
            if mk is None: continue
            q = float(lg["qty"]); ep = float(lg["entry_price"])
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
            log.info("Income rows inserted", n=len(rows))
        else:
            log.info("Income (none)")

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
        log.info("Worker started")
        while not self._stop.is_set():
            job = self.store.fetch_next_job()
            if not job:
                time.sleep(idle_sleep); continue
            jid = job["job_id"]; task = job["task"]
            log.debug("job.start", job_id=jid, task=task)
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
                log.debug("job.done", job_id=jid, task=task)
            except Exception as e:
                log.exc(e, job_id=jid, task=task)
                self.store.finish_job(jid, ok=False, error=str(e))

    def _do_price_backfill(self, job_id: str, payload: dict):
        pid = payload["position_id"];
        user_ts = int(payload["user_ts"])
        position = self.store.get_position(pid)
        if not position:
            raise ValueError("position not found")

        intended_usd = float(position["target_usd"])
        num_sym = position["num"]
        den_raw = position["den"]  # may be None/"" when single-leg
        dir_sign = int(position["dir_sign"])  # +1 LONG ; -1 SHORT

        # detect single-leg (no denominator)
        den_is_none = (den_raw is None)
        den_str = ("" if den_is_none else str(den_raw).strip().upper())
        single_leg = den_is_none or den_str in ("", "1", "UNIT")

        # always backfill NUM
        pp_num = self.oracle.price_at(num_sym, user_ts)

        def _signed_qty(symbol: str, usd: float, px: float, sign: int) -> float:
            q_abs = self.mc.step_round_qty(symbol, abs(usd) / px)
            return sign * q_abs

        if single_leg:
            # Single leg: qty sign follows dir_sign (+ long / - short)
            q_num = _signed_qty(num_sym, intended_usd, pp_num.price, +1 if dir_sign > 0 else -1)

            self.store.upsert_leg({
                "position_id": pid, "symbol": num_sym, "qty": q_num,
                "entry_price": pp_num.price, "entry_price_ts": pp_num.price_ts,
                "price_method": pp_num.method, "note": None
            })

            self.store.finish_job(job_id, ok=True)
            log.info("Backfill complete (single-leg)", position_id=pid, q_num=q_num)
            return

        # Two-leg (RV) flow
        den_sym = den_str
        pp_den = self.oracle.price_at(den_sym, user_ts)

        # For RV pair: LONG dir => +num / -den ; SHORT dir => -num / +den
        q_num = _signed_qty(num_sym, intended_usd, pp_num.price, +1 if dir_sign > 0 else -1)
        q_den = _signed_qty(den_sym, intended_usd, pp_den.price, -1 if dir_sign > 0 else +1)

        self.store.upsert_leg({
            "position_id": pid, "symbol": num_sym, "qty": q_num,
            "entry_price": pp_num.price, "entry_price_ts": pp_num.price_ts,
            "price_method": pp_num.method, "note": None
        })
        self.store.upsert_leg({
            "position_id": pid, "symbol": den_sym, "qty": q_den,
            "entry_price": pp_den.price, "entry_price_ts": pp_den.price_ts,
            "price_method": pp_den.method, "note": None
        })

        self.store.finish_job(job_id, ok=True)
        log.info("Backfill complete (pair)", position_id=pid, q_num=q_num, q_den=q_den)


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


# =======================
# ===== TELEGRAM ===
# =======================

@dataclass
class CommandContext:
    chat_id: int
    text: str
    # give handlers access to your services:
    store: Storage
    positionbook: PositionBook
    mc: MarketCatalog
    oracle: PriceOracle
    api: BinanceUM
    alerts: AlertsEngine
    reporter: Reporter


class CommandRegistry:
    def __init__(self):
        # keys are ('@'|'!', command_name)
        self._handlers: Dict[Tuple[str,str], Callable[[CommandContext, Dict[str,str]], str]] = {}

    def at(self, name: str):
        """Decorator for '@' (GET) commands."""
        def deco(fn):
            self._handlers[('@', name.lower())] = fn
            return fn
        return deco

    def bang(self, name: str):
        """Decorator for '!' (SET) commands."""
        def deco(fn):
            self._handlers[('!', name.lower())] = fn
            return fn
        return deco

    # FULL METHOD REPLACEMENT in class CommandRegistry
    def dispatch(self, ctx: CommandContext) -> str:
        s = (ctx.text or "").strip()
        log.debug("dispatch.enter", text=s)
        if not s or s[0] not in ("@", "!"):
            log.debug("dispatch.exit", reason="not-a-command")
            return "Unrecognized. Use @help for available commands."

        prefix = s[0]
        parts = s.split(None, 1)
        if not parts:
            log.debug("dispatch.exit", reason="empty-after-prefix")
            return "Empty command."

        head = parts[0][1:].lower()
        tail = parts[1].strip() if len(parts) > 1 else ""
        handler = self._handlers.get((prefix, head))

        if not handler:
            log.warn("dispatch.unknown", prefix=prefix, head=head)
            return f"Unknown {prefix}{head}. Try @help."

        args: Dict[str, str] = {}
        if prefix == '@':
            for tok in tail.split():
                if ":" in tok:
                    k, v = tok.split(":", 1)
                    args[k.strip().lower()] = v.strip()
                elif tok:
                    log.debug("dispatch.ignored-token", token=tok)
        else:
            args["_"] = tail

        log.debug("dispatch.call", cmd=head, prefix=prefix, args=args)
        try:
            out = handler(ctx, args)
            log.debug("dispatch.ok", cmd=head)
            return out
        except Exception as e:
            log.exc(e, where="dispatch.handler", cmd=head)
            return f"Error: {e}"


class TelegramBot:
    """
    Thin Telegram adapter. No internal enable flag.
    """
    def __init__(self, book: PositionBook):
        self.book = book
        self._registry = None

    def set_registry(self, registry: CommandRegistry):
        self._registry = registry

    # ---- generic text handler (still uses registry for non-@ paths) ----
    async def _on_text(self, update, context):
        msg = (update.message.text or "").strip()
        chat_id = update.effective_chat.id
        log.debug("tg.on_text", chat_id=chat_id, text=msg)

        engine = context.application.bot_data.get("engine")
        cmd_ctx = CommandContext(
            chat_id=chat_id, text=msg,
            store=engine.store, positionbook=engine.positionbook, mc=engine.mc,
            oracle=engine.oracle, api=engine.api, alerts=engine.alerts, reporter=engine.reporter,
        )

        try:
            out = self._registry.dispatch(cmd_ctx)
            await context.bot.send_message(chat_id=chat_id, text=out or "OK")
        except Exception as e:
            log.exc(e, where="tg.on_text.dispatch")
            await context.bot.send_message(chat_id=chat_id, text=f"Error: {e}")

# --- add this lightweight internal scheduler somewhere in bot_api.py ---

import threading, time
from typing import Callable, Dict

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
                    log.exc(e, where="scheduler.job", job=name)
            time.sleep(0.5)


class BotEngine:
    """
    Wires Telegram <-> your domain services.
    - Spins the background Worker (price backfills, etc.)
    - Handles '@' commands via TelegramBot parsers
    - Schedules spontaneous heartbeat messages ("are you ok?")
    """
    def __init__(self, cfg: AppConfig):
        from thinkers import ThinkerManager

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

        # PTB application (only if Telegram is enabled)
        self._app = None

        # thinkers
        self.thinkers: Optional[ThinkerManager] = None

        tz = cfg.TZ_LOCAL
        Clock.set_tz(tz)
        log.info("Clock TZ set", tz=str(tz))


    # --------------------------------
    # Building + wiring (single place)
    # --------------------------------
    def _build_parts(self):
        from thinkers import ThinkerManager

        # Clock / TZ once
        tz = getattr(self.cfg, "TZ_LOCAL", None)
        if tz is not None:
            Clock.set_tz(tz)
            log.info("Clock TZ set", tz=str(tz))

        # Core deps
        self.api    = self.api    or BinanceUM(self.cfg)
        self.store  = self.store  or Storage(self.cfg.DB_PATH or "./rv.sqlite3")
        self.mc     = self.mc     or MarketCatalog(self.api)
        self.oracle = self.oracle or PriceOracle(self.cfg, self.api)
        self.positionbook = self.positionbook or PositionBook(self.store, self.mc, self.oracle)
        self.alerts = self.alerts or AlertsEngine(self.store, self.positionbook)
        self.reporter = self.reporter or Reporter()
        self.worker   = self.worker   or Worker(self.cfg, self.store, self.api, self.mc, self.oracle)
        self.scheduler= self.scheduler or InternalScheduler()

        # Preload catalog to avoid first-use hiccups
        try:
            self.mc.load()
        except Exception as e:
            log.exc(e, where="engine.build.mc.load")

        # Registry (commands) — single canonical instance
        self._registry = self._registry or build_registry()

        # Thinkers (manager uses engine services)
        if not self.thinkers:
            self.thinkers = ThinkerManager(self.cfg, self.store, self.api, self.mc, self.oracle, services=self)

        # Telegram (optional)
        if self.cfg.TELEGRAM_ENABLED:
            try:
                from telegram.ext import Application, MessageHandler, CommandHandler, filters
                self.tgbot = self.tgbot or TelegramBot(self.positionbook)
                self.tgbot.set_registry(self._registry)

                self._app = Application.builder().token(self.cfg.TELEGRAM_TOKEN).build()
                self._app.bot_data["engine"] = self

                # Handlers
                self._app.add_handler(CommandHandler("start", self._cmd_start))
                self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text))

                log.info("Telegram: application prepared")
            except Exception as e:
                log.exc(e, where="engine.telegram.init")
                log.warn("Telegram disabled due to initialization error")

        # Schedule periodic jobs (always via our own scheduler)
        tick_sec = int(self.cfg.KLINES_POLL_SEC or 15)
        self.scheduler.schedule_every("thinkers.tick", tick_sec, self._job_thinkers, run_immediately=True)
        self.scheduler.schedule_every("heartbeat", 3600, self._job_heartbeat, run_immediately=False)


    def set_registry(self, registry: CommandRegistry):
        self._registry = registry

    # ---------- interface ----------
    def send_text(self, text: str) -> None:
        if not self.cfg.TELEGRAM_ENABLED:
            log.info("ALERT", text=text);
            return
        assert self._app
        self._app.create_task(self._app.bot.send_message(chat_id=self.cfg.TELEGRAM_CHAT_ID, text=text))

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
        log.info("Worker thread started")

        self.scheduler.start()

        # If Telegram prepared, run polling (non-blocking loop handled by PTB)
        if self._app:
            log.info("Telegram polling…")
            # Run PTB in a thread so console/other UIs can coexist
            threading.Thread(target=lambda: self._app.run_polling(close_loop=False),
                             name="rv-telegram", daemon=True).start()
        else:
            log.info("Engine running without Telegram")

    def stop(self):
        try:
            if self._app:
                self._app.stop()
        except Exception as e:
            log.exc(e, where="engine.stop.telegram")
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
                    log.info("ALERT", text=msg)
            except Exception as e:
                log.exc(e, where="emit_alert")
        else:
            log.info("ALERT", text=msg)

    # inside class BotEngine

    def make_cmd_ctx(self, text: str, chat_id: int = 0) -> CommandContext:
        """Unified factory for CommandContext (Telegram or console)."""
        return CommandContext(
            chat_id=chat_id,
            text=text,
            store=self.store,
            positionbook=self.positionbook,
            mc=self.mc,
            oracle=self.oracle,
            api=self.api,
            alerts=self.alerts,
            reporter=self.reporter,
        )

    def dispatch_command(self, text: str, chat_id: int = 0) -> str:
        """Build context and route through the registry."""
        ctx = self.make_cmd_ctx(text=text, chat_id=chat_id)
        return self._registry.dispatch(ctx)

    # ---------- telegram handlers ----------
    async def _cmd_start(self, update, context):
        await update.message.reply_text("RV bot ready. Try: @positions")

    async def _on_text(self, update, context):
        msg = (update.message.text or "").strip()
        chat_id = update.effective_chat.id
        log.debug("engine.on_text", chat_id=chat_id, text=msg)
        try:
            out = self.dispatch_command(msg, chat_id=chat_id)
            self.send_text(out or "OK")
        except Exception as e:
            log.exc(e, where="engine.on_text.dispatch")
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
            log.exc(e, where="job.heartbeat")

    def _job_thinkers(self):
        try:
            n = self.thinkers.run_once()
            if n:
                log.debug("thinkers.cycle", actions=n)
        except Exception as e:
            log.exc(e, where="job.thinkers")

    # ---------- execution helpers ----------
    def _exec_positions(self, args: Dict) -> str:
        """
        Executes @positions against *your DB* (RV pairs). Summarizes or lists.
        """
        status = args["status"]
        what = args["what"]
        limit = args["limit"]

        # Filter positions by status
        if status == "open":
            rows = self.store.list_open_positions()
        elif status == "closed":
            rows = self.store.con.execute("SELECT * FROM positions WHERE status='CLOSED'").fetchall()
        else:
            rows = self.store.con.execute("SELECT * FROM positions").fetchall()

        # Optional position_id or pair filter
        pid = args.get("position_id")
        if pid:
            rows = [r for r in rows if r["position_id"] == pid]
        if args.get("pair"):
            num, den = args["pair"]
            def _match(r):
                if den is None:
                    # single-instrument vs USDT: match either leg == sym normalized
                    legs = self.store.get_legs(r["position_id"])
                    return any(l["symbol"].startswith(num) for l in legs)
                return (r["num"].startswith(num) and r["den"].startswith(den))
            rows = [r for r in rows if _match(r)]

        # Marks snapshot for current PNL
        valid = {"summary", "full", "legs"}

        if what not in valid:
            return f"Unknown 'what': {what}. Try one of: {', '.join(sorted(valid))}"

        # ChatGPT: gotta retrieve from klines cache, not api directly-
        marks = latest_marks_for_positions(self.api, self.store, rows)

        if what == "summary":
            return self.reporter.fmt_positions_summary(self.store, self.positionbook, rows, marks)
        if what == "full":
            return self.reporter.fmt_positions_full(self.store, self.positionbook, rows, marks, limit)
        # what == "legs"
        return self.reporter.fmt_positions_legs(self.store, self.positionbook, rows, marks, limit)

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
                log.debug("mark snapshot fail", symbol=s, err=str(e))
        return marks


# ===== COMMAND DECLARATIONS =====

def build_registry() -> CommandRegistry:
    R = CommandRegistry()

    @R.at("help")
    def _help(ctx: CommandContext, args: Dict[str,str]) -> str:
        return (
            "Commands:\n"
            "  @open                 – list open RV positions (summary)\n"
            "  @positions [filters]  – advanced view; e.g. @positions what:full limit:20\n"
            "  !open <spec>          – open/add RV position (e.g. '!open 2025-11-10T13:44:05 STRK/ETH -5000 note: x')\n"
            "  !close <position_id>      – close a recorded position\n"
        )

    @R.at("open")
    def _at_open(ctx: CommandContext, args: Dict[str,str]) -> str:
        # alias to @positions status:open summary
        text_alias = "@positions status:open what:summary"
        ctx2 = CommandContext(**{**ctx.__dict__, "text": text_alias})
        return R.dispatch(ctx2)

    @R.at("positions")
    def _at_positions(ctx: CommandContext, args: Dict[str, str]) -> str:
        # defaults
        status = args.get("status", "open")
        what = args.get("what", "summary")
        sort_ = args.get("sort", "pnl")  # (kept for future; unused here)
        limit = int(args.get("limit", "100"))
        position_id = args.get("position_id")
        pair = args.get("pair")

        exec_args = {
            "status": status, "what": what, "sort": sort_, "limit": limit,
            "position_id": position_id, "pair": None
        }
        if pair:
            up = pair.upper()
            if "/" in up:
                num, den = (x.strip() for x in up.split("/", 1))
                exec_args["pair"] = (num, den)
            else:
                exec_args["pair"] = (up, None)

        # call the pure helper
        return exec_positions(ctx.store, ctx.positionbook, ctx.api, ctx.reporter, exec_args)

        return _exec_positions(exec_args)

    @R.bang("open")
    def _bang_open(ctx: CommandContext, args: Dict[str,str]) -> str:
        """
        !open <ISO|epoch_ms> <NUM/DEN | SYMBOL> <±usd> [note:...]
        Example:
          !open 2025-11-10T13:44:05 STRK/ETH -5000 note:rv test
          !open 2025-11-10T13:44:05 ETHUSDT +3000
        """
        tail = args.get("_","").strip()
        if not tail:
            return "Usage: !open <ts> <NUM/DEN|SYMBOL> <±usd> [note:...]"
        parts = tail.split()
        if len(parts) < 3:
            return "Usage: !open <ts> <NUM/DEN|SYMBOL> <±usd> [note:...]"
        ts_raw, pair_raw, usd_raw = parts[0], parts[1], parts[2]
        note = ""
        if "note:" in tail:
            note = tail.split("note:",1)[1].strip()

        # ts
        ts_ms = parse_when(ts_raw)

        # parse pair vs single
        if "/" in pair_raw.upper():
            num_tok, den_tok = pair_raw.upper().split("/",1)
            num = ctx.mc.normalize(num_tok)
            den = ctx.mc.normalize(den_tok)
        else:
            num = ctx.mc.normalize(pair_raw.upper())
            den = None

        usd = int(float(usd_raw))
        pid = ctx.positionbook.open_position(num, den, usd, ts_ms, note=note)
        return f"Opened pair {pid}: {num}/{den} target=${abs(usd):.0f} (queued price backfill)."

    @R.at("thinkers")
    def _at_thinkers(ctx: CommandContext, args: Dict[str,str]) -> str:
        rows = ctx.store.list_thinkers()
        if not rows:
            return "No thinkers."
        out = []
        for r in rows:
            out.append(f"#{r['id']} {r['kind']} enabled={r['enabled']} cfg={r['config_json']}")
        return "\n".join(out)

    @R.bang("thinker-enable")
    def _bang_thinker_enable(ctx: CommandContext, args: Dict[str,str]) -> str:
        tid = args.get("_","").strip()
        if not tid.isdigit():
            return "Usage: !thinker-enable <id>"
        ctx.store.update_thinker_enabled(int(tid), True)
        return f"Thinker #{tid} enabled."

    @R.bang("thinker-disable")
    def _bang_thinker_disable(ctx: CommandContext, args: Dict[str,str]) -> str:
        tid = args.get("_","").strip()
        if not tid.isdigit():
            return "Usage: !thinker-disable <id>"
        ctx.store.update_thinker_enabled(int(tid), False)
        return f"Thinker #{tid} disabled."

    @R.bang("thinker-rm")
    def _bang_thinker_rm(ctx: CommandContext, args: Dict[str,str]) -> str:
        tid = args.get("_","").strip()
        if not tid.isdigit():
            return "Usage: !thinker-rm <id>"
        ctx.store.delete_thinker(int(tid))
        return f"Thinker #{tid} deleted."

    @R.bang("alert")
    def _bang_alert(ctx: CommandContext, args: Dict[str,str]) -> str:
        """
        !alert <SYMBOL> >= <PRICE> [msg:...]
        !alert <SYMBOL> <= <PRICE> [msg:...]
        """
        tail = args.get("_","").strip()
        parts = tail.split()
        if len(parts) < 3:
            return "Usage: !alert <SYMBOL> (>=|<=) <PRICE> [msg:...]"
        sym, op, pr = parts[0].upper(), parts[1], parts[2]
        if op not in (">=", "<="):
            return "Op must be >= or <="
        direction = "ABOVE" if op == ">=" else "BELOW"
        try:
            price = float(pr)
        except:
            return "Bad price."
        msg = ""
        if "msg:" in tail:
            msg = tail.split("msg:",1)[1].strip()
        cfg = {"symbol": sym, "direction": direction, "price": price, "message": msg}
        tid = ctx.store.insert_thinker("THRESHOLD_ALERT", cfg)
        return f"Thinker #{tid} THRESHOLD_ALERT set for {sym} {direction} {price}"

    @R.bang("psar")
    def _bang_psar(ctx: CommandContext, args: Dict[str,str]) -> str:
        """
        !psar <position_id> <symbol> <LONG|SHORT> [af:0.02] [max:0.2] [win:200]
        """
        tail = args.get("_","").strip()
        parts = tail.split()
        if len(parts) < 3:
            return "Usage: !psar <position_id> <symbol> <LONG|SHORT> [af:0.02] [max:0.2] [win:200]"
        pid, sym, d = parts[0], parts[1].upper(), parts[2].upper()
        if d not in ("LONG","SHORT"):
            return "Direction must be LONG|SHORT"
        # optional kvs
        kv = {"af":0.02, "max_af":0.2, "window_min":200}
        for tok in parts[3:]:
            if ":" in tok:
                k,v = tok.split(":",1)
                k = k.lower().strip()
                v = v.strip()
                if k == "af": kv["af"] = float(v)
                elif k in ("max","max_af"): kv["max_af"] = float(v)
                elif k in ("win","window","window_min"): kv["window_min"] = int(v)
        cfg = {"position_id": pid, "symbol": sym, "direction": d, **kv}
        tid = ctx.store.insert_thinker("PSAR_STOP", cfg)
        return f"Thinker #{tid} PSAR_STOP set for {pid}/{sym} dir={d} af={kv['af']} max={kv['max_af']} win={kv['window_min']}"

    # ---------- @jobs: list DB jobs ----------
    @R.at("jobs")
    def _at_jobs(ctx: CommandContext, args: Dict[str, str]) -> str:
        """
        @jobs [state:PENDING|RUNNING|DONE|ERR] [limit:50]
        """
        state = args.get("state")
        limit = int(args.get("limit", "50"))

        rows = ctx.store.list_jobs(state=state, limit=limit)
        if not rows:
            return "No jobs."

        def _fmt_ts(ms: int) -> str:
            # ISO UTC, seconds precision
            return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat(timespec="seconds")

        lines = []
        for r in rows:
            err = (r["last_error"] or "")
            if len(err) > 120:
                err = err[:117] + "..."
            lines.append(
                f"{r['job_id']}  {r['state']}  {r['task']}  attempts={r['attempts']} "
                f"pos={r['position_id'] or '-'}  "
                f"created={_fmt_ts(r['created_ts'])}  updated={_fmt_ts(r['updated_ts'])}"
                + (f"\n  err: {err}" if r["state"] == "ERR" and err else "")
            )
        return "\n".join(lines)

    # ---------- !retry_jobs: retry failed jobs ----------
    @R.bang("retry_jobs")
    def _bang_retry_jobs(ctx: CommandContext, args: Dict[str, str]) -> str:
        """
        !retry_jobs [id:<job_id>] [limit:N]
          - With id: retries one job.
          - Without id: retries all ERR jobs (optionally limited).
        """
        jid = args.get("id")
        if jid:
            ok = ctx.store.retry_job(jid)
            return f"{'Retried' if ok else 'Not found'}: {jid}"

        limit = args.get("limit")
        n = ctx.store.retry_failed_jobs(limit=int(limit) if limit else None)
        return f"Retried {n} failed job(s)."

    return R


def latest_marks_for_positions(api, store, rows) -> Dict[str,float]:
    syms = {lg["symbol"] for r in rows for lg in store.get_legs(r["position_id"]) if lg["symbol"]}
    now = Clock.now_utc_ms(); out={}
    for s in syms:
        try:
            mk = api.mark_price_klines(s, "1m", now-60_000, now)
            if mk: out[s] = float(mk[-1][4])
        except Exception as e:
            log.debug("mark snapshot fail", symbol=s, err=str(e))
    return out


def pnl_for_position(store, positionbook, position_id, marks) -> float:
    res = positionbook.pnl_position(position_id, marks)
    return res["pnl_usd"] if res["ok"] else 0.0


def exec_positions(store, positionbook, api, reporter, args) -> str:
    """
    Pure function used by @positions. Mirrors BotEngine._exec_positions but without self.
    args keys: status, what, limit, position_id, pair
    """
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
    marks = latest_marks_for_positions(api, store, rows)
    if what == "summary":
        return reporter.fmt_positions_summary(store, positionbook, rows, marks)
    if what == "full":
        return reporter.fmt_positions_full(store, positionbook, rows, marks, limit)
    if what == "legs":
        return reporter.fmt_positions_legs(store, positionbook, rows, marks, limit)

    # Signal mis-usage clearly (your earlier convention)
    raise RuntimeError('ChatGPT: unknown "what". Use one of summary|full|legs')

