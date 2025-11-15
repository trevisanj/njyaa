#!/usr/bin/env python3
# FILE: thinkers.py

from __future__ import annotations
import json, math, sqlite3, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Iterable, Tuple

from bot_api import Storage, BinanceUM, MarketCatalog, PriceOracle
from common import log, Clock, AppConfig
from contracts import EngineServices

# -----------------------------
# Thinker Protocol + Envelope
# -----------------------------

class Thinker(Protocol):
    """
    Contract for pluggable 'thinkers' (analysis/alert/stop logic).

    A Thinker must be **pure-ish** w.r.t. inputs; it can read anything from the
    provided context and may emit zero or more 'actions' (alerts, logs, orders later).
    It should NOT perform direct I/O side-effects (Telegram, DB writes). Return actions instead.

    Lifecycle:
      - init(...) happens when the manager constructs instances
      - tick(ctx, now_ms) called each engine cycle
    """
    kind: str  # stable identifier, e.g. "THRESHOLD_ALERT", "PSAR_STOP"

    def init(self, config: Dict[str, Any]) -> None: ...
    def tick(self, ctx: "ThinkerContext", now_ms: int) -> List["ThinkerAction"]: ...


@dataclass
class ThinkerAction:
    """
    Unified action envelope emitted by thinkers.

    type: "ALERT" | "LOG" | (future: "ORDER")
    level: freeform ("INFO","WARN","CRIT") for ALERT/LOG convenience
    text: human-readable message
    payload: structured dict (for logs, state, or downstream integrations)
    """
    type: str
    level: str
    text: str
    payload: Dict[str, Any]


@dataclass
class ThinkerRow:
    """
    DB materialization of a thinker.
    """
    id: int
    kind: str
    enabled: int
    config_json: str
    runtime_json: str
    created_ts: int
    updated_ts: int

    def config(self) -> Dict[str, Any]:
        try:
            return json.loads(self.config_json or "{}")
        except Exception:
            return {}

    def runtime(self) -> Dict[str, Any]:
        try:
            return json.loads(self.runtime_json or "{}")
        except Exception:
            return {}


# -----------------------------
# Context passed to thinkers
# -----------------------------

@dataclass
class ThinkerContext:
    """
    Read-only façade for a thinker to fetch what it needs without coupling to big objects.

    Access paths:
      - price snapshots via PriceOracle or direct klines through api
      - marks via api.mark_price_klines(...)
      - storage for *reads* (thinkers don't write their own state directly)
      - catalog for symbol normalization (if needed)
    """
    cfg: AppConfig
    store: Storage
    api: BinanceUM
    mc: MarketCatalog
    oracle: PriceOracle

    # convenience helpers
    def mark_close_now(self, symbol: str) -> Optional[float]:
        """
        Return last mark close (1m) for symbol, or None on failure.
        """
        try:
            now = Clock.now_utc_ms()
            mk = self.api.mark_price_klines(symbol, "1m", now-60_000, now)
            if isinstance(mk, list) and mk:
                return float(mk[-1][4])
        except Exception as e:
            log.debug("ctx.mark_close_now.fail", symbol=symbol, err=str(e))
        return None

    def minute_klines(self, symbol: str, minutes: int = 200) -> List[List[Any]]:
        """
        Fetch last `minutes` mark-price klines @ 1m (open,high,low,close,...).
        """
        now = Clock.now_utc_ms()
        start = now - max(2, minutes) * 60_000
        return self.api.mark_price_klines(symbol, "1m", start, now)


# --------------------------------
# Thinker Implementations (examples)
# --------------------------------

class ThresholdAlertThinker:
    """
    Simple price threshold alert:
      config = {
        "symbol": "ETHUSDT",
        "direction": "ABOVE" | "BELOW",
        "price": 4000.0,
        "message": "ETH crosses 4k",
      }

    Runtime memory (managed by manager):
      { "fired": false, "last_price": 3981.2, "last_fire_ts": 0 }
    """
    kind = "THRESHOLD_ALERT"

    def __init__(self):
        self._cfg: Dict[str, Any] = {}

    def init(self, config: Dict[str, Any]) -> None:
        self._cfg = dict(config or {})
        mandatory = ("symbol", "direction", "price")
        for k in mandatory:
            if k not in self._cfg:
                raise ValueError(f"ThresholdAlertThinker missing '{k}'")

        d = self._cfg["direction"].upper()
        if d not in ("ABOVE", "BELOW"):
            raise ValueError("direction must be ABOVE or BELOW")
        self._cfg["direction"] = d
        self._cfg["price"] = float(self._cfg["price"])

    def tick(self, ctx: ThinkerContext, now_ms: int) -> List[ThinkerAction]:
        sym = self._cfg["symbol"]
        pr = ctx.mark_close_now(sym)
        if pr is None:
            return []

        msg = self._cfg.get("message") or f"{sym} {self._cfg['direction']} {self._cfg['price']}"
        thr = self._cfg["price"]; dir_ = self._cfg["direction"]

        hit = (pr >= thr) if dir_ == "ABOVE" else (pr <= thr)
        if not hit:
            return [ThinkerAction("LOG", "DEBUG", f"[thr] {sym}={pr:.4f} vs {dir_} {thr}", {"symbol": sym, "price": pr})]

        return [ThinkerAction(
            type="ALERT",
            level="INFO",
            text=f"{msg} | last={pr:.4f}",
            payload={"symbol": sym, "price": pr, "threshold": thr, "direction": dir_}
        )]


class PSARStopThinker:
    """
    Minimal Parabolic SAR trailing stop for a specific position.

    config = {
      "position_id": "<hex>",
      "symbol": "ETHUSDT",
      "direction": "LONG" | "SHORT",   # relative to stop logic
      "af": 0.02,
      "max_af": 0.2,
      "window_min": 200   # how many last 1m bars to pull (for smoothing)
    }

    Runtime we persist (manager saves it for us):
      {
        "psar": float,
        "ep": float,
        "af": float,
        "trend": "UP"|"DOWN",
        "last_candle_open": int,
        "last_alert_ts": int
      }

    Fires ALERT when price crosses PSAR against trend.
    """
    kind = "PSAR_STOP"

    def __init__(self):
        self._cfg: Dict[str, Any] = {}

    def init(self, config: Dict[str, Any]) -> None:
        self._cfg = dict(config or {})
        for k in ("position_id", "symbol", "direction"):
            if k not in self._cfg:
                raise ValueError(f"PSAR missing '{k}'")
        self._cfg["af"] = float(self._cfg.get("af", 0.02))
        self._cfg["max_af"] = float(self._cfg.get("max_af", 0.2))
        self._cfg["window_min"] = int(self._cfg.get("window_min", 200))
        d = self._cfg["direction"].upper()
        if d not in ("LONG", "SHORT"):
            raise ValueError("direction must be LONG or SHORT")
        self._cfg["direction"] = d

    # classic PSAR step on OHLC stream
    @staticmethod
    def _psar_series(ohlc: List[Tuple[float,float,float,float]], af0: float, afmax: float):
        """
        Return last (psar, ep, af, trend) after iterating series.

        ohlc: list of (o,h,l,c)
        """
        if len(ohlc) < 5:
            raise ValueError("Need at least 5 bars for PSAR bootstrap")

        # Bootstrap from first bars
        h0 = max(x[1] for x in ohlc[:5]); l0 = min(x[2] for x in ohlc[:5])
        up = True  # start bullish by default; will flip quickly if wrong
        ep = h0 if up else l0
        af = af0
        psar = l0 if up else h0

        for i in range(5, len(ohlc)):
            o,h,l,c = ohlc[i]
            prev_psar = psar
            prev_ep = ep
            prev_up = up

            # move PSAR
            psar = prev_psar + af * (prev_ep - prev_psar)

            # clamp against last/prev extremities
            if up:
                psar = min(psar, min(ohlc[i-1][2], ohlc[i-2][2]))  # clamp to last 2 lows
            else:
                psar = max(psar, max(ohlc[i-1][1], ohlc[i-2][1]))  # clamp to last 2 highs

            # trend check
            if up:
                if l < psar:  # flip to downtrend
                    up = False
                    psar = prev_ep
                    ep = l
                    af = af0
                else:
                    if h > prev_ep:
                        ep = h
                        af = min(af + af0, afmax)
                    else:
                        ep = prev_ep
            else:
                if h > psar:  # flip to uptrend
                    up = True
                    psar = prev_ep
                    ep = h
                    af = af0
                else:
                    if l < prev_ep:
                        ep = l
                        af = min(af + af0, afmax)
                    else:
                        ep = prev_ep

        return psar, ep, af, ("UP" if up else "DOWN")

    def tick(self, ctx: ThinkerContext, now_ms: int) -> List[ThinkerAction]:
        sym = self._cfg["symbol"]
        mins = self._cfg["window_min"]
        rows = ctx.minute_klines(sym, mins)
        if not rows:
            return []

        # Build OHLC tuples (using mark price klines)
        # rows: [open_time, open, high, low, close, volume, close_time, ...]
        ohlc = [(float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rows]
        psar, ep, af, trend = self._psar_series(ohlc, self._cfg["af"], self._cfg["max_af"])
        last_close = float(rows[-1][4])

        # Direction semantics:
        # - LONG: stop triggers if close < psar
        # - SHORT: stop triggers if close > psar
        hit = (last_close < psar) if self._cfg["direction"] == "LONG" else (last_close > psar)

        payload = {
            "symbol": sym, "psar": psar, "ep": ep, "af": af, "trend": trend,
            "last_close": last_close, "direction": self._cfg["direction"]
        }

        if hit:
            return [ThinkerAction(
                type="ALERT",
                level="WARN",
                text=f"[PSAR] {sym} {self._cfg['direction']} stop hit: close={last_close:.4f} vs psar={psar:.4f}",
                payload=payload
            )]

        return [ThinkerAction(
            type="LOG",
            level="DEBUG",
            text=f"[PSAR] {sym} {self._cfg['direction']} ok: close={last_close:.4f} psar={psar:.4f} trend={trend}",
            payload=payload
        )]


# -----------------------------
# Thinker Registry/Factory
# -----------------------------

class ThinkerFactory:
    """
    Maps kind -> constructor.
    Register your new kinds here.
    """
    _map = {
        ThresholdAlertThinker.kind: ThresholdAlertThinker,
        PSARStopThinker.kind: PSARStopThinker,
    }

    @classmethod
    def create(cls, kind: str) -> Thinker:
        if kind not in cls._map:
            raise ValueError(f"Unknown thinker kind '{kind}'")
        return cls._map[kind]()


# -----------------------------
# Manager (DB + wiring + loop)
# -----------------------------

class ThinkerManager:
    """
    Loads thinkers from DB, instantiates them, runs periodic ticks,
    collects actions and writes thinker_state_log entries, and emits alerts via provided callback.

    You own the 'emit_alert' function (e.g., Telegram send).
    """

    def __init__(self, cfg: AppConfig, store: Storage, api: BinanceUM,
                 mc: MarketCatalog, oracle: PriceOracle,
                 services: EngineServices,
                 ):
        self.cfg, self.store, self.api, self.mc, self.oracle = cfg, store, api, mc, oracle
        self.services = services
        self._instances: Dict[int, Thinker] = {}    # id -> instance
        self._configs: Dict[int, Dict[str, Any]] = {}  # id -> config dict

    # --- DB schema bootstrap (idempotent); call from Storage._init_db ideally ---
    @staticmethod
    def ensure_schema(con: sqlite3.Connection):
        cur = con.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS thinkers (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          kind TEXT NOT NULL,
          enabled INTEGER NOT NULL DEFAULT 1,
          config_json TEXT NOT NULL,
          runtime_json TEXT,
          created_ts INTEGER NOT NULL,
          updated_ts INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_thinkers_kind ON thinkers(kind);

        CREATE TABLE IF NOT EXISTS thinker_state_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          thinker_id INTEGER NOT NULL,
          ts INTEGER NOT NULL,
          level TEXT NOT NULL,
          message TEXT NOT NULL,
          payload_json TEXT,
          FOREIGN KEY(thinker_id) REFERENCES thinkers(id) ON DELETE CASCADE
        );
        """)

        con.commit()

    # --- load/instantiate
    def _load(self) -> List[ThinkerRow]:
        rows = self.store.con.execute("SELECT * FROM thinkers WHERE enabled=1 ORDER BY id").fetchall()
        out: List[ThinkerRow] = []
        for r in rows:
            out.append(ThinkerRow(
                id=int(r["id"]), kind=r["kind"], enabled=int(r["enabled"]),
                config_json=r["config_json"] or "{}", runtime_json=r["runtime_json"] or "{}",
                created_ts=int(r["created_ts"]), updated_ts=int(r["updated_ts"])
            ))
        return out

    def _ensure_instantiated(self, tr: ThinkerRow):
        if tr.id in self._instances:
            return
        inst = ThinkerFactory.create(tr.kind)
        inst.init(tr.config())
        self._instances[tr.id] = inst
        self._configs[tr.id] = tr.config()
        log.info("thinker.ready", id=tr.id, kind=tr.kind)

    # --- run one cycle
    def run_once(self, now_ms: Optional[int] = None) -> int:
        """
        Execute one full pass of enabled thinkers.
        Returns number of actions emitted.
        """
        now = now_ms or Clock.now_utc_ms()
        ctx = ThinkerContext(self.cfg, self.store, self.api, self.mc, self.oracle)

        actions = 0
        for tr in self._load():
            self._ensure_instantiated(tr)
            inst = self._instances[tr.id]
            try:
                out = inst.tick(ctx, now)
            except Exception as e:
                log.exc(e, where="thinker.tick", thinker_id=tr.id, kind=tr.kind)
                self._log_state(tr.id, now, "ERROR", f"tick failed: {e}", {})
                continue

            if not out:
                continue

            for act in out:
                actions += 1
                # Persist to state log
                self._log_state(tr.id, now, act.level, act.text, act.payload)

                # Emit alerts externally
                if act.type == "ALERT":
                    try:
                        self.services.emit_alert(f"{act.text}")
                    except Exception as e:
                        log.exc(e, where="thinker.emit_alert", thinker_id=tr.id)

            # Optional: allow thinkers to persist small runtime snapshots (e.g. last psar)
            # Convention: if action payload has "_runtime", we persist it.
            # (Lightweight pattern to avoid deep couplings.)
            _rts = [a.payload.get("_runtime") for a in out if isinstance(a.payload, dict) and a.payload.get("_runtime")]
            if _rts:
                try:
                    rt = tr.runtime()
                    rt.update(_rts[-1])  # last wins
                    self.store.con.execute("UPDATE thinkers SET runtime_json=?, updated_ts=? WHERE id=?",
                                           (json.dumps(rt, ensure_ascii=False), now, tr.id))
                    self.store.con.commit()
                except Exception as e:
                    log.exc(e, where="thinker.persist_runtime", thinker_id=tr.id)

        return actions

    def _log_state(self, thinker_id: int, ts: int, level: str, message: str, payload: Dict[str, Any]):
        try:
            self.store.con.execute("""INSERT INTO thinker_state_log(thinker_id, ts, level, message, payload_json)
                                      VALUES(?,?,?,?,?)""",
                                   (thinker_id, ts, level, message, json.dumps(payload or {}, ensure_ascii=False)))
            self.store.con.commit()
        except Exception as e:
            log.exc(e, where="thinker_state_log.insert", thinker_id=thinker_id)
