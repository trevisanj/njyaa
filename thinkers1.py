#!/usr/bin/env python3
# FILE: thinkers1.py

from __future__ import annotations
import json, math, sqlite3, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Iterable, Tuple

from bot_api import Storage, BinanceUM, MarketCatalog, PriceOracle, PositionBook
from klines_cache import KlinesCache
from common import log, Clock, AppConfig
from contracts import EngineServices
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from common import Log, sublog

# -----------------------------
# Thinker Registry/Factory
# -----------------------------

import sys, inspect

# -----------------------------
# Manager (DB + wiring + loop)
# -----------------------------

class ThinkerManager:
    """
    Loads thinkers from DB, instantiates them, runs periodic ticks,
    collects actions and writes thinker_state_log entries, and emits alerts via provided callback.

    You own the 'emit_alert' function (e.g., Telegram send).
    """

    def __init__(self, eng: EngineServices):
        self.eng = eng
        self.store = eng.store
        self._instances: Dict[int, ThinkerBase] = {}    # id -> instance
        self._configs: Dict[int, Dict[str, Any]] = {}  # id -> config dict
        self.log = sublog("thinking")

        self.ctx = ThinkerContext(
            log=self.log,
            cfg=eng.cfg, store=eng.store, mc=eng.mc, oracle=eng.oracle,
            positionbook=eng.positionbook, kc=eng.kc,)

        self.factory = ThinkerFactory()

    # --- load/instantiate
    # TODO what when thinkers are activated/deactivated/deleted when the system is running?
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
        inst = self.factory.create(tr.kind)
        inst.init(tr.config())
        self._instances[tr.id] = inst
        self._configs[tr.id] = tr.config()
        self.log.info("thinker.ready", id=tr.id, kind=tr.kind)

    # --- run one cycle
    def run_once(self, now_ms: Optional[int] = None) -> int:
        """
        Execute one full pass of enabled thinkers.
        Returns number of actions emitted.
        """
        now = now_ms or Clock.now_utc_ms()

        actions = 0
        rows = self._load()
        self.log.debug(f"Thinking ... {len(rows)} thinkers ...")
        for tr in rows:
            self._ensure_instantiated(tr)
            inst = self._instances[tr.id]
            try:
                out = inst.tick(self.ctx, now)
            except Exception as e:
                log().exc(e, where="thinker.tick", thinker_id=tr.id, kind=tr.kind)
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
                        self.eng.emit_alert(f"{act.text}")
                    except Exception as e:
                        log().exc(e, where="thinker.emit_alert", thinker_id=tr.id)

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
                    log().exc(e, where="thinker.persist_runtime", thinker_id=tr.id)

        return actions

    def _log_state(self, thinker_id: int, ts: int, level: str, message: str, payload: Dict[str, Any]):
        try:
            self.store.con.execute("""INSERT INTO thinker_state_log(thinker_id, ts, level, message, payload_json)
                                      VALUES(?,?,?,?,?)""",
                                   (thinker_id, ts, level, message, json.dumps(payload or {}, ensure_ascii=False)))
            self.store.con.commit()
        except Exception as e:
            self.log.exc(e, where="thinker_state_log.insert", thinker_id=thinker_id)


# -----------------------------
# Thinker Context
# -----------------------------

@dataclass
class ThinkerContext:
    """
    Shared lightweight read-only view of the trading environment
    accessible by all Thinkers each tick.
    """
    log: Log
    cfg: AppConfig
    store: Storage
    mc: MarketCatalog
    oracle: PriceOracle
    positionbook: PositionBook
    kc: KlinesCache

# -----------------------------
# Thinker Factory
# -----------------------------

class ThinkerFactory:
    def __init__(self):
        self._map = {}
        import thinkers2
        module = sys.modules["thinkers2"]  # self reference
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if hasattr(obj, "kind") and getattr(obj, "kind", None):
                self._map[obj.kind] = obj

    def create(self, kind: str) -> ThinkerBase:
        cls = self._map.get(kind)
        if not cls:
            raise ValueError(f"Unknown thinker kind: {kind}")
        return cls()


# -----------------------------
# Thinker Protocol + Envelope
# -----------------------------

class ThinkerBase(ABC):
    kind: str  # e.g. "THRESHOLD_ALERT", "PSAR_STOP"

    @abstractmethod
    def init(self, config: Dict[str, Any]) -> None: ...

    @abstractmethod
    def tick(self, ctx: ThinkerContext, now_ms: int) -> List["ThinkerAction"]: ...


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
