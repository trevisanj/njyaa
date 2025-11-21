#!/usr/bin/env python3
# FILE: thinkers1.py

from __future__ import annotations
import json, math, sqlite3, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Iterable, Tuple, Union, TYPE_CHECKING

from setuptools.msvc import msvc14_get_vc_env

from bot_api import Storage, BinanceUM, MarketCatalog, PriceOracle, PositionBook
from klines_cache import KlinesCache
from common import log, Clock, AppConfig
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from common import Log, sublog
import common

if TYPE_CHECKING:
    from commands import CO
    from bot_api import BotEngine

# -----------------------------
# Thinker Registry/Factory
# -----------------------------

import sys, inspect

def _parse_json(blob: Optional[str]) -> Dict[str, Any]:
    try:
        return json.loads(blob or "{}")
    except Exception:
        return {}

# -----------------------------
# Manager (DB + wiring + loop)
# -----------------------------

class ThinkerManager:
    """
    Loads thinkers from DB, instantiates them, runs periodic ticks,
    collects actions and writes thinker_state_log entries, and emits alerts via provided callback.

    You own the 'emit_alert' function (e.g., Telegram send).
    """

    def __init__(self, eng: BotEngine):
        self.eng = eng
        self.store = eng.store
        self._instances: Dict[int, ThinkerBase] = {}    # id -> instance
        self._configs: Dict[int, Dict[str, Any]] = {}  # id -> config dict
        self.log = sublog("thinking")
        self.factory = ThinkerFactory(self, eng)

    # --- load/instantiate
    # TODO what when thinkers are activated/deactivated/deleted when the system is running?
    def _load(self) -> List[sqlite3.Row]:
        return self.store.con.execute("SELECT * FROM thinkers WHERE enabled=1 ORDER BY id").fetchall()

    def _ensure_instantiated(self, tr: sqlite3.Row):
        tid = int(tr["id"])
        if tid in self._instances:
            return
        inst = self.factory.create(tr["kind"])
        config = _parse_json(tr["config_json"])
        inst.init(config)
        runtime = _parse_json(tr["runtime_json"])
        inst.attach_runtime(tid, runtime)
        self._instances[tid] = inst
        self._configs[tid] = config
        self.log.info("thinker.ready", id=tid, kind=tr["kind"])

    # --- run one cycle
    def run_once(self, now_ms: Optional[int] = None) -> Tuple[int, int]:
        """
        Execute one full pass of enabled thinkers.
        Returns number of actions emitted.
        """
        now = now_ms or Clock.now_utc_ms()

        n_ok, n_fail = 0, 0
        rows = self._load()
        self.log.debug(f"Thinking ... {len(rows)} thinkers ...")
        for tr in rows:
            self._ensure_instantiated(tr)
            tid = int(tr["id"])
            inst = self._instances[tid]
            try:
                msg = inst.tick(now)
                if msg:
                    self.log_event(tid, "INFO", str(msg))
                n_ok += 1
            except Exception as e:
                log().exc(e, where="thinker.tick", thinker_id=tid, kind=tr["kind"])
                self.store.log_thinker_event(tid, "ERROR", f"tick failed: {e}", {})
                n_fail += 1

        return n_ok, n_fail

    def log_event(self, tid, level, message, **kwargs):
        assert level in common.LV
        self.store.log_thinker_event(tid, level, message, payload=kwargs)
        log()._emit(level, message, **kwargs)


# -----------------------------
# Thinker Factory
# -----------------------------

class ThinkerFactory:
    def __init__(self, tm: ThinkerManager, eng: BotEngine):
        self._map = {}
        self._kinds = []
        self.tm = tm
        self.eng = eng
        import thinkers2
        module = sys.modules["thinkers2"]  # self reference
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, ThinkerBase) and cls != ThinkerBase:
                assert cls.kind, f"Please be kind enough to specify {cls.__name__}.kind"
                self._map[cls.kind] = cls
                self._kinds.append(cls.kind)
        assert self._map, "No thinkers found, this must be a bug"

    def create(self, kind: Union[str, int]) -> ThinkerBase:
        """
        Create new thinker identified by kind

        Arguments:
            kind: thinker kind or index within self._kinds

        Returns:
            new thinker
        """
        if isinstance(kind, int):
            n = len(self._kinds)
            if kind >= n:
                raise ValueError(f"Invalid kind index: {kind} (maximum: {n})")
            kind = self._kinds[kind]

        cls = self._map.get(kind)
        if not cls:
            raise ValueError(f"Unknown thinker kind: {kind}")
        return cls(self.tm, self.eng)

    def kinds(self):
        return self._kinds


# -----------------------------
# Thinker Protocol + Envelope
# -----------------------------

class ThinkerBase(ABC):
    kind: str = None  # e.g. "THRESHOLD_ALERT", "PSAR_STOP"
    required_fields: Tuple[str, ...] = ()

    def __init__(self, tm: ThinkerManager, eng: BotEngine):
        self.tm = tm
        self.eng = eng
        self._cfg: Dict[str, Any] = {}
        self._runtime: Dict[str, Any] = {}
        self._thinker_id: Optional[int] = None

    def init(self, config: Dict[str, Any]) -> None:
        self._cfg = dict(config or {})
        self._runtime = {}
        self._on_init()
        missing = [k for k in self.required_fields if k not in self._cfg]
        if missing:
            raise ValueError(f"{self.__class__.__name__} missing {', '.join(missing)}")

    def _set_def_cfg(self, cfg: Dict[str, Any]) -> None:
        assert isinstance(cfg, dict)
        for k, v in cfg.items():
            if k not in self._cfg:
                self._cfg[k] = v

    def attach_runtime(self, thinker_id: int, runtime: Dict[str, Any]):
        self._thinker_id = int(thinker_id)
        self._runtime = dict(runtime or {})

    def runtime(self) -> Dict[str, Any]:
        return self._runtime

    def save_runtime(self):
        if self._store is None or self._thinker_id is None:
            return
        self.eng.store.update_thinker_runtime(self._thinker_id, self._runtime)

    def notify(self, level: str, msg: str, send=False, **kwargs) -> None:
        self.tm.log_event(self._thinker_id, level, msg, **kwargs)
        if send:
            self.eng.send_text(msg)

    def _on_init(self) -> None:
        """Hook for subclasses after config validation."""
        return

    @abstractmethod
    def tick(self, now_ms: int): ...
