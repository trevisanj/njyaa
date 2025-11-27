#!/usr/bin/env python3
# FILE: thinkers1.py

from __future__ import annotations
import json, math, sqlite3, time, threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Iterable, Tuple, Union, TYPE_CHECKING, get_type_hints, Literal
from bot_api import Storage, BinanceUM, MarketCatalog, PriceOracle, PositionBook
from klines_cache import KlinesCache
from common import log, Clock, AppConfig, coerce_to_type
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from common import Log, sublog
import common
from dataclasses import dataclass, fields, is_dataclass

if TYPE_CHECKING:
    from commands import CO
    from bot_api import BotEngine

# -----------------------------
# Thinker Registry/Factory
# -----------------------------

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
        self._reload_pending: set[int] = set()
        self._disable_pending: set[int] = set()
        self._messages: Dict[int, list] = {}
        self._lock = threading.RLock()
        self.log = sublog("thinking")
        self.factory = ThinkerFactory(self, eng)

    def get_in_carbonite(self, thinker_id: int, expected_kind=None) -> "ThinkerBase":
        """Returns thinker frozen in carbonite (cannot be tick()'ed)

        Args:
            thinker_id:
            expected_kind: if passed, will raise TypeError if thinker's type does not match expected_kind
        """
        tr = self.eng.store.get_thinker(thinker_id)
        if expected_kind and tr["kind"] != expected_kind:
            raise TypeError(f"Thinker #{thinker_id} is a {tr['kind']} (expected {expected_kind})")
        inst = self.factory.create(tr["kind"])
        inst._init_from_row(tr)
        inst._in_carbonite = True
        return inst

    # --- load/instantiate
    def _load(self) -> List[sqlite3.Row]:
        return self.store.con.execute("SELECT * FROM thinkers WHERE enabled=1 ORDER BY id").fetchall()

    def reload(self, thinker_id: int):
        tid = int(thinker_id)
        tr = self.store.get_thinker(tid)
        with self._lock:
            self._reload_pending.discard(tid)
            if tr and tr["enabled"]:
                self._reload_pending.add(tid)

    def _ensure_instantiated(self, tr: sqlite3.Row):
        tid = int(tr["id"])
        with self._lock:
            if tid in self._reload_pending:
                self._instances.pop(tid, None)
                self._reload_pending.discard(tid)
            if tid in self._instances:
                return
            inst = self.factory.create(tr["kind"])
            inst._init_from_row(tr)
            self._instances[tid] = inst
            self.log.info("thinker.ready", id=tid, kind=tr["kind"])

    # --- run one cycle
    def run_once(self, now_ms: Optional[int] = None) -> Tuple[int, int]:
        """
        Execute one full pass of enabled thinkers.
        Returns number of actions emitted.
        """
        now = now_ms or Clock.now_utc_ms()

        n_ok, n_fail = 0, 0
        with self._lock:
            disable_batch = list(self._disable_pending)
        for tid in disable_batch:
            self._drop(tid, None)

        rows = self._load()
        self.log.debug(f"Thinking ... {len(rows)} thinkers ...")
        for tr in rows:
            tid = int(tr["id"])
            with self._lock:
                if tid in self._disable_pending:
                    self._drop(tid, tr)
                    continue

            if self.eng.stopping():
                log().info("tm.run_once.cancelled", reason="Engine stopping")
                break

            self._ensure_instantiated(tr)
            inst = self._instances[tid]
            try:
                result = inst.tick(now)
                if result is not None:
                    self.log_event(tid, "INFO", "thinker.result", result=str(result), thinker_id=tid, kind=tr["kind"])
                n_ok += 1
            except Exception as e:
                log().exc(e, where="thinker.tick", thinker_id=tid, kind=tr["kind"])
                self.store.log_thinker_event(tid, "ERROR", f"tick failed: {e}", {})
                n_fail += 1

        return n_ok, n_fail

    def _drop(self, tid: int, tr=None):
        self._disable_pending.discard(tid)
        removed = self._instances.pop(tid, None)
        kind = removed.kind if removed else (tr["kind"] if tr is not None else "?")
        self._reload_pending.discard(tid)
        self.log.info("thinker.disabled", id=tid, kind=kind)

    def log_event(self, tid, level, message, **kwargs):
        assert level in common.LV
        self.store.log_thinker_event(tid, level, message, payload=kwargs)
        log()._emit(level, message, **kwargs)

    def disable(self, thinker_id: int):
        """Mark a thinker to be disabled/purged on the next run_once pass."""
        tid = int(thinker_id)
        with self._lock:
            self._disable_pending.add(tid)


# -----------------------------
# Thinker Factory
# -----------------------------

class ThinkerFactory:
    def __init__(self, tm: ThinkerManager, eng: BotEngine):
        self._map = {}
        self._kinds = []
        self.tm = tm
        self.eng = eng
        import thinkers2  # noqa: F401  ensure subclasses register
        classes = sorted(ThinkerBase.__subclasses__(), key=lambda cls: cls.__name__)
        for cls in classes:
            assert cls.kind, f"Please be kind enough to specify {cls.__name__}.kind"
            assert cls.kind not in self._map, f"Duplicate thinker kind: {cls.kind}"
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

    def cls_for(self, kind: str):
        return self._map[kind]


# -----------------------------
# Thinker Protocol + Envelope
# -----------------------------

class ThinkerBase(ABC):
    kind: str = None  # e.g. "THRESHOLD_ALERT", "PSAR_STOP"
    Config: Optional[type] = None  # dataclass for typed config

    def __init__(self, tm: ThinkerManager, eng: BotEngine):
        self.tm = tm
        self.eng = eng
        self._cfg: Dict[str, Any] = {}
        self._runtime: Dict[str, Any] = {}
        self._thinker_id: Optional[int] = None
        self._in_carbonite = False
        self._tick_count = 0

    def _set_def_cfg(self, cfg: Dict[str, Any]) -> None:
        assert isinstance(cfg, dict)
        for k, v in cfg.items():
            if k not in self._cfg:
                self._cfg[k] = v

    def _init_from_row(self, tr) -> None:
        tid = int(tr["id"])
        config = _parse_json(tr["config_json"])
        runtime = _parse_json(tr["runtime_json"])
        self._thinker_id = int(tid)
        self._cfg = self._build_cfg(config or {})
        self._runtime = dict(runtime or {})
        self._on_init()

    def runtime(self) -> Dict[str, Any]:
        return self._runtime

    def save_runtime(self):
        assert self._thinker_id is not None
        self.eng.store.update_thinker_runtime(self._thinker_id, self._runtime)

    def save_config(self):
        assert self._thinker_id is not None
        self.eng.store.update_thinker_config(self._thinker_id, self._cfg)

    def notify(self, level: str, msg: str, send=False, **kwargs) -> None:
        self.tm.log_event(self._thinker_id, level, msg, **kwargs)
        if send:  # TODO decouple message and logging
            self.eng.send_text(msg)

    def _on_init(self) -> None:
        """Hook for subclasses after config validation."""
        return

    def _build_cfg(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        ConfigCls = getattr(self, "Config", None)
        if ConfigCls:
            assert is_dataclass(ConfigCls), "Config must be a dataclass"
            hints = get_type_hints(ConfigCls)
            kwargs: Dict[str, Any] = {}
            for f in fields(ConfigCls):
                name = f.name
                if name in cfg:
                    kwargs[name] = coerce_to_type(cfg[name], hints.get(name, Any))
                else:
                    kwargs[name] = getattr(ConfigCls, name, f.default)
            obj = ConfigCls(**kwargs)
            cfg_dict: Dict[str, Any] = {f.name: getattr(obj, f.name) for f in fields(ConfigCls)}
            for k, v in cfg.items():
                if k not in cfg_dict:
                    cfg_dict[k] = v
            return cfg_dict
        return dict(cfg)

    def tick(self, now_ms: int):
        if self._in_carbonite:
            raise RuntimeError(f"This {self.__class__.__name__} is frozen in carbonite")
        ret = self.on_tick(now_ms)
        self._tick_count += 1
        return ret


    @abstractmethod
    def on_tick(self, now_ms: int): ...
