# common.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from zoneinfo import ZoneInfo
import sys, json, threading
from datetime import datetime, timezone

__all__ = [ "AppConfig", "Clock", "Log", "log", "set_global_logger", ]

# =======================
# ====== CONFIG =========
# =======================

@dataclass
class AppConfig:
    """
    Unified configuration container for the trading system.
    All fields are optional — instantiate with only what’s needed.
    """

    # --- General system ---
    # Local timezone for reporting, logs, scheduling.
    TZ_LOCAL: Optional[ZoneInfo] = None

    # Path for general-purpose database (used by other modules).
    DB_PATH: Optional[str] = None

    # Logging verbosity level: "DEBUG", "INFO", "WARNING", etc.
    LOG_LEVEL: Optional[str] = "INFO"

    # How many seconds of prices to backfill on startup.
    PRICE_BACKFILL_WINDOW_SEC: Optional[int] = None

    # Maximum backfill depth in seconds to prevent overload.
    PRICE_BACKFILL_MAX_SEC: Optional[int] = None

    # Polling interval (seconds) for mark price updates.
    MARK_POLL_SEC: Optional[int] = None

    # Polling interval (seconds) for income / funding / realized PnL.
    INCOME_POLL_SEC: Optional[int] = None

    # Digits to display when reporting USD values.
    REPORT_USD_DIGITS: Optional[int] = 2

    # --- Telegram alerts ---
    # Enable or disable Telegram integration.
    TELEGRAM_ENABLED: Optional[bool] = False

    # Telegram bot token.
    TELEGRAM_TOKEN: Optional[str] = None

    # Telegram chat ID to send alerts to.
    TELEGRAM_CHAT_ID: Optional[str] = None

    # --- Binance stuff ---
    # Binance API key.
    BINANCE_KEY: Optional[str] = None

    # Binance API secret.
    BINANCE_SEC: Optional[str] = None

    # Binance receive window in milliseconds (default: 60_000).
    RECV_WINDOW_MS: Optional[int] = 60_000

    # --- Klines cache subsystem ---
    # Path to SQLite database for cached candles.
    KLINES_CACHE_DB_PATH: Optional[str] = None

    # Number of most recent bars to keep per (symbol, timeframe).
    KLINES_CACHE_KEEP_BARS: Optional[int] = None

    # Polling interval (seconds) for refreshing klines from API.
    KLINES_POLL_SEC: Optional[int] = None

    # List of timeframes to maintain in cache, e.g. ["1m", "5m", "1h"].
    KLINES_TIMEFRAMES: Optional[List[str]] = field(default_factory=list)

    # Max number of bars to fetch per API call.
    KLINES_FETCH_LIMIT: Optional[int] = None


# =======================
# ====== CLOCK ==========
# =======================

class Clock:
    """
    Centralized time helpers.
    - Uses an injectable local TZ (default None) for naive ISO strings.
    - No hidden dependency on Config/AppConfig to avoid import cycles.
    """
    _tz_local: ZoneInfo = None

    @classmethod
    def set_tz(cls, tz: ZoneInfo):
        cls._tz_local = tz

    @classmethod
    def get_tz(cls) -> Optional[ZoneInfo]:
        return cls._tz_local

    @staticmethod
    def now_utc_ms() -> int:
        return int(datetime.now(timezone.utc).timestamp() * 1000)

    @classmethod
    def from_local_iso_to_utc_ms(cls, s: str) -> int:
        """
        Accepts:
          - ISO8601 string; if tz-naive, interpret in the injected local TZ.
          - Integers as epoch seconds or milliseconds are **not** parsed here; callers
            can just pass ints directly where needed.
        """
        try:
            dt = datetime.fromisoformat(s)
        except Exception as e:
            raise ValueError(f"Bad timestamp format: {s}") from e
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=cls._tz_local)
        return int(dt.astimezone(timezone.utc).timestamp() * 1000)


# =======================
# ====== TELEMETRY ======
# =======================

class Log:
    LV = {"DEBUG":10, "INFO":20, "WARN":30, "ERROR":40}

    def __init__(self, level="INFO", stream=None, json_mode=False, name=None, context=None):
        self.stream = stream or sys.stdout
        self.level  = self.LV.get(str(level).upper(), 20)
        self.json_mode = bool(json_mode)
        self.name   = name  # e.g., 'rv.worker'
        self.ctx    = dict(context or {})
        self._lock  = threading.Lock()

    # ---- composition ----
    def set_level(self, level:str):
        self.level = self.LV.get(str(level).upper(), self.level)
        return self

    def child(self, name:str):
        full = f"{self.name}.{name}" if self.name else name
        return Log(level=self.level_name, stream=self.stream, json_mode=self.json_mode, name=full, context=self.ctx)

    def bind(self, **extra):
        # returns a logger with default context merged in
        return Log(level=self.level_name, stream=self.stream, json_mode=self.json_mode, name=self.name,
                   context={**self.ctx, **extra})

    # ---- emitters ----
    @property
    def level_name(self):
        for k,v in self.LV.items():
            if v == self.level: return k
        # not found (custom), return numeric
        return str(self.level)

    def _emit(self, lvname:str, msg:str, **fields):
        if self.LV[lvname] < self.level:
            return
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        merged = {**self.ctx, **fields} if (fields or self.ctx) else None
        with self._lock:
            if self.json_mode:
                payload = {"ts": ts, "level": lvname, "name": self.name, "msg": msg}
                if merged: payload.update(merged)
                self.stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            else:
                name_part = f" {self.name}" if self.name else ""
                ctx_part = (" " + json.dumps(merged, ensure_ascii=False)) if merged else ""
                self.stream.write(f"[{ts}] {lvname}{name_part} {msg}{ctx_part}\n")
            self.stream.flush()

    def debug(self, m, **k): self._emit("DEBUG", m, **k)
    def info(self, m, **k):  self._emit("INFO",  m, **k)
    def warn(self, m, **k):  self._emit("WARN",  m, **k)
    def error(self, m, **k): self._emit("ERROR", m, **k)

    def exc(self, e: Exception, **k):
        import traceback
        tb = traceback.format_exc()
        if self.json_mode:
            self.error("exception", err=str(e), traceback=tb, **k)
        else:
            self._emit("ERROR", f"Exception:\n{tb}", **k)


# ---- global logger + override hook ----
log = Log("DEBUG", name="rv")  # default singleton used across API

def set_global_logger(new_log: Log):
    """Call this from your main to replace the API's global `log`."""
    global log
    log = new_log
