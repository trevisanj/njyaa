# common.py
from __future__ import annotations
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
import sys, json
from typing import List, Optional
import re
from datetime import datetime, timezone, timedelta, tzinfo
import threading
from typing import Optional, Union


__all__ = [ "AppConfig", "Clock", "Log", "log", "set_global_logger", "sublog", "tf_ms", "parse_when",
            "fmt_ts_ms", "fmt_ts_s"]

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

    # Polling interval (seconds) to refreshing klines, then think
    THINKING_SEC: Optional[int] = None

    # List of timeframes to maintain in cache, e.g. ["1m", "5m", "1h"].
    KLINES_TIMEFRAMES: Optional[List[str]] = field(default_factory=list)

    # Max number of bars to fetch per API call.
    KLINES_FETCH_LIMIT: Optional[int] = None

    # Auxiliary symbols to be fetched
    AUX_SYMBOLS: Optional[List[str]] = None


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
_log = Log("DEBUG", name="rv")  # default singleton used across API

def set_global_logger(new_log: Log):
    """Call this from your main to replace the API's global `log`."""
    global _log
    _log = new_log

def log() -> Log:
    return _log

def sublog(name, **ctx) -> Log:
    """Create a child logger sharing global config."""
    return _log.child(name).bind(**ctx)


# ---- tiny TF helpers ----
_TF_MS = {
    "1s": 1_000, "3s": 3_000, "5s": 5_000, "15s": 15_000, "30s": 30_000,
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
    "1w": 604_800_000,
}
def tf_ms(tf: str) -> int:
    """
    Return timeframe length in milliseconds.

    Args:
        tf: Timeframe string like '1m', '5m', '1h', '1d'.

    Returns:
        Milliseconds for the timeframe.

    Raises:
        ValueError: If `tf` is unknown.
    """
    if tf not in _TF_MS:
        raise ValueError(f"Unknown timeframe '{tf}'. Supported: {', '.join(_TF_MS)}")
    return _TF_MS[tf]


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




def fmt_ts_ms(ts_ms: Union[int, float, str],
              tz: Optional[tzinfo] = None,
              timespec: str = "seconds") -> str:
    """
    Convert epoch milliseconds to an ISO-8601 string.

    Args:
        ts_ms: Timestamp in **milliseconds** since epoch.
        tz: Output timezone (defaults to Clock.tz if set, else UTC).
        timespec: Passed to datetime.isoformat (e.g. "seconds", "milliseconds").

    Returns:
        ISO string like '2025-11-16T12:34:56-03:00'.
    """
    ms = int(float(ts_ms))
    z = tz or Clock.get_tz()
    return datetime.fromtimestamp(ms / 1000.0, tz=z).isoformat(timespec=timespec)


def fmt_ts_s(ts_s: Union[int, float, str],
             tz: Optional[tzinfo] = None,
             timespec: str = "seconds") -> str:
    """
    Convert epoch **seconds** to an ISO-8601 string (convenience wrapper).
    """
    s = int(float(ts_s))
    z = tz or Clock.get_tz()
    return datetime.fromtimestamp(s, tz=z).isoformat(timespec=timespec)
