# common.py
from __future__ import annotations
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
import sys, json
from typing import List, Optional, Any, Literal, Union
import re
from datetime import datetime, timezone, timedelta, tzinfo
import threading
import math

__all__ = [
    "AppConfig",
    "Clock",
    "Log",
    "log",
    "set_global_logger",
    "sublog",
    "tf_ms",
    "parse_when",
    "ts_human",
    "LV",
    "coerce_to_type",
    "pct_of",
    "leg_pnl",
    "fmt_pair",
    "PP_CTX",
    "SSTRAT_CTX",
    "ATTACHED_AT",
    "SSTRAT_KIND",
    "LAST_TS",
    "THOUGHT",
    "NOW_MS",
    "LAST_MOVE_ALERT_TS",
    "LAST_HIT_ALERT_TS",
    "is_sane",
    "float2str",
    "str_exc",
    "IND_STATES",
    "TooFewDataPoints",
    "LatestPriceFail",
    "STATE_TS",
    "TICK_COUNT",
]

class TooFewDataPoints(Exception):
    pass

class LatestPriceFail(RuntimeError):
    """Raised when the latest price cannot be retrieved."""
    pass


LV = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

# Runtime key for per-position context in trailing thinkers
PP_CTX = "pp_ctx"
# Runtime key for per-strategy context under a position
SSTRAT_CTX = "sstrat_ctx"
# Strategy kind key
SSTRAT_KIND = "sstrat_kind"
# Last timestamp key
LAST_TS = "last_ts"
# Timestamp of last saved indicator states
STATE_TS = "state_ts"
# Indicator states key
IND_STATES = "ind_states"
# *User* time of attachment of a position into a thinker
ATTACHED_AT = "attached_at"
# Number of bars needed for all indicators in a sstrat to work
WINDOW_SIZE = "window_size"
# Current computed snapshot/thought key
THOUGHT = "thought"
NOW_MS = "now_ms"
LAST_MOVE_ALERT_TS = "last_move_alert_ts"
LAST_HIT_ALERT_TS = "last_hit_alert_ts"
TICK_COUNT = "tick_count"

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
    LOG_FILE_PATH: Optional[str] = None
    LOG_TO_FILE: Optional[bool] = False
    LOG_TO_STDOUT: Optional[bool] = True

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
    CONSOLE_ENABLED: Optional[bool] = False
    CONSOLE_MODE: Optional[str] = "prompt"   # prompt|curses

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
    # Path to SQLite database for indicator history cache.
    INDICATOR_HISTORY_DB_PATH: Optional[str] = None

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


HOUR_SEP = "-"

def ts_human(ms: int | datetime | None) -> str:
    """Human timestamp from ms (or datetime) in local tz."""
    if ms is None:
        return "?"
    tz = Clock.get_tz()
    if isinstance(ms, datetime):
        dt = ms.astimezone(tz)
    else:
        dt = datetime.fromtimestamp(int(ms)/1000, tz=tz)
    return dt.strftime("%Y%m%d{}%H:%M:%S").format(HOUR_SEP)


# =======================
# ====== TELEMETRY ======
# =======================

class Log:
    def __init__(self, level="INFO", stream=None, json_mode=False, name=None, context=None):
        assert level in LV
        self.stream = stream or sys.stdout
        self.level  = LV[level]
        self.json_mode = bool(json_mode)
        self.name   = name  # e.g., 'njyaa.worker'
        self.ctx    = dict(context or {})
        self._lock  = threading.Lock()

    # ---- composition ----
    def set_level(self, level:str):
        self.level = LV.get(str(level).upper(), self.level)
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
        for k,v in LV.items():
            if v == self.level: return k
        # not found (custom), return numeric
        return str(self.level)

    def _emit(self, lvname:str, msg:str, **fields):
        if LV[lvname] < self.level:
            return
        ts = ts_human(Clock.now_utc_ms())
        merged = {**self.ctx, **fields} if (fields or self.ctx) else None
        with self._lock:
            if self.json_mode:
                payload = {"ts": ts, "level": lvname, "name": self.name, "thread": threading.current_thread().name, "msg": msg}
                if merged: payload.update(merged)
                self.stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            else:
                name_part = f" {self.name}" if self.name else ""
                thread_part = f" [{threading.current_thread().name}]"
                ctx_part = (" " + json.dumps(merged, ensure_ascii=False)) if merged else ""
                self.stream.write(f"[{ts}] {lvname}{name_part}{thread_part} {msg}{ctx_part}\n")
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
_log = Log("INFO", name="njyaa")  # default singleton used across API

def set_global_logger(new_log: Log):
    """Call this from your main to replace the API's global `log`."""
    global _log
    _log = new_log

def log() -> Log:
    return _log

def sublog(name, **ctx) -> Log:
    """Create a child logger sharing global config."""
    return _log.child(name).bind(**ctx)


def coerce_to_type(val: Any, typ: Any) -> Any:
    """Best-effort coercion for thinker config fields based on type hints."""
    origin = getattr(typ, "__origin__", None)
    args = getattr(typ, "__args__", ())

    # Literal[...] handling
    if origin is Literal:
        base_types = [type(a) for a in args if a is not None]
        if base_types:
            try:
                val = coerce_to_type(val, base_types[0])
            except Exception:
                pass
        if val not in args:
            raise ValueError(f"Invalid literal {val} (allowed: {args})")
        return val

    # Optional[X] -> treat as Union[X,None]
    if origin is Union and len(args) == 2 and type(None) in args:
        other = args[0] if args[1] is type(None) else args[1]
        if val in (None, "None", "none"):
            return None
        return coerce_to_type(val, other)

    # Primitive coercions
    if typ in (int, float, str, bool):
        if typ is bool:
            if isinstance(val, str):
                if val.lower() in ("true", "1", "yes", "on"):
                    return True
                if val.lower() in ("false", "0", "no", "off"):
                    return False
            return bool(val)
        try:
            return typ(val)
        except Exception:
            raise ValueError(f"Expected {typ.__name__} got {val}")

    return val


def pct_of(val: Any, base: Any) -> Optional[float]:
    """Return val/base as float percentage (0-1)."""
    try:
        if val is None or base is None:
            return None
        base_f = float(base)
        if base_f == 0.0:
            return None
        return float(val) / base_f
    except Exception:
        return None


def leg_pnl(entry: Optional[float], qty: Optional[float], mark: Optional[float]) -> Optional[float]:
    """Return leg PnL in quote terms."""
    if entry is None or qty is None or mark is None:
        return None
    return (float(mark) - float(entry)) * float(qty)


def fmt_pair(num: str, den: str | None, printer_friendly: bool = True) -> str:
    """Human-friendly pair string for display."""
    assert num
    if not den:
        return num
    if printer_friendly and num.endswith("USDT") and den.endswith("USDT"):
        return f"{num.removesuffix('USDT')}/{den.removesuffix('USDT')}"
    return f"{num}/{den}"


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


def float2str(
    x: float,
    *,
    max_total: int = 7,
    max_frac_lt1: int = 5,
    sig_small: int = 4,
) -> str:
    """
    Pragmatic float formatter:
      - if |x| >= 1: show all digits before dot, cap fractional digits so total digits ~ max_total
      - if 0.1 <= |x| < 1: up to max_frac_lt1 fractional digits
      - if |x| < 0.1: show sig_small significant digits after leading zeros


def str_exc(e: Exception) -> str:
    return f"{e.__class__.__name__}: {e}"
    """
    if x == 0 or not isinstance(x, (int, float)):
        return str(x)
    sign = "-" if x < 0 else ""
    ax = abs(x)
    if ax >= 1:
        s_int = f"{int(ax)}"
        frac_digits = max(0, max_total - len(s_int))
        fmt = f"{{:.{frac_digits}f}}" if frac_digits > 0 else "{:.0f}"
        return sign + fmt.format(ax)
    if ax >= 0.1:
        return sign + f"{ax:.{max_frac_lt1}f}".rstrip("0").rstrip(".")
    # ax < 0.1: compute leading zeros after decimal and keep sig_small significant digits
    leading_zeros = int(math.floor(-math.log10(ax))) if ax > 0 else 0
    frac_digits = max(0, leading_zeros + sig_small)
    return sign + f"{ax:.{frac_digits}f}".rstrip("0").rstrip(".")


def is_sane(value) -> bool:
    """
    Return True if value is not None/NaN/inf.
    Accepts numeric types; non-numeric raises.
    """
    if value is None:
        return False
    try:
        f = float(value)
    except Exception:
        raise
    return math.isfinite(f)


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
