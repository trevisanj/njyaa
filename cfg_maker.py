"""
Common AppConfig builder to be shared between Telegram and console ui
"""

from zoneinfo import ZoneInfo
import os, sys

from bot_api import (
    AppConfig, log, Clock
)

def _get_env(varname, default, show=True):
    env_varname = f"NJYAA_{varname}"
    value = os.environ.get(env_varname)
    if value:
        show_value = value if show else "..."
        log().info(f"njyaa: Read ({env_varname}: '{show_value!r}') --> cfg.{varname}")
        return value
    return default

def _get_env_bool(varname, default) -> bool:
    """Like _get_env() but tries to parse value into a bool"""

    value = _get_env(varname, None)

    if value:
        v = value.strip().lower()

        truthy = {"1", "true", "t", "yes", "y"}
        falsy = {"0", "false", "f", "no", "n"}

        if v in truthy:
            return True
        if v in falsy:
            return False

        raise ValueError(f"Invalid boolean env value: '{value!r}")

    assert isinstance(default, bool)
    return default

def make_cfg():
    sys.path.append("/home/j/yp/saccakeys")

    cfg = AppConfig(
        TZ_LOCAL=ZoneInfo("America/Fortaleza"),
        RECV_WINDOW_MS=60_000,
        DB_PATH="njyaa_db.sqlite",
        LOG_LEVEL=_get_env("LOG_LEVEL", "INFO"),
        PRICE_BACKFILL_WINDOW_SEC=5,
        PRICE_BACKFILL_MAX_SEC=120,
        MARK_POLL_SEC=10,
        INCOME_POLL_SEC=60,
        REPORT_USD_DIGITS=2,
        TELEGRAM_ENABLED=_get_env_bool("TELEGRAM_ENABLED", False),
        CONSOLE_ENABLED=_get_env_bool("CONSOLE_ENABLED", False),
        TELEGRAM_CHAT_ID=_get_env("TELEGRAM_CHAT_ID", None, show=False),
        TELEGRAM_TOKEN=_get_env("TELEGRAM_TOKEN", None, show=False),
        BINANCE_KEY=_get_env("BINANCE_KEY", None, show=False),
        BINANCE_SEC=_get_env("BINANCE_SEC", None, show=False),

        # ---- klines cache ----
        KLINES_CACHE_DB_PATH="./njyaa_cache.sqlite",
        INDICATOR_HISTORY_DB_PATH="./njyaa_indicator.sqlite",
        KLINES_CACHE_KEEP_BARS=2000,
        KLINES_TIMEFRAMES=["1m", "1h", "1d"],
        KLINES_FETCH_LIMIT=1000,

        THINKING_SEC=10,

        AUX_SYMBOLS=[],
    )

    if all([cfg.TELEGRAM_CHAT_ID, cfg.TELEGRAM_TOKEN, cfg.BINANCE_KEY, cfg.BINANCE_SEC]):
        pass
    else:
        sys.path.insert(0, "/home/j/yp/saccakeys")

        from saccakeys import keys as _keys

        B = _keys.apikeys["binance"]
        T = _keys.apikeys["mettabot"]

        if not cfg.TELEGRAM_CHAT_ID: cfg.TELEGRAM_CHAT_ID = T[1]
        if not cfg.TELEGRAM_TOKEN: cfg.TELEGRAM_TOKEN = T[0]
        if not cfg.BINANCE_KEY: cfg.BINANCE_KEY = B[0]
        if not cfg.BINANCE_SEC: cfg.BINANCE_SEC = B[1]

    return cfg
