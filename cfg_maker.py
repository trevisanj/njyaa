"""
Common AppConfig builder to be shared between Telegram and console ui
"""

from zoneinfo import ZoneInfo
import os, sys

from bot_api import (
    AppConfig, log, Clock
)

def _get_env(varname, default):
    env_varname = f"NJYAA_{varname}"
    value = os.environ.get(varname)
    if value:
        log().info(f"Read ({env_varname}: '{value!r}') --> cfg.{varname}")
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
    from saccakeys import keys as _keys

    B = _keys.apikeys["binance"]
    T = _keys.apikeys["mettabot"]

    cfg = AppConfig(
        TZ_LOCAL=ZoneInfo("America/Fortaleza"),
        RECV_WINDOW_MS=60_000,
        DB_PATH=_get_env("DB_PATH", "njyaa_db.sqlite"),
        LOG_LEVEL=_get_env("LOG_LEVEL", "INFO"),
        PRICE_BACKFILL_WINDOW_SEC=5,
        PRICE_BACKFILL_MAX_SEC=120,
        MARK_POLL_SEC=10,
        INCOME_POLL_SEC=60,
        REPORT_USD_DIGITS=2,
        TELEGRAM_ENABLED=_get_env_bool("TELEGRAM_ENABLED", False),
        CONSOLE_ENABLED=_get_env_bool("CONSOLE_ENABLED", True),
        TELEGRAM_TOKEN=T[0],
        TELEGRAM_CHAT_ID=T[1],
        BINANCE_KEY=B[0],
        BINANCE_SEC=B[1],

        # ---- NEW: klines cache ----
        KLINES_CACHE_DB_PATH=_get_env("KLINES_CACHE_DB_PATH", "./njyaa_cache.sqlite"),
        KLINES_CACHE_KEEP_BARS=int(_get_env("KLINES_CACHE_KEEP_BARS", "2000")),
        THINKING_SEC=int(_get_env("KLINES_POLL_SEC", "10")),
        KLINES_TIMEFRAMES=["1m", "1h", "1d"],
        KLINES_FETCH_LIMIT=int(_get_env("KLINES_FETCH_LIMIT", "1000")),

        AUX_SYMBOLS=[],
    )

    return cfg
