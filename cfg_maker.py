"""
Common AppConfig builder to be shared between Telegram and console ui
"""

from zoneinfo import ZoneInfo
import os, sys

from bot_api import (
    AppConfig, log, Clock
)

def make_cfg():
    sys.path.append("/home/j/yp/saccakeys")
    from saccakeys import keys as _keys

    B = _keys.apikeys["binance"]
    T = _keys.apikeys["mettabot"]

    cfg = AppConfig(
        TZ_LOCAL=ZoneInfo("America/Fortaleza"),
        RECV_WINDOW_MS=60_000,
        DB_PATH=os.environ.get("RV_SQLITE", "./rv.sqlite"),
        LOG_LEVEL=os.environ.get("RV_LOG", "INFO"),
        PRICE_BACKFILL_WINDOW_SEC=5,
        PRICE_BACKFILL_MAX_SEC=120,
        MARK_POLL_SEC=10,
        INCOME_POLL_SEC=60,
        REPORT_USD_DIGITS=2,
        TELEGRAM_ENABLED=False,
        CONSOLE_ENABLED=False,
        TELEGRAM_TOKEN=T[0],
        TELEGRAM_CHAT_ID=T[1],
        BINANCE_KEY=B[0],
        BINANCE_SEC=B[1],

        # ---- NEW: klines cache ----
        KLINES_CACHE_DB_PATH=os.environ.get("RV_CACHE_SQLITE", "./rv_cache.sqlite"),
        KLINES_CACHE_KEEP_BARS=int(os.environ.get("RV_CACHE_KEEP_BARS", "2000")),
        THINKING_SEC=int(os.environ.get("RV_KLINES_POLL_SEC", "10")),
        KLINES_TIMEFRAMES=(os.environ.get("RV_KLINES_TFS", "1m").split(",")),
        KLINES_FETCH_LIMIT=int(os.environ.get("RV_KLINES_FETCH_LIMIT", "1000")),

        AUX_SYMBOLS=[],
    )

    return cfg
