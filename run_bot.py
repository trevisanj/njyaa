#!/usr/bin/env python3
# FILE: run_bot_bk02.py

import signal
from zoneinfo import ZoneInfo
import os, sys

from bot_api import (
    AppConfig, log,
    BinanceUM, Storage, MarketCatalog, PriceOracle, PositionBook,
    AlertsEngine, Reporter, TelegramBot, Worker,
    BotEngine, build_registry, Clock
)

def main():
    # run_bot.py (main)

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
        TELEGRAM_ENABLED=True,
        TELEGRAM_TOKEN=T[0],
        TELEGRAM_CHAT_ID=T[1],
        BINANCE_KEY=B[0],
        BINANCE_SEC=B[1],

        # ---- NEW: klines cache ----
        KLINES_CACHE_DB_PATH=os.environ.get("RV_CACHE_SQLITE", "./rv_cache.sqlite"),
        KLINES_CACHE_KEEP_BARS=int(os.environ.get("RV_CACHE_KEEP_BARS", "2000")),
        KLINES_POLL_SEC=int(os.environ.get("RV_KLINES_POLL_SEC", "10")),
        KLINES_TIMEFRAMES=(os.environ.get("RV_KLINES_TFS", "1m").split(",")),
        KLINES_FETCH_LIMIT=int(os.environ.get("RV_KLINES_FETCH_LIMIT", "1000")),
    )
    Clock.set_tz(cfg.TZ_LOCAL)
    log.info("Clock TZ set", tz=str(cfg.TZ_LOCAL))

    # 2) Core services
    api = BinanceUM(cfg)
    store = Storage(path=cfg.DB_PATH)
    mc = MarketCatalog(api); mc.load()
    oracle = PriceOracle(cfg, api)
    positionbook = PositionBook(store, mc, oracle)
    alerts = AlertsEngine(store, positionbook)
    reporter = Reporter()
    tgbot = TelegramBot(positionbook)  # or
    worker = Worker(cfg, store, api, mc, oracle)

    # 3) Command registry (single place to add/change commands)
    registry = build_registry()

    # 4) Engine wiring
    engine = BotEngine(cfg, api, store, mc, oracle, positionbook, alerts, reporter, tgbot, worker)
    engine.set_registry(registry)
    tgbot.set_registry(registry)

    # 5) Graceful shutdown hooks
    def _graceful_exit(signum, frame):
        log.info("Signal received, stopping...", sign=signum)
        try:
            engine.stop()
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    # 6) Run (blocks)
    engine.start()

if __name__ == "__main__":
    main()
