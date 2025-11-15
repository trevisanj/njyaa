# contracts.py
from typing import Protocol

# forward declarations (no circular imports)
if False:  # type-checkers only
    from storage import Storage
    from binance_um import BinanceUM
    from bot_api import MarketCatalog, PriceOracle, PositionBook, AlertsEngine, Reporter
    from common import AppConfig
    from klines_cache import KlinesCache

class EngineServices(Protocol):
    # --- core services ---
    cfg: "AppConfig"
    store: "Storage"
    api: "BinanceUM"
    mc: "MarketCatalog"
    oracle: "PriceOracle"
    positionbook: "PositionBook"
    alerts: "AlertsEngine"
    reporter: "Reporter"
    kc: "KlinesCache"


    def emit_alert(self, msg: str) -> None: ...
    # add more as you truly need them later:
    # def send_image(self, path: str, caption: str | None = None) -> None: ...
    # def schedule_job(self, name: str, delay_sec: int, payload: dict) -> None: ...
