
#!/usr/bin/env python3
# FILE: thinkers1.py

from __future__ import annotations
import json, math, sqlite3, time, random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Iterable, Tuple, TYPE_CHECKING

from bot_api import Storage, BinanceUM, MarketCatalog, PriceOracle
from commands import OCMarkDown
from common import log, Clock, AppConfig, leg_pnl
from thinkers1 import ThinkerBase
import enghelpers as eh
import risk_report

if TYPE_CHECKING:
    from bot_api import BotEngine


# --------------------------------
# Thinker Implementations (examples)
# --------------------------------

class ThresholdAlertThinker(ThinkerBase):
    """
    Simple price threshold alert:
      config = {
        "symbol": "ETHUSDT",
        "direction": "ABOVE" | "BELOW",
        "price": 4000.0,
        "message": "ETH crosses 4k",
      }

    Runtime memory (managed by manager):
      { "fired": false, "last_price": 3981.2, "last_fire_ts": 0 }
    """
    kind = "THRESHOLD_ALERT"

    @dataclass
    class Config:
        symbol: str = field(default="BTCUSDT", metadata={"help": "Market symbol to watch"})
        direction: Literal["ABOVE", "BELOW"] = field(default="ABOVE", metadata={"help": "Trigger when price is ABOVE or BELOW threshold"})
        price: float = field(default=100_000.0, metadata={"help": "Trigger threshold price"})
        message: str = field(default="", metadata={"help": "Optional custom alert message"})

    def _on_init(self) -> None:
        d = self._cfg["direction"].upper()
        if d not in ("ABOVE", "BELOW"):
            raise ValueError("direction must be ABOVE or BELOW")
        self._cfg["direction"] = d
        self._cfg["price"] = float(self._cfg["price"])

    def tick(self, now_ms: int) -> Any:
        sym = self._cfg["symbol"]
        pr = _mark_close_now(self.eng, sym)
        if pr is None:
            return

        msg = self._cfg.get("message") or f"{sym} {self._cfg['direction']} {self._cfg['price']}"
        thr = self._cfg["price"]; dir_ = self._cfg["direction"]

        hit = (pr >= thr) if dir_ == "ABOVE" else (pr <= thr)
        if not hit:
            self.notify("DEBUG", f"[thr] {sym}={pr:.4f} vs {dir_} {thr}",
                        send=True, symbol=sym, direction=dir_, price=pr)
            return

        self.notify("INFO", f"{msg} | last={pr:.4f}", send=True,
                    symbol=sym, direction=dir_, price=pr, threshold=thr)
        return 1


class PSARStopThinker(ThinkerBase):
    """
    Minimal Parabolic SAR trailing stop for a specific position.

    config = {
      "position_id": "<hex>",
      "symbol": "ETHUSDT",
      "direction": "LONG" | "SHORT",   # relative to stop logic
      "af": 0.02,
      "max_af": 0.2,
      "window_min": 200   # how many last 1m bars to pull (for smoothing)
    }

    Runtime we persist (manager saves it for us):
      {
        "psar": float,
        "ep": float,
        "af": float,
        "trend": "UP"|"DOWN",
        "last_candle_open": int,
      "last_alert_ts": int
      }

    Fires ALERT when price crosses PSAR against trend.
    """
    kind = "PSAR_STOP"
    @dataclass
    class Config:
        position_id: str = field(default="0", metadata={"help": "Target position id (string/hex)"})
        symbol: str = field(default="BTCUSDT", metadata={"help": "Symbol to monitor for PSAR stops"})
        direction: Literal["LONG", "SHORT"] = field(default="LONG", metadata={"help": "LONG triggers on close < psar; SHORT triggers on close > psar"})
        af: float = field(default=0.02, metadata={"help": "Acceleration factor step"})
        max_af: float = field(default=0.2, metadata={"help": "Maximum acceleration factor"})
        window_min: int = field(default=200, metadata={"help": "Number of 1m bars to pull for PSAR"})

    def _on_init(self) -> None:
        self._cfg["af"] = float(self._cfg["af"])
        self._cfg["max_af"] = float(self._cfg["max_af"])
        self._cfg["window_min"] = int(self._cfg["window_min"])
        d = self._cfg["direction"].upper()
        if d not in ("LONG", "SHORT"):
            raise ValueError("direction must be LONG or SHORT")
        self._cfg["direction"] = d

    # classic PSAR step on OHLC stream
    @staticmethod
    def _psar_series(ohlc: List[Tuple[float,float,float,float]], af0: float, afmax: float):
        """
        Return last (psar, ep, af, trend) after iterating series.

        ohlc: list of (o,h,l,c)
        """
        if len(ohlc) < 5:
            raise ValueError("Need at least 5 bars for PSAR bootstrap")

        # Bootstrap from first bars
        h0 = max(x[1] for x in ohlc[:5]); l0 = min(x[2] for x in ohlc[:5])
        up = True  # start bullish by default; will flip quickly if wrong
        ep = h0 if up else l0
        af = af0
        psar = l0 if up else h0

        for i in range(5, len(ohlc)):
            o,h,l,c = ohlc[i]
            prev_psar = psar
            prev_ep = ep
            prev_up = up

            # move PSAR
            psar = prev_psar + af * (prev_ep - prev_psar)

            # clamp against last/prev extremities
            if up:
                psar = min(psar, min(ohlc[i-1][2], ohlc[i-2][2]))  # clamp to last 2 lows
            else:
                psar = max(psar, max(ohlc[i-1][1], ohlc[i-2][1]))  # clamp to last 2 highs

            # trend check
            if up:
                if l < psar:  # flip to downtrend
                    up = False
                    psar = prev_ep
                    ep = l
                    af = af0
                else:
                    if h > prev_ep:
                        ep = h
                        af = min(af + af0, afmax)
                    else:
                        ep = prev_ep
            else:
                if h > psar:  # flip to uptrend
                    up = True
                    psar = prev_ep
                    ep = h
                    af = af0
                else:
                    if l < prev_ep:
                        ep = l
                        af = min(af + af0, afmax)
                    else:
                        ep = prev_ep

        return psar, ep, af, ("UP" if up else "DOWN")

    def tick(self, now_ms: int):
        sym = self._cfg["symbol"]
        mins = self._cfg["window_min"]
        rows = _minute_klines(self.eng, sym, mins)
        if not rows:
            return 0

        # Build OHLC tuples (using mark price klines)
        # rows: [open_time, open, high, low, close, volume, close_time, ...]
        ohlc = [(float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rows]
        psar, ep, af, trend = self._psar_series(ohlc, self._cfg["af"], self._cfg["max_af"])
        last_close = float(rows[-1][4])

        # Direction semantics:
        # - LONG: stop triggers if close < psar
        # - SHORT: stop triggers if close > psar
        hit = (last_close < psar) if self._cfg["direction"] == "LONG" else (last_close > psar)

        if hit:
            self.notify("WARN", f"[PSAR] {sym} {self._cfg['direction']} stop hit: close={last_close:.4f} vs psar={psar:.4f}",
                        symbol=sym, psar=psar, ep=ep, af=af, trend=trend, last_close=last_close,
                        direction=self._cfg["direction"])
        else:
            self.notify("DEBUG", f"[PSAR] {sym} {self._cfg['direction']} ok: close={last_close:.4f} psar={psar:.4f} trend={trend}",
                        symbol=sym, psar=psar, ep=ep, af=af, trend=trend, last_close=last_close,
                        direction=self._cfg["direction"])
        return 1


def _mark_close_now(eng: BotEngine, symbol: str) -> Optional[float]:
    return eh.last_cached_price(eng, symbol)


def _minute_klines(eng: BotEngine, symbol: str, mins: int) -> List:
    return eng.kc.last_n(symbol, "1m", mins, include_live=True, asc=True)


class RiskThinker(ThinkerBase):
    """
    Monitors exposure vs leverage and per-position drawdowns vs risk budget.

    config = {
      "warn_exposure_ratio": 1.0,    # alert if exposure > ratio * max_exposure
      "warn_loss_mult": 1.0,         # alert if pnl < -loss_mult * risk_budget
      "min_alert_interval_ms": 300000
    }
    """
    kind = "RISK_MONITOR"

    def __init__(self, tm: "ThinkerManager", eng: BotEngine):
        self._last_alert_ms: int = 0
        super().__init__(tm, eng)
        self._last_alert_ms: int = 0

    @dataclass
    class Config:
        warn_exposure_ratio: float = field(default=1.0, metadata={"help": "Alert when exposure exceeds ratio * max_exposure "})
        warn_loss_mult: float = field(default=1.0, metadata={"help": "Alert when PnL <= -mult * risk_budget"})
        min_alert_interval_ms: int = field(default=300_000, metadata={"help": "Cooldown between alerts (ms)"})

    def _on_init(self) -> None:
        pass

    def tick(self, now_ms: int):
        thresholds = risk_report.RiskThresholds(
            warn_exposure_ratio=self._cfg["warn_exposure_ratio"],
            warn_loss_mult=self._cfg["warn_loss_mult"],
        )
        report = risk_report.build_risk_report(self.eng, thresholds)

        if not report.alerts:
            return

        if now_ms - self._last_alert_ms < self._cfg["min_alert_interval_ms"]:
            self.notify(
                "DEBUG",
                "[risk] alerts suppressed (cooldown)",
                alerts=[a.message for a in report.alerts],
                cooldown_ms=self._cfg["min_alert_interval_ms"],
            )
            return None

        self._last_alert_ms = now_ms
        self.notify(
            "WARN",
            "[risk] " + "; ".join(a.message for a in report.alerts),
            send=True,
            exposure=report.total_exposure,
            max_exposure=report.max_exposure,
            ref_balance=report.ref_balance,
            leverage=report.leverage,
            # positions=[p.__dict__ for p in report.positions],
        )
        return None

class HorseWithNoName(ThinkerBase):
    """Proof-of-concept thinker"""
    kind = "HORSE_WITH_NO_NAME"

    @dataclass
    class Config:
        prob: float = field(default=0.1, metadata={"help": "Probability of sending a lyric each tick (0-1)"})

    LYRICS = [
"On the first part of the journey",
"I was looking at all the life",
"There were plants and birds and rocks and things",
"There was sand and hills and rings",
"The first thing I met was a fly with a buzz",
"And the sky with no clouds",
"The heat was hot, and the ground was dry",
"But the air was full of sound",
"I've been through the desert",
"On a horse with no name",
"It felt good to be out of the rain",
"In the desert, you can remember your name",
"'Cause there ain't no one for to give you no pain",
"After two days in the desert sun",
"My skin began to turn red",
"After three days in the desert fun",
"I was looking at a river bed",
"And the story it told of a river that flowed",
"Made me sad to think it was dead",
"After nine days, I let the horse run free",
"'Cause the desert had turned to sea",
"There were plants and birds and rocks and things",
"There was sand and hills and rings",
"The ocean is a desert with it's life underground",
"And a perfect disguise above",
"Under the cities lies a heart made of ground",
"But the humans will give no love",
    ]

    def _on_init(self):
        self._cfg["prob"] = float(self._cfg["prob"])

    def tick(self, now_ms: int):
        if random.random() < self._cfg["prob"]:
            line = random.choice(self.LYRICS)

            self.notify(
                "WARN",
                f"HORSE SAYS: ((({line})))",
                send=True,
            )
            # self.eng._render_co(OCMarkDown(f"> {line}"))
