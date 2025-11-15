
#!/usr/bin/env python3
# FILE: thinkers1.py

from __future__ import annotations
import json, math, sqlite3, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Iterable, Tuple

from bot_api import Storage, BinanceUM, MarketCatalog, PriceOracle
from common import log, Clock, AppConfig
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from thinkers1 import ThinkerBase, ThinkerAction, ThinkerContext

# -----------------------------
# Thinker Registry/Factory
# -----------------------------

import inspect, sys


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

    def __init__(self):
        self._cfg: Dict[str, Any] = {}

    def init(self, config: Dict[str, Any]) -> None:
        self._cfg = dict(config or {})
        mandatory = ("symbol", "direction", "price")
        for k in mandatory:
            if k not in self._cfg:
                raise ValueError(f"ThresholdAlertThinker missing '{k}'")

        d = self._cfg["direction"].upper()
        if d not in ("ABOVE", "BELOW"):
            raise ValueError("direction must be ABOVE or BELOW")
        self._cfg["direction"] = d
        self._cfg["price"] = float(self._cfg["price"])

    def tick(self, ctx: ThinkerContext, now_ms: int) -> List[ThinkerAction]:
        ctx.log.info("ASDKASJUKHDASJKSDHFAJKHFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
        sym = self._cfg["symbol"]
        pr = ctx.mark_close_now(sym)
        if pr is None:
            return []

        msg = self._cfg.get("message") or f"{sym} {self._cfg['direction']} {self._cfg['price']}"
        thr = self._cfg["price"]; dir_ = self._cfg["direction"]

        hit = (pr >= thr) if dir_ == "ABOVE" else (pr <= thr)
        if not hit:
            return [ThinkerAction("LOG", "DEBUG", f"[thr] {sym}={pr:.4f} vs {dir_} {thr}", {"symbol": sym, "price": pr})]

        return [ThinkerAction(
            type="ALERT",
            level="INFO",
            text=f"{msg} | last={pr:.4f}",
            payload={"symbol": sym, "price": pr, "threshold": thr, "direction": dir_}
        )]


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

    def __init__(self):
        self._cfg: Dict[str, Any] = {}

    def init(self, config: Dict[str, Any]) -> None:
        self._cfg = dict(config or {})
        for k in ("position_id", "symbol", "direction"):
            if k not in self._cfg:
                raise ValueError(f"PSAR missing '{k}'")
        self._cfg["af"] = float(self._cfg.get("af", 0.02))
        self._cfg["max_af"] = float(self._cfg.get("max_af", 0.2))
        self._cfg["window_min"] = int(self._cfg.get("window_min", 200))
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

    def tick(self, ctx: ThinkerContext, now_ms: int) -> List[ThinkerAction]:
        sym = self._cfg["symbol"]
        mins = self._cfg["window_min"]
        rows = ctx.minute_klines(sym, mins)
        if not rows:
            return []

        # Build OHLC tuples (using mark price klines)
        # rows: [open_time, open, high, low, close, volume, close_time, ...]
        ohlc = [(float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rows]
        psar, ep, af, trend = self._psar_series(ohlc, self._cfg["af"], self._cfg["max_af"])
        last_close = float(rows[-1][4])

        # Direction semantics:
        # - LONG: stop triggers if close < psar
        # - SHORT: stop triggers if close > psar
        hit = (last_close < psar) if self._cfg["direction"] == "LONG" else (last_close > psar)

        payload = {
            "symbol": sym, "psar": psar, "ep": ep, "af": af, "trend": trend,
            "last_close": last_close, "direction": self._cfg["direction"]
        }

        if hit:
            return [ThinkerAction(
                type="ALERT",
                level="WARN",
                text=f"[PSAR] {sym} {self._cfg['direction']} stop hit: close={last_close:.4f} vs psar={psar:.4f}",
                payload=payload
            )]

        return [ThinkerAction(
            type="LOG",
            level="DEBUG",
            text=f"[PSAR] {sym} {self._cfg['direction']} ok: close={last_close:.4f} psar={psar:.4f} trend={trend}",
            payload=payload
        )]


