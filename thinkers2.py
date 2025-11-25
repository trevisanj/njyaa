
#!/usr/bin/env python3
# FILE: thinkers1.py

from __future__ import annotations
import json, math, sqlite3, time, random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Iterable, Tuple, TYPE_CHECKING, Literal, Union

from bot_api import Storage, BinanceUM, MarketCatalog, PriceOracle
from commands import OCMarkDown
from common import log, Clock, AppConfig, leg_pnl, tf_ms, ts_human
from thinkers1 import ThinkerBase
import risk_report
from indicator_engines import StopStrategy, SSPSAR
from trailing_policies import evaluate_policy
import pandas as pd
import numpy as np
import enghelpers as eh
import engclasses as ec


if TYPE_CHECKING:
    from bot_api import BotEngine


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

    def on_tick(self, now_ms: int) -> Any:
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


def _mark_close_now(eng: BotEngine, symbol: str) -> Optional[float]:
    return eng.kc.last_cached_price(symbol)


def _minute_klines(eng: BotEngine, symbol: str, mins: int) -> Dict[str, List[Any]]:
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

    def on_tick(self, now_ms: int):
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


# ========= Trailing stop orchestrator =========


def _agg_stop(side: int, stops: List[Optional[float]]) -> Optional[float]:
    usable = [s for s in stops if s is not None]
    if not usable:
        return None
    return max(usable) if side > 0 else min(usable)


def _stop_improved(side: int, prev: Optional[float], new: Optional[float], min_move_bp: float) -> bool:
    if new is None:
        return False
    if prev is None:
        return True
    if side > 0:
        return (new - prev) * 10_000 / prev >= min_move_bp
    return (prev - new) * 10_000 / prev >= min_move_bp


class TrailingStopThinker(ThinkerBase):
    """
    Orchestrates per-position trailing policies fed by indicator steppers.
    Attachments + state live in this thinker's runtime, keyed by position id.
    """
    kind = "TRAILING_STOP"

    @dataclass
    class Config:
        timeframe: str = field(default="1d", metadata={"help": "Timeframe for indicator klines"})
        min_move_bp: float = field(default=1.0, metadata={"help": "Minimum bps improvement to log stop moves"})
        alert_cooldown_ms: int = field(default=60_000, metadata={"help": "Cooldown between repeated hit alerts"})
        sstrat: str = field(default="SSPSAR", metadata={"help": "Stop strategy kind"})

    def _on_init(self) -> None:
        self._runtime.setdefault("positions", {})

    def _pair_bars(self, pos: ec.Position, start_ts: int, end_ts: Optional[int] = None) -> pd.DataFrame:
        """Returns OHLCV data for configured timeframe"""
        return self.eng.kc.pair_bars(pos.num, pos.den, self._cfg["timeframe"], start_ts, end_ts)

    def on_tick(self, now_ms: int):
        # ---------- tick-level setup ----------

        # runtime keys
        PP_CTX = "pp_ctx"

        pp_ctx = self._runtime.get(PP_CTX, {})  # per-position runtime state map
        if not pp_ctx:
            # no positions attached
            return 0

        min_move_bp = float(self._cfg["min_move_bp"])  # min bps improvement to log
        tf = self._cfg["timeframe"]  # timeframe string
        thinker_id = self._thinker_id  # cached id for history rows
        dirty = False  # whether we must save runtime at end

        processed = 0  # counter of processed attachments
        for pid_str, p_ctx in pp_ctx.items():
            # ---------- fetch position + validate attachment ----------
            if p_ctx.get("invalid"):
                # position does not exist or is closed (see below)
                continue

            pos = self.eng.store.get_position(int(pid_str))  # Position object (or None)
            num_den = pos.get_pair()

            if not pos or pos.status != "OPEN":
                # mark invalid if missing/closed; skip
                p_ctx["invalid"] = True
                p_ctx["invalid_msg"] = "Closed" if pos else "Inexistent"
                pp_ctx[pid_str] = p_ctx
                dirty = True
                continue

            # ---------- build kline window ----------
            need_n = max(int(p_ctx.get("lookback_bars", 200)), 5)  # minimum bars required
            tfms = tf_ms(self._cfg["timeframe"])  # timeframe in ms
            anchor_ts = p_ctx.get("last_ts") or p_ctx.get("attached_at_ms")  # last processed or attach time
            start_ts = max(0, int(anchor_ts) - need_n * tfms)  # start window so we have enough bars
            bars = self._pair_bars(pos, start_ts, None)  # fetch num[/den] bars as dataframe
            if bars.empty or len(bars) < need_n:
                log().debug(f"[trail] Missing klines {tf}, can't calculate stop", num_den=num_den, position_id=pos.id)
                continue

            # ---------- run strategy ----------
            sstrat_rt = p_ctx.setdefault("sstrat", {})
            sstrat_kind = p_ctx.get("sstrat_kind", self._cfg.get("sstrat", "SSPSAR"))
            strat = StopStrategy.from_kind(sstrat_kind, self.eng, self, sstrat_rt, pos, sstrat_kind)
            strat.run(bars)
            stop_info = strat.on_get_stop_info()

            stop_series = stop_info.get("stop") if stop_info else None
            if stop_series is None or len(stop_series) == 0 or np.all(np.isnan(stop_series)):
                continue
            latest_stop = stop_series[-1]
            price = bars["Close"].iloc[-1]
            hit = price <= latest_stop if pos.dir_sign > 0 else price >= latest_stop

            prev_stop_val = stop_info.get("prev_stop_value") if stop_info else None
            if _stop_improved(pos.dir_sign, prev_stop_val, latest_stop, min_move_bp):
                self.notify("INFO", f"[trail] {num_den} stop -> {latest_stop:.4f} ({'LONG' if pos.dir_sign>0 else 'SHORT'})", send=True,
                            symbol=num_den, stop=latest_stop, prev=prev_stop_val, price=price)
            if hit:
                self.notify("WARN", f"[trail] {num_den} stop hit @ {price:.4f} vs {latest_stop:.4f}", send=True,
                            symbol=num_den, stop=latest_stop, price=price)

            p_ctx["last_ts"] = sstrat_rt.get("last_ts", int(bars.index[-1].value // 1_000_000))
            pp_ctx[pid_str] = p_ctx
            processed += 1
            dirty = True

        if dirty:
            self.save_runtime()  # persist runtime if touched
        return


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

    def on_tick(self, now_ms: int):
        if random.random() < self._cfg["prob"]:
            line = random.choice(self.LYRICS)

            self.notify(
                "WARN",
                f"HORSE SAYS: ((({line})))",
                send=True,
            )
            # self.eng._render_co(OCMarkDown(f"> {line}"))
