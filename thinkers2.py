
#!/usr/bin/env python3
# FILE: thinkers1.py

from __future__ import annotations
import json, math, sqlite3, time, random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Iterable, Tuple, TYPE_CHECKING, Literal, Union

from bot_api import Storage, BinanceUM, MarketCatalog, PriceOracle
from commands import OCMarkDown
from common import (log, Clock, AppConfig, leg_pnl, tf_ms, ts_human, PP_CTX, SSTRAT_CTX, SSTRAT_KIND, ATTACHED_AT,
                    LAST_TS, WINDOW_SIZE, float2str, TooFewDataPoints, THOUGHT, NOW_MS, LAST_MOVE_ALERT_TS,
                    LAST_HIT_ALERT_TS)
from thinkers1 import ThinkerBase
import risk_report
from sstrats import StopStrategy, SSPSAR
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
    return eng.kc.last_n(symbol, "1m", mins, include_live=True)


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



# TODO garbage in runtime, see below:
#  {
#    "pp_ctx": {
#      "4": {
#        "attached_at": 1764159582999,
#        "cfg": {},                     <------------------------- GARBAGE
#        "last_alert_ts": 1764246160301,
#        "sstrat_ctx": {
#          "cfg": {
#            "fraction": 0.01
#          },
#          "ind_states": {
#            "1764241200000": {
#              "price_fraction": {
#                "last_value": 1.5440524497363975e-06
#              },
#              "stopper": {
#                "stop": 1.5440524497363975e-06
#              }
#            }
#          },
#          "last_ts": 1764244800000,
#          "state_ts": -1,            <---------------------------- GARBAGE
#          "window_size": 1
#        },
#        "sstrat_kind": "SSFRACTION"
#      }
#    }
#  }
class TrailingStopThinker(ThinkerBase):
    """
    Orchestrates per-position trailing policies fed by indicator steppers.
    Attachments + state live in this thinker's runtime, keyed by position id.
    ```
    """
    kind = "TRAILING_STOP"

    @dataclass
    class Config:
        timeframe: str = field(default="1d", metadata={"help": "Timeframe for indicator klines"})
        min_move_bp: float = field(default=1.0, metadata={"help": "Minimum bps improvement to log stop moves"})
        alert_cooldown_ms: int = field(default=60_000, metadata={"help": "Cooldown between repeated hit alerts"})
        sstrat_kind: str = field(default="SSPSAR", metadata={"help": "Stop strategy kind"})

    def get_timeframe(self):
        # won't make this a property so that it doesn't suppress errors
        return self._cfg["timeframe"]

    def _on_init(self) -> None:
        self._runtime.setdefault(PP_CTX, {})
        self._strats: Dict[str, StopStrategy] = {}

    def _reset(self, targets):
        """Reset strategy/runtime for selected positions or all."""

        pp_ctx = self._runtime.setdefault(PP_CTX, {})
        if targets == "all" or "all" in targets:
            target_keys = list(pp_ctx.keys())
        else:
            target_keys = [str(pid) for pid in targets if str(pid) in pp_ctx]
        for pid_str in target_keys:
            p_ctx = pp_ctx.get(pid_str) or {}
            p_ctx.pop(SSTRAT_CTX, None)
            log().debug("trail.reset", pid=pid_str)
            # TODO maybe this reset is not deleting the indicator history as it should, i am seeing ghosts, eventually
            self.eng.ih.delete_by_thinker_position(self._thinker_id, int(pid_str))

    def on_tick(self, now_ms: int):
        # ---------- tick-level setup ----------

        resets = self._runtime.pop("reset", None)
        if resets:
            self._reset(resets)
            self.save_runtime()

        pp_ctx = self._runtime.setdefault(PP_CTX, {})
        if not pp_ctx:
            return 0
        log().debug("trail.tick.start", attachments=len(pp_ctx))

        min_move_bp = float(self._cfg["min_move_bp"])  # min bps improvement to log
        dirty = False  # whether we must save runtime at end

        processed = 0  # counter of processed attachments
        for pid_str, p_ctx in pp_ctx.items():
            # TODO runtime is being saved only once for all positions, so if one fucks you, others lose new state

            # ---------- fetch position + validate attachment ----------
            if p_ctx.get("invalid"):
                # position does not exist or is closed (see below)
                continue

            pos = self.eng.store.get_position(int(pid_str))  # Position object (or None)
            num_den = pos.get_pair()

            def stamp():
                # Produces a "stamp" of the reporting context
                return f"t#{self._thinker_id} p#{pid_str} {num_den} sstrat={sstrat.name}"

            log().debug("trail.tick.pos", pid=pid_str, num_den=num_den)

            if not pos or pos.status != "OPEN":
                # mark invalid if missing/closed; skip
                p_ctx["invalid"] = True
                p_ctx["invalid_msg"] = "Closed" if pos else "Inexistent"
                pp_ctx[pid_str] = p_ctx
                self._strats.pop(pid_str, None)
                dirty = True
                continue

            # ------------ retrieve/initialize strategy -------------
            sstrat_ctx = p_ctx.setdefault(SSTRAT_CTX, {})
            sstrat_kind = p_ctx.get(SSTRAT_KIND, self._cfg["sstrat_kind"])
            sstrat = self._strats.get(pid_str)
            if sstrat is None:
                # TODO: sstrat needs a unique name, i guess
                sstrat_name = sstrat_kind
                sstrat = StopStrategy.from_kind(sstrat_kind, self.eng, self, sstrat_ctx, pos, sstrat_name,
                                                p_ctx[ATTACHED_AT])
                self._strats[pid_str] = sstrat
            elif sstrat.kind != sstrat_kind:
                raise RuntimeError(f"sstrat is a {sstrat.kind} but should be {sstrat_kind}")
            else:
                sstrat.ctx = sstrat_ctx
                sstrat.pos = pos

            # ---------- run strategy ----------
            log().debug("trail.tick.run_sstrat", stamp=stamp())
            try:
                sstrat.run()
                dirty = True
            except TooFewDataPoints as e:
                log().exc(e, stamp=stamp())
                continue

            # ----------- stop interpretation section -----------
            thought = p_ctx.setdefault(THOUGHT, {})
            last_move_alert_ts = thought.get(LAST_MOVE_ALERT_TS, 0)
            last_hit_alert_ts = thought.get(LAST_HIT_ALERT_TS, 0)

            stop_info = sstrat.get_stop_info()
            stop_series = stop_info["value"]
            flag_series = stop_info["flag"]

            price = sstrat.last_bars["Close"].iloc[-1]
            hit = bool(flag_series[-1] == 1)
            latest_stop = stop_series[-1]
            prev_stop_val = stop_series[-2] if len(stop_series) >= 2 else None

            now = now_ms
            cooldown_ms = int(self._cfg["alert_cooldown_ms"])

            if _stop_improved(pos.dir_sign, prev_stop_val, latest_stop, min_move_bp):
                msg = f"🟢 [trail] {stamp()} stop -> {float2str(latest_stop)} ({'LONG' if pos.dir_sign>0 else 'SHORT'})"
                self.notify("INFO", msg, send=True,
                            num_den=num_den, stop=latest_stop, prev=prev_stop_val, price=price)
                last_move_alert_ts = now
            if hit and (now - last_hit_alert_ts) >= cooldown_ms:
                self.notify("WARN", f"🛑 [trail] {stamp()} stop hit @ {float2str(price)} vs {float2str(latest_stop)}",
                            send=True, num_den=num_den, stop=latest_stop, price=price)
                last_hit_alert_ts = now
            p_ctx["last_alert_ts"] = max(last_move_alert_ts, last_hit_alert_ts)
            thought.update({
                LAST_MOVE_ALERT_TS: last_move_alert_ts,
                LAST_HIT_ALERT_TS: last_hit_alert_ts,
            })
            log().debug("trail.tick.done", pid=pid_str, stop=latest_stop, prev=prev_stop_val, hit=hit)

            pp_ctx[pid_str] = p_ctx
            processed += 1
            dirty = True

        if dirty:
            # self._runtime[PP_CTX] = pp_ctx
            self.save_runtime()  # persist runtime if touched
            log().info("trail.tick.saved_runtime", thinker_id=self._thinker_id)
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
