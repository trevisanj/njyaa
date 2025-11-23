
#!/usr/bin/env python3
# FILE: thinkers1.py

from __future__ import annotations
import json, math, sqlite3, time, random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Iterable, Tuple, TYPE_CHECKING, Literal

from bot_api import Storage, BinanceUM, MarketCatalog, PriceOracle
from commands import OCMarkDown
from common import log, Clock, AppConfig, leg_pnl
from thinkers1 import ThinkerBase
import enghelpers as eh
import risk_report
from indicator_engines import step_psar, step_atr, PSARState, ATRState
from trailing_policies import evaluate_policy

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


# ========= Trailing stop orchestrator =========


def _bars_from_rows(rows: List) -> List[tuple[int, float, float, float, float]]:
    return [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rows]


def _agg_stop(side: str, stops: List[Optional[float]]) -> Optional[float]:
    usable = [s for s in stops if s is not None]
    if not usable:
        return None
    return max(usable) if side == "LONG" else min(usable)


def _stop_improved(side: str, prev: Optional[float], new: Optional[float], min_move_bp: float) -> bool:
    if new is None:
        return False
    if prev is None:
        return True
    if side == "LONG":
        return (new - prev) * 10_000 / prev >= min_move_bp
    return (prev - new) * 10_000 / prev >= min_move_bp


class TrailingStopThinker(ThinkerBase):
    """
    Orchestrates per-position trailing policies fed by indicator steppers.
    Attachments live in storage.exit_attachments.
    """
    kind = "TRAILING_STOP"

    @dataclass
    class Config:
        timeframe: str = field(default="1m", metadata={"help": "Timeframe for indicator klines"})
        min_move_bp: float = field(default=1.0, metadata={"help": "Minimum bps improvement to log stop moves"})
        alert_cooldown_ms: int = field(default=60_000, metadata={"help": "Cooldown between repeated hit alerts"})

    def _on_init(self) -> None:
        pass

    def _fetch_bars(self, symbol: str, n: int) -> List[tuple[int, float, float, float, float]]:
        rows = self.eng.kc.last_n(symbol, self._cfg["timeframe"], n=n, include_live=True, asc=True)
        return _bars_from_rows(rows) if rows else []

    def _indicator_configs(self, policies: List[dict]) -> Dict[str, dict]:
        cfgs: Dict[str, dict] = {}
        for p in policies:
            ind = p.get("indicator") or p.get("name") or p.get("policy") or "psar"
            if ind not in cfgs:
                cfgs[ind] = dict(p.get("indicator_cfg") or {})
        return cfgs

    def _run_indicator(self, ind: str, bars: List[tuple[int, float, float, float, float]], cfg: dict,
                       state: Optional[dict]):
        if ind == "psar":
            st = PSARState(**state) if state else None
            ns, traces = step_psar(bars, st, cfg.get("af", 0.02), cfg.get("max_af", 0.2))
            return ns.__dict__, traces, traces[-1]["psar"] if traces else (state["psar"] if state else None)
        if ind == "atr":
            st = ATRState(**state) if state else None
            ns, traces = step_atr(bars, st, int(cfg.get("period", 14)))
            return ns.__dict__, traces, traces[-1]["atr"] if traces else (state["atr"] if state else None)
        raise ValueError(f"Unknown indicator: {ind}")

    def _indicator_history_rows(self, tid: int, pid: int, name: str, traces: List[dict]) -> List[dict]:
        rows = []
        for t in traces:
            v = t.get("psar") if "psar" in t else t.get("atr")
            rows.append({
                "thinker_id": tid,
                "position_id": pid,
                "name": name,
                "ts_ms": t["ts_ms"],
                "value": v,
                "price": t.get("c"),
                "aux": {k: t[k] for k in ("ep", "af", "trend", "tr") if k in t},
            })
        return rows

    def _position_snapshot(self, pos_row, price: float) -> Dict[str, Any]:
        side = "LONG" if int(pos_row["dir_sign"]) > 0 else "SHORT"
        return {
            "side": side,
            "last_price": price,
            "position_id": int(pos_row["position_id"]),
            "num": pos_row["num"],
            "den": pos_row["den"],
        }

    def tick(self, now_ms: int):
        attachments = self.eng.store.list_exit_attachments()
        if not attachments:
            return 0
        processed = 0
        min_move_bp = float(self._cfg["min_move_bp"])
        tf = self._cfg["timeframe"]
        tid = int(self._thinker_id or 0)

        for att in attachments:
            pid = int(att["position_id"])
            pos = self.eng.store.get_position(pid)
            if not pos or pos["status"] != "OPEN":
                continue
            symbol = pos["num"]
            price = eh.last_cached_price(self.eng, symbol)
            if price is None:
                continue

            policies = att["policies"]
            ind_cfgs = self._indicator_configs(policies)

            bars = self._fetch_bars(symbol, max(att["lookback_bars"], 5))
            if len(bars) < 5:
                self.notify("DEBUG", f"[trail] {symbol} missing klines {tf}", symbol=symbol)
                continue

            indicators: Dict[str, Dict[str, Any]] = {}
            history_rows: List[dict] = []
            for ind_name, cfg in ind_cfgs.items():
                st_row = self.eng.store.get_indicator_state(tid, pid, ind_name)
                st_payload = st_row["state"] if st_row else None
                try:
                    ns, traces, latest_val = self._run_indicator(ind_name, bars, cfg, st_payload)
                except Exception as e:
                    self.notify("WARN", f"[trail] {symbol} indicator {ind_name} failed: {e}", symbol=symbol)
                    continue
                self.eng.store.upsert_indicator_state(tid, pid, ind_name, ns, traces[-1]["ts_ms"] if traces else now_ms)
                history_rows.extend(self._indicator_history_rows(tid, pid, ind_name, traces))
                indicators[ind_name] = {"value": latest_val, "ts_ms": traces[-1]["ts_ms"] if traces else st_row["ts_ms"], "raw": ns}

            if history_rows:
                self.eng.store.insert_indicator_history(history_rows)
            history_rows = []

            if not indicators:
                continue

            snap = self._position_snapshot(pos, price)
            suggested: List[Dict[str, Any]] = []
            for p in policies:
                pname = p.get("policy") or p.get("name")
                assert pname, "policy name required"
                ind_name = p.get("indicator") or "psar"
                prev = self.eng.store.get_trailing_stop_state(pid, pname)
                prev_stop = prev["stop"] if prev else None
                try:
                    decision = evaluate_policy(pname, p, snap, indicators, prev_stop)
                except Exception as e:
                    self.notify("WARN", f"[trail] {symbol} policy {pname} failed: {e}", symbol=symbol)
                    continue
                suggestion = decision.get("suggested_stop")
                self.eng.store.set_trailing_stop_state(pid, pname, suggestion, {"reason": decision.get("reason")}, now_ms)
                history_rows.append({
                    "thinker_id": tid,
                    "position_id": pid,
                    "name": f"trail:{pname}",
                    "ts_ms": indicators.get(ind_name, {}).get("ts_ms", now_ms),
                    "value": suggestion,
                    "price": price,
                    "aux": {"policy": pname, "source": ind_name},
                })
                if suggestion is not None:
                    suggested.append({"policy": pname, "stop": suggestion})

            if history_rows:
                self.eng.store.insert_indicator_history(history_rows)

            stops = [s["stop"] for s in suggested]
            agg = _agg_stop(snap["side"], stops)
            prev_agg = self.eng.store.get_trailing_stop_state(pid, "__agg__")
            prev_stop = prev_agg["stop"] if prev_agg else None

            if agg is not None:
                hit = price <= agg if snap["side"] == "LONG" else price >= agg
                meta = prev_agg.get("meta") if prev_agg else {}
                if _stop_improved(snap["side"], prev_stop, agg, min_move_bp):
                    self.notify("INFO", f"[trail] {symbol} stop -> {agg:.4f} ({snap['side']})", send=True,
                                symbol=symbol, stop=agg, prev=prev_stop)
                if hit:
                    last_alert_ts = (meta or {}).get("last_alert_ts")
                    if not last_alert_ts or now_ms - int(last_alert_ts) >= self._cfg["alert_cooldown_ms"]:
                        self.notify("WARN", f"[trail] {symbol} stop hit @ {price:.4f} vs {agg:.4f}", send=True,
                                    symbol=symbol, stop=agg, price=price)
                        meta = dict(meta or {})
                        meta["last_alert_ts"] = now_ms
                history_rows.append({
                    "thinker_id": tid,
                    "position_id": pid,
                    "name": "trail:agg",
                    "ts_ms": now_ms,
                    "value": agg,
                    "price": price,
                    "aux": {"side": snap["side"]},
                })
                self.eng.store.set_trailing_stop_state(pid, "__agg__", agg, meta, now_ms)

            if history_rows:
                self.eng.store.insert_indicator_history(history_rows)

            processed += 1
        return processed


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
