
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
from indicator_engines import run_indicator, ind_name_to_cls, BaseIndicator
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

    def _on_init(self) -> None:
        self._runtime.setdefault("positions", {})

    def _indicator_configs(self, policies: List[dict], pos: ec.Position) -> Dict[str, dict]:
        """Returns dict {indicator_name: cfg, ...}"""
        cfgs = {policy["ind_name"]: self._policy_get_indicator_config(policy, pos) for policy in policies}
        return cfgs

    def _policy_get_indicator_config(self, policy, pos: ec.Position) -> Union[Dict[Any], None]:
        """Returns indicator configuration Dict for a policy.

        If Policy were a class, this would be one of its methods"""
        ind_name = policy["ind_name"]
        cfg = policy.get("ind_cfg")
        if cfg is None:
            # first time: we initialize
            cls: BaseIndicator = ind_name_to_cls(ind_name)
            cfg = cls.default_cfg(position=pos)
        return cfg

    def _indicator_history_rows(self, tid: int, pid: int, name: str, outputs: Dict[str, Any], bars: pd.DataFrame, start_idx: int) -> List[dict]:
        rows = []
        val_arr = outputs.get("value")
        if val_arr is None:
            return rows
        idxs = np.where(~np.isnan(val_arr))[0]
        for pos in idxs:
            if pos < start_idx:
                continue
            open_ts = int(bars.index[pos].value // 1_000_000)
            aux = {}
            for k in ("ep", "af", "trend", "tr"):
                arr = outputs.get(k)
                if arr is not None and pos < len(arr):
                    aux[k] = arr[pos]
            rows.append({
                "thinker_id": tid,
                "position_id": pid,
                "name": name,
                "open_ts": open_ts,
                "value": val_arr[pos],
                "price": float(bars.iloc[pos]["Close"]),
                "aux": aux,
            })
        return rows

    # def _position_snapshot(self, pos_row, price: float) -> Dict[str, Any]:
    #     return {
    #         "side": int(pos_row["dir_sign"]),
    #         "last_price": price,
    #         "position_id": int(pos_row["position_id"]),
    #         "num": pos_row["num"],
    #         "den": pos_row["den"],
    #     }

    def _pair_bars(self, pos: ec.Position, start_ts: int, end_ts: Optional[int] = None) -> pd.DataFrame:
        """Returns OHLCV data for configured timeframe"""
        return self.eng.kc.pair_bars(pos.num, pos.den, self._cfg["timeframe"], start_ts, end_ts)

    def on_tick(self, now_ms: int):
        pp_ctx = self._runtime.get("positions", {})  # positions context
        if not pp_ctx:
            # no positions attached
            return 0
        processed = 0
        min_move_bp = float(self._cfg["min_move_bp"])
        tf = self._cfg["timeframe"]
        thinker_id = self._thinker_id
        dirty = False  # save runtime required?

        for pid_str, ctx in pp_ctx.items():
            if ctx.get("invalid"):
                # position does not exist or is closed (se below)
                continue

            pos = self.eng.store.get_position(int(pid_str))

            if not pos or pos.status != "OPEN":
                ctx["invalid"] = True
                ctx["invalid_msg"] = "Closed" if pos else "Inexistent"
                pp_ctx[pid_str] = ctx
                dirty = True
                continue

            den = pos["den"]

            policies = ctx.get("policies") or []

            ind_cfgs = self._indicator_configs(policies, pos)

            need_n = max(int(ctx.get("lookback_bars", 200)), 5)
            tfms = tf_ms(self._cfg["timeframe"])
            anchor_ts = ctx.get("last_ts") or ctx.get("attached_at_ms")
            start_ts = max(0, int(anchor_ts) - need_n * tfms)
            bars = self._pair_bars(pos, start_ts, None)
            if bars.empty or len(bars) < need_n:
                log().debug(f"[trail] Missing klines {tf}, can't calculate stop", symbol=pos.get_pair(), position_id=pos.id)
                continue

            indicators: Dict[str, Dict[str, Any]] = {}
            history_rows: List[dict] = []
            ind_states = ctx.setdefault("indicators", {})
            ctx_last_ts = ctx.get("last_ts")
            start_idx = 0

            anchor_ts = ctx_last_ts if ctx_last_ts is not None else ctx["attached_at_ms"]
            if anchor_ts is not None:
                dt = pd.to_datetime(int(anchor_ts), unit="ms")
                start_idx = int(bars.index.searchsorted(dt, side="right"))
                if start_idx >= len(bars):
                    self.notify("DEBUG", "trail.anchor_clamped", anchor_ts=anchor_ts, last_bar=int(bars.index[-1].value // 1_000_000), pid=pid, symbol=symbol)
                    start_idx = len(bars) - 1
            if start_idx >= len(bars):
                pp_ctx[pid_str] = ctx
                continue

            last_ts_seen = ctx_last_ts
            for ind_name, cfg in ind_cfgs.items():
                st_info = ind_states.get(ind_name, {})
                st_payload = st_info.get("state")
                try:
                    ns, outputs = run_indicator(ind_name, bars, cfg, st_payload, start_idx=start_idx)
                except Exception as e:
                    self.notify("WARN", f"[trail] {symbol} indicator {ind_name} failed: {e}", symbol=symbol)
                    continue
                val_arr = outputs.get("value")
                ts_val = last_ts_seen
                if val_arr is not None and np.any(~np.isnan(val_arr)):
                    last_idx = int(np.nanmax(np.where(~np.isnan(val_arr), np.arange(len(val_arr)), -1)))
                    ts_val = int(bars.index[last_idx].value // 1_000_000)
                ind_states[ind_name] = {"state": ns}
                history_rows.extend(self._indicator_history_rows(thinker_id, pid, ind_name, outputs, bars, start_idx))
                # latest value = last non-nan in value array
                latest_val = None
                if val_arr is not None and np.any(~np.isnan(val_arr)):
                    latest_val = float(val_arr[~np.isnan(val_arr)][-1])
                indicators[ind_name] = {"value": latest_val, "raw": ns, "open_ts": ts_val}
                last_ts_seen = ts_val

            if history_rows:
                self.eng.ih.insert_history(history_rows)
            history_rows = []

            if not indicators or not policies:
                continue

            snap = self._position_snapshot(pos, bars["Close"].iloc[-1])
            trailing = ctx.setdefault("trailing", {})
            suggested: List[Dict[str, Any]] = []
            for p in policies:
                pname = p.get("policy") or p.get("name")
                assert pname, "policy name required"
                ind_name = p.get("indicator") or pname
                prev_stop = (trailing.get(pname) or {}).get("stop")
                try:
                    decision = evaluate_policy(pname, p, snap, indicators, prev_stop)
                except Exception as e:
                    self.notify("WARN", f"[trail] {symbol} policy {pname} failed: {e}", symbol=symbol)
                    continue
                suggestion = decision.get("suggested_stop")
                trailing[pname] = {"stop": suggestion, "meta": {"reason": decision.get("reason")}, "open_ts": now_ms}
                history_rows.append({
                    "thinker_id": thinker_id,
                    "position_id": pid,
                    "name": f"trail:{pname}",
                    "open_ts": indicators.get(ind_name, {}).get("open_ts", now_ms),
                    "value": suggestion,
                    "price": snap["last_price"],
                    "aux": {"policy": pname, "source": ind_name},
                })
                if suggestion is not None:
                    suggested.append({"policy": pname, "stop": suggestion})

            if history_rows:
                self.eng.ih.insert_history(history_rows)

            stops = [s["stop"] for s in suggested]
            agg = _agg_stop(snap["side"], stops)
            prev_agg = trailing.get("__agg__") or {}
            prev_stop = prev_agg.get("stop")

            if agg is not None:
                price = snap["last_price"]
                hit = price <= agg if snap["side"] > 0 else price >= agg
                meta = dict(prev_agg.get("meta") or {})
                if _stop_improved(snap["side"], prev_stop, agg, min_move_bp):
                    self.notify("INFO", f"[trail] {symbol} stop -> {agg:.4f} ({'LONG' if snap['side']>0 else 'SHORT'})", send=True,
                                symbol=symbol, stop=agg, prev=prev_stop, price=price)
                if hit:
                    last_alert_ts = meta.get("last_alert_ts")
                    if not last_alert_ts or now_ms - int(last_alert_ts) >= self._cfg["alert_cooldown_ms"]:
                        self.notify("WARN", f"[trail] {symbol} stop hit @ {price:.4f} vs {agg:.4f}", send=True,
                                    symbol=symbol, stop=agg, price=price)
                        meta["last_alert_ts"] = now_ms
                trailing["__agg__"] = {"stop": agg, "meta": meta, "open_ts": now_ms}
                history_rows.append({
                    "thinker_id": thinker_id,
                    "position_id": pid,
                    "name": "trail:agg",
                    "open_ts": now_ms,
                    "value": agg,
                    "price": price,
                    "aux": {"side": snap["side"]},
                })

            if history_rows:
                self.eng.ih.insert_history(history_rows)

            pp_ctx[pid_str] = ctx
            processed += 1
            dirty = True
            latest_bar_ts = int(bars.index[-1].value // 1_000_000)
            ctx["last_ts"] = max(last_ts_seen or latest_bar_ts, latest_bar_ts)

        if dirty:
            self.save_runtime()
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
