#!/usr/bin/env python3
# FILE: indicator_engines.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, Sequence
import numpy as np
import pandas as pd
import copy
from common import LAST_TS, log, LOOKBACK_BARS, tf_ms, TooFewDataPoints, ts_human, IND_STATES, STATE_TS
import engclasses as ec
from collections import OrderedDict


class BaseIndicator:
    """Minimal base for indicator steppers."""
    kind: str = "BASE"

    def __init__(self, cfg: dict, name: str, sstrat: "StopStrategy"):
        self.sstrat = sstrat
        self.cfg = cfg or {}
        self.name = name
        self.state: Optional[dict] = None
        self.outputs: Optional[Dict[str, np.ndarray]] = None
        self.on_init()

    @classmethod
    def default_cfg(cls, *, position: ec.Position) -> dict:
        return {}

    @classmethod
    def lookback_bars(cls, cfg: dict) -> int:
        return 1

    @classmethod
    def from_kind(cls, kind: str, name, cfg: dict, sstrat):
        if kind not in INDICATOR_CLASSES:
            raise ValueError(f"Unknown indicator kind: {kind}")
        return INDICATOR_CLASSES[kind](cfg, name, sstrat)

    def on_init(self):
        """Inherit to create new instance variables etc."""
        pass

    def run(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Execute indicator, storing outputs for later inspection; returns outputs only."""
        new_state, outs = self.on_run(self.sstrat._start_idx, self.sstrat._end_idx, *args, **kwargs)
        self.outputs = outs
        self.state = new_state
        return outs

    def on_run(self, start_idx: int, end_idx: int, *args, **kwargs) -> Tuple[dict, Dict[str, np.ndarray]]:
        raise NotImplementedError

    def get_my_state(self):
        return self.sstrat.get_ind_state(self.name)


class PSARIndicator(BaseIndicator):
    """Parabolic SAR indicator with configurable initial trend."""
    kind = "psar"

    @classmethod
    def lookback_bars(cls, cfg: dict) -> int:
        return 5

    @classmethod
    def default_cfg(cls, *, position: ec.Position) -> dict:
        trend = "UP" if position.side > 1 else "DOWN"
        return {"af": 0.02, "max_af": 0.2, "initial_trend": trend}

    # noinspection PyMethodOverriding
    def on_run(self, start_idx: int, end_idx: int, df: pd.DataFrame, *args, **kwargs) -> Tuple[dict, Dict[str, np.ndarray]]:
        af0 = self.cfg["af"]
        afmax = self.cfg["max_af"]
        init_up = True if str(self.cfg["initial_trend"]).upper() == "UP" else False

        o_arr = df["Open"].values
        h_arr = df["High"].values
        l_arr = df["Low"].values
        c_arr = df["Close"].values

        n_total = len(df)
        val_arr = np.full(n_total, np.nan, dtype=float)

        state = self.get_my_state()
        stopped = False
        stop_val: Optional[float] = None
        stopped_count = int(state.get("stopped_count", 0)) if state else 0

        if state is None:
            need = self.lookback_bars(self.cfg)
            win_start = start_idx - need
            h0 = np.max(h_arr[win_start:start_idx])
            l0 = np.min(l_arr[win_start:start_idx])
            up = init_up
            ep = h0 if up else l0
            af = af0
            psar = l0 if up else h0
        else:
            psar = state["psar"]
            ep = state["ep"]
            af = state["af"]
            trend = state.get("trend")
            stopped = state["stopped"] or trend == "STOPPED"
            up = True if trend == "UP" else False
            stop_val = psar if stopped else None
            stopped_count = state["stopped_count"]

        if stopped and stop_val is not None:
            for i in range(start_idx, end_idx):
                val_arr[i] = stop_val
            stopped_count += max(0, end_idx - start_idx)
            new_state = {"psar": psar, "ep": ep, "af": af, "trend": ("STOPPED" if stopped else ("UP" if up else "DOWN")),
                         "stopped": stopped, "stopped_count": stopped_count}
            outputs = {"value": val_arr}
            return new_state, outputs

        for i in range(start_idx, end_idx):
            o, h, l, c = o_arr[i], h_arr[i], l_arr[i], c_arr[i]
            prev_psar = psar
            prev_ep = ep
            prev_up = up

            psar = prev_psar + af * (prev_ep - prev_psar)
            if up:
                if i >= 1:
                    psar = min(psar, l_arr[i - 1])
                if i >= 2:
                    psar = min(psar, l_arr[i - 2])
            else:
                if i >= 1:
                    psar = max(psar, h_arr[i - 1])
                if i >= 2:
                    psar = max(psar, h_arr[i - 2])

            if up:
                if l < psar:
                    stop_val = psar
                    stopped = True
                else:
                    if h > prev_ep:
                        ep = h
                        af = min(af + af0, afmax)
                    else:
                        ep = prev_ep
            else:
                if h > psar:
                    stop_val = psar
                    stopped = True
                else:
                    if l < prev_ep:
                        ep = l
                        af = min(af + af0, afmax)
                    else:
                        ep = prev_ep

            val_arr[i] = psar

            if stopped and stop_val is not None:
                stopped_count += (end_idx - i)
                for j in range(i, end_idx):
                    val_arr[j] = stop_val
                psar = stop_val
                up = prev_up
                break

        new_state = {"psar": psar, "ep": ep, "af": af, "trend": ("STOPPED" if stopped else ("UP" if up else "DOWN")),
                     "stopped": stopped, "stopped_count": stopped_count}
        outputs = {"value": val_arr}
        return new_state, outputs


class ATRIndicator(BaseIndicator):
    """Wilder ATR indicator."""
    kind = "atr"

    @classmethod
    def lookback_bars(cls, cfg: dict) -> int:
        return max(cfg["period"] + 1, 2)

    @classmethod
    def default_cfg(cls, *, position: ec.Position) -> dict:
        return {"period": 14}

    # noinspection PyMethodOverriding
    def on_run(self, start_idx: int, end_idx: int, df: pd.DataFrame, *args, **kwargs) -> Tuple[dict, Dict[str, np.ndarray]]:
        """
        Compute ATR; returns (new_state, outputs).
        outputs: {"value", "tr"} aligned to df.index (NaNs where unchanged).
        """
        period = int(self.cfg["period"])
        need = self.lookback_bars(self.cfg)
        state = self.get_my_state()

        h_arr = df["High"].values
        l_arr = df["Low"].values
        c_arr = df["Close"].values

        n_total = len(df)
        val_arr = np.full(n_total, np.nan, dtype=float)
        idx = start_idx

        if state is None:
            prev_close = c_arr[0]
            trs: list[float] = []
            for i in range(1, need):
                h = h_arr[i]; l = l_arr[i]; c = c_arr[i]
                tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
                trs.append(tr)
                prev_close = c
            atr = sum(trs) / period
            idx = max(idx, need)
        else:
            atr = state["atr"]
            prev_close = state["prev_close"]

        for i in range(idx, end_idx):
            h = h_arr[i]; l = l_arr[i]; c = c_arr[i]
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            atr = ((atr * (period - 1)) + tr) / period
            prev_close = c
            val_arr[i] = atr

        new_state = {"atr": atr, "prev_close": prev_close}
        outputs = {"value": val_arr}
        return new_state, outputs


class StopperIndicator(BaseIndicator):
    """
    Protective stop ratchet based on upstream proposed levels.
    cfg:
      side: +1 for long, -1 for short
    run inputs:
      bars: DataFrame with Close
      values: np.ndarray of proposed stop levels (same length as bars)
    outputs:
      value: ratcheted stop series
      flag: 1.0 when stop is hit, NaN otherwise
    """
    kind = "stopper"

    @classmethod
    def default_cfg(cls, *, position: ec.Position) -> dict:
        return {"side": position.side}

    def on_init(self):
        self._stop_info = None

    # noinspection PyMethodOverriding
    def on_run(self, start_idx: int, end_idx: int, df: pd.DataFrame, values: np.ndarray, *args, **kwargs) \
            -> Tuple[dict, Dict[str, np.ndarray]]:
        if df is None or df.empty:
            raise ValueError("bars required")
        side = int(self.cfg.get("side") or 0)
        if side == 0:
            raise ValueError("Stopper side must be +1 or -1")
        n_total = len(df)
        val_arr = np.full(n_total, np.nan, dtype=float)
        flag_arr = np.full(n_total, np.nan, dtype=float)
        stop = None if self.state is None else self.state.get("stop")
        closes = df["Close"].values

        for i in range(start_idx, end_idx):
            candidate = values[i] if values is not None and i < len(values) else np.nan
            if not np.isnan(candidate):
                if stop is None:
                    stop = candidate
                else:
                    stop = max(stop, candidate) if side > 0 else min(stop, candidate)
            val_arr[i] = stop if stop is not None else np.nan
            if stop is None:
                continue
            hit = closes[i] <= stop if side > 0 else closes[i] >= stop
            if hit:
                flag_arr[i] = 1.0

        new_state = {"stop": stop}
        # makes it retrievable to the sstrat later
        self._stop_info = {"value": val_arr, "flag": flag_arr}
        return new_state, self._stop_info


class StopStrategy:
    """
    Base stop strategy:
      - holds indicators (instantiated once)
      - owns runtime slice (states, cfg)
      - runs indicators, saves their state/history, and returns suggested stop + history rows
    """
    kind: str = "BASE_STRAT"
    def __init__(self, eng, thinker, ctx: dict, pos, name: str, attached_at):
        self.eng = eng
        self.thinker = thinker
        self.ctx = ctx
        self.pos = pos
        self.name = name
        self.timeframe = thinker.get_timeframe()
        self.attached_at = attached_at
        self.inds: Dict[str, BaseIndicator] = {}
        # {ts_ms: {ind_name: state, ...}, ...} with maximum 2 elements
        self.ind_states = ctx.setdefault(IND_STATES, OrderedDict())
        self._stopper: Optional["StopperIndicator"] = None
        self._stop_info: Optional[Dict[str, Sequence]] = None
        self.cfg = ctx.setdefault("cfg", {})
        self.history_rows: list[dict] = []

        ctx.setdefault(STATE_TS, -1)

        self.setup()

    ind_map = None  # subclass can set to list of kinds or dict {kind: name}

    def on_conf_ind(self):
        """Hook to tweak indicator configs after auto-build."""

    def on_get_default_cfg(self, ind_kind: str) -> dict:
        """
        Return default cfg for a given indicator kind.
        Override if strategy needs custom defaults.
        """
        cls = INDICATOR_CLASSES[ind_kind]
        return cls.default_cfg(position=self.pos)

    def setup(self):
        """Convenience wrapper to configure indicators and perform extra setup."""
        self._build_indicators()
        self.on_conf_ind()

    def _build_indicators(self):
        items = [(k, v) for k, v in self.ind_map.items()] if isinstance(self.ind_map, dict) else \
                [(k, k) for k in self.ind_map]
        for kind, name in items:
            cfg = self.on_get_default_cfg(kind)
            ind = self.inds[name] = BaseIndicator.from_kind(kind, name, cfg, self)
            if kind == "stopper":
                # Grabs last stopper
                self._stopper = ind

        assert self._stopper, f"sstrat.init.fail.no_stopper: sstrat_kind={self.kind}"

    def on_run(self, bars: pd.DataFrame):
        """Override this to implement the actual strategy"""
        raise NotImplementedError

    def get_stop_info(self):
        """Return dict with value/flag keys from last run."""
        return self._stopper._stop_info

    def get_lookback_bars(self) -> int:
        """Return minimum bars needed based on contained indicators."""
        if not LOOKBACK_BARS in self.ctx:
            self.ctx[LOOKBACK_BARS] = max(ind.__class__.lookback_bars(ind.cfg) for ind in self.inds.values())
        return self.ctx[LOOKBACK_BARS]

    def run(self):
        """
        Execute strategy: compute start_idx, run indicators/stop logic, persist state/history.
        """
        last_ts = self.ctx.get(LAST_TS)
        lookback_bars = self.get_lookback_bars()
        tfms = tf_ms(self.timeframe)

        # -------------- fetches bars from cache
        anchor_ts = last_ts or self.attached_at
        start_ts = max(0, int(anchor_ts) - lookback_bars * tfms)  # start window so we have enough bars
        self.last_bars = bars = self.eng.kc.pair_bars(self.pos.num, self.pos.den, self.timeframe, start_ts)
        # TODO perhaps lookback_bars+1 here
        if bars.empty or len(bars) < lookback_bars:
            raise TooFewDataPoints(f"[trail] Not enough klines ({len(bars)} < {lookback_bars}), can't run strategy!")
        log().debug("sstrat.bars", sstrat_kind=self.kind, n_bars=len(bars), lookback=lookback_bars,
                    anchor_ts=ts_human(anchor_ts), position_id=self.pos.id, )
        self._v_ts = bars.index.astype("int64") // 1_000_000  # for get_ind_state()

        # pass 1: finalized bars (exclude last bar)
        bars_final = bars.iloc[:-1]
        if not bars_final.empty:
            start_idx_final = self._calc_start_idx(bars_final, last_ts)
            self._run_pass("final", bars_final, start_idx_final, persist_state=True, save_history=True)

        # pass 2: live view (tail, do not persist state)

        # Note: indicators shouldn't write into their states ever
        # ctx_states = copy.deepcopy(self.ctx[IND_STATES])
        # for name, ind in self.inds.items():
        # if name in ctx_states:
        #     ind.state = copy.deepcopy(ctx_states[name])

        tail = bars.iloc[-lookback_bars:] if lookback_bars < len(bars) else bars
        start_idx_live = lookback_bars-1
        self._run_pass("live", tail, start_idx_live, persist_state=False, save_history=True)

        # restore persisted states (no change)
        ctx_states = self.ctx[IND_STATES]
        for name, ind in self.inds.items():
            if name in ctx_states:
                st = ctx_states[name]
                ind.state = st[1] if isinstance(st, tuple) else st
        return None

    def _calc_start_idx(self, bars: pd.DataFrame, last_ts: Optional[int]) -> int:
        if last_ts is None:
            return 0
        dt = pd.to_datetime(int(last_ts), unit="ms")
        start_idx = int(bars.index.searchsorted(dt, side="left"))
        if start_idx >= len(bars):
            start_idx = max(0, len(bars) - 1)
        return start_idx

    def _run_pass(self, label: str, bars: pd.DataFrame, start_idx: int, *, persist_state: bool, save_history: bool):
        log().debug(f"sstrat.run.{label}", strat=self.kind, position_id=self.pos.id,
                    start_idx=start_idx, last_ts=self.ctx.get(LAST_TS), rows=len(bars))
        self._start_idx = start_idx
        self._end_idx = len(bars)
        (self.on_run
         (bars))
        if persist_state:
            ts_ms = int(bars.index[-1].value // 1_000_000)
            self.ctx[STATE_TS] = ts_ms
            self.ctx[IND_STATES] = {name: ind.state for name, ind in self.inds.items()}
            self.ctx[LAST_TS] = ts_ms
        if save_history:
            ts_slice = bars.index[start_idx:]
            ts_ms = ts_slice.astype("int64") // 1_000_000
            for ind_name, ind in self.inds.items():
                if not ind.outputs:
                    continue
                for out_name, arr in ind.outputs.items():
                    vals = arr[start_idx:]
                    log().debug(f"sstrat.hist.write.{label}", strat=self.kind, ind=ind_name, out=out_name,
                                position_id=self.pos.id, points=len(vals))
                    self.eng.ih.insert_history2(self.thinker._thinker_id, self.pos.id, f"{ind_name}-{out_name}", ts_ms,
                                                vals,)

    @classmethod
    def from_kind(cls, kind: str, eng, thinker, ctx: dict, pos, name, attached_at: int):
        if kind not in SSTRAT_CLASSES:
            raise ValueError(f"Unknown strategy kind: {kind}")
        strat_cls = SSTRAT_CLASSES[kind]
        return strat_cls(eng, thinker, ctx, pos, name, attached_at)

    def get_ind_state(self, ind_name: str):
        idx = self._start_idx
        ts_ms = int(self._v_ts[idx - 1]) if idx > 0 else "OUT_OF_BOUNDS"
        have_ts = self.ctx[STATE_TS]

        if ts_ms == have_ts:
            return self.ctx[IND_STATES][ind_name]

        log().debug("sstrat.state.miss", ind=ind_name, idx_minus_1=idx-1, ts=ts_human(ts_ms), have_ts=have_ts)
        return None


class SSPSAR(StopStrategy):
    """
    Stop strategy: PSAR feeding Stopper (protective, one-sided).
    """
    kind = "SSPSAR"
    ind_map = ["psar", "stopper"]

    def on_run(self, bars: pd.DataFrame):
        psar = self.inds["psar"]
        stopper = self.inds["stopper"]

        psar_out = psar.run(bars)
        _ = stopper.run(bars, psar_out["value"])



SSTRAT_CLASSES = {cls.kind: cls for cls in StopStrategy.__subclasses__()}


# registry
INDICATOR_CLASSES = {cls.kind: cls for cls in BaseIndicator.__subclasses__()}
