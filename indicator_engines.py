#!/usr/bin/env python3
# FILE: indicator_engines.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, Sequence
import numpy as np
import pandas as pd
import copy
from common import LAST_TS, log, WINDOW_SIZE, tf_ms, TooFewDataPoints, ts_human, IND_STATES, STATE_TS
import engclasses as ec
from collections import OrderedDict


class BaseIndicator:
    """Minimal base for indicator steppers."""
    kind: str = "BASE"

    def __init__(self, cfg: dict, name: str, sstrat: "StopStrategy"):
        self.sstrat = sstrat
        self.cfg = cfg or {}
        self.name = name
        self._temp_state: Optional[dict] = None
        self.outputs: Optional[Dict[str, np.ndarray]] = None
        self.on_init()

    @classmethod
    def default_cfg(cls, *, position: ec.Position) -> dict:
        return {}

    @classmethod
    def window_size(cls, cfg: dict) -> int:
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
        self._temp_state = new_state
        return outs

    def on_run(self, start_idx: int, end_idx: int, *args, **kwargs) -> Tuple[dict, Dict[str, np.ndarray]]:
        raise NotImplementedError

    def get_my_state(self):
        return self.sstrat.get_ind_state(self.name)

    def n_total(self) -> int:
        return len(self.sstrat._v_ts)


class PSARIndicator(BaseIndicator):
    """Parabolic SAR indicator with configurable initial trend."""
    kind = "psar"

    @classmethod
    def window_size(cls, cfg: dict) -> int:
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

        n_total = self.n_total()
        val_arr = np.full(n_total, np.nan, dtype=float)

        state = self.get_my_state()
        stopped = False
        stop_val: Optional[float] = None
        stopped_count = int(state.get("stopped_count", 0)) if state else 0

        if state is None:
            win_size = self.window_size(self.cfg)
            win_start = start_idx - (win_size - 1)
            h0 = np.max(h_arr[win_start:start_idx])
            l0 = np.min(l_arr[win_start:start_idx])
            up = init_up
            ep = h0 if up else l0
            af = af0
            psar = l0 if up else h0  # todo include stopped_count here
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
    def window_size(cls, cfg: dict) -> int:
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
        need = self.window_size(self.cfg)
        state = self.get_my_state()

        h_arr = df["High"].values
        l_arr = df["Low"].values
        c_arr = df["Close"].values

        n_total = self.n_total()
        val_arr = np.full(n_total, np.nan, dtype=float)
        idx = start_idx

        if state is None:
            seed_start = start_idx - (need - 1)
            prev_close = c_arr[seed_start]
            trs: list[float] = []
            for i in range(seed_start + 1, seed_start + need):
                h = h_arr[i]; l = l_arr[i]; c = c_arr[i]
                tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
                trs.append(tr)
                prev_close = c
            atr = sum(trs) / period
            idx = seed_start + need
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
        state = self.get_my_state()
        side = self.cfg["side"]
        n_total = len(df)
        val_arr = np.full(n_total, np.nan, dtype=float)
        flag_arr = np.full(n_total, np.nan, dtype=float)
        stop = None if state is None else state["stop"]
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

    def get_window_size(self) -> int:
        """Return minimum bars needed based on contained indicators."""
        if not WINDOW_SIZE in self.ctx:
            self.ctx[WINDOW_SIZE] = max(ind.__class__.window_size(ind.cfg) for ind in self.inds.values())
        return self.ctx[WINDOW_SIZE]

    def run(self):
        """
        Execute strategy: compute start_idx, run indicators/stop logic, persist state/history.
        """
        last_ts = self.ctx.get(LAST_TS)
        window_size = self.get_window_size()
        tfms = tf_ms(self.timeframe)

        # ------ reads bars from cache
        anchor_ts = last_ts or self.attached_at
        start_ts = int(anchor_ts) - (window_size + 1) * tfms  # start window so we have enough bars\

        # reads bars
        self.last_bars = bars = self.eng.kc.pair_bars(self.pos.num, self.pos.den, self.timeframe, start_ts)
        n_bars = len(bars)
        log().debug("sstrat.bars", sstrat_kind=self.kind, n_bars=n_bars, window_size=window_size,
                    anchor_ts=ts_human(anchor_ts), position_id=self.pos.id, )
        v_ts = bars.index.astype("int64") // 1_000_000  # for get_ind_state()

        # start_idx corresponds to a timestamp <= anchor_ts
        start_idx = v_ts.searchsorted(anchor_ts, side="right")-1
        if start_idx < window_size-1:
            # TODO: Consider recovering (put this in a loop to increase the start_ts lookback for a few tries)
            raise TooFewDataPoints(f"[trail] Not enough klines (window_size={window_size}, start_idx={start_idx}), can't run strategy!")

        if start_idx < n_bars-1:
            # Pass 1: only if there is more than 1 data point to be calculated
            self._run_pass(bars[:-1], start_idx, True)

        # Pass 2: live bar; does not save state
        self._run_pass(bars[-window_size:], window_size-1, False)

        ts_ms = int(v_ts[-1])
        self.ctx[LAST_TS] = ts_ms

        return None

    def _run_pass(self, bars, start_idx: int, save_state: bool):
        self._start_idx = start_idx
        self._end_idx = len(bars)
        self._v_ts = bars.index.astype("int64") // 1_000_000  # for get_ind_state()


        # Runs the stop strategy
        self.on_run(bars)

        if save_state:
            # saves indictor states
            ts_ms = int(self._v_ts[-1])
            self.ind_states[ts_ms] = {name: ind._temp_state for name, ind in self.inds.items()}

        ts_ms = self._v_ts[start_idx:]
        for ind_name, ind in self.inds.items():
            if not ind.outputs:
                continue
            for out_name, arr in ind.outputs.items():
                vals = arr[start_idx:]
                log().debug(f"sstrat.history.write", strat=self.kind, ind=ind_name, out=out_name,
                            position_id=self.pos.id, points=len(vals))
                self.eng.ih.insert_history2(self.thinker._thinker_id, self.pos.id, f"{ind_name}-{out_name}", ts_ms,
                                            vals, )

    @classmethod
    def from_kind(cls, kind: str, eng, thinker, ctx: dict, pos, name, attached_at: int):
        if kind not in SSTRAT_CLASSES:
            raise ValueError(f"Unknown strategy kind: {kind}")
        strat_cls = SSTRAT_CLASSES[kind]
        return strat_cls(eng, thinker, ctx, pos, name, attached_at)

    def get_ind_state(self, ind_name: str):
        idx = self._start_idx
        ts_ms = int(self._v_ts[idx - 1]) if idx > 0 else "OUT_OF_BOUNDS"

        states = self.ind_states.get(ts_ms)
        if states:
            return states[ind_name]

        log().debug("sstrat.state.miss", ind=ind_name, idx_minus_1=idx-1, ts=ts_human(ts_ms), have_ts=list(self.ind_states.keys()))
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
