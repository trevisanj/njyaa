#!/usr/bin/env python3
# FILE: indicators.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, Sequence
import numpy as np
import pandas as pd
import engclasses as ec


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

        if state is None:
            win_size = self.window_size(self.cfg)
            win_start = start_idx - (win_size - 1)
            h0 = np.max(h_arr[win_start:start_idx])
            l0 = np.min(l_arr[win_start:start_idx])
            up = init_up
            ep = h0 if up else l0
            af = af0
            psar = l0 if up else h0
            stopped_count = 0
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
                     "stopped": stopped, "stopped_count": int(stopped_count)}
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


class TrailingPercentIndicator(BaseIndicator):
    """
    Percentage trail off recent extremes: long => below low, short => above high.
    cfg:
      fraction: decimal fraction (e.g. 0.01 for 1%)
      side: +1 long / -1 short
    """
    kind = "trail_pct"

    @classmethod
    def default_cfg(cls, *, position: ec.Position) -> dict:
        return {"fraction": 0.01, "side": position.side}

    @classmethod
    def window_size(cls, cfg: dict) -> int:
        return 1

    # noinspection PyMethodOverriding
    def on_run(self, start_idx: int, end_idx: int, df: pd.DataFrame, *args, **kwargs) -> Tuple[dict, Dict[str, np.ndarray]]:
        fraction = float(self.cfg["fraction"])
        side = int(self.cfg["side"])
        h_arr = df["High"].values
        l_arr = df["Low"].values
        n_total = self.n_total()
        val_arr = np.full(n_total, np.nan, dtype=float)

        for i in range(start_idx, end_idx):
            if side > 0:
                val_arr[i] = l_arr[i] * (1 - fraction)
            else:
                val_arr[i] = h_arr[i] * (1 + fraction)

        last_val = val_arr[end_idx - 1] if end_idx > start_idx else np.nan
        new_state = {"last_value": last_val}
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
        n_total = self.n_total()
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
        self._stop_info = {"value": val_arr, "flag": flag_arr}
        return new_state, self._stop_info


INDICATOR_CLASSES = {cls.kind: cls for cls in BaseIndicator.__subclasses__()}
