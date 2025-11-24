#!/usr/bin/env python3
# FILE: indicator_engines.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd



class BaseIndicator:
    """Minimal base for indicator steppers."""

    @classmethod
    def default_cfg(cls, *, position=None) -> dict:
        return {}

    @classmethod
    def lookback_bars(cls, cfg: dict) -> int:
        return 1



class PSARIndicator(BaseIndicator):
    """Parabolic SAR indicator with configurable initial trend."""

    @classmethod
    def lookback_bars(cls, cfg: dict) -> int:
        return 5

    @classmethod
    def default_cfg(cls, *, position=None) -> dict:
        side = None
        if position and isinstance(position, dict):
            side = position.get("side")
        trend = "UP" if side == "LONG" else ("DOWN" if side == "SHORT" else "UP")
        return {"af": 0.02, "max_af": 0.2, "initial_trend": trend}

    @classmethod
    def run(cls, df: pd.DataFrame, state: Optional[dict], cfg: dict, start_idx: int = 0) -> Tuple[dict, Dict[str, np.ndarray]]:
        """
        Compute PSAR; returns (new_state, outputs).
        outputs: {"value", "ep", "af", "trend"} aligned to df.index (NaNs where unchanged).
        """
        if df is None or df.empty:
            raise ValueError("bars required")
        if start_idx >= len(df):
            return state or {}, {"value": np.full(len(df), np.nan)}

        af0 = cfg["af"]
        afmax = cfg["max_af"]
        init_up = True if str(cfg["initial_trend"]).upper() == "UP" else False

        o_arr = df["Open"].values
        h_arr = df["High"].values
        l_arr = df["Low"].values
        c_arr = df["Close"].values

        n_total = len(df)
        val_arr = np.full(n_total, np.nan, dtype=float)
        ep_arr = np.full(n_total, np.nan, dtype=float)
        af_arr = np.full(n_total, np.nan, dtype=float)
        trend_arr = np.full(n_total, None, dtype=object)

        stopped = False
        stop_val: Optional[float] = None
        stopped_count = int(state.get("stopped_count", 0)) if state else 0

        if state is None:
            if start_idx != 0:
                start_idx = 0
            need = cls.lookback_bars(cfg)
            if len(df) < need:
                raise ValueError(f"Need at least {need} bars to bootstrap PSAR")
            h0 = float(np.max(h_arr[:need]))
            l0 = float(np.min(l_arr[:need]))
            up = init_up
            ep = h0 if up else l0
            af = af0
            psar = l0 if up else h0
            idx = need
        else:
            psar = float(state["psar"])
            ep = float(state["ep"])
            af = float(state["af"])
            trend = state.get("trend")
            stopped = bool(state.get("stopped", False)) or trend == "STOPPED"
            up = True if trend == "UP" else False
            stop_val = psar if stopped else None
            stopped_count = int(state.get("stopped_count", 0))
            idx = max(start_idx, 0)

        if stopped and stop_val is not None:
            for i in range(idx, n_total):
                val_arr[i] = stop_val
                ep_arr[i] = ep
                af_arr[i] = af
                trend_arr[i] = "STOPPED"
            stopped_count += max(0, n_total - idx)
            return {"psar": stop_val, "ep": ep, "af": af, "trend": "STOPPED", "stopped": True, "stopped_count": stopped_count}, {
                "value": val_arr,
                "ep": ep_arr,
                "af": af_arr,
                "trend": trend_arr,
            }

        for i in range(idx, n_total):
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
            ep_arr[i] = ep
            af_arr[i] = af
            trend_arr[i] = "UP" if up else "DOWN"

            if stopped and stop_val is not None:
                stopped_count += (n_total - i)
                for j in range(i, n_total):
                    val_arr[j] = stop_val
                    ep_arr[j] = ep
                    af_arr[j] = af
                    trend_arr[j] = "STOPPED"
                psar = stop_val
                up = prev_up
                break

        new_state = {"psar": psar, "ep": ep, "af": af, "trend": ("STOPPED" if stopped else ("UP" if up else "DOWN")), "stopped": stopped, "stopped_count": stopped_count}
        outputs = {"value": val_arr, "ep": ep_arr, "af": af_arr, "trend": trend_arr}
        return new_state, outputs


class ATRIndicator(BaseIndicator):
    """Wilder ATR indicator."""

    @classmethod
    def lookback_bars(cls, cfg: dict) -> int:
        return max(cfg["period"] + 1, 2)

    @classmethod
    def default_cfg(cls, *, position=None) -> dict:
        return {"period": 14}

    @classmethod
    def run(cls, df: pd.DataFrame, state: Optional[dict], cfg: dict, start_idx: int = 0) -> Tuple[dict, Dict[str, np.ndarray]]:
        """
        Compute ATR; returns (new_state, outputs).
        outputs: {"value", "tr"} aligned to df.index (NaNs where unchanged).
        """
        if df is None or df.empty:
            raise ValueError("bars required")
        if start_idx >= len(df):
            return state or {}, {"value": np.full(len(df), np.nan)}

        base_cfg = cls.default_cfg()
        cfg = {**base_cfg, **(cfg or {})}
        period = int(cfg["period"])
        need = cls.lookback_bars(cfg)

        h_arr = df["High"].values
        l_arr = df["Low"].values
        c_arr = df["Close"].values
        o_arr = df["Open"].values

        n_total = len(df)
        val_arr = np.full(n_total, np.nan, dtype=float)
        tr_arr = np.full(n_total, np.nan, dtype=float)
        idx = max(start_idx, 0)

        if state is None:
            if len(df) < need:
                raise ValueError(f"Need at least {need} bars to bootstrap ATR(p={period})")
            prev_close = c_arr[0]
            trs: list[float] = []
            for i in range(1, need):
                h = h_arr[i]; l = l_arr[i]; c = c_arr[i]
                tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
                trs.append(tr)
                prev_close = c
            atr = sum(trs) / float(period)
            idx = need
        else:
            atr = float(state["atr"])
            prev_close = float(state["prev_close"])

        for i in range(idx, n_total):
            h = h_arr[i]; l = l_arr[i]; c = c_arr[i]
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            atr = ((atr * (period - 1)) + tr) / period
            prev_close = c
            val_arr[i] = atr
            tr_arr[i] = tr

        new_state = {"atr": atr, "prev_close": prev_close}
        outputs = {"value": val_arr, "tr": tr_arr}
        return new_state, outputs


INDICATOR_DISPATCH = {
    "psar": PSARIndicator,
    "atr": ATRIndicator,
}

def ind_name_to_cls(name):
    return INDICATOR_DISPATCH[name]

def run_indicator(name: str, bars: pd.DataFrame, cfg: dict, state: Optional[dict], start_idx: int = 0) -> Tuple[dict, Dict[str, np.ndarray]]:
    cls = INDICATOR_DISPATCH.get(name)
    if not cls:
        raise ValueError(f"Unknown indicator: {name}")
    return cls.run(bars, state, cfg, start_idx=start_idx)
