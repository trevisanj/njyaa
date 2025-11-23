#!/usr/bin/env python3
# FILE: indicator_engines.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd


class PSARIndicator:
    """Parabolic SAR indicator with configurable initial trend."""

    @classmethod
    def lookback_bars(cls, cfg: dict) -> int:
        return 5

    @classmethod
    def default_cfg(cls) -> dict:
        return {"af": 0.02, "max_af": 0.2, "initial_trend": "UP"}

    @classmethod
    def run(cls, df: pd.DataFrame, state: Optional[dict], cfg: dict, start_idx: int = 0) -> Tuple[dict, Dict[str, np.ndarray], Optional[float]]:
        """
        Compute PSAR; returns (new_state, outputs, latest_value).
        outputs: {"value", "ep", "af", "trend"} aligned to df.index (NaNs where unchanged).
        """
        base_cfg = cls.default_cfg()
        cfg = {**base_cfg, **(cfg or {})}
        if df is None or df.empty:
            raise ValueError("bars required")
        if start_idx >= len(df):
            return state or {}, {"value": np.full(len(df), np.nan)}, None

        af0 = cfg["af"]
        afmax = cfg["max_af"]
        init_up = True if str(cfg["initial_trend"]).upper() == "UP" else False

        o_arr = df["Open"].values
        h_arr = df["High"].values
        l_arr = df["Low"].values
        c_arr = df["Close"].values
        ts_arr = (df.index.view("int64") // 1_000_000).astype(np.int64)

        n_total = len(df)
        val_arr = np.full(n_total, np.nan, dtype=float)
        ep_arr = np.full(n_total, np.nan, dtype=float)
        af_arr = np.full(n_total, np.nan, dtype=float)
        trend_arr = np.full(n_total, None, dtype=object)

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
            up = True if state["trend"] == "UP" else False
            idx = max(start_idx, 0)

        for i in range(idx, len(ts_arr)):
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
                if h > psar:
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

            val_arr[i] = psar
            ep_arr[i] = ep
            af_arr[i] = af
            trend_arr[i] = "UP" if up else "DOWN"

        latest = None
        if np.any(~np.isnan(val_arr)):
            latest = float(val_arr[~np.isnan(val_arr)][-1])

        last_idx = int(np.nanmax(np.where(~np.isnan(val_arr), np.arange(len(val_arr)), -1))) if np.any(~np.isnan(val_arr)) else (len(df) - 1)
        last_open_ms = int(df.index[last_idx].value // 1_000_000)
        new_state = {"psar": psar, "ep": ep, "af": af, "trend": ("UP" if up else "DOWN"), "last_open_ms": last_open_ms}
        outputs = {"value": val_arr, "ep": ep_arr, "af": af_arr, "trend": trend_arr}
        return new_state, outputs, latest


class ATRIndicator:
    """Wilder ATR indicator."""

    @classmethod
    def lookback_bars(cls, cfg: dict) -> int:
        period = int(cfg.get("period", 14))
        return max(period + 1, 2)

    @classmethod
    def default_cfg(cls) -> dict:
        return {"period": 14}

    @classmethod
    def run(cls, df: pd.DataFrame, state: Optional[dict], cfg: dict, start_idx: int = 0) -> Tuple[dict, Dict[str, np.ndarray], Optional[float]]:
        """
        Compute ATR; returns (new_state, outputs, latest_value).
        outputs: {"value", "tr"} aligned to df.index (NaNs where unchanged).
        """
        base_cfg = cls.default_cfg()
        cfg = {**base_cfg, **(cfg or {})}
        if df is None or df.empty:
            raise ValueError("bars required")
        if start_idx >= len(df):
            return state or {}, {"value": np.full(len(df), np.nan)}, None

        period = int(cfg["period"])
        need = cls.lookback_bars(cfg)

        h_arr = df["High"].values
        l_arr = df["Low"].values
        c_arr = df["Close"].values
        o_arr = df["Open"].values
        ts_arr = (df.index.view("int64") // 1_000_000).astype(np.int64)

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

        for i in range(idx, len(ts_arr)):
            h = h_arr[i]; l = l_arr[i]; c = c_arr[i]
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            atr = ((atr * (period - 1)) + tr) / period
            prev_close = c
            val_arr[i] = atr
            tr_arr[i] = tr

        latest = None
        if np.any(~np.isnan(val_arr)):
            latest = float(val_arr[~np.isnan(val_arr)][-1])

        last_idx = int(np.nanmax(np.where(~np.isnan(val_arr), np.arange(len(val_arr)), -1))) if np.any(~np.isnan(val_arr)) else (len(df) - 1)
        new_state = {"atr": atr, "prev_close": prev_close, "last_open_ms": int(df.index[last_idx].value // 1_000_000)}
        outputs = {"value": val_arr, "tr": tr_arr}
        return new_state, outputs, latest


INDICATOR_DISPATCH = {
    "psar": PSARIndicator,
    "atr": ATRIndicator,
}


def run_indicator(name: str, bars: pd.DataFrame, cfg: dict, state: Optional[dict], start_idx: int = 0) -> Tuple[dict, Dict[str, np.ndarray], Optional[float]]:
    cls = INDICATOR_DISPATCH.get(name)
    if not cls:
        raise ValueError(f"Unknown indicator: {name}")
    return cls.run(bars, state, cfg, start_idx=start_idx)
