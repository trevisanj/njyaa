#!/usr/bin/env python3
# FILE: indicator_engines.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# Types:
# Bar = (open_ts_ms, open, high, low, close)
Bar = Tuple[int, float, float, float, float]


@dataclass
class PSARState:
    psar: float
    ep: float
    af: float
    trend: str   # "UP" or "DOWN"
    last_open_ms: int


def step_psar(bars: List[Bar], state: Optional[PSARState], af0: float, afmax: float) -> Tuple[PSARState, List[Dict[str, Any]]]:
    """
    Advance PSAR over a sequence of bars (ascending by open_ts).
    If state is None, bootstrap from the first 5 bars (needs at least 5).
    Returns updated state and per-bar trace for logging/plotting.
    """
    if not bars:
        raise ValueError("bars required")
    if state is None and len(bars) < 5:
        raise ValueError("Need at least 5 bars to bootstrap PSAR")

    traces: List[Dict[str, Any]] = []

    idx = 0
    if state is None:
        first = bars[:5]
        h0 = max(b[2] for b in first)
        l0 = min(b[3] for b in first)
        up = True
        ep = h0 if up else l0
        af = af0
        psar = l0 if up else h0
        idx = 5
    else:
        psar = float(state.psar)
        ep = float(state.ep)
        af = float(state.af)
        up = True if state.trend == "UP" else False
        # only process bars strictly newer than last_open_ms
        while idx < len(bars) and bars[idx][0] <= state.last_open_ms:
            idx += 1

    for i in range(idx, len(bars)):
        o, h, l, c = bars[i][1], bars[i][2], bars[i][3], bars[i][4]
        prev_psar = psar
        prev_ep = ep
        prev_up = up

        psar = prev_psar + af * (prev_ep - prev_psar)
        if up:
            # clamp to last 2 lows
            if i >= 1:
                psar = min(psar, bars[i - 1][3])
            if i >= 2:
                psar = min(psar, bars[i - 2][3])
        else:
            # clamp to last 2 highs
            if i >= 1:
                psar = max(psar, bars[i - 1][2])
            if i >= 2:
                psar = max(psar, bars[i - 2][2])

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

        traces.append({
            "ts_ms": bars[i][0],
            "psar": psar,
            "value": psar,
            "ep": ep,
            "af": af,
            "trend": "UP" if up else "DOWN",
            "o": o,
            "h": h,
            "l": l,
            "c": c,
        })

    if not traces:
        # nothing new; return previous state unchanged
        last_ts = state.last_open_ms if state else bars[-1][0]
        return PSARState(psar=psar, ep=ep, af=af, trend=("UP" if up else "DOWN"), last_open_ms=last_ts), traces

    last_open_ms = traces[-1]["ts_ms"]
    new_state = PSARState(psar=psar, ep=ep, af=af, trend=("UP" if up else "DOWN"), last_open_ms=last_open_ms)
    return new_state, traces


@dataclass
class ATRState:
    atr: float
    prev_close: float
    last_open_ms: int


def step_atr(bars: List[Bar], state: Optional[ATRState], period: int) -> Tuple[ATRState, List[Dict[str, Any]]]:
    """
    Wilder ATR stepper. Requires period+1 bars when bootstrapping.
    Returns updated state and per-bar trace with TR/ATR.
    """
    if period <= 1:
        raise ValueError("period must be > 1")
    if not bars:
        raise ValueError("bars required")

    traces: List[Dict[str, Any]] = []
    idx = 0

    if state is None:
        if len(bars) < period + 1:
            raise ValueError(f"Need at least {period+1} bars to bootstrap ATR(p={period})")
        # Bootstrap using first period TR average
        prev_close = bars[0][4]
        trs: List[float] = []
        for i in range(1, period + 1):
            _, _, h, l, c = bars[i]
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            trs.append(tr)
            prev_close = c
        atr = sum(trs) / float(period)
        idx = period + 1
    else:
        atr = float(state.atr)
        prev_close = float(state.prev_close)
        # advance only newer bars
        while idx < len(bars) and bars[idx][0] <= state.last_open_ms:
            idx += 1

    for i in range(idx, len(bars)):
        _, _, h, l, c = bars[i]
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        atr = ((atr * (period - 1)) + tr) / period
        prev_close = c
        traces.append({
            "ts_ms": bars[i][0],
            "atr": atr,
            "value": atr,
            "tr": tr,
            "o": bars[i][1],
            "h": h,
            "l": l,
            "c": c,
        })

    if not traces:
        last_ts = state.last_open_ms if state else bars[-1][0]
        return ATRState(atr=atr, prev_close=prev_close, last_open_ms=last_ts), traces

    new_state = ATRState(atr=atr, prev_close=prev_close, last_open_ms=traces[-1]["ts_ms"])
    return new_state, traces


def run_indicator(name: str, bars: List[Bar], cfg: dict, state: Optional[dict]) -> Tuple[dict, List[Dict[str, Any]], Optional[float]]:
    """
    Generic indicator dispatcher. Returns (state_dict, traces, latest_value).
    """
    if name == "psar":
        st = PSARState(**state) if state else None
        ns, traces = step_psar(bars, st, cfg.get("af", 0.02), cfg.get("max_af", 0.2))
        latest = traces[-1]["value"] if traces else (state["psar"] if state else None)
        return ns.__dict__, traces, latest
    if name == "atr":
        st = ATRState(**state) if state else None
        ns, traces = step_atr(bars, st, int(cfg.get("period", 14)))
        latest = traces[-1]["value"] if traces else (state["atr"] if state else None)
        return ns.__dict__, traces, latest
    raise ValueError(f"Unknown indicator: {name}")
