#!/usr/bin/env python3
# FILE: trailing_policies.py
from __future__ import annotations
from typing import Dict, Any, Optional


def _protective_stop_long(candidate: Optional[float], prev_stop: Optional[float]) -> Optional[float]:
    if candidate is None:
        return prev_stop
    if prev_stop is None:
        return candidate
    return max(prev_stop, candidate)


def _protective_stop_short(candidate: Optional[float], prev_stop: Optional[float]) -> Optional[float]:
    if candidate is None:
        return prev_stop
    if prev_stop is None:
        return candidate
    return min(prev_stop, candidate)


def eval_psar_lock(cfg: Dict[str, Any], position: Dict[str, Any], indicators: Dict[str, Dict[str, Any]],
                   prev_stop: Optional[float]) -> Dict[str, Any]:
    """
    Stop follows latest PSAR level; only ratchets in profit direction.
    cfg: {indicator: "psar"}
    position: {side: "LONG"/"SHORT", last_price: float}
    indicators: {"psar": {"value": float, "ts_ms": int, "raw": {...}}}
    """
    name = cfg.get("indicator", "psar")
    psar = indicators.get(name, {})
    lvl = psar.get("value")
    side = position["side"]
    stop = _protective_stop_long(lvl, prev_stop) if side == "LONG" else _protective_stop_short(lvl, prev_stop)
    return {
        "policy": "psar_lock",
        "suggested_stop": stop,
        "source_level": lvl,
        "indicator": name,
        "indicator_ts": psar.get("ts_ms"),
        "reason": "psar_follow",
    }


def eval_atr_trail(cfg: Dict[str, Any], position: Dict[str, Any], indicators: Dict[str, Dict[str, Any]],
                   prev_stop: Optional[float]) -> Dict[str, Any]:
    """
    ATR-based trail: stop = close - k*ATR (long) or close + k*ATR (short), then ratchet.
    cfg: {indicator: "atr", k: float}
    position: {side, last_price}
    indicators: {"atr": {"value": float, "ts_ms": int}}
    """
    name = cfg.get("indicator", "atr")
    k = float(cfg.get("k", 2.0))
    atr = indicators.get(name, {})
    atr_v = atr.get("value")
    px = position["last_price"]
    stop_candidate = None
    stop = prev_stop
    reason = "atr_missing"
    if atr_v is not None and px is not None:
        stop_candidate = px - k * atr_v if position["side"] == "LONG" else px + k * atr_v
        stop = _protective_stop_long(stop_candidate, prev_stop) if position["side"] == "LONG" else _protective_stop_short(stop_candidate, prev_stop)
        reason = "atr_trail"
    result = {
        "policy": "atr_trail",
        "suggested_stop": stop,
        "source_level": stop_candidate,
        "indicator": name,
        "indicator_ts": atr.get("ts_ms"),
        "reason": reason,
    }
    return result


POLICY_DISPATCH = {
    "psar_lock": eval_psar_lock,
    "atr_trail": eval_atr_trail,
}


def evaluate_policy(policy_name: str, cfg: Dict[str, Any], position: Dict[str, Any], indicators: Dict[str, Dict[str, Any]],
                    prev_stop: Optional[float]) -> Dict[str, Any]:
    fn = POLICY_DISPATCH.get(policy_name)
    if not fn:
        raise ValueError(f"Unknown policy: {policy_name}")
    return fn(cfg, position, indicators, prev_stop)
