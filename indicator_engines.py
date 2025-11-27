#!/usr/bin/env python3
# FILE: indicator_engines.py
"""
Shim module kept for backward compatibility.
Indicators now live in indicators.py, stop strategies in sstrats.py.
"""
from __future__ import annotations
from indicators import BaseIndicator, PSARIndicator, ATRIndicator, TrailingPercentIndicator, StopperIndicator, INDICATOR_CLASSES
from sstrats import StopStrategy, SSPSAR, SSATR, SSTRAT_CLASSES

__all__ = [
    "BaseIndicator",
    "PSARIndicator",
    "ATRIndicator",
    "TrailingPercentIndicator",
    "StopperIndicator",
    "INDICATOR_CLASSES",
    "StopStrategy",
    "SSPSAR",
    "SSATR",
    "SSTRAT_CLASSES",
]
