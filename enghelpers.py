from __future__ import annotations
import hashlib, math
from typing import Any
from typing import Callable, Dict, Tuple, Optional
from dataclasses import dataclass, field
import sys, json, threading
from common import *
from typing import Callable, List, Optional
from storage import Storage
import re
from datetime import datetime, timezone, timedelta
from binance_um import BinanceUM
import threading, time
from typing import Callable, Dict
from klines_cache import KlinesCache, rows_to_dataframe
import tempfile, os, io
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

if False:
    from bot_api import BotEngine


def render_chart(eng: BotEngine, symbol: str, timeframe: str, outdir: str = "/tmp") -> str:
    """Render a candlestick+volume chart from KlinesCache → PNG. Returns file path."""
    kc = eng.kc
    rows = kc.last_n(symbol, timeframe, n=200)
    if not rows:
        raise ValueError(f"No cached klines for {symbol} {timeframe}")

    df = rows_to_dataframe(rows)

    # Simple indicator (20-period MA)
    df["MA20"] = df["Close"].rolling(window=20).mean()

    # Dark theme (helps when charts are viewed in Telegram clients with dark UI).
    colors = mpf.make_marketcolors(
        up="tab:green",
        down="tab:red",
        wick={"up": "tab:green", "down": "tab:red"},
        edge="inherit",
        volume="inherit",
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=colors,
        y_on_right=True,
        facecolor="#0d1117",
        edgecolor="#111111",
        gridcolor="#1f2933",
        gridstyle="--",
    )

    fig, axlist = mpf.plot(
        df,
        type="candle",
        mav=(20,),
        volume=True,
        style=style,
        title=f"{symbol} {timeframe} – last {len(df)} candles",
        returnfig=True,
        figsize=(9, 6),
    )

    out_path = os.path.join(outdir, f"chart_{symbol}_{timeframe}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log().info("render_chart.success", out_path=out_path)
    return out_path


def parse_pair_or_single(eng: BotEngine, raw: str) -> Tuple[str, Optional[str]]:
    """
    Parse a trading spec into (num_symbol, den_symbol_or_None).

    Accepts:
      - "ETH/STRK"            -> ("ETHUSDT", "STRKUSDT")
      - "ETHUSDT/STRKUSDT"    -> ("ETHUSDT", "STRKUSDT")
      - "ETH" or "ETHUSDT"    -> ("ETHUSDT", None)     # single-leg
      - "ETH/" or "ETH/1"     -> ("ETHUSDT", None)     # explicit single-leg

    Rules:
      - Uses mc.normalize() for both sides.
      - Denominator tokens "1", "UNIT", "" mean single-leg.
      - Plain "USDT" as denominator is **not** allowed (ambiguous). Omit the denominator instead.

    Raises:
      ValueError with a precise message on bad input.
    """
    if not raw or not isinstance(raw, str):
        raise ValueError("Empty pair/symbol.")

    s = raw.strip().upper()

    # Single token (no slash): single-leg
    if "/" not in s:
        try:
            num = eng.mc.normalize(s)
        except Exception as e:
            raise ValueError(f"Unknown/unsupported symbol or base asset: {raw}") from e
        return num, None

    # Pair form
    left, right = (t.strip() for t in s.split("/", 1))

    if not left:
        raise ValueError("Missing numerator before '/'.")

    # Explicit single-leg hints on the right side
    if right in ("", "1", "UNIT", "-"):
        try:
            num = eng.mc.normalize(left)
        except Exception as e:
            raise ValueError(f"Unknown/unsupported numerator: {left}") from e
        return num, None

    if right == "USDT":
        # Denominator must be a PERPETUAL symbol, not the quote token.
        raise ValueError("Denominator 'USDT' is not valid. For single-leg, omit the denominator (e.g., 'ETH' or 'ETHUSDT').")

    # Proper pair: normalize both sides
    try:
        num = eng.mc.normalize(left)
    except Exception as e:
        raise ValueError(f"Unknown/unsupported numerator: {left}") from e

    try:
        den = eng.mc.normalize(right)
    except Exception as e:
        raise ValueError(f"Unknown/unsupported denominator: {right}") from e

    return num, den
