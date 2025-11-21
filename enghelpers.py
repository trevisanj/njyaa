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


def _close_series(eng: BotEngine, symbol: str, timeframe: str, n: int):
    """Return a Close-price Series for a symbol/timeframe."""
    rows = eng.kc.last_n(symbol, timeframe, n=n)
    if not rows:
        raise ValueError(f"No cached klines for {symbol} {timeframe}")
    df = rows_to_dataframe(rows)
    s = df["Close"].copy()
    s.name = symbol
    return s


def divide_series(num: pd.Series, den: pd.Series, name: Optional[str] = None) -> pd.Series:
    """Construct ratio series with index intersection."""
    aligned_num, aligned_den = num.align(den, join="inner")
    if aligned_num.empty:
        raise ValueError("No overlapping timestamps between numerator and denominator series.")
    ratio = aligned_num / aligned_den
    ratio.name = name or f"{num.name}/{den.name}"
    return ratio


def pair_close_series(eng: BotEngine, pair_or_symbol: str, timeframe: str, n: int) -> pd.Series:
    """Return Close-price series for NUM[/DEN], dividing when denominator is provided."""
    num, den = parse_pair_or_single(eng, pair_or_symbol)
    num_series = _close_series(eng, num, timeframe, n=n)
    if not den:
        return num_series
    den_series = _close_series(eng, den, timeframe, n=n)
    return divide_series(num_series, den_series, name=f"{num}/{den}")


def render_chart(eng: BotEngine, symbol: str, timeframe: str, n: int = 200, outdir: str = "/tmp") -> str:
    """Render a candlestick+volume chart from KlinesCache → PNG. Returns file path."""
    kc = eng.kc
    rows = kc.last_n(symbol, timeframe, n=n)
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
        title=f"{symbol} {timeframe} – last {len(df)} of {n} candles",
        returnfig=True,
        figsize=(9, 6),
    )

    out_path = os.path.join(outdir, f"chart_{symbol}_{timeframe}_{n}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log().info("render_chart.success", out_path=out_path)
    return out_path


def render_ratio_chart(eng: BotEngine, pair_or_symbol: str, timeframe: str, n: int = 200,
                       outdir: str = "/tmp") -> str:
    """Render a Close-price line + MA for a symbol or ratio."""
    series = pair_close_series(eng, pair_or_symbol, timeframe, n=n)
    ma = series.rolling(window=20).mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(series.index, series, label=series.name or pair_or_symbol, color="deepskyblue", linewidth=1.6)
    ax.plot(ma.index, ma, label="MA20", color="orange", linewidth=1.2)
    ax.set_title(f"{series.name or pair_or_symbol} {timeframe} – last {len(series)} of {n} closes")
    ax.set_xlabel("Time")
    ax.set_ylabel("Close")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    fname = f"chart_rv_{(series.name or pair_or_symbol).replace('/', '-')}_{timeframe}_{n}.png"
    out_path = os.path.join(outdir, fname)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log().info("render_ratio_chart.success", out_path=out_path, pair=pair_or_symbol, timeframe=timeframe, n=n)
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
