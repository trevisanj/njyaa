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
import numpy as np

if False:
    from bot_api import BotEngine


def last_cached_price(eng: "BotEngine", symbol: str) -> Optional[float]:
    kc = eng.kc
    tfs = eng.cfg.KLINES_TIMEFRAMES or ["1m"]
    for tf in tfs:
        r = kc.last_row(symbol, tf)
        if r and r["close"] is not None:
            return float(r["close"])
    return None


def _close_series(eng: BotEngine, symbol: str, timeframe: str, n: int):
    """Return a Close-price Series for a symbol/timeframe."""
    cols = eng.kc.last_n(symbol, timeframe, n=n)
    if not cols["close"]:
        raise ValueError(f"No cached klines for {symbol} {timeframe}")
    df = rows_to_dataframe(cols)
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


def klines_cache_summary(eng: BotEngine):
    """
    Return klines cache summary rows [(symbol, timeframe, n)] ordered by symbol + timeframe ms.
    """
    cur = eng.kc.con.execute(
        """
        SELECT k.symbol, k.timeframe, COUNT(*) AS n
        FROM klines k
        JOIN timeframe_ms t ON t.timeframe = k.timeframe
        GROUP BY k.symbol, k.timeframe
        ORDER BY k.symbol, t.ms
        """
    )
    return cur.fetchall()


def render_chart(eng: BotEngine, symbol: str, timeframe: str, n: int = 200, outdir: str = "/tmp") -> str:
    """Render a candlestick+volume chart from KlinesCache → PNG. Returns file path."""
    kc = eng.kc
    cols = kc.last_n(symbol, timeframe, n=n)
    if not cols["close"]:
        raise ValueError(f"No cached klines for {symbol} {timeframe}")

    df = rows_to_dataframe(cols)

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

    fname = f"chart_NJYAA_{(series.name or pair_or_symbol).replace('/', '-')}_{timeframe}_{n}.png"
    out_path = os.path.join(outdir, fname)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log().info("render_ratio_chart.success", out_path=out_path, pair=pair_or_symbol, timeframe=timeframe, n=n)
    return out_path


def render_indicator_history_chart(eng: "BotEngine", thinker_id: int, position_id: int, indicator_name: str, symbol: str,
                                   timeframe: str = "1m", n: int = 300, outdir: str = "/tmp") -> str:
    """Overlay price closes with recorded indicator history."""
    cols = eng.kc.last_n(symbol, timeframe, n=n, include_live=True, asc=True)
    if not cols["close"]:
        raise ValueError(f"No klines for {symbol} {timeframe}")
    df = rows_to_dataframe(cols)
    hist = eng.ih.list_history(thinker_id, position_id, indicator_name, limit=5000, columns=None)
    if not hist["ts_ms"]:
        raise ValueError("No history rows")
    hdf = pd.DataFrame(hist)
    hdf["dt"] = pd.to_datetime(hdf["ts_ms"], unit="ms")
    df_idx = pd.to_datetime(df.index)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_idx, df["Close"], label="Close", color="steelblue", linewidth=1.4)
    ax.plot(hdf["dt"], hdf["value"], label=indicator_name, color="tomato", linewidth=1.2)
    ax.set_title(f"{symbol} {indicator_name} history (pos {position_id})")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fname = f"hist_{indicator_name}_{symbol}_{timeframe}_{position_id}.png".replace("/", "-")
    out_path = os.path.join(outdir, fname)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log().info("render_indicator_history_chart", path=out_path, position_id=position_id, indicator=indicator_name)
    return out_path


def pnl_time_series(eng: "BotEngine", position_ids: list[int], timeframe: str,
                    start_ms: int | None, end_ms: int | None):
    """
    Build a PnL time series per position (and total) using cached klines.

    Returns:
      df: pandas.DataFrame indexed by ts, columns per position_id + "TOTAL" (floats, NaN when missing)
      report: dict with missing/issue details
    meta: dict with time bounds used
    """
    assert eng.store and eng.kc and eng.positionbook
    tfm = tf_ms(timeframe)
    now_ms = Clock.now_utc_ms()

    positions = []
    missing_positions = []
    for pid in position_ids:
        row = eng.store.get_position(pid)
        if row:
            positions.append(row)
        else:
            missing_positions.append(pid)
    if not positions:
        raise ValueError("No positions to process.")

    legs_by_pid = {int(p["position_id"]): eng.store.get_legs(p["position_id"]) for p in positions}
    pos_meta: dict[int, dict] = {}
    open_bounds: list[int] = []
    close_bounds: list[int] = []
    for p in positions:
        pid = int(p["position_id"])
        num = str(p["num"])
        den = p["den"]
        open_ms = int(p["user_ts"] or p["created_ts"])
        closed_raw = p["closed_ts"] if "closed_ts" in p.keys() else None
        closed_ms = int(closed_raw) if closed_raw is not None else None
        open_bounds.append(open_ms)
        close_bounds.append(closed_ms or now_ms)
        lbl = f"{pid} {fmt_pair(num, den)}"
        pos_meta[pid] = {
            "label": lbl,
            "num": num,
            "den": den,
            "open_ms": open_ms,
            "closed_ms": closed_ms,
            "status": p["status"],
        }

    # window based on union of position lifetimes unless explicitly overridden
    end_ts = end_ms or (max(close_bounds) if close_bounds else now_ms)
    start_ts = start_ms or (min(open_bounds) if open_bounds else end_ts - tfm * 200)
    missing_qty_entry: list[int] = []
    symbols = set()
    for pid, legs in legs_by_pid.items():
        for lg in legs:
            if lg["qty"] is None or lg["entry_price"] is None:
                missing_qty_entry.append(pid)
                break
        symbols.update(lg["symbol"] for lg in legs if lg["symbol"])

    price_rows: dict[str, list[tuple[int, float]]] = {}
    symbols_missing_klines = []
    for sym in symbols:
        cols = eng.kc.range_by_close(sym, timeframe, start_ts - tfm, end_ts, include_live=False, columns=["close_ts", "close"])
        if not cols["close_ts"]:
            symbols_missing_klines.append(sym)
            continue
        price_rows[sym] = list(zip(cols["close_ts"], cols["close"]))

    # Build time axis from available klines
    ts_set = set()
    for rows in price_rows.values():
        ts_set.update(int(ts) for ts, _ in rows)
    timeline = sorted(ts_set)
    if not timeline:
        raise ValueError("No klines in requested window.")

    # Build per-symbol price lookup at close_ts
    price_map = {sym: {int(ts): float(close) for ts, close in rows} for sym, rows in price_rows.items()}

    data = {pid: [] for pid in legs_by_pid}
    total = []
    missing_prices: list[dict] = []

    for ts in timeline:
        tick_total = 0.0
        for pid, legs in legs_by_pid.items():
            # skip if leg data incomplete
            if pid in missing_qty_entry:
                data[pid].append(np.nan)
                continue
            span = pos_meta[pid]
            open_ms = span["open_ms"]
            closed_ms = span["closed_ms"]
            if ts < open_ms:
                data[pid].append(np.nan)
                continue
            if closed_ms is not None and ts > closed_ms:
                data[pid].append(np.nan)
                continue
            prices_for_pid = {}
            missing_any = False
            for lg in legs:
                sym = lg["symbol"]
                price = price_map.get(sym, {}).get(ts)
                if price is None:
                    missing_prices.append({"position_id": pid, "symbol": sym, "ts": ts})
                    missing_any = True
                    break
                prices_for_pid[sym] = price
            if missing_any:
                data[pid].append(np.nan)
                continue
            pnl = sum((prices_for_pid[lg["symbol"]] - float(lg["entry_price"])) * float(lg["qty"]) for lg in legs)
            data[pid].append(pnl)
            tick_total += pnl
        total.append(tick_total if data else np.nan)

    df = pd.DataFrame(data, index=pd.to_datetime(timeline, unit="ms"))
    df["TOTAL"] = total

    report = {
        "missing_positions": missing_positions,
        "missing_qty_entry": sorted(set(missing_qty_entry)),
        "symbols_missing_klines": symbols_missing_klines,
        "missing_prices": missing_prices,
    }
    meta = {"start_ms": start_ts, "end_ms": end_ts, "timeframe": timeframe, "positions": pos_meta}
    return df, report, meta


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
