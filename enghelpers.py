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
import tempfile, os, io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import pandas as pd
import numpy as np

if False:
    from bot_api import BotEngine


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
    df = kc.last_n(symbol, timeframe, n=n, fmt="ohlcv")
    if df.empty:
        raise ValueError(f"No cached klines for {symbol} {timeframe}")

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
    num, den = parse_pair_or_single(eng, pair_or_symbol)
    tfm = tf_ms(timeframe)
    now_ms = Clock.now_utc_ms()
    start_ts = max(0, now_ms - n * tfm)
    df = eng.kc.pair_bars(num, den, timeframe, start_ts, None)
    if df.empty:
        raise ValueError(f"No klines for {pair_or_symbol} {timeframe}")
    series = df["Close"]
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
                                   timeframe: str = "1d", n: int = 300, outdir: str = "/tmp") -> str:
    """Overlay price closes with recorded indicator history."""
    num, den = parse_pair_or_single(eng, symbol)
    tfm = tf_ms(timeframe)
    now_ms = Clock.now_utc_ms()
    start_ts = max(0, now_ms - n * tfm)
    price_df = eng.kc.pair_bars(num, den, timeframe, start_ts, None)
    if price_df.empty:
        raise ValueError(f"No klines for {symbol} {timeframe}")

    hdf = eng.ih.range_by_ts(thinker_id, position_id, indicator_name, start_open_ts=start_ts, fmt="dataframe")
    if hdf is None or hdf.empty:
        raise ValueError("No history rows")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(price_df.index, price_df["Close"], label=f"{symbol} Close", color="steelblue", linewidth=1.4)
    ax.plot(hdf.index, hdf["value"], label=indicator_name, color="tomato", linestyle="none", marker="x", markersize=6)
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


def render_chart_ind(eng: "BotEngine", thinker_id: int, position_id: int, indicator_names: list[str],
                     symbol: str, timeframe: str, pos_start_ms: int, pos_end_ms: int,
                     start_ms: int | None = None, end_ms: int | None = None, n: Optional[int] = None,
                     outdir: str = "/tmp") -> str:
    """
    Plot price candles and multiple indicator histories for a position.
    """
    start_bound = None
    end_bound = None
    if n is not None:
        start_bound = start_ms if start_ms is not None else pos_start_ms
        end_bound = end_ms if end_ms is not None else pos_end_ms
    assert end_bound is None or start_bound is None or start_bound <= end_bound, "start_ts must be before end_ts"

    num, den = parse_pair_or_single(eng, symbol)
    # --- fetch all indicator dfs and collect bounds
    dfs = []
    # ts bounds for the klines will be defined by indicator data retrieved
    min_ts, max_ts = None, None
    for name in indicator_names:
        df = eng.ih.window(thinker_id, position_id, name, start_ts=start_bound, end_ts=end_bound,
                           fmt="dataframe", n=n)
        if df is None or df.empty:
            continue
        dfs.append((name, df))
        ts_min = int(df.index[0].value // 1_000_000)
        ts_max = int(df.index[-1].value // 1_000_000)
        min_ts = ts_min if min_ts is None else min(min_ts, ts_min)
        max_ts = ts_max if max_ts is None else max(max_ts, ts_max)
    if not dfs:
        raise ValueError("No indicator history rows")

    price_df = eng.kc.pair_bars(num, den, timeframe, min_ts, max_ts or None)
    if price_df.empty:
        raise ValueError(f"No klines for {symbol} {timeframe}")

    fig, axlist = mpf.plot(price_df, type="candle", returnfig=True,
                           title=f"{symbol} indicators ({', '.join(indicator_names)})", show_nontrading=True)
    ax = axlist[0] if isinstance(axlist, (list, tuple)) else axlist
    for name, df in dfs:
        x = mdates.date2num(df.index.to_pydatetime())
        ax.scatter(x, df["value"], marker="x", s=20, label=name)
    ax.legend()
    out_path = os.path.join(outdir, f"chart_ind_{thinker_id}_{position_id}_{symbol}_{timeframe}.png".replace("/", "-"))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log().info("render_chart_ind", path=out_path, position_id=position_id, indicators=indicator_names)
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
        position = eng.store.get_position(pid)
        if position:
            positions.append(position)
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
        closed_raw = p["closed_ts"] if "closed_ts" in p.keys() else None
        closed_ms = int(closed_raw) if closed_raw is not None else None
        open_bounds.append(p.user_ts)
        close_bounds.append(closed_ms or now_ms)
        lbl = f"{pid} {p.get_pair()}"
        pos_meta[pid] = {
            "label": lbl,
            "num": p.num,
            "den": p.den,
            "open_ms": p.user_ts,
            "closed_ms": closed_ms,
            "status": p["status"],
        }

    # window based on union of position lifetimes unless explicitly overridden
    end_ts = end_ms or (max(close_bounds) if close_bounds else now_ms)
    start_ts = start_ms or (min(open_bounds) if open_bounds else end_ts - tfm * 200)
    end_arg = None if (end_ms is None and end_ts == now_ms) else end_ts
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
        cols = eng.kc.range_by_close(sym, timeframe, start_ts - tfm, end_arg, include_live=False, columns=["close_ts", "close"])
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
