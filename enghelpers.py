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
# inside bot_api.py (or a separate enghelpers.py helper imported there)
import tempfile, os, io
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

if False:
    from bot_api import BotEngine


def latest_prices_for_positions(eng: BotEngine, rows) -> Dict[str,float]:
    """
    Use KlinesCache for a lightweight mark snapshot.
    # Falls back to API if cache empty.

    Caveats:
      - if price not found for symbol, fills with None
    """
    syms = {leg["symbol"] for r in rows for leg in eng.store.get_legs(r["position_id"]) if leg["symbol"]}
    out: Dict[str, float] = {}
    if not syms:
        return out

    # Prefer 1m if configured; else take the first configured tf; else default "1m".
    tfs = list(eng.cfg.KLINES_TIMEFRAMES or [])
    tf = "1m" if "1m" in tfs or not tfs else tfs[0]

    for s in syms:
        try:
            last = eng.kc.last_n(s, tf, n=1, include_live=True, asc=True)
            out[s] = float(last[0]["close"]) if last else None

            # if last:
            #     out[s] = float(last[0]["close"])
            #     continue
            # # fallback if cache is cold
            # now = Clock.now_utc_ms()
            # mk = eng.api.mark_price_klines(s, "1m", now-60_000, now)
            # if mk:
            #     out[s] = float(mk[-1][4])
        except Exception as e:
            log().debug("latest_prices.snapshot.fail", symbol=s, err=str(e))
    return out


# TODO get rid of this wrapper, treat None return values appropriately
def pnl_for_position(eng: BotEngine, position_id, prices) -> float:
    res = eng.positionbook.pnl_position(position_id, prices)
    return res["pnl_usd"] if res["ok"] else 0.0


def exec_positions(eng: BotEngine, args) -> str:
    """
    Pure function used by @positions
    args keys: status, what, limit, position_id, pair
    """
    store = eng.store
    reporter = eng.reporter
    positionbook = eng.positionbook

    status = args.get("status", "open")
    what   = args.get("what", "summary")
    limit  = int(args.get("limit", 100))
    pair   = args.get("pair")
    pid    = args.get("position_id")

    # rows by status
    if status == "open":
        rows = store.list_open_positions()
    elif status == "closed":
        rows = store.con.execute("SELECT * FROM positions WHERE status='CLOSED'").fetchall()
    else:
        rows = store.con.execute("SELECT * FROM positions").fetchall()

    # optional filters
    if pid:
        rows = [r for r in rows if r["position_id"] == pid]

    if pair:
        num, den = pair
        def _match(r):
            if den is None:
                legs = store.get_legs(r["position_id"])
                return any((lg["symbol"] or "").startswith(num) for lg in legs)
            return (r["num"].startswith(num) and r["den"].startswith(den))
        rows = [r for r in rows if _match(r)]

    # marks & render
    prices = latest_prices_for_positions(eng, rows)
    if what == "summary":
        return reporter.fmt_positions_summary(store, positionbook, rows, prices)
    if what == "full":
        return reporter.fmt_positions_full(store, positionbook, rows, prices, limit)
    if what == "legs":
        return reporter.fmt_positions_legs(store, positionbook, rows, prices, limit)

    # Signal mis-usage clearly (your earlier convention)
    raise RuntimeError('ChatGPT: unknown "what". Use one of summary|full|legs')


def render_chart(eng: BotEngine, symbol: str, timeframe: str, outdir: str = "/tmp") -> str:
    """Render a candlestick+volume chart from KlinesCache → PNG. Returns file path."""
    kc = eng.kc
    rows = kc.last_n(symbol, timeframe, n=200)
    if not rows:
        raise ValueError(f"No cached klines for {symbol} {timeframe}")

    df = rows_to_dataframe(rows)

    # Simple indicator (20-period MA)
    df["MA20"] = df["Close"].rolling(window=20).mean()

    style = mpf.make_mpf_style(base_mpf_style="binance", y_on_right=True)

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
