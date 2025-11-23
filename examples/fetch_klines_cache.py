#!/usr/bin/env python3
# Quick helper to pull a slice of klines from the cache into a variable.
# Uses raw SQLite for a fast dict-of-lists result.

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Sequence

# ---- hard-wired knobs ----
ROOT = Path(__file__).resolve().parent.parent
KLINES_CACHE_DB_PATH = ROOT / "njyaa_cache.sqlite"
SYMBOL = "BTCUSDT"
TIMEFRAME = "1d"
N = 200
INCLUDE_LIVE = False
COLUMNS = [
    "symbol",
    "timeframe",
    "open_ts",
    "close_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "finalized",
]


def fetch_columns_direct(db_path: Path, symbol: str, timeframe: str, n: int, include_live: bool) -> tuple[List[sqlite3.Row], Dict[str, List[Any]]]:
    """Fetch klines as both row list and column dict using raw SQLite (fast)."""
    con = sqlite3.connect(db_path)
    live_clause = "" if include_live else "AND finalized=1"
    select_cols = ",".join(COLUMNS)
    q = f"""
        SELECT {select_cols}
        FROM klines
        WHERE symbol=? AND timeframe=? {live_clause}
        ORDER BY open_ts DESC
        LIMIT ?
    """
    rows_desc = con.execute(q, (symbol, timeframe, n)).fetchall()  # list of tuples
    con.close()
    rows = list(reversed(rows_desc))  # oldest→newest
    if rows:
        transposed: Sequence[Sequence[Any]] = list(zip(*rows))
        cols: Dict[str, List[Any]] = {k: list(col) for k, col in zip(COLUMNS, transposed)}
    else:
        cols = {k: [] for k in COLUMNS}
    # rebuild Row-like dicts for callers that expect mapping access
    row_objs = [dict(zip(COLUMNS, r)) for r in rows]
    return row_objs, cols


def main():
    assert KLINES_CACHE_DB_PATH.exists(), f"Cache DB missing: {KLINES_CACHE_DB_PATH}"

    history, cols = fetch_columns_direct(
        db_path=KLINES_CACHE_DB_PATH,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        n=N,
        include_live=INCLUDE_LIVE,
    )

    # keep the variable in scope; print a tiny summary for convenience
    print(f"history: {len(history)} rows for {SYMBOL} {TIMEFRAME} (include_live={INCLUDE_LIVE})")
    print({k: len(v) for k, v in cols.items()})
    if history:
        first = history[0]
        last = history[-1]
        print(f"range: {first['open_ts']} -> {last['close_ts']}")

    return history, cols


if __name__ == "__main__":
    history, cols = main()
