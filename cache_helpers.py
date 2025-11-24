import sqlite3, time, math
from typing import Iterable, List, Tuple, Optional, Dict, Any, TYPE_CHECKING, Sequence
from dataclasses import dataclass
import threading
from common import tf_ms
import pandas as pd

"""
Cache helper utilities shared by caches (klines, indicator history, etc.).

Provides fast conversions between SQLite rows/columnar dicts and pandas DataFrames
with a consistent Date index derived from timestamp columns.
"""

OHLCV_COLS: List[str] = ["open_ts", "open", "high", "low", "close", "volume"]


def rows_to_ohlcv(rows, columns=None) -> pd.DataFrame:
    """
    Convert rows to OHLCV DataFrame with Date index.

    Args:
          rows: [sqlite3.Row, ...]
          columns: defaults to OHLCV_COLS
    """
    if columns is None:
        columns = OHLCV_COLS
    columnar = rows_to_columnar(rows, columns)
    return columnar_to_ohlcv(columnar)


def columnar_to_ohlcv(columnar: dict[str, list[Any]]) -> pd.DataFrame:
    """Columnar → OHLCV DataFrame with Date index."""
    if not columnar:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"]).set_index(pd.DatetimeIndex([], name="Date"))

    df = pd.DataFrame(data={s[0].upper() + s[1:]: columnar[s] for s in ["open", "high", "low", "close", "volume"]},
                      index=pd.to_datetime(columnar["open_ts"], unit="ms"))
    df.index.name = "Date"
    return df


def rows_to_columnar(rows: Sequence[Sequence[Any]], columns: Sequence[str]) -> Dict[str, List[Any]]:
    """Convert sequence of DB rows to dict-of-lists using provided column order."""
    if not rows:
        cols = list(columns) if columns is not None else []
        return {k: [] for k in cols}
    cols = list(columns) if columns is not None else list(rows[0].keys())
    data = zip(*rows)
    return {k: list(col) for k, col in zip(cols, data)}


def rows_to_generic_df(rows, columns: Optional[List[str]] = None, ts_name: str = "open_ts") -> pd.DataFrame:
    """
    Minimal converter: rows/columnar → DataFrame, sets Date index from ts_name.

    Keeps all other columns intact; raises if the timestamp column is missing.
    """
    if isinstance(rows, dict):
        columnar = rows
    else:
        if not rows:
            return pd.DataFrame().set_index(pd.DatetimeIndex([], name="Date"))
        if columns is None:
            try:
                columns = list(rows[0].keys())
            except Exception:
                columns = None
        cols = columns or []
        columnar = rows_to_columnar(rows, cols)

    if ts_name not in columnar:
        raise ValueError(f"rows_to_generic_df: missing timestamp column {ts_name}")
    idx = pd.to_datetime(columnar.pop(ts_name), unit="ms")
    df = pd.DataFrame(columnar)
    df.index = idx
    df.index.name = "Date"
    return df
