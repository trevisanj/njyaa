#!/usr/bin/env python3
# FILE: klines_cache.py

from __future__ import annotations
import sqlite3, time, math
from typing import Iterable, List, Tuple, Optional, Dict, Any, TYPE_CHECKING, Sequence
from dataclasses import dataclass
import threading
from common import tf_ms
import pandas as pd
from cache_helpers import rows_to_ohlcv, rows_to_columnar, rows_to_generic_df

if TYPE_CHECKING:
    from bot_api import BotEngine


# Default column set for klines queries (ts + OHLCV + finalized/meta)
KLINE_COLS: List[str] = [
    # "symbol",
    # "timeframe",
    "open_ts",
    "close_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    # "finalized",
]



@dataclass
class KlineRow:
    """
    Immutable row representing a single kline/candle.

    Fields:
        symbol: Market symbol, e.g. 'ETHUSDT'.
        timeframe: Interval string (e.g. '1m').
        open_ts: Candle open timestamp (ms, inclusive).
        close_ts: Candle close timestamp (ms, exclusive).
        o, h, l, c: Open/High/Low/Close prices (float).
        v: Base-asset volume.
        finalized: 1 if the candle is closed; 0 if it’s the live candle.

    Notes:
        - `close_ts` is stored explicitly for fast “at-close” queries and backtests.
        - In production, only the most recent candle for a (symbol,timeframe) should be live.
    """
    symbol: str
    timeframe: str
    open_ts: int
    close_ts: int
    o: float
    h: float
    l: float
    c: float
    v: float
    finalized: int  # 0/1


class KlinesCache:
    """
    Single-table kline cache.

    Design:
        - PK: (symbol, timeframe, open_ts)
        - Secondary index: (symbol, timeframe, close_ts)
        - Stores a 'finalized' bit so you can query "closed-only" vs "include live" quickly.

    Typical usage:
        cache = KlinesCache(eng)
        last_open = cache.latest_open_ts("ETHUSDT", "1m")
        rows = cache.last_n("ETHUSDT", "1m", n=500, include_live=False)  # returns columnar dict
    """

    def __init__(self, eng: BotEngine):
        """Open/create the SQLite cache."""
        path_ = eng.cfg.KLINES_CACHE_DB_PATH
        assert path_, "AppConfig must define KLINES_CACHE_DB_PATH"
        self.eng = eng
        self.cfg = eng.cfg
        self.api = eng.api
        self.db_path = path_

        self.con = sqlite3.connect(self.db_path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self._ensure_schema()

        # Mechanism to keep "now" consistent within batch normalization
        self._last_now_ms = None

        # DB lock for concurrent writers (WAL allows concurrent readers)
        self._db_lock = threading.Lock()

    # ---------- schema ----------
    def _ensure_schema(self):
        c = self.con.cursor()
        c.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS klines(
          symbol     TEXT NOT NULL,
          timeframe  TEXT NOT NULL,
          open_ts    INTEGER NOT NULL,
          close_ts   INTEGER NOT NULL,
          open       REAL NOT NULL,
          high       REAL NOT NULL,
          low        REAL NOT NULL,
          close      REAL NOT NULL,
          volume     REAL NOT NULL,
          finalized  INTEGER NOT NULL,
          PRIMARY KEY(symbol, timeframe, open_ts)
        );
        CREATE INDEX IF NOT EXISTS idx_klines_close
          ON klines(symbol, timeframe, close_ts);
        CREATE INDEX IF NOT EXISTS ix_klines_sym_tf_ts
          ON klines(symbol, timeframe, open_ts DESC);
        CREATE INDEX IF NOT EXISTS ix_klines_final_close
          ON klines(finalized, close_ts);
        CREATE TABLE IF NOT EXISTS timeframe_ms(
          timeframe TEXT PRIMARY KEY,
          ms INTEGER NOT NULL
        );
        """)
        c.executemany(
            "INSERT OR IGNORE INTO timeframe_ms(timeframe, ms) VALUES (?,?)",
            [
                ("1m", 60_000),
                ("3m", 180_000),
                ("5m", 300_000),
                ("15m", 900_000),
                ("30m", 1_800_000),
                ("1h", 3_600_000),
                ("2h", 7_200_000),
                ("4h", 14_400_000),
                ("6h", 21_600_000),
                ("8h", 28_800_000),
                ("12h", 43_200_000),
                ("1d", 86_400_000),
                ("3d", 259_200_000),
                ("1w", 604_800_000),
            ],
        )
        self.con.commit()

    def _ensure_api(self):
        """
        Ensure the cache has access to an exchange API when required.
        """
        if self.api is None:
            raise RuntimeError("This operation requires an API instance (eng.api is None).")

    def close(self):
        """Close the underlying SQLite connection (best-effort)."""
        try:
            self.con.close()
        except:
            pass

    # ---------- reads (now returning sqlite3.Row dict-like) ----------
    def count(self, symbol: str | None = None, timeframe: str | None = None) -> int:
        """
        Count cached klines.

        Parameters
        ----------
        symbol : str, optional
            If provided, only count rows for this symbol.
        timeframe : str, optional
            If provided, only count rows for this timeframe.

        Returns
        -------
        int
            Number of matching rows.
        """
        cur = self.con.cursor()
        sql = "SELECT COUNT(*) FROM klines"
        args: List[Any] = []
        where: List[str] = []
        if symbol:
            where.append("symbol = ?")
            args.append(symbol.upper())
        if timeframe:
            where.append("timeframe = ?")
            args.append(timeframe)
        if where:
            sql += " WHERE " + " AND ".join(where)
        cur.execute(sql, args)
        return int(cur.fetchone()[0])

    def last_cached_price(self, symbol: str) -> Optional[float]:
        """
        Latest close for the given symbol scanning configured timeframes (finer first).
        """
        tfs = self.cfg.KLINES_TIMEFRAMES or ["1m"]
        for tf in reversed(tfs):
            r = self.last_row(symbol, tf)
            if r and r["close"] is not None:
                return float(r["close"])
        return None

    def latest_open_ts(self, symbol: str, timeframe: str) -> Optional[int]:
        """
        Get the latest known candle’s open_ts for (symbol,timeframe).

        Returns:
            open_ts in ms, or None if no rows yet.

        Use cases:
            - Incremental fetch: pass this to your fetcher.
        """
        r = self.con.execute(
            """SELECT open_ts FROM klines
               WHERE symbol=? AND timeframe=?
               ORDER BY open_ts DESC LIMIT 1""",
            (symbol, timeframe)
        ).fetchone()
        return int(r["open_ts"]) if r else None

    def last_n(self, symbol: str, timeframe: str, n: int, columns: Optional[List[str]] = None,
               include_live: bool = True, asc: bool = True, fmt: str = "columnar"):
        """
        Fetch the last N klines for (symbol,timeframe).

        fmt: "columnar" (dict of lists), "dataframe", or "ohlcv".
        """
        cols = columns or KLINE_COLS
        live_clause = "" if include_live else "AND finalized=1"
        select_cols = ",".join(cols)
        q = f"""
            SELECT {select_cols}
            FROM klines
            WHERE symbol=? AND timeframe=? {live_clause}
            ORDER BY open_ts {'ASC' if asc else 'DESC'}
            LIMIT ?
        """
        rows = self.con.execute(q, (symbol, timeframe, n)).fetchall()
        if not asc:
            rows = list(reversed(rows))
        if fmt == "dataframe":
            return rows_to_generic_df(rows, columns=cols)
        if fmt == "ohlcv":
            return rows_to_ohlcv(rows, columns=cols)
        return rows_to_columnar(rows, cols)

    def window(self, symbol: str, timeframe: str, start_open_ts: Optional[int] = None,
               end_open_ts: Optional[int] = None, columns: Optional[List[str]] = None,
               include_live: bool = True, asc: bool = True, fmt: str = "columnar", n: Optional[int] = None):
        """
        Return klines either bounded by timestamps (open) or, when n is provided, the latest n rows.
        """
        if n is None:
            return self.range_by_open(symbol, timeframe, start_open_ts, end_open_ts, columns=columns,
                                      include_live=include_live, asc=asc, fmt=fmt)
        return self.last_n(symbol, timeframe, n, columns=columns, include_live=include_live, asc=asc, fmt=fmt)

    def range_by_open(self, symbol: str, timeframe: str, start_open_ts: Optional[int] = None,
                      end_open_ts: Optional[int] = None, columns: Optional[List[str]] = None,
                      include_live: bool = True, asc: bool = True, fmt: str = "columnar"):
        """Get klines in [start_open_ts, end_open_ts] by open timestamp."""
        cols = columns or KLINE_COLS
        select_cols = ",".join(cols)
        where = ["symbol=?", "timeframe=?"]
        args: list[Any] = [symbol, timeframe]
        if start_open_ts is not None:
            where.append("open_ts>=?")
            args.append(start_open_ts)
        if end_open_ts is not None:
            where.append("open_ts<=?")
            args.append(end_open_ts)
        if not include_live:
            where.append("finalized=1")
        q = f"""SELECT {select_cols}
                FROM klines
                WHERE {' AND '.join(where)}
                ORDER BY open_ts {'ASC' if asc else 'DESC'}"""
        rows = self.con.execute(q, args).fetchall()
        if fmt == "dataframe":
            return rows_to_generic_df(rows, columns=cols)
        if fmt == "ohlcv":
            return rows_to_ohlcv(rows, columns=cols)
        return rows_to_columnar(rows, cols)

    def range_by_close(self, symbol: str, timeframe: str, start_close_ts: int, end_close_ts: Optional[int] = None,
                       columns: Optional[List[str]] = None, include_live: bool = True, asc: bool = True,
                       fmt: str = "columnar"):
        """Get klines whose close_ts falls in [start_close_ts, end_close_ts]."""
        cols = columns or KLINE_COLS
        select_cols = ",".join(cols)
        where = ["symbol=?", "timeframe=?", "close_ts>=?"]
        args: list[Any] = [symbol, timeframe, start_close_ts]
        if end_close_ts is not None:
            where.append("close_ts<=?")
            args.append(end_close_ts)
        if not include_live:
            where.append("finalized=1")
        q = f"""SELECT {select_cols}
                FROM klines
                WHERE {' AND '.join(where)}
                ORDER BY open_ts {'ASC' if asc else 'DESC'}"""
        rows = self.con.execute(q, args).fetchall()
        if fmt == "dataframe":
            return rows_to_generic_df(rows, columns=cols)
        if fmt == "ohlcv":
            return rows_to_ohlcv(rows, columns=cols)
        return rows_to_columnar(rows, cols)

    def pair_bars(self, num: str, den: Optional[str], timeframe: str, start_open_ts: Optional[int] = None,
                  end_open_ts: Optional[int] = None, include_live: bool = True, n: Optional[int] = None):
        """
        Return OHLC for num[/den], aligned by open_ts. If n is provided, fetch the latest n bars ignoring timestamp
        bounds.
        """
        getter = lambda sym: self.window(sym, timeframe, start_open_ts, end_open_ts, include_live=include_live,
                                         fmt="ohlcv", n=n)
        if not den:
            return getter(num)

        _num_df = getter(num)
        _den_df = getter(den)

        if _num_df.empty or _den_df.empty:
            return _num_df

        out_df, den_df = _num_df.align(_den_df, join="inner")

        if out_df.empty:
            return out_df

        for col in ("Open", "High", "Low", "Close"):
            out_df[col] = out_df[col] / den_df[col]
        out_df["Low"] = out_df[["Low", "Open", "Close"]].min(axis=1)
        out_df["High"] = out_df[["High", "Open", "Close"]].max(axis=1)
        out_df["Volume"] = out_df["Volume"]
        return out_df

    def last_row(self, symbol: str, timeframe: str) -> Optional[sqlite3.Row]:
        """
        Get the newest single row for (symbol,timeframe).

        Returns:
            sqlite3.Row (dict-like) or None.
        """
        r = self.con.execute(
            """SELECT symbol, timeframe, open_ts, close_ts,
                      open, high, low, close, volume, finalized
               FROM klines
               WHERE symbol=? AND timeframe=?
               ORDER BY open_ts DESC LIMIT 1""",
            (symbol, timeframe)
        ).fetchone()
        return r

    # ---------- writes ----------
    def upsert(self, row: KlineRow):
        """
        Insert or update a single kline row by (symbol,timeframe,open_ts).

        Overwrites: close_ts, O/H/L/C/V, finalized.
        """
        q = """INSERT INTO klines(symbol,timeframe,open_ts,close_ts,open,high,low,close,volume,finalized)
               VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(symbol,timeframe,open_ts) DO UPDATE SET
                 close_ts=excluded.close_ts,
                 open=excluded.open, high=excluded.high, low=excluded.low,
                 close=excluded.close, volume=excluded.volume,
                 finalized=excluded.finalized"""
        with self._db_lock:
            self.con.execute(q, (row.symbol,row.timeframe,row.open_ts,row.close_ts,
                                 row.o,row.h,row.l,row.c,row.v,row.finalized))
            self.con.commit()

    def bulk_upsert(self, rows: Iterable[KlineRow]):
        """
        Insert/update many klines at once (fast path).
        """
        q = """INSERT INTO klines(symbol,timeframe,open_ts,close_ts,open,high,low,close,volume,finalized)
               VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(symbol,timeframe,open_ts) DO UPDATE SET
                 close_ts=excluded.close_ts,
                 open=excluded.open, high=excluded.high, low=excluded.low,
                 close=excluded.close, volume=excluded.volume,
                 finalized=excluded.finalized"""
        payload = [
            (r.symbol,r.timeframe,r.open_ts,r.close_ts,r.o,r.h,r.l,r.c,r.v,r.finalized)
            for r in rows
        ]
        if not payload:
            return
        with self._db_lock:
            self.con.executemany(q, payload)
            self.con.commit()

    def finalize_due(self, now_ms: Optional[int] = None) -> int:
        """
        Flip live bars to finalized where their close_ts has passed.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        with self._db_lock:
            cur = self.con.cursor()
            cur.execute("""UPDATE klines
                           SET finalized=1
                           WHERE finalized=0 AND close_ts <= ?""", (now_ms,))
            self.con.commit()
            return cur.rowcount

    def prune_keep_last_n(self, keep_n: int,
                          symbol: Optional[str] = None,
                          timeframe: Optional[str] = None) -> int:
        """
        Keep only the newest N candles per group.
        """
        assert (symbol and timeframe) or (not symbol and not timeframe), \
            "Either both symbol and timeframe must be provided, or neither."
        with self._db_lock:
            cur = self.con.cursor()
            if symbol and timeframe:
                cur.execute("""
                WITH ranked AS (
                  SELECT open_ts
                  FROM klines
                  WHERE symbol=? AND timeframe=?
                  ORDER BY open_ts DESC
                  LIMIT -1 OFFSET ?
                )
                DELETE FROM klines
                WHERE symbol=? AND timeframe=? AND open_ts IN (SELECT open_ts FROM ranked)
                """, (symbol, timeframe, keep_n, symbol, timeframe))
            else:
                cur.execute("""
                DELETE FROM klines
                WHERE (symbol,timeframe,open_ts) NOT IN (
                  SELECT symbol,timeframe,open_ts FROM (
                    SELECT symbol,timeframe,open_ts,
                           ROW_NUMBER() OVER (PARTITION BY symbol,timeframe ORDER BY open_ts DESC) AS rn
                    FROM klines
                  ) WHERE rn <= ?
                )""", (keep_n,))
            self.con.commit()
            return cur.rowcount

    def ingest_klines(self, symbol: str, timeframe: str,
                      klines: list, now_ms) -> int:
        """
        Normalize and persist klines already fetched externally (e.g. via BinanceUM).

        Args:
            symbol: e.g. 'ETHUSDT'
            timeframe: e.g. '1m'
            klines: iterable of raw Binance-style rows:
                [open_ts, open, high, low, close, volume, close_ts, ...]
            now_ms: timestamp for finalization logic.

        Returns:
            Number of rows inserted/upserted.

        Notes:
            - Marks all but the latest candle as finalized=1.
            - The latest is finalized only if close_ts <= now_ms.
            - Safe to call repeatedly (uses bulk_upsert).
        """
        if not klines:
            return 0

        tfm = tf_ms(timeframe)
        now_ms = now_ms or int(time.time() * 1000)

        rows: list[KlineRow] = []
        for k in klines:
            o_ts = int(k[0])
            c_ts = o_ts + tfm
            rows.append(KlineRow(
                symbol=symbol,
                timeframe=timeframe,
                open_ts=o_ts,
                close_ts=c_ts,
                o=float(k[1]),
                h=float(k[2]),
                l=float(k[3]),
                c=float(k[4]),
                v=float(k[5]),
                finalized=1,
            ))

        # Un-finalize live candle if still open
        if rows:
            last = rows[-1]
            if last.close_ts > now_ms:
                last.finalized = 0

        self.bulk_upsert(rows)
        return len(rows)
