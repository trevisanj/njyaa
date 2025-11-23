#!/usr/bin/env python3
# FILE: klines_cache.py

from __future__ import annotations
import sqlite3, time, math
from typing import Iterable, List, Tuple, Optional, Dict, Any, TYPE_CHECKING, Sequence
from dataclasses import dataclass
import threading
from common import tf_ms

if TYPE_CHECKING:
    from bot_api import BotEngine

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


# Default column set for klines queries (ts + OHLCV + finalized/meta)
KLINE_COLS: List[str] = [
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

    # ---------- writes ----------
    def upsert(self, row: KlineRow):
        """
        Insert or update a single kline row by (symbol,timeframe,open_ts).

        Overwrites: close_ts, O/H/L/C/V, finalized.

        Args:
            row: KlineRow to upsert.
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

        Args:
            rows: Iterable of KlineRow.

        Notes:
            - Use this after fetching a batch from the exchange.
            - Last candle in the batch is usually the live one; set `finalized` accordingly before passing.
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

        Args:
            now_ms: Reference clock (ms). Defaults to current time.

        Returns:
            Number of rows updated.

        Use cases:
            - You fetched live bars minutes ago; call this to mark anything that is now definitively closed.
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

        Args:
            keep_n: Number of newest candles to keep per (symbol,timeframe).
            symbol: If provided with timeframe, prune only that pair.
            timeframe: If provided with symbol, prune only that pair.

        Returns:
            Number of rows deleted.

        Notes:
            - With both symbol and timeframe: prunes only that stream.
            - With neither: prunes for all (symbol,timeframe) groups.
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

    def last_n(
        self,
        symbol: str,
        timeframe: str,
        n: int,
        columns: Optional[List[str]] = None,
        include_live: bool = True,
        asc: bool = True,
    ) -> Dict[str, List[Any]]:
        """
        Fetch the last N klines for (symbol,timeframe) as columnar dict-of-lists.

        Args:
            symbol: Symbol, e.g. 'ETHUSDT'.
            timeframe: Interval string, e.g. '1m'.
            n: Count to return (from the newest backward).
            columns: Which columns to return; defaults to KLINE_COLS (ts + OHLCV + meta).
            include_live: If False, only finalized candles are returned.
            asc: If True, results are ordered oldest→newest; if False, newest→oldest.
        """
        cols = columns or KLINE_COLS
        live_clause = "" if include_live else "AND finalized=1"
        select_cols = ",".join(cols)
        q = f"""
            SELECT {select_cols}
            FROM klines
            WHERE symbol=? AND timeframe=? {live_clause}
            ORDER BY open_ts DESC
            LIMIT ?
        """
        rows = self.con.execute(q, (symbol, timeframe, n)).fetchall()
        if asc:
            rows = list(reversed(rows))
        return _rows_to_columnar(rows, cols)

    def range_by_open(
        self,
        symbol: str,
        timeframe: str,
        start_open_ts: int,
        end_open_ts: int,
        columns: Optional[List[str]] = None,
        include_live: bool = True,
    ) -> Dict[str, List[Any]]:
        """
        Get klines in [start_open_ts, end_open_ts) by open timestamp (columnar).
        """
        cols = columns or KLINE_COLS
        live_clause = "" if include_live else "AND finalized=1"
        select_cols = ",".join(cols)
        q = f"""SELECT {select_cols}
                FROM klines
                WHERE symbol=? AND timeframe=?
                  AND open_ts>=? AND open_ts<? {live_clause}
                ORDER BY open_ts ASC"""
        rows = self.con.execute(q, (symbol, timeframe, start_open_ts, end_open_ts)).fetchall()
        return _rows_to_columnar(rows, cols)

    def range_by_close(
        self,
        symbol: str,
        timeframe: str,
        start_close_ts: int,
        end_close_ts: int,
        columns: Optional[List[str]] = None,
        include_live: bool = True,
    ) -> Dict[str, List[Any]]:
        """
        Get klines whose close_ts falls in (start_close_ts, end_close_ts] (columnar).
        """
        cols = columns or KLINE_COLS
        live_clause = "" if include_live else "AND finalized=1"
        select_cols = ",".join(cols)
        q = f"""SELECT {select_cols}
                FROM klines
                WHERE symbol=? AND timeframe=?
                  AND close_ts>? AND close_ts<=? {live_clause}
                ORDER BY open_ts ASC"""
        rows = self.con.execute(q, (symbol, timeframe, start_close_ts, end_close_ts)).fetchall()
        return _rows_to_columnar(rows, cols)

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

    # ---------- internals ----------
    @staticmethod
    def _row_to_dataclass(r: sqlite3.Row) -> KlineRow:
        """
        Convert sqlite3.Row to KlineRow.
        (Kept for compatibility; not used by current read methods.)
        """
        return KlineRow(
            symbol=r["symbol"], timeframe=r["timeframe"],
            open_ts=int(r["open_ts"]), close_ts=int(r["close_ts"]),
            o=float(r["open"]), h=float(r["high"]), l=float(r["low"]),
            c=float(r["close"]), v=float(r["volume"]),
            finalized=int(r["finalized"])
        )

    def _norm_one(self, x, timeframe: str | None = None):
        """
        Normalize one kline record into canonical tuple for database insertion.

        Parameters
        ----------
        x : dict | tuple
            Kline data. May omit `close_ts` and/or `finalized`.
        timeframe : str, optional
            If provided, overrides or fills missing timeframe in the input.

        Behavior
        --------
        - `close_ts` is computed automatically if missing: open_ts + timeframe_ms.
        - `finalized` defaults to 0 if close_ts > exchange-synced now_ms, else 1.
        - Validates timestamp ordering and timeframe existence.

        Returns
        -------
        tuple
            (symbol, timeframe, open_ts, close_ts, open, high, low, close, volume, finalized)
        """
        # -- extract raw values --
        if isinstance(x, dict):
            symbol = str(x["symbol"]).upper()
            tf = x.get("timeframe")
            if tf is None:
                if timeframe is None:
                    raise ValueError("If record's timeframe is not defined, timeframe argument must be set")
                tf = timeframe
            open_ts = int(x["open_ts"])
            close_ts = int(x.get("close_ts") or 0)
            o = float(x["open"])
            h = float(x["high"])
            l = float(x["low"])
            c = float(x["close"])
            v = float(x["volume"])
            fin = x.get("finalized")
        else:
            if len(x) < 9:
                raise ValueError(f"Tuple must have ≥9 elements, got {len(x)}")
            symbol, tf, open_ts, *rest = x
            tf = str(timeframe or tf)
            open_ts = int(open_ts)
            if len(rest) == 6:  # missing close_ts
                o, h, l, c, v, fin = rest
                close_ts = 0
            elif len(rest) == 7:
                close_ts, o, h, l, c, v, fin = rest
            else:
                raise ValueError("Tuple shape not recognized for _norm_one()")
            o, h, l, c, v = float(o), float(h), float(l), float(c), float(v)
            fin = None if fin is None else (1 if bool(fin) else 0)

        # -- derive timeframe and ms offset --
        tfm = tf_ms(tf)

        # -- fill close_ts if missing --
        if not close_ts:
            close_ts = open_ts + tfm

        # -- derive finalized if missing --
        if fin is None:
            if self._last_now_ms is None:
                self._ensure_api()
                self._last_now_ms = self.api.now_ms()
            fin = 0 if close_ts > self._last_now_ms else 1
        else:
            fin = 1 if bool(fin) else 0

        # -- sanity --
        if close_ts <= open_ts:
            raise ValueError(f"close_ts({close_ts}) <= open_ts({open_ts}) for {symbol} {tf}")

        return (symbol, tf, open_ts, close_ts, o, h, l, c, v, fin)

    def upsert_batch(self, rows, timeframe: str | None = None) -> int:
        """
        Upsert a batch of klines.

        Parameters
        ----------
        rows : Iterable[Tuple | Dict]
            Each item represents one kline. Accepted shapes:

            Tuple (10 fields):
                (symbol:str, timeframe:str, open_ts:int, close_ts:int,
                 open:float, high:float, low:float, close:float,
                 volume:float, finalized:int|bool)

            Dict keys:
                {
                  "symbol", "timeframe", "open_ts", "close_ts",
                  "open", "high", "low", "close", "volume", "finalized"
                }

            Other dict keys:
                "timeframe": required if timeframe not passed; overwritten if timeframe passed
                "close_ts": calculated automatically if not passed
                "finalized": calculated automatically if not passed

        timeframe : str, optional
            If provided, this value overrides the `timeframe` field in every row.

        Returns
        -------
        int
            Number of rows inserted or updated (SQLite's `changes()` across the transaction).
        """
        rows = list(rows)
        if not rows:
            return 0

        self._last_now_ms = None
        batch = [self._norm_one(r, timeframe) for r in rows]

        sql = """
        INSERT INTO klines
          (symbol, timeframe, open_ts, close_ts,
           open, high, low, close, volume, finalized)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, timeframe, open_ts) DO UPDATE SET
           close_ts = excluded.close_ts,
           open     = excluded.open,
           high     = excluded.high,
           low      = excluded.low,
           close    = excluded.close,
           volume   = excluded.volume,
           finalized= excluded.finalized
        """

        with self._db_lock:
            cur = self.con.cursor()
            try:
                cur.execute("BEGIN")
                cur.executemany(sql, batch)
                cur.execute("SELECT changes()")
                changed = cur.fetchone()[0] or 0
                self.con.commit()
                return int(changed)
            except Exception:
                self.con.rollback()
                raise


# ---------- standalone helper (rows -> pandas.DataFrame) ----------
def rows_to_dataframe(rows):
    """
    Zero-copy-ish path from iterable[sqlite3.Row|dict] -> DataFrame.

    - Uses pandas.from_records to vectorize ingestion.
    - Keeps only open,high,low,close,volume.
    - Renames to ['Open','High','Low','Close','Volume'].
    - Index = Date (from open_ts, ms). No sorting.
    - Sets df.attrs['symbol'] / ['timeframe'] when unique.
    """
    import pandas as pd

    if isinstance(rows, dict):
        rows_dict = rows
        if not rows_dict:
            df = pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
            df.index.name = "Date"
            df.attrs["symbol"] = None
            df.attrs["timeframe"] = None
            return df
        df = pd.DataFrame(rows_dict)
        cols = list(rows_dict.keys())
    else:
        rows = list(rows)
        if not rows:
            df = pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
            df.index.name = "Date"
            df.attrs["symbol"] = None
            df.attrs["timeframe"] = None
            return df

        # Let pandas ingest mapping/rows in C
        # Derive columns from first row if possible (sqlite3.Row supports .keys())
        try:
            cols = list(rows[0].keys())
        except Exception:
            cols = None  # let pandas infer

        df = pd.DataFrame.from_records(rows, columns=cols)

    # Minimal columns present check
    needed = ["open_ts", "open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"rows_to_dataframe: missing columns {missing}")

    # Build target view without extra copies
    # Pop open_ts to avoid an extra column copy later
    open_ts = df.pop("open_ts").to_numpy(dtype="int64", copy=False)

    # Keep only required price/vol columns in one pass
    df = df.loc[:, ["open", "high", "low", "close", "volume"]]

    # Rename in-place (O(1))
    df.columns = ["Open", "High", "Low", "Close", "Volume"]

    # Set Date index (vectorized)
    df.index = pd.to_datetime(open_ts, unit="ms", utc=False)
    df.index.name = "Date"

    # Attach metadata only if unique (cheap nunique on small columns)
    for meta in ("symbol", "timeframe"):
        if meta in (cols or df.columns):
            s = df.get(meta)
            # if meta was popped earlier, it won't exist; fallback to building from rows
            if s is None:
                try:
                    series = pd.Series([r[meta] if isinstance(r, dict) else r[meta] for r in rows])
                except Exception:
                    series = None
            else:
                series = s
            if series is not None and series.nunique(dropna=True) == 1:
                df.attrs[meta] = series.iloc[0]
            else:
                df.attrs[meta] = None
        else:
            df.attrs[meta] = None

    return df


def _rows_to_columnar(rows: Sequence[Sequence[Any]], columns: Sequence[str]) -> Dict[str, List[Any]]:
    """Convert sequence of DB rows to dict-of-lists using provided column order."""
    cols: Dict[str, List[Any]] = {k: [] for k in columns}
    if not rows:
        return cols
    for r in rows:
        for k, v in zip(columns, r):
            cols[k].append(v)
    return cols
