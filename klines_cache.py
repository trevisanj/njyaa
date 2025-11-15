#!/usr/bin/env python3
# FILE: klines_cache.py

from __future__ import annotations
import sqlite3, time, math, requests
from typing import Iterable, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from contracts import EngineServices
from common import tf_ms
import threading

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
        cache = KlinesCache("./rv_cache.sqlite")
        last_open = cache.latest_open_ts("ETHUSDT", "1m")
        cache.fetch_and_cache(api, "ETHUSDT", "1m", since_open_ts=last_open, max_bars=1000)
        rows = cache.last_n("ETHUSDT", "1m", n=500, include_live=False)
    """

    def __init__(self, eng: EngineServices):
        """
        Open/create the SQLite cache.

        Args:
            cfg: AppConfig-like object containing at least KLINES_CACHE_DB_PATH.
            api: Option1al BinanceUM-like object (used for clock sync, fetches, etc.).
        """
        self.eng = eng
        self.cfg = eng.cfg
        self.api = eng.api

        self.db_path = self.cfg.KLINES_CACHE_DB_PATH
        if not self.db_path:
            raise ValueError("AppConfig must define KLINES_CACHE_DB_PATH")

        self.con = sqlite3.connect(self.db_path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self._ensure_schema()

        # Mechanism to keep "now" consistent in a batch operations. Needs to be manually reset before batch operations
        self._last_now_ms = None

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
        """)
        self.con.commit()
    def close(self):
        """Close the underlying SQLite connection (best-effort)."""
        try: self.con.close()
        except: pass

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
            self.con.execute(q, (row.symbol,row.timeframe,row.open_ts,row.close_ts,row.o,row.h,row.l,row.c,row.v,row.finalized))
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
        with self._db_lock:
            self.con.executemany(q, [
                (r.symbol,r.timeframe,r.open_ts,r.close_ts,r.o,r.h,r.l,r.c,r.v,r.finalized) for r in rows
            ])
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
            now_ms = int(time.time()*1000)
        with self._db_lock:
            cur = self.con.cursor()
            cur.execute("""UPDATE klines
                           SET finalized=1
                           WHERE finalized=0 AND close_ts <= ?""", (now_ms,))
            self.con.commit()
            return cur.rowcount

    def prune_keep_last_n(self, keep_n: int, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> int:
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
                cur.execute(f"""
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

    # ---------- reads ----------
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
        args = []
        where = []
        if symbol:
            where.append("symbol = ?")
            args.append(symbol.upper())
        if timeframe:
            where.append("timeframe = ?")
            args.append(timeframe)
        if where:
            sql += " WHERE " + " AND ".join(where)
        cur.execute(sql, args)
        return cur.fetchone()[0]

    def latest_open_ts(self, symbol: str, timeframe: str) -> Optional[int]:
        """
        Get the latest known candle’s open_ts for (symbol,timeframe).

        Returns:
            open_ts in ms, or None if no rows yet.

        Use cases:
            - Incremental fetch: pass this to `fetch_and_cache(..., since_open_ts=...)`.
        """
        r = self.con.execute("""SELECT open_ts FROM klines
                                WHERE symbol=? AND timeframe=?
                                ORDER BY open_ts DESC LIMIT 1""", (symbol,timeframe)).fetchone()
        return int(r["open_ts"]) if r else None

    def last_n(self, symbol: str, timeframe: str, n: int,
               include_live: bool = True, asc: bool = True) -> List[KlineRow]:
        """
        Fetch the last N klines for (symbol,timeframe).

        Args:
            symbol: Symbol, e.g. 'ETHUSDT'.
            timeframe: Interval string, e.g. '1m'.
            n: Count to return (from the newest backward).
            include_live: If False, only finalized candles are returned.
            asc: If True, results are ordered oldest→newest; if False, newest→oldest.

        Returns:
            List[KlineRow] of length ≤ n, correctly representing the last N bars.
        """
        live_clause = "" if include_live else "AND finalized=1"

        # fetch newest first, then reverse if ascending requested
        q = f"""
            SELECT * FROM klines
            WHERE symbol=? AND timeframe=? {live_clause}
            ORDER BY open_ts DESC
            LIMIT ?
        """
        rows = self.con.execute(q, (symbol, timeframe, n)).fetchall()
        out = [self._row_to_dataclass(r) for r in rows]
        return list(reversed(out)) if asc else out

    def range_by_open(self, symbol: str, timeframe: str, start_open_ts: int, end_open_ts: int, include_live: bool = True) -> List[KlineRow]:
        """
        Get klines in [start_open_ts, end_open_ts) by open timestamp.

        Args:
            symbol: Symbol.
            timeframe: Interval.
            start_open_ts: Inclusive open_ts lower bound (ms).
            end_open_ts: Exclusive open_ts upper bound (ms).
            include_live: If False, returns closed candles only.

        Returns:
            List[KlineRow] ordered by open_ts ASC.

        Notes:
            - Good for live-trading logic that keys off candle starts.
        """
        q = """SELECT * FROM klines
               WHERE symbol=? AND timeframe=? AND open_ts>=? AND open_ts<? {live}
               ORDER BY open_ts ASC"""
        live_clause = "" if include_live else "AND finalized=1"
        rows = self.con.execute(q.format(live=live_clause),
                                (symbol,timeframe,start_open_ts,end_open_ts)).fetchall()
        return [self._row_to_dataclass(r) for r in rows]

    def range_by_close(self, symbol: str, timeframe: str, start_close_ts: int, end_close_ts: int, include_live: bool = True) -> List[KlineRow]:
        """
        Get klines whose close_ts falls in (start_close_ts, end_close_ts].

        Args:
            symbol: Symbol.
            timeframe: Interval.
            start_close_ts: Exclusive lower bound (ms).
            end_close_ts: Inclusive upper bound (ms).
            include_live: If False, returns closed candles only.

        Returns:
            List[KlineRow] ordered by open_ts ASC.

        Notes:
            - Ideal for “signal at close” logic and backtesting windows.
        """
        q = """SELECT * FROM klines
               WHERE symbol=? AND timeframe=? AND close_ts> ? AND close_ts<=? {live}
               ORDER BY open_ts ASC"""
        live_clause = "" if include_live else "AND finalized=1"
        rows = self.con.execute(q.format(live=live_clause),
                                (symbol,timeframe,start_close_ts,end_close_ts)).fetchall()
        return [self._row_to_dataclass(r) for r in rows]

    def last_row(self, symbol: str, timeframe: str) -> Optional[KlineRow]:
        """
        Get the newest single row for (symbol,timeframe).

        Returns:
            KlineRow or None.
        """
        r = self.con.execute("""SELECT * FROM klines
                                WHERE symbol=? AND timeframe=?
                                ORDER BY open_ts DESC LIMIT 1""", (symbol,timeframe)).fetchone()
        return self._row_to_dataclass(r) if r else None

    def ingest_klines(self, symbol: str, timeframe: str, klines: list, now_ms) -> int:
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

        # rows.sort(key=lambda r: r.open_ts)  -- unnecessary sorting

        # un-finalize live candle
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
                    raise ValueError(f"If record's timeframe is not defined, timeframe argument must be set")
                else:
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
            # -- determine 'now' using exchange-synced time --
            if self._last_now_ms is None:
                self._ensure_api()
                self._last_now_ms = self.api.now_ms()

            fin = 0 if close_ts > self._last_now_ms else 1
        else:
            fin = 1 if bool(fin) else 0

        # -- basic sanity check --
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

            Required dict keys:
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
            This is useful when all klines belong to the same timeframe and the source
            data (e.g., Binance API) doesn’t include it explicitly.

        Returns
        -------
        int
            Number of rows inserted or updated (SQLite's `changes()` across the transaction).

        Behavior
        --------
        - Uses a single transaction for the whole batch.
        - Employs UPSERT on UNIQUE(symbol, timeframe, open_ts).
        - Ignores empty input (returns 0).
        - Minimal validation: checks tuple length and timeframe validity.
        - If `timeframe` argument is provided, overwrites any per-row value.

        Raises
        ------
        ValueError
            If a tuple has wrong arity, required dict keys are missing,
            or timeframe is unknown.
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
