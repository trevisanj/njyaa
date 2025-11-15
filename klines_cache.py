#!/usr/bin/env python3
# FILE: klines_cache.py

from __future__ import annotations
import sqlite3, time, math, requests
from typing import Iterable, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


# ---- tiny TF helpers ----
_TF_MS = {
    "1s": 1_000, "3s": 3_000, "5s": 5_000, "15s": 15_000, "30s": 30_000,
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
    "1w": 604_800_000,
}
def tf_ms(tf: str) -> int:
    """
    Return timeframe length in milliseconds.

    Args:
        tf: Timeframe string like '1m', '5m', '1h', '1d'.

    Returns:
        Milliseconds for the timeframe.

    Raises:
        ValueError: If `tf` is unknown.
    """
    if tf not in _TF_MS:
        raise ValueError(f"Unknown timeframe '{tf}'. Supported: {', '.join(_TF_MS)}")
    return _TF_MS[tf]

@dataclass(frozen=True)
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

    def __init__(self, cfg, api=None):
        """
        Open/create the SQLite cache.

        Args:
            cfg: AppConfig-like object containing at least KLINES_CACHE_DB_PATH.
            api: Optional BinanceUM-like object (used for clock sync, fetches, etc.).
        """
        self.cfg = cfg
        self.api = api

        self.db_path = getattr(cfg, "KLINES_CACHE_DB_PATH", None)
        if not self.db_path:
            raise ValueError("AppConfig must define KLINES_CACHE_DB_PATH")

        self.con = sqlite3.connect(self.db_path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self._ensure_schema()

        # Mechanism to keep "now" consistent in a batch operations. Needs to be manually reset before batch operations
        self._last_now_ms = None

    def _ensure_api(self):
        """
        Ensure the cache has access to a BinanceUM-like API when required.

        Raises
        ------
        RuntimeError
            If no API object is set.
        """
        if self.api is None:
            raise RuntimeError("This operation requires an API instance. "
                               "Initialize KlinesCache with api=BinanceUM(...) first.")

    # ---------- schema ----------
    def _ensure_schema(self):
        """
        Create tables and indexes if they don't exist.
        - WAL enabled for concurrency/perf.
        - Secondary index on close_ts for at-close scans.
        """
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
        cur = self.con.cursor()
        cur.execute("""UPDATE klines
                       SET finalized=1
                       WHERE finalized=0 AND close_ts <= ?""", (now_ms,))
        self.con.commit()
        return cur.rowcount

    def prune_keep_last_n(self, symbol: Optional[str], timeframe: Optional[str], keep_n: int) -> int:
        """
        Keep only the newest N candles per group.

        Args:
            symbol: If provided with timeframe, prune only that pair.
            timeframe: If provided with symbol, prune only that pair.
            keep_n: Number of newest candles to keep per (symbol,timeframe).

        Returns:
            Number of rows deleted.

        Notes:
            - With both symbol and timeframe: prunes only that stream.
            - With neither: prunes for all (symbol,timeframe) groups.
        """
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

    # ---------- fetch + ingest ----------
    def fetch_and_cache(self, symbol: str, timeframe: str,
                        since_open_ts: Optional[int], max_bars: int = 1000) -> int:
        """
        Pull fresh klines from Binance and store them, safely overwriting the live candle.

        Strategy:
            - If since_open_ts is None: pull the most recent window (up to `max_bars`).
            - Else: overlap by one candle (start = since_open_ts - tf_ms(timeframe))
                    so the previously-live bar is overwritten with its final values.

        Args:
            symbol: e.g. 'ETHUSDT'.
            timeframe: e.g. '1m'.
            since_open_ts: Last known open_ts, or None for cold start.
            max_bars: Upper bound on bars to fetch.

        Returns:
            Number of rows upserted.

        Caveats:
            - Sets finalized=1 for all but the last fetched candle.
            - The last candle is marked finalized=1 only if its close_ts ≤ now.
        """

        self._ensure_api()
        api = self.api

        tfm = tf_ms(timeframe)
        now_ms = api.now_ms()
        if since_open_ts is None:
            # pull most recent window
            end = now_ms
            limit = max_bars
            start = None
        else:
            # overlap
            start = max(0, since_open_ts - tfm)
            end = now_ms
            limit = max_bars

        klines = self.api.klines(symbol, timeframe, start, end, limit)
        raw = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in klines]

        if not raw:
            return 0

        rows: List[KlineRow] = []
        for k in raw:
            o_ts = int(k[0])
            c_ts = o_ts + tfm
            rows.append(KlineRow(
                symbol=symbol, timeframe=timeframe, open_ts=o_ts, close_ts=c_ts,
                o=float(k[1]), h=float(k[2]), l=float(k[3]), c=float(k[4]),
                v=float(k[5]), finalized=0  # default to live; we’ll flip below
            ))

        # mark all but last as finalized=1 (they are closed), last stays 0 unless its close in the past
        rows.sort(key=lambda r: r.open_ts)
        for r in rows[:-1]:
            object.__setattr__(r, "finalized", 1)
        if rows:
            last = rows[-1]
            if last.close_ts <= now_ms:
                object.__setattr__(last, "finalized", 1)

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


# ---------------------------
# __main__: offline-first smoke test for KLinesCache (NO argparse)
# ---------------------------
if __name__ == "__main__":
    """
    What this does:
      1) Opens/creates the cache DB (DB_PATH).
      2) Synthesizes a batch of 1m candles for SYMBOL (or fetches live if LIVE_MODE=True).
      3) Upserts them; then upserts an overlapping tail to test idempotency + live-bar overwrite.
      4) Reads back last N, a small time-window, and latest bar.
      5) Prunes to keep only KEEP newest bars.

    Tweak the GLOBALS below. No CLI, no surprises.
    """
    
    import os
    import time as _time
    from bot_api import AppConfig
    from zoneinfo import ZoneInfo


    def ascii_chart(bars, symbol: str = "", timeframe: str = "", height: int = 20):
        """
        Render an ASCII multi-line candlestick-style chart in the terminal.

        Parameters
        ----------
        bars : list[KlineRow]
            Sequence of KlineRow objects (must have .o, .h, .l, .c, .finalized).
        symbol : str, optional
            Symbol label for the chart title.
        timeframe : str, optional
            Timeframe label for the chart title.
        height : int, default 20
            Vertical resolution in rows (more = smoother).

        Behavior
        --------
        - Uses O/H/L/C to draw vertical wicks and candle bodies.
        - Filled blocks (`█`) represent closed candles.
        - Thin glyphs (`┆`) mark live (unfinalized) candles.
        - Scales automatically to visible price range.
        - Prints numeric scale on the left.
        """
        if not bars:
            print("[ascii_chart] No data.")
            return

        highs = [r.h for r in bars]
        lows = [r.l for r in bars]
        hi, lo = max(highs), min(lows)
        rng = max(hi - lo, 1e-12)
        step = rng / (height - 1)

        width = len(bars)
        grid = [[" " for _ in range(width)] for _ in range(height)]

        for i, r in enumerate(bars):
            hi_row = int((r.h - lo) / step)
            lo_row = int((r.l - lo) / step)
            op_row = int((r.o - lo) / step)
            cl_row = int((r.c - lo) / step)
            top = max(op_row, cl_row)
            bot = min(op_row, cl_row)
            for y in range(lo_row, hi_row + 1):
                if bot <= y <= top:
                    grid[y][i] = "█" if r.finalized else "┆"
                elif grid[y][i] == " ":
                    grid[y][i] = "│"

        print(f"\nASCII Chart ({symbol} {timeframe}) — {len(bars)} bars")
        for y in reversed(range(height)):
            level = lo + y * step
            if y % (height // 5) == 0 or y == 0:
                label = f"{level:>10.4f} ┤"
            else:
                label = " " * 12
            print(label + "".join(grid[y]))
        print(" " * 12 + "─" * width)
        print(" " * 12 + "0" + " " * (width - 2) + f"{width:>3d}")
        print(f"range: {lo:.4f} – {hi:.4f}")


    # --------- GLOBALS (tune here) ---------
    TZ_LOCAL = ZoneInfo("America/Fortaleza")
    DB_PATH   = os.environ.get("RV_KCACHE", "./rv_klines_cache.sqlite")
    SYMBOL    = "ETHUSDT"
    TF        = "1m"       # "1m","5m","15m","1h","4h","1d"
    N_BARS    = 300        # initial synthetic (or live) bars
    OVERLAP   = 5          # how many bars to overlap in the 2nd upsert
    KEEP      = 200        # prune to keep last KEEP bars
    LIVE_MODE = True      # set True to use Binance; see LiveFetcher notes
    LIVE_MINUTES = 180     # if LIVE_MODE=True and you prefer minutes span instead of N_BARS

    CHANGE = False         # If False, changes to database will be skipped

    # --------- interval mapping ----------
    if TF not in _TF_MS:
        raise SystemExit(f"Unsupported timeframe '{TF}'. Supported: {', '.join(sorted(_TF_MS))}")
    TF_MS = _TF_MS[TF]


    # ---------- Offline fake fetcher ----------
    class FakeFetcher:
        def __init__(self, seed_price=1800.0, drift=0.00015, wobble=0.002):
            self.seed = float(seed_price)
            self.drift = float(drift)
            self.wobble = float(wobble)

        def klines(self, symbol: str, interval: str, start_ms: int, end_ms: int):
            """Return list of Binance-like klines: [open_ts, open, high, low, close, volume, close_ts]."""
            step = _TF_MS[interval]
            start = start_ms - (start_ms % step)
            if end_ms <= start:
                return []
            out, p, t, i = [], self.seed, start, 0
            while t < end_ms:
                wig = (1 + self.wobble * ((hash((symbol, t)) % 200) / 100.0 - 1.0))
                o = p
                c = max(0.01, o * (1 + self.drift)) * wig
                h = max(o, c) * 1.001
                l = min(o, c) * 0.999
                v = 100 + (i % 17)
                out.append([t, f"{o:.2f}", f"{h:.2f}", f"{l:.2f}", f"{c:.2f}", f"{v:.3f}", t + step - 1])
                p, t, i = c, t + step, i + 1
            return out

    # ---------- Optional live fetcher (uses your bot_api_02.BinanceUM) ----------
    now_ms = int(time.time()*1000)
    if LIVE_MODE and CHANGE:
        from bot_api import BinanceUM  # relies on your project layout
        import sys as _sys

        _sys.path.append("/home/j/yp/saccakeys")
        from saccakeys import keys as _keys

        B = _keys.apikeys["binance"]
        cfg_api = AppConfig(
            RECV_WINDOW_MS=60_000,
            BINANCE_KEY=B[0],
            BINANCE_SEC=B[1],
        )

        api = BinanceUM(cfg_api)
        now_ms = api.now_ms()

    class LiveFetcher:
        """
        Minimal adapter. Requires saccakeys and your BinanceUM wrapper.
        """
        def __init__(self):
            pass

        def klines(self, symbol: str, interval: str, start_ms: int, end_ms: int):
            rows = api.klines(symbol, interval, start_ms, end_ms)
            out = []
            for r in rows:
                # futures kline: [open_time, open, high, low, close, volume, close_time, ...]
                out.append([r[0], r[1], r[2], r[3], r[4], r[5], r[6]])
            return out

    # ---------- Build cache ----------
    cfg_cache = AppConfig(TZ_LOCAL,
                          KLINES_CACHE_DB_PATH=DB_PATH)

    cache = KlinesCache(cfg_cache)

    # ---------- Compose first window ----------
    if LIVE_MODE and LIVE_MINUTES:
        span_ms = LIVE_MINUTES * 60_000
        n_bars  = max(2, LIVE_MINUTES)  # for logs only
    else:
        n_bars  = max(10, N_BARS)
        span_ms = n_bars * TF_MS

    start_ms = now_ms - span_ms
    end_ms   = now_ms

    fetcher = LiveFetcher() if LIVE_MODE else FakeFetcher()

    # ---------- Fetch + upsert first batch ----------
    if CHANGE:
        kl = fetcher.klines(SYMBOL, TF, start_ms, end_ms)
        batch = []
        for k in kl:
            open_ts = int(k[0])
            o, h, l, c = map(float, (k[1], k[2], k[3], k[4]))
            vol = float(k[5])
            close_ts = open_ts + TF_MS
            finalized = 0 if close_ts > now_ms else 1  # last bar likely live
            batch.append({
                "symbol": SYMBOL,
                "open_ts": open_ts, "close_ts": close_ts,
                "open": o, "high": h, "low": l, "close": c, "volume": vol,
                "finalized": finalized,
            })


        cache.upsert_batch(batch, timeframe=TF)
        total1 = cache.count(SYMBOL, TF)
        print(f"[1] Upserted first batch: {len(batch)} bars -> cache count {total1}")

        # ---------- Overlapping second batch (tail overwrite) ----------
        if len(batch) >= OVERLAP + 2:
            tail_from = batch[-(OVERLAP + 2)]["open_ts"]
            tail_to   = now_ms + TF_MS
            kl2 = fetcher.klines(SYMBOL, TF, tail_from, tail_to)
            batch2 = []
            for k in kl2:
                open_ts = int(k[0])
                o, h, l, c = map(float, (k[1], k[2], k[3], k[4]))
                vol = float(k[5])
                close_ts = open_ts + TF_MS
                finalized = 0 if close_ts > now_ms else 1
                batch2.append({
                    "symbol": SYMBOL, "tf": TF,
                    "open_ts": open_ts, "close_ts": close_ts,
                    "open": o, "high": h, "low": l, "close": c, "volume": vol,
                    "finalized": finalized,
                })
            cache.upsert_batch(batch2, timeframe=TF)
            total2 = cache.count(SYMBOL, TF)
            print(f"[2] Upserted overlapping tail: {len(batch2)} bars -> cache count {total2}")

    # ---------- Readbacks ----------
    last_10 = cache.last_n(SYMBOL, TF, n=10)
    print(f"[3] last_n=10 -> {len(last_10)} rows | finalized flags:", [r.finalized for r in last_10])

    win_start = now_ms - 5 * TF_MS
    win = cache.range_by_open(SYMBOL, TF, start_open_ts=win_start, end_open_ts=now_ms + 1)
    print(f"[4] range_by_open(last ~5 bars) -> {len(win)} rows | first_open={win[0].open_ts if win else None}")

    latest = cache.last_row(SYMBOL, TF)
    print(
        f"[5] last_row -> open_ts={latest.open_ts if latest else None}, finalized={latest.finalized if latest else None}")

    if CHANGE:
        before = cache.count(SYMBOL, TF)
        cache.prune_keep_last_n(SYMBOL, TF, keep_n=KEEP)
        after = cache.count(SYMBOL, TF)
        print(f"[6] prune_keep_last_n({KEEP}) -> {before} -> {after}")

        newest = cache.last_row(SYMBOL, TF)
        assert newest is not None, "Latest bar missing after prune"
        print("Last row:", newest)

    print("[OK] KLinesCache smoke test complete (globals mode).")


    # ---------- ASCII CHART ----------
    bars = cache.last_n(SYMBOL, TF, n=100)
    ascii_chart(bars, SYMBOL, TF, height=25)
