#!/usr/bin/env python3
# FILE: indicator_history.py
from __future__ import annotations
import json
import sqlite3
import threading
from typing import List, Optional
from cache_helpers import rows_to_columnar, rows_to_generic_df


DEFAULT_COLUMNS = ["thinker_id", "position_id", "name", "open_ts", "value", "aux_json"]


class IndicatorHistory:
    """
    Lightweight time-series store for indicator outputs.
    Separate DB to avoid bloating the main state DB; no FKs.
    PK: (thinker_id, position_id, name, ts_ms)
    """
    def __init__(self, db_path: str):
        assert db_path, "INDICATOR_HISTORY_DB_PATH missing"
        self.db_path = db_path
        self.con = sqlite3.connect(self.db_path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._ensure_schema()

    # ---------- lifecycle ----------
    def _ensure_schema(self):
        c = self.con.cursor()
        c.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS indicator_history(
          thinker_id  INTEGER NOT NULL,
          position_id INTEGER NOT NULL,
          name        TEXT    NOT NULL,
          open_ts     INTEGER NOT NULL,
          value    REAL,
          aux_json TEXT,
          PRIMARY KEY(thinker_id, position_id, name, open_ts)
        );
        CREATE INDEX IF NOT EXISTS ix_ind_hist_ts ON indicator_history(name, open_ts);
        """)
        self.con.commit()

    def close(self):
        self.con.close()

    # ---------- writes ----------
    def insert_history(self, rows: List[dict]) -> int:
        """Bulk insert history rows."""
        if not rows:
            return 0
        payload = [
            (
                int(r["thinker_id"]),
                int(r["position_id"]),
                r["name"],
                int(r["open_ts"]),
                r["value"],
                json.dumps(r["aux"] or {}, ensure_ascii=False),
            )
            for r in rows
        ]
        with self._lock:
            cur = self.con.cursor()
            cur.executemany(
                """
                INSERT OR REPLACE INTO indicator_history(thinker_id,position_id,name,open_ts,value,aux_json)
                VALUES(?,?,?,?,?,?)
                """,
                payload,
            )
            self.con.commit()
            return cur.rowcount or 0

    def delete_by_position(self, position_id: int) -> int:
        with self._lock:
            cur = self.con.cursor()
            cur.execute("DELETE FROM indicator_history WHERE position_id=?", (int(position_id),))
            self.con.commit()
            return cur.rowcount or 0

    def delete_by_thinker(self, thinker_id: int) -> int:
        with self._lock:
            cur = self.con.cursor()
            cur.execute("DELETE FROM indicator_history WHERE thinker_id=?", (int(thinker_id),))
            self.con.commit()
            return cur.rowcount or 0

    # ---------- reads ----------
    def last_n(self, thinker_id: int, position_id: int, name: str, n: int = 200, columns: Optional[List[str]] = None,
               asc: bool = True, fmt: str = "columnar"):
        """Return last N rows for (thinker,position,name)."""
        assert fmt in ("columnar", "dataframe")
        cols = columns or DEFAULT_COLUMNS
        order = "ASC" if asc else "DESC"
        select_cols = ",".join(cols)
        with self._lock:
            rows = self.con.execute(
                f"""
                SELECT {select_cols}
                FROM indicator_history
                WHERE thinker_id=? AND position_id=? AND name=?
                ORDER BY open_ts {order}
                LIMIT ?
                """,
                (int(thinker_id), int(position_id), name, int(n)),
            ).fetchall()
        return self._format_rows(rows, cols, fmt)

    def range_by_ts(self, thinker_id: int, position_id: int, name: str, start_open_ts: Optional[int] = None,
                    end_open_ts: Optional[int] = None, columns: Optional[List[str]] = None, asc: bool = True,
                    limit: int = 5000, fmt: str = "columnar"):
        """Return rows within optional [start_ts, end_ts] bounds."""
        assert fmt in ("columnar", "dataframe")
        cols = columns or DEFAULT_COLUMNS
        select_cols = ",".join(cols)
        where = ["thinker_id=?", "position_id=?", "name=?"]
        params = [int(thinker_id), int(position_id), name]
        if start_open_ts is not None:
            params.append(int(start_open_ts))
            where.append("open_ts>=?")
        if end_open_ts is not None:
            params.append(int(end_open_ts))
            where.append("open_ts<?")
        order = "ASC" if asc else "DESC"
        where_clause = " AND ".join(where)
        query = f"""
            SELECT {select_cols}
            FROM indicator_history
            WHERE {where_clause}
            ORDER BY open_ts {order}
            LIMIT ?
        """
        params.append(int(limit))
        with self._lock:
            rows = self.con.execute(query, params).fetchall()
        return self._format_rows(rows, cols, fmt)

    def list_indicators(self, thinker_id: Optional[int] = None, position_id: Optional[int] = None, fmt: str = "columnar"):
        """Distinct indicator names grouped by thinker/position."""
        where = []
        params: list = []
        if thinker_id is not None:
            where.append("thinker_id=?")
            params.append(int(thinker_id))
        if position_id is not None:
            where.append("position_id=?")
            params.append(int(position_id))
        where_clause = ("WHERE " + " AND ".join(where)) if where else ""
        with self._lock:
            rows = self.con.execute(
                f"""SELECT DISTINCT thinker_id, position_id, name
                    FROM indicator_history
                    {where_clause}
                    ORDER BY thinker_id, position_id, name""",
                params,
            ).fetchall()
        cols = ["thinker_id", "position_id", "name"]
        result = rows_to_columnar(rows, cols)
        if fmt == "dataframe":
            import pandas as pd
            result = pd.DataFrame.from_records(rows, columns=cols)
        return result

    def _format_rows(self, rows, columns: List[str], fmt: str):
        columnar = rows_to_columnar(rows, columns)
        if fmt == "dataframe":
            return rows_to_generic_df(dict(columnar), columns=columns, ts_name="open_ts")
        return columnar
