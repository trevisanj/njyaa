#!/usr/bin/env python3
# FILE: indicator_history.py
from __future__ import annotations
import sqlite3, json, threading
from typing import List, Optional, Dict


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

    def _ensure_schema(self):
        c = self.con.cursor()
        c.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS indicator_history(
          thinker_id  INTEGER NOT NULL,
          position_id INTEGER NOT NULL,
          name        TEXT    NOT NULL,
          ts_ms       INTEGER NOT NULL,
          value       REAL,
          price       REAL,
          aux_json    TEXT,
          PRIMARY KEY(thinker_id, position_id, name, ts_ms)
        );
        CREATE INDEX IF NOT EXISTS ix_ind_hist_ts ON indicator_history(name, ts_ms);
        """)
        self.con.commit()

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass

    def insert_history(self, rows: List[dict]) -> int:
        """Bulk insert history rows."""
        if not rows:
            return 0
        payload = [
            (
                int(r["thinker_id"]),
                int(r["position_id"]),
                r["name"],
                int(r["ts_ms"]),
                r.get("value"),
                r.get("price"),
                json.dumps(r.get("aux") or {}, ensure_ascii=False),
            )
            for r in rows
        ]
        with self._lock:
            cur = self.con.cursor()
            cur.executemany(
                """
                INSERT OR REPLACE INTO indicator_history(thinker_id,position_id,name,ts_ms,value,price,aux_json)
                VALUES(?,?,?,?,?,?,?)
                """,
                payload,
            )
            self.con.commit()
            return cur.rowcount or 0

    def last_n(self, thinker_id: int, position_id: int, name: str, n: int = 200, asc: bool = False) -> List[dict]:
        """Return last N rows for (thinker,position,name)."""
        order = "ASC" if asc else "DESC"
        rows = self.con.execute(
            f"""
            SELECT thinker_id,position_id,name,ts_ms,value,price,aux_json
            FROM indicator_history
            WHERE thinker_id=? AND position_id=? AND name=?
            ORDER BY ts_ms {order}
            LIMIT ?
            """,
            (int(thinker_id), int(position_id), name, int(n)),
        ).fetchall()
        out = []
        for r in rows:
            try:
                out.append({
                    "thinker_id": int(r["thinker_id"]),
                    "position_id": int(r["position_id"]),
                    "name": r["name"],
                    "ts_ms": int(r["ts_ms"]),
                    "value": r["value"],
                    "price": r["price"],
                    "aux": json.loads(r["aux_json"]) if r["aux_json"] else {},
                })
            except Exception:
                continue
        return list(reversed(out)) if asc else out

    def list_history(self, thinker_id: int, position_id: int, name: str, since_ms: Optional[int] = None, limit: int = 500) -> List[dict]:
        """Return history rows ascending."""
        q = """
            SELECT thinker_id,position_id,name,ts_ms,value,price,aux_json
            FROM indicator_history
            WHERE thinker_id=? AND position_id=? AND name=?
        """
        params = [int(thinker_id), int(position_id), name]
        if since_ms is not None:
            q += " AND ts_ms>=?"
            params.append(int(since_ms))
        q += " ORDER BY ts_ms ASC LIMIT ?"
        params.append(int(limit))
        rows = self.con.execute(q, params).fetchall()
        out: List[dict] = []
        for r in rows:
            try:
                out.append({
                    "thinker_id": int(r["thinker_id"]),
                    "position_id": int(r["position_id"]),
                    "name": r["name"],
                    "ts_ms": int(r["ts_ms"]),
                    "value": r["value"],
                    "price": r["price"],
                    "aux": json.loads(r["aux_json"]) if r["aux_json"] else {},
                })
            except Exception:
                continue
        return out

    def range_by_ts(self, thinker_id: int, position_id: int, name: str, start_ts: int, end_ts: int, limit: int = 5000) -> List[dict]:
        """Return rows in [start_ts, end_ts] ascending."""
        rows = self.con.execute(
            """
            SELECT thinker_id,position_id,name,ts_ms,value,price,aux_json
            FROM indicator_history
            WHERE thinker_id=? AND position_id=? AND name=? AND ts_ms BETWEEN ? AND ?
            ORDER BY ts_ms ASC
            LIMIT ?
            """,
            (int(thinker_id), int(position_id), name, int(start_ts), int(end_ts), int(limit)),
        ).fetchall()
        out = []
        for r in rows:
            try:
                out.append({
                    "thinker_id": int(r["thinker_id"]),
                    "position_id": int(r["position_id"]),
                    "name": r["name"],
                    "ts_ms": int(r["ts_ms"]),
                    "value": r["value"],
                    "price": r["price"],
                    "aux": json.loads(r["aux_json"]) if r["aux_json"] else {},
                })
            except Exception:
                continue
        return out

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
