# storage.py
from __future__ import annotations
import sqlite3
import json
import threading
import time
from contextlib import contextmanager
from typing import List, Optional, Any
from common import Clock, log

DEFAULT_CONFIG = {
    "reference_balance": 10_000.0,
    "leverage": 1.0,
    "default_risk": 0.02,
}


# =======================
# ====== STORAGE ========
# =======================

class Storage:
    """
    Thread-safe SQLite storage layer.
    - Single connection with WAL, busy_timeout, foreign_keys=ON.
    - Re-entrant lock guards all write operations.
    - `txn(write=True)` starts BEGIN IMMEDIATE (writer lock) and commits/rolls back.
    - Reads are lightweight and generally safe under WAL; we still allow using txn() when desired.
    """

    def __init__(self, path: str):
        self.path = path
        # One connection shared across threads (guarded by _db_lock for writes)
        self.con = sqlite3.connect(self.path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self._db_lock = threading.RLock()
        self._indicator_history = None  # optional hook to external history store
        self._init_db()
        self._configure_pragmas()

    def _configure_pragmas(self):
        # Pragmas applied once per connection
        cur = self.con.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA busy_timeout=5000;")  # ms
        self.con.commit()

    def _init_db(self):
        c = self.con.cursor()
        c.executescript("""
        -- ========== positions ==========
        CREATE TABLE IF NOT EXISTS positions (
          position_id   INTEGER PRIMARY KEY AUTOINCREMENT,
          num           TEXT    NOT NULL,
          den           TEXT,                    -- NULL => single-leg
          dir_sign      INTEGER NOT NULL,        -- +1 long, -1 short (meaning for single-leg)
          target_usd    REAL    NOT NULL,
          risk          REAL    NOT NULL DEFAULT 0.02,
          user_ts       INTEGER NOT NULL,        -- user-declared timestamp (ms)
          closed_ts     INTEGER,                 -- when status transitions to CLOSED (ms)
          status        TEXT    NOT NULL DEFAULT 'OPEN',
          note          TEXT,
          created_ts    INTEGER NOT NULL
        );
        -- Uniqueness of a conceptual “position”
        CREATE UNIQUE INDEX IF NOT EXISTS ux_positions_signature
          ON positions(num, den, dir_sign, target_usd, user_ts);

        -- ============ legs ============
        CREATE TABLE IF NOT EXISTS legs (
          leg_id          INTEGER PRIMARY KEY AUTOINCREMENT,
          position_id     INTEGER NOT NULL,
          symbol          TEXT    NOT NULL,
          qty             REAL,                  -- signed; NULL until backfilled
          entry_price     REAL,                  -- NULL until backfilled
          entry_price_ts  INTEGER,               -- NULL until backfilled
          price_method    TEXT,
          need_backfill   INTEGER NOT NULL DEFAULT 1,  -- 1 => needs qty/price fill
          note            TEXT,
          UNIQUE(position_id, symbol),
          FOREIGN KEY(position_id) REFERENCES positions(position_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_legs_pos ON legs(position_id);

        -- ============= jobs =============
        CREATE TABLE IF NOT EXISTS jobs(
          job_id      TEXT PRIMARY KEY,
          position_id INTEGER,
          task        TEXT NOT NULL,
          payload     TEXT,
          state       TEXT NOT NULL,
          attempts    INTEGER NOT NULL DEFAULT 0,
          last_error  TEXT,
          created_ts  INTEGER NOT NULL,
          updated_ts  INTEGER NOT NULL,
          worker_id   TEXT
        );

        -- ============= marks/pnl =========
        CREATE TABLE IF NOT EXISTS marks(
          ts INTEGER NOT NULL, symbol TEXT NOT NULL, mark_price REAL NOT NULL,
          PRIMARY KEY (ts, symbol)
        );
        CREATE TABLE IF NOT EXISTS pnl_events(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts INTEGER NOT NULL, symbol TEXT NOT NULL, type TEXT NOT NULL, amount REAL NOT NULL
        );

        -- ============= thinkers ==========
        CREATE TABLE IF NOT EXISTS thinkers (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          kind TEXT NOT NULL,
          enabled INTEGER NOT NULL DEFAULT 1,
          config_json TEXT NOT NULL,
          runtime_json TEXT,
          created_ts INTEGER NOT NULL,
          updated_ts INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_thinkers_kind ON thinkers(kind);

        CREATE TABLE IF NOT EXISTS thinker_state_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          thinker_id INTEGER NOT NULL,
          ts INTEGER NOT NULL,
          level TEXT NOT NULL,
          message TEXT NOT NULL,
          payload_json TEXT,
          FOREIGN KEY(thinker_id) REFERENCES thinkers(id) ON DELETE CASCADE
        );

        -- ============= indicator history ==========
        CREATE TABLE IF NOT EXISTS indicator_history (
          thinker_id  INTEGER NOT NULL,
          position_id INTEGER NOT NULL,
          name        TEXT    NOT NULL,
          ts_ms       INTEGER NOT NULL,
          value       REAL,
          price       REAL,
          aux_json    TEXT,
          PRIMARY KEY(thinker_id, position_id, name, ts_ms),
          FOREIGN KEY(position_id) REFERENCES positions(position_id) ON DELETE CASCADE,
          FOREIGN KEY(thinker_id) REFERENCES thinkers(id) ON DELETE CASCADE
        );

        -- ============= config (singleton) ==========
        CREATE TABLE IF NOT EXISTS config (
          id INTEGER PRIMARY KEY CHECK (id = 1),
          reference_balance REAL NOT NULL,
          leverage          REAL NOT NULL,
          default_risk      REAL NOT NULL,
          updated_ts        INTEGER NOT NULL,
          updated_by        TEXT
        );
        """)

        self.con.commit()

        # lightweight migrations for new columns/tables
        self._ensure_column("positions", "risk", "REAL NOT NULL DEFAULT 0.02")
        self._ensure_column("positions", "closed_ts", "INTEGER")
        self._ensure_config_row()
        self._ensure_indicator_tables()

    # -------- transaction manager --------
    @contextmanager
    def txn(self, write: bool = False):
        """
        Transaction context.
        - write=False: no BEGIN; useful for multi-step reads under the same cursor.
        - write=True: BEGIN IMMEDIATE, then COMMIT/ROLLBACK.
        """
        if write:
            with self._db_lock:
                cur = self.con.cursor()
                cur.execute("BEGIN IMMEDIATE;")
                try:
                    yield cur
                    self.con.commit()
                except Exception:
                    self.con.rollback()
                    raise
        else:
            # For read-only multi-step sequences (no lock required)
            cur = self.con.cursor()
            try:
                yield cur
            finally:
                # no commit needed
                pass

    # -------- schema helpers --------
    def _ensure_column(self, table: str, column: str, definition: str):
        """Add column if missing (idempotent)."""
        try:
            cur = self.con.execute(f"PRAGMA table_info({table});")
            cols = [r[1] for r in cur.fetchall()]
            if column in cols:
                return
            self.con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition};")
            self.con.commit()
        except Exception as e:
            log().exc(e, where="storage.ensure_column", table=table, column=column)

    def _ensure_config_row(self):
        """Create singleton config row with defaults if missing."""
        try:
            row = self.con.execute("SELECT id FROM config WHERE id=1").fetchone()
            if row:
                return
            ts = Clock.now_utc_ms()
            self.con.execute(
                "INSERT INTO config(id,reference_balance,leverage,default_risk,updated_ts,updated_by)"
                " VALUES(1,?,?,?,?,?)",
                (DEFAULT_CONFIG["reference_balance"], DEFAULT_CONFIG["leverage"],
                 DEFAULT_CONFIG["default_risk"], int(ts), "bootstrap")
            )
            self.con.commit()
        except Exception as e:
            log().exc(e, where="storage.ensure_config_row")

    def _ensure_indicator_tables(self):
        """Indicator history moved to a separate DB; drop legacy table if present."""
        try:
            with self.txn(write=True) as cur:
                cur.execute("DROP TABLE IF EXISTS indicator_history")
        except Exception as e:
            log().exc(e, where="storage.drop_legacy_indicator_history")

    def get_config(self) -> dict:
        """Return singleton config (expects row to exist)."""
        row = self.con.execute(
            "SELECT reference_balance, leverage, default_risk, updated_ts, updated_by FROM config WHERE id=1"
        ).fetchone()
        if not row:
            self._ensure_config_row()
            row = self.con.execute(
                "SELECT reference_balance, leverage, default_risk, updated_ts, updated_by FROM config WHERE id=1"
            ).fetchone()
        if not row:
            raise RuntimeError("config row missing")
        return {
            "reference_balance": float(row["reference_balance"]),
            "leverage": float(row["leverage"]),
            "default_risk": float(row["default_risk"]),
            "updated_ts": int(row["updated_ts"]) if row["updated_ts"] is not None else None,
            "updated_by": row["updated_by"],
        }

    def update_config(self, fields: dict) -> int:
        """Update singleton config row. Returns number of columns updated (0 if none)."""
        allowed = {"reference_balance", "leverage", "default_risk", "updated_by"}
        cols = {k: v for k, v in fields.items() if k in allowed}
        if not cols:
            return 0
        sets = ", ".join(f"{k}=?" for k in cols.keys())
        params = list(cols.values())
        params.append(Clock.now_utc_ms())
        try:
            with self.txn(write=True) as cur:
                cur.execute(f"UPDATE config SET {sets}, updated_ts=? WHERE id=1", params)
                if cur.rowcount == 0:
                    self._ensure_config_row()
                    cur.execute(f"UPDATE config SET {sets}, updated_ts=? WHERE id=1", params)
            return len(cols)
        except Exception as e:
            log().exc(e, where="storage.update_config")
            return 0

    # --- positions (auto-increment, unique signature) ---
    def get_or_create_position(self, num: str, den: Optional[str], dir_sign: int,
                               target_usd: float, risk: float, user_ts: int,
                               status: str = "OPEN", note: Optional[str] = None) -> int:
        ts = Clock.now_utc_ms()
        with self.txn(write=True) as cur:
            try:
                cur.execute("""
                    INSERT INTO positions (num,den,dir_sign,target_usd,risk,user_ts,status,note,created_ts)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (num, den, int(dir_sign), float(target_usd), float(risk), int(user_ts), status, note, ts))
                return int(cur.lastrowid)
            except sqlite3.IntegrityError:
                # already exists per UX index; fetch id
                r = cur.execute("""
                    SELECT position_id FROM positions
                    WHERE num=? AND den IS ? AND dir_sign=? AND target_usd=? AND user_ts=?
                """, (num, den, int(dir_sign), float(target_usd), int(user_ts))).fetchone()
                if not r:
                    raise
                return int(r["position_id"])

    def get_position(self, position_id: int, fmt: str = "obj") -> Optional[Any]:
        row = self.con.execute("SELECT * FROM positions WHERE position_id=?", (int(position_id),)).fetchone()
        if row is None:
            return None
        if fmt == "row":
            return row
        from engclasses import Position  # lazy to avoid circular import at module load
        return Position.from_row(row)

    def list_open_positions(self) -> List[sqlite3.Row]:
        return self.con.execute("SELECT * FROM positions WHERE status='OPEN'").fetchall()

    def close_position(self, position_id: int):
        with self.txn(write=True) as cur:
            ts = Clock.now_utc_ms()
            cur.execute(
                "UPDATE positions SET status='CLOSED', closed_ts=? WHERE position_id=?",
                (ts, int(position_id)),
            )

    def set_indicator_history(self, ih):
        """Attach external IndicatorHistory for cleanup hooks."""
        self._indicator_history = ih

    # --- legs (stubs + fulfill) ---
    def ensure_leg_stub(self, position_id: int, symbol: str):
        """Create the leg if absent, marked as needing backfill."""
        with self.txn(write=True) as cur:
            cur.execute("""
                INSERT OR IGNORE INTO legs(position_id, symbol, need_backfill)
                VALUES (?, ?, 1)
            """, (int(position_id), symbol))

    def fulfill_leg(self, position_id: int, symbol: str, qty: float, price: float, price_ts: int, method: str):
        """Fill qty/price and clear backfill flag."""
        with self.txn(write=True) as cur:
            cur.execute("""
                UPDATE legs
                SET qty = ?, entry_price = ?, entry_price_ts = ?, price_method = ?, need_backfill = 0
                WHERE position_id = ? AND symbol = ?
            """, (float(qty), float(price), int(price_ts), method, int(position_id), symbol))

    def legs_needing_backfill(self, position_id: int):
        return self.con.execute("""
            SELECT * FROM legs WHERE position_id=? AND need_backfill=1
        """, (int(position_id),)).fetchall()

    # --- positions upsert (kept) ---
    def upsert_position(self, row: dict):
        q = """INSERT OR IGNORE INTO positions(position_id,num,den,dir_sign,target_usd,risk,user_ts,status,note,created_ts)
               VALUES(:position_id,:num,:den,:dir_sign,:target_usd,:risk,:user_ts,:status,:note,:created_ts)"""
        if "closed_ts" in row:
            q = """INSERT OR IGNORE INTO positions(position_id,num,den,dir_sign,target_usd,risk,user_ts,closed_ts,status,note,created_ts)
                   VALUES(:position_id,:num,:den,:dir_sign,:target_usd,:risk,:user_ts,:closed_ts,:status,:note,:created_ts)"""
        with self.txn(write=True) as cur:
            cur.execute(q, row)

    # --- legs append (kept as-is) ---
    def upsert_leg(self, row: dict):
        """
        Insert a NEW leg row (signed qty). This is append-only by design.
        row keys: position_id, symbol, qty (signed), entry_price, entry_price_ts, price_method, note
        """
        q = """INSERT INTO legs(position_id,symbol,qty,entry_price,entry_price_ts,price_method,note)
               VALUES(:position_id,:symbol,:qty,:entry_price,:entry_price_ts,:price_method,:note)"""
        with self.txn(write=True) as cur:
            cur.execute(q, row)

    def get_legs(self, position_id: int) -> List[sqlite3.Row]:
        return self.con.execute("SELECT * FROM legs WHERE position_id=? ORDER BY leg_id", (int(position_id),)).fetchall()

    def delete_position_completely(self, position_id: int) -> int:
        """
        Permanently remove a position and all related data (legs, jobs, etc.).

        Returns:
            int: total number of rows deleted across tables (best-effort).

        Notes:
            - Idempotent: if the id is missing, deletes 0 rows.
            - Wrapped in a single transaction; will rollback on any failure.
            - Extend here if you add new tables referencing positions (alerts, notes, etc.).
        """
        with self.txn(write=True) as cur:
            # Delete dependents first (order matters if you add FKs later)
            cur.execute("DELETE FROM legs WHERE position_id = ?", (position_id,))
            n_legs = cur.rowcount or 0

            cur.execute("DELETE FROM jobs WHERE position_id = ?", (position_id,))
            n_jobs = cur.rowcount or 0

            # If you have an alerts/rules table tied to positions, delete here too:
            # cur.execute("DELETE FROM alerts WHERE position_id = ?", (position_id,))
            # n_alerts = cur.rowcount or 0

            cur.execute("DELETE FROM positions WHERE position_id = ?", (position_id,))
            n_pos = cur.rowcount or 0

            log().info("position.deleted", position_id=position_id,
                       rows=dict(legs=n_legs, jobs=n_jobs, positions=n_pos))
            total = (n_legs + n_jobs + n_pos)
        # cleanup indicator history if hooked
        try:
            if self._indicator_history:
                total += self._indicator_history.delete_by_position(position_id)
        except Exception as e:
            log().exc(e, where="storage.delete_position.indicator_history", position_id=position_id)
        return total

    # --- jobs ---
    def enqueue_job(self, job_id: str, task: str, payload: dict, position_id: Optional[str] = None):
        ts = Clock.now_utc_ms()
        row = {
            "job_id": job_id, "position_id": position_id, "task": task,
            "payload": json.dumps(payload), "state": "PENDING",
            "attempts": 0, "last_error": None, "created_ts": ts, "updated_ts": ts
        }
        with self.txn(write=True) as cur:
            cur.execute("""INSERT OR IGNORE INTO jobs(job_id,position_id,task,payload,state,attempts,last_error,created_ts,updated_ts)
                           VALUES(:job_id,:position_id,:task,:payload,:state,:attempts,:last_error,:created_ts,:updated_ts)""", row)

    def recover_stale_jobs(self) -> int:
        with self.txn(write=True) as cur:
            cur.execute("UPDATE jobs SET state='PENDING' WHERE state='RUNNING'")
            n = cur.rowcount or 0
        if n:
            log().warn("jobs.recovered", count=n)
        return n

    def fetch_next_job(self) -> Optional[sqlite3.Row]:
        """
        Atomically claim the next PENDING job by flipping to RUNNING inside a write txn.
        Uses BEGIN IMMEDIATE to avoid writer races.
        """
        with self.txn(write=True) as cur:
            job = cur.execute(
                "SELECT * FROM jobs WHERE state='PENDING' ORDER BY created_ts LIMIT 1"
            ).fetchone()
            if job:
                cur.execute(
                    "UPDATE jobs SET state='RUNNING', updated_ts=? WHERE job_id=?",
                    (Clock.now_utc_ms(), job["job_id"])
                )
            return job

    def finish_job(self, job_id: str, ok: bool = True, error: str | None = None):
        st = "DONE" if ok else "ERR"
        with self.txn(write=True) as cur:
            cur.execute(
                "UPDATE jobs SET state=?, attempts=attempts+1, last_error=?, updated_ts=? WHERE job_id=?",
                (st, error, Clock.now_utc_ms(), job_id)
            )

    # --- marks & income ---
    def insert_mark(self, ts: int, symbol: str, price: float):
        with self.txn(write=True) as cur:
            cur.execute("INSERT OR IGNORE INTO marks(ts,symbol,mark_price) VALUES(?,?,?)", (ts, symbol, price))

    def insert_income(self, rows: List[dict]):
        with self.txn(write=True) as cur:
            for r in rows:
                cur.execute("""INSERT INTO pnl_events(ts,symbol,type,amount) VALUES(?,?,?,?)""",
                            (r["time"], r.get("symbol", "UNKNOWN"), r["incomeType"], float(r["income"])))

    # ----- THINKERS CRUD / HELPERS -----

    def insert_thinker(self, kind: str, config: dict) -> int:
        ts = Clock.now_utc_ms()
        with self.txn(write=True) as cur:
            cur.execute("""INSERT INTO thinkers(kind,enabled,config_json,runtime_json,created_ts,updated_ts)
                           VALUES(?,?,?,?,?,?)""",
                        (kind, 1, json.dumps(config, ensure_ascii=False), "{}", ts, ts))
            return int(cur.lastrowid)

    def get_thinker(self, thinker_id: int) -> Optional[sqlite3.Row]:
        return self.con.execute("SELECT * FROM thinkers WHERE id=?", (thinker_id,)).fetchone()

    def update_thinker_enabled(self, thinker_id: int, enabled: bool):
        with self.txn(write=True) as cur:
            cur.execute("UPDATE thinkers SET enabled=?, updated_ts=? WHERE id=?",
                        (1 if enabled else 0, Clock.now_utc_ms(), thinker_id))

    def update_thinker_config(self, thinker_id: int, config: dict):
        with self.txn(write=True) as cur:
            cur.execute("""UPDATE thinkers SET config_json=?, updated_ts=? WHERE id=?""",
                        (json.dumps(config or {}, ensure_ascii=False), Clock.now_utc_ms(), thinker_id))

    def delete_thinker(self, thinker_id: int):
        with self.txn(write=True) as cur:
            cur.execute("DELETE FROM thinkers WHERE id=?", (thinker_id,))

    def list_thinkers(self) -> List[sqlite3.Row]:
        return self.con.execute("SELECT * FROM thinkers ORDER BY id").fetchall()

    def update_thinker_runtime(self, thinker_id: int, runtime: dict) -> None:
        try:
            payload = json.dumps(runtime or {}, ensure_ascii=False)
        except Exception as e:
            pretty = repr(runtime)
            log().error(f"thinker.runtime.json.dump.failed: {e}\nruntime={pretty}")
            raise
        with self.txn(write=True) as cur:
            cur.execute(
                """UPDATE thinkers SET runtime_json=?, updated_ts=? WHERE id=?""",
                (payload, Clock.now_utc_ms(), int(thinker_id)),
            )

    def log_thinker_event(self, thinker_id: int, level: str, message: str, payload: dict | None = None):
        with self.txn(write=True) as cur:
            cur.execute("""INSERT INTO thinker_state_log(thinker_id, ts, level, message, payload_json)
                           VALUES(?,?,?,?,?)""",
                        (thinker_id, Clock.now_utc_ms(), level, message,
                         json.dumps(payload or {}, ensure_ascii=False)))

    # ----- indicator + exit/trailing helpers -----

    # ------------------- JOBS (inspection / retries) -------------------

    def list_jobs(self, state: Optional[str] = None, limit: int = 50) -> List[sqlite3.Row]:
        q = "SELECT job_id, position_id, task, state, attempts, last_error, created_ts, updated_ts FROM jobs"
        params = []
        if state:
            q += " WHERE state=?"
            params.append(state.upper())
        q += " ORDER BY created_ts DESC LIMIT ?"
        params.append(int(limit))
        return self.con.execute(q, params).fetchall()

    def retry_job(self, job_id: str) -> bool:
        with self.txn(write=True) as cur:
            cur.execute("UPDATE jobs SET state='PENDING', updated_ts=? WHERE job_id=?",
                        (Clock.now_utc_ms(), job_id))
            return cur.rowcount > 0

    def retry_failed_jobs(self, limit: Optional[int] = None) -> int:
        # select first to honor an optional limit
        q = "SELECT job_id FROM jobs WHERE state='ERR' ORDER BY updated_ts DESC"
        rows = self.con.execute(q + (" LIMIT ?" if limit is not None else ""),
                                ((int(limit),) if limit is not None else ())).fetchall()
        n = 0
        for r in rows:
            if self.retry_job(r["job_id"]):
                n += 1
        return n

    # -------- generic UPDATE helper (kept, now locked) --------
    def sql_update(self, table: str, pk_field: str, pk_value: int, fields: dict) -> int:
        """Perform a parameterized UPDATE; returns rowcount."""
        if not fields:
            return 0
        cols = sorted(fields.keys())
        sets = ", ".join(f"{c}=?" for c in cols)
        args = [fields[c] for c in cols] + [pk_value]
        with self.txn(write=True) as cur:
            cur.execute(f"UPDATE {table} SET {sets} WHERE {pk_field}=?", args)
            return cur.rowcount
