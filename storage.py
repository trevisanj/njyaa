# storage.py
from __future__ import annotations
import sqlite3
import json
from common import Clock
from typing import List, Optional
from common import log


# =======================
# ====== STORAGE ========
# =======================

class Storage:
    def __init__(self, path):
        self.path = path
        self.con = sqlite3.connect(self.path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self._init_db()
        self._set_wal()

    def _set_wal(self):
        cur = self.con.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
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
          user_ts       INTEGER NOT NULL,        -- user-declared timestamp (ms)
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
        """)
        self.con.commit()

    # --- positions (auto-increment, unique signature) ---
    def get_or_create_position(self, num: str, den: Optional[str], dir_sign: int,
                               target_usd: float, user_ts: int,
                               status: str = "OPEN", note: Optional[str] = None) -> int:
        ts = Clock.now_utc_ms()
        cur = self.con.cursor()
        try:
            cur.execute("""
                INSERT INTO positions (num,den,dir_sign,target_usd,user_ts,status,note,created_ts)
                VALUES (?,?,?,?,?,?,?,?)
            """, (num, den, int(dir_sign), float(target_usd), int(user_ts), status, note, ts))
            self.con.commit()
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

    def get_position(self, position_id: int):
        return self.con.execute("SELECT * FROM positions WHERE position_id=?", (int(position_id),)).fetchone()

    def list_open_positions(self):
        return self.con.execute("SELECT * FROM positions WHERE status='OPEN'").fetchall()

    def close_position(self, position_id: int):
        self.con.execute("UPDATE positions SET status='CLOSED' WHERE position_id=?", (int(position_id),))
        self.con.commit()

    # --- legs (stubs + fulfill) ---
    def ensure_leg_stub(self, position_id: int, symbol: str):
        """Create the leg if absent, marked as needing backfill."""
        self.con.execute("""
            INSERT OR IGNORE INTO legs(position_id, symbol, need_backfill)
            VALUES (?, ?, 1)
        """, (int(position_id), symbol))
        self.con.commit()

    def fulfill_leg(self, position_id: int, symbol: str, qty: float, price: float, price_ts: int, method: str):
        """Fill qty/price and clear backfill flag."""
        self.con.execute("""
            UPDATE legs
            SET qty = ?, entry_price = ?, entry_price_ts = ?, price_method = ?, need_backfill = 0
            WHERE position_id = ? AND symbol = ?
        """, (float(qty), float(price), int(price_ts), method, int(position_id), symbol))
        self.con.commit()

    def legs_needing_backfill(self, position_id: int):
        return self.con.execute("""
            SELECT * FROM legs WHERE position_id=? AND need_backfill=1
        """, (int(position_id),)).fetchall()

    # --- positions
    def upsert_position(self, row: dict):
        q = """INSERT OR IGNORE INTO positions(position_id,num,den,dir_sign,target_usd,user_ts,status,note,created_ts)
               VALUES(:position_id,:num,:den,:dir_sign,:target_usd,:user_ts,:status,:note,:created_ts)"""
        self.con.execute(q, row); self.con.commit()

    def get_position(self, position_id:str) -> Optional[sqlite3.Row]:
        r = self.con.execute("SELECT * FROM positions WHERE position_id=?", (position_id,)).fetchone()
        return r

    def list_open_positions(self) -> List[sqlite3.Row]:
        return self.con.execute("SELECT * FROM positions WHERE status='OPEN'").fetchall()

    def close_position(self, position_id:str):
        self.con.execute("UPDATE positions SET status='CLOSED' WHERE position_id=?", (position_id,))
        self.con.commit()

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
        cur = self.con.cursor()
        try:
            cur.execute("BEGIN")

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

            self.con.commit()
            log().info("position.deleted", position_id=position_id,
                       rows=dict(legs=n_legs, jobs=n_jobs, positions=n_pos))
            return (n_legs + n_jobs + n_pos)
        except Exception as e:
            self.con.rollback()
            log().exc(e, where="storage.delete_position_completely", position_id=position_id)
            raise

    # --- legs
    def upsert_leg(self, row: dict):
        """
        Insert a NEW leg row (signed qty). This is append-only by design.
        row keys: position_id, symbol, qty (signed), entry_price, entry_price_ts, price_method, note
        """
        q = """INSERT INTO legs(position_id,symbol,qty,entry_price,entry_price_ts,price_method,note)
               VALUES(:position_id,:symbol,:qty,:entry_price,:entry_price_ts,:price_method,:note)"""
        self.con.execute(q, row)
        self.con.commit()

    def get_legs(self, position_id: str) -> List[sqlite3.Row]:
        return self.con.execute("SELECT * FROM legs WHERE position_id=? ORDER BY leg_id", (position_id,)).fetchall()

    # --- jobs
    def enqueue_job(self, job_id:str, task:str, payload:dict, position_id:Optional[str]=None):
        ts = Clock.now_utc_ms()
        row = {
            "job_id": job_id, "position_id": position_id, "task": task,
            "payload": json.dumps(payload), "state": "PENDING",
            "attempts": 0, "last_error": None, "created_ts": ts, "updated_ts": ts
        }
        self.con.execute("""INSERT OR IGNORE INTO jobs(job_id,position_id,task,payload,state,attempts,last_error,created_ts,updated_ts)
                            VALUES(:job_id,:position_id,:task,:payload,:state,:attempts,:last_error,:created_ts,:updated_ts)""", row)
        self.con.commit()

    def recover_stale_jobs(self):
        cur = self.con.execute("UPDATE jobs SET state='PENDING' WHERE state='RUNNING'")
        n = cur.rowcount
        self.con.commit()
        if n:
            log().warn("jobs.recovered", count=n)
        return n

    # Storage
    def fetch_next_job(self) -> Optional[sqlite3.Row]:
        cur = self.con.cursor()
        cur.execute("BEGIN IMMEDIATE TRANSACTION;")
        job = cur.execute("SELECT * FROM jobs WHERE state='PENDING' ORDER BY created_ts LIMIT 1").fetchone()
        if job:
            cur.execute("UPDATE jobs SET state='RUNNING', updated_ts=? WHERE job_id=?", (Clock.now_utc_ms(), job["job_id"]))
        self.con.commit()
        return job

    def finish_job(self, job_id:str, ok=True, error:str=None):
        st = "DONE" if ok else "ERR"
        self.con.execute("UPDATE jobs SET state=?, attempts=attempts+1, last_error=?, updated_ts=? WHERE job_id=?",
                         (st, error, Clock.now_utc_ms(), job_id))
        self.con.commit()

    # --- marks & income
    def insert_mark(self, ts:int, symbol:str, price:float):
        self.con.execute("INSERT OR IGNORE INTO marks(ts,symbol,mark_price) VALUES(?,?,?)", (ts,symbol,price))
        self.con.commit()

    def insert_income(self, rows: List[dict]):
        cur = self.con.cursor()
        for r in rows:
            cur.execute("""INSERT INTO pnl_events(ts,symbol,type,amount) VALUES(?,?,?,?)""",
                        (r["time"], r.get("symbol","UNKNOWN"), r["incomeType"], float(r["income"])))
        self.con.commit()

    # ----- THINKERS CRUD / HELPERS -----

    def insert_thinker(self, kind:str, config:dict) -> int:
        ts = Clock.now_utc_ms()
        cur = self.con.cursor()
        cur.execute("""INSERT INTO thinkers(kind,enabled,config_json,runtime_json,created_ts,updated_ts)
                       VALUES(?,?,?,?,?,?)""",
                    (kind, 1, json.dumps(config, ensure_ascii=False), "{}", ts, ts))
        self.con.commit()
        return int(cur.lastrowid)

    def update_thinker_enabled(self, thinker_id:int, enabled:bool):
        self.con.execute("UPDATE thinkers SET enabled=?, updated_ts=? WHERE id=?",
                         (1 if enabled else 0, Clock.now_utc_ms(), thinker_id))
        self.con.commit()

    def delete_thinker(self, thinker_id:int):
        self.con.execute("DELETE FROM thinkers WHERE id=?", (thinker_id,))
        self.con.commit()

    def list_thinkers(self) -> List[sqlite3.Row]:
        return self.con.execute("SELECT * FROM thinkers ORDER BY id").fetchall()

    def log_thinker_event(self, thinker_id:int, level:str, message:str, payload:dict|None=None):
        self.con.execute("""INSERT INTO thinker_state_log(thinker_id, ts, level, message, payload_json)
                            VALUES(?,?,?,?,?)""",
                         (thinker_id, Clock.now_utc_ms(), level, message, json.dumps(payload or {}, ensure_ascii=False)))
        self.con.commit()


    # ------------------- JOBS

    # --- jobs inspection / retries ---
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
        cur = self.con.cursor()
        cur.execute("UPDATE jobs SET state='PENDING', updated_ts=? WHERE job_id=?",
                    (Clock.now_utc_ms(), job_id))
        self.con.commit()
        return cur.rowcount > 0

    def retry_failed_jobs(self, limit: Optional[int] = None) -> int:
        # select first to honor an optional limit
        q = "SELECT job_id FROM jobs WHERE state='ERR' ORDER BY updated_ts DESC"
        if limit is not None:
            q += " LIMIT ?"
            rows = self.con.execute(q, (int(limit),)).fetchall()
        else:
            rows = self.con.execute(q).fetchall()
        n = 0
        for r in rows:
            if self.retry_job(r["job_id"]):
                n += 1
        return n


    def sql_update(self, table: str, pk_field: str, pk_value: int, fields: dict) -> int:
        """Perform a parameterized UPDATE; returns rowcount."""
        if not fields:
            return 0
        cols = sorted(fields.keys())
        sets = ", ".join(f"{c}=?" for c in cols)
        args = [fields[c] for c in cols] + [pk_value]
        cur = self.con.cursor()
        cur.execute(f"UPDATE {table} SET {sets} WHERE {pk_field}=?", args)
        self.con.commit()
        return cur.rowcount
