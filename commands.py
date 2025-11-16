# commands.py
from __future__ import annotations
import hashlib, math
from typing import Any
from typing import Callable, Dict, Tuple, Optional
from dataclasses import dataclass, field
import sys, json, threading
from common import *
from typing import Callable, List, Optional
from storage import Storage
import re
from datetime import datetime, timezone, timedelta
from binance_um import BinanceUM
import threading, time
from typing import Callable, Dict
from bot_api import BotEngine
import enghelpers as eh


class CommandRegistry:
    """
    Unified command registry with positional argspec + key:value options.

    - Register handlers via @R.at(name, argspec=[...], options=[...]) or
      @R.bang(name, argspec=[...], options=[...])

    - Dispatcher parses:
        * positionals in order, then
        * remaining tokens as key:value options (only those declared).

    - Auto-help: builds usage + summary from registered metadata/docstrings.
    """
    def __init__(self):
        # keys are ('@'|'!', command_name)
        self._handlers: Dict[Tuple[str, str], Callable[[BotEngine, Dict[str, str]], str]] = {}
        self._meta: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # ---- decorators ----
    def at(self, name: str, argspec: List[str] = None, options: List[str] = None):
        return self._reg(('@', name.lower()), argspec, options)

    def bang(self, name: str, argspec: List[str] = None, options: List[str] = None):
        return self._reg(('!', name.lower()), argspec, options)

    def _reg(self, key: Tuple[str, str], argspec: Optional[List[str]], options: Optional[List[str]]):
        def deco(fn):
            self._handlers[key] = fn
            # Extract first docstring line as summary (if present)
            doc = (fn.__doc__ or "").strip()
            summary = ""
            if doc:
                summary = doc.splitlines()[0].strip()
            self._meta[key] = {
                "argspec": list(argspec or []),
                "options": set(options or []),
                "summary": summary,
                "name": key[1],
                "prefix": key[0],
            }
            return fn
        return deco

    # ---- dispatcher ----
    def dispatch(self, eng: BotEngine, msg: str) -> str:
        s = (msg or "").strip()
        log().debug("dispatch.enter", text=s)
        if not s or s[0] not in ("@", "!"):
            log().debug("dispatch.exit", reason="not-a-command")
            return self._help_text()

        prefix = s[0]
        parts = s.split(None, 1)
        if not parts:
            log().debug("dispatch.exit", reason="empty-after-prefix")
            return "Empty command."

        head = parts[0][1:].lower()
        tail = parts[1].strip() if len(parts) > 1 else ""
        handler = self._handlers.get((prefix, head))
        meta = self._meta.get((prefix, head))

        if not handler:
            log().warn("dispatch.unknown", prefix=prefix, head=head)
            return f"Unknown {prefix}{head}. Try @help."

        # Parse args according to argspec/options
        args = self._parse_args(tail, meta)
        missing = [a for a in meta["argspec"] if a not in args]
        if missing:
            return self._usage_line(meta, reason=f"Missing: {', '.join(missing)}")

        log().debug("dispatch.call", cmd=head, prefix=prefix, args=args)
        try:
            out = handler(eng, args)
            log().debug("dispatch.ok", cmd=head)
            return out
        except Exception as e:
            log().exc(e, where="dispatch.handler", cmd=head)
            return f"Error: {e}"

    # ---- parsing helpers ----
    def _parse_args(self, tail: str, meta: Dict[str, Any]) -> Dict[str, str]:
        argspec: List[str] = meta.get("argspec", [])
        options: set = meta.get("options", set())

        toks = tail.split() if tail else []
        result: Dict[str, str] = {}

        # Fill positionals
        for name in argspec:
            if toks:
                result[name] = toks.pop(0)
            else:
                break

        # Remaining tokens as key:value (only declared options)
        for tok in toks:
            if ":" in tok:
                k, v = tok.split(":", 1)
                k = k.strip().lower()
                v = v.strip()
                if not options or k in options:
                    result[k] = v
            else:
                log().debug("dispatch.ignored-token", token=tok)

        return result

    # ---- help/usage ----
    def _usage_line(self, meta: Dict[str, Any], reason: Optional[str] = None) -> str:
        pre = meta["prefix"]
        nm = meta["name"]
        pos = " ".join(f"<{a}>" for a in meta["argspec"])
        opt = ""
        if meta["options"]:
            opt_items = " ".join(f"[{o}:…]" for o in sorted(meta["options"]))
            opt = (" " + opt_items) if opt_items else ""
        line = f"{pre}{nm}" + (f" {pos}" if pos else "") + opt
        if reason:
            return f"{reason}\n{line}"
        return line

    def _help_text(self) -> str:
        """
        Build a help listing from registered commands.
        Shows: @name/!name — usage — first doc line (if any).
        """
        lines: List[str] = ["Commands:"]
        # sort by prefix then name
        for (prefix, name) in sorted(self._handlers.keys()):
            meta = self._meta.get((prefix, name), {})
            usage = self._usage_line(meta)
            summary = meta.get("summary") or ""
            if summary:
                lines.append(f"  {usage}\n    {summary}\n")
            else:
                lines.append(f"  {usage}\n")
        return "\n".join(lines)


def build_registry() -> CommandRegistry:
    R = CommandRegistry()

    # ----------------------- HELP -----------------------
    # TODO improve help formatting
    @R.at("help")
    def _help(eng: BotEngine, args: Dict[str, str]) -> str:
        """Show this help."""
        return R._help_text()

    # ----------------------- OPEN (alias) -----------------------
    @R.at("open")
    def _at_open(eng: BotEngine, args: Dict[str, str]) -> str:
        """List open RV positions (summary). Alias for: @positions status:open detail:2"""
        return R.dispatch(eng, "@positions status:open detail:2")

    # ----------------------- POSITIONS (detail levels) -----------------------
    @R.at("positions", options=["status", "detail", "sort", "limit", "position_id", "pair"])
    def _at_positions(eng: BotEngine, args: Dict[str, str]) -> str:
        """
        Positions report with detail levels.

        detail: 1 = summary
                2 = summary + per-position overview (with opened datetime & signed target)
                3 = level 2 + per-symbol leg summary (agg qty, WAP, last price, PnL)
                4 = level 2 + list every leg (timestamp, entry, last, qty, PnL)

        Examples:
          @positions detail:1
          @positions status:closed detail:2
          @positions pair:STRK/ETH detail:3 limit:20
        """
        store = eng.store

        status = args.get("status", "open")
        # Back-compat: map what: to detail
        detail = int(args.get("detail", 2))
        limit = int(args.get("limit", "100"))
        pid = args.get("position_id")
        raw_pair = args.get("pair")
        pair = None
        if raw_pair:
            pair = eh.parse_pair_or_single(eng, raw_pair.upper())

        # fetch rows by status
        if status == "open":
            rows = store.list_open_positions()
        elif status == "closed":
            rows = store.con.execute("SELECT * FROM positions WHERE status='CLOSED'").fetchall()
        else:
            rows = store.con.execute("SELECT * FROM positions").fetchall()

        # filters
        if pid:
            rows = [r for r in rows if r["position_id"] == int(pid)]

        if pair:
            num, den = pair
            def _match(r):
                if den is None:
                    legs = store.get_legs(r["position_id"])
                    return any((lg["symbol"] or "").startswith(num) for lg in legs)
                return (r["num"].startswith(num) and r["den"].startswith(den))
            rows = [r for r in rows if _match(r)]

        # Precompute last cached prices for all involved symbols
        involved_syms = set()
        for r in rows:
            for lg in store.get_legs(r["position_id"]):
                if lg["symbol"]:
                    involved_syms.add(lg["symbol"])
        marks: Dict[str, Optional[float]] = {s: _last_cached_price(eng, s) for s in involved_syms}

        # ---------- detail 1: summary ----------
        total_target = 0.0
        total_pnl = 0.0
        pnl_missing_count = 0
        for r in rows:
            total_target += _position_signed_target(r)
            # Sum position PnL from legs with None-safe handling
            pos_pnl = 0.0
            pos_missing = 0
            for lg in store.get_legs(r["position_id"]):
                m = marks.get(lg["symbol"])
                pnl = _leg_pnl(lg["entry_price"], lg["qty"], m)
                if pnl is None:
                    pos_missing += 1
                    continue
                pos_pnl += pnl
            total_pnl += pos_pnl
            pnl_missing_count += (1 if pos_missing else 0)

        lines: List[str] = []
        lines.append(
            f"Positions: {len(rows)} | Target ≈ ${_fmt_num(total_target, 2)} | "
            f"PNL ≈ ${_fmt_num(total_pnl, 2)}"
            + (f" (PnL incomplete for {pnl_missing_count} position(s))" if pnl_missing_count else "")
        )
        if detail == 1:
            return "\n".join(lines)

        # ---------- detail 2+: per-position overview ----------
        count = 0
        for r in rows:
            if count >= limit:
                break
            pid = r["position_id"]
            opened_ms = r["user_ts"] or r["created_ts"]
            opened_str = _ts_human(opened_ms)
            signed_target = _position_signed_target(r)

            # compute position pnl and missing flag
            pos_pnl = 0.0
            any_missing = False
            for lg in store.get_legs(pid):
                m = marks.get(lg["symbol"])
                pnl = _leg_pnl(lg["entry_price"], lg["qty"], m)
                if pnl is None:
                    any_missing = True
                else:
                    pos_pnl += pnl

            pnl_str = _fmt_num(pos_pnl, 2)
            if any_missing:
                pnl_str += " (incomplete)"

            lines.append(
                f"{pid} {r['num']} / {r['den'] or '-'} "
                f"status={r['status']} opened={opened_str} "
                f"signed_target=${_fmt_num(signed_target, 2)} PNL=${pnl_str}"
            )

            # ---------- detail 3: per-symbol leg summary ----------
            if detail == 3:
                # aggregate per symbol: signed qty sum; WAP by abs(qty) as weight
                by_sym: Dict[str, Dict[str, float]] = {}
                missing_map: Dict[str, bool] = {}
                for lg in store.get_legs(pid):
                    s = lg["symbol"]
                    q = lg["qty"]
                    ep = lg["entry_price"]
                    if s not in by_sym:
                        by_sym[s] = {"qty": 0.0, "wap_num": 0.0, "wap_den": 0.0, "pnl": 0.0}
                        missing_map[s] = False
                    if q is not None:
                        by_sym[s]["qty"] += float(q)
                    if ep is not None and q is not None:
                        w = abs(float(q))
                        by_sym[s]["wap_num"] += float(ep) * w
                        by_sym[s]["wap_den"] += w
                    # pnl accumulation
                    pnl = _leg_pnl(ep, q, marks.get(s))
                    if pnl is None:
                        missing_map[s] = True
                    else:
                        by_sym[s]["pnl"] += pnl

                for s, acc in by_sym.items():
                    wap = (acc["wap_num"] / acc["wap_den"]) if acc["wap_den"] > 0 else None
                    last = marks.get(s)
                    pnl_s = _fmt_num(acc["pnl"], 2)
                    if missing_map.get(s):
                        pnl_s += " (incomplete)"
                    lines.append(
                        f"  - {s}  qty={_fmt_qty(acc['qty'])}  entry≈{_fmt_num(wap, 6)}  "
                        f"last={_fmt_num(last, 6)}  pnl=${pnl_s}"
                    )

            # ---------- detail 4: list every leg ----------
            if detail == 4:
                for lg in store.get_legs(pid):
                    ts = lg["entry_price_ts"]
                    ts_h = _ts_human(ts)
                    last = marks.get(lg["symbol"])
                    pnl = _leg_pnl(lg["entry_price"], lg["qty"], last)
                    pnl_str = _fmt_num(pnl, 2)
                    if pnl is None:
                        pnl_str = "? (missing price/entry/qty)"
                    lines.append(
                        f"  - {lg['symbol']}  t={ts_h}  qty={_fmt_qty(lg['qty'])}  "
                        f"entry={_fmt_num(lg['entry_price'], 6)}  last={_fmt_num(last, 6)}  pnl=${pnl_str}"
                    )

            count += 1

        return "\n".join(lines)

    # ----------------------- !OPEN POSITION -----------------------
    @R.bang("open", argspec=["pair", "ts", "usd"], options=["note"])
    def _bang_open(eng: BotEngine, args: Dict[str, str]) -> str:
        """
        Open/add an RV position.

        Usage:
          !open <ISO|epoch_ms> <NUM/DEN|SYMBOL> <±usd> [note:...]
        Examples:
          !open STRK/ETH 2025-11-10T13:44:05  -5000 note:rv test
          !open ETHUSDT  2025-11-10T13:44:05 +3000
        """
        ts_ms = parse_when(args["ts"])
        num, den = eh.parse_pair_or_single(eng, args["pair"])
        usd = int(float(args["usd"]))
        note = args.get("note", "")

        pid = eng.positionbook.open_position(num, den, usd, ts_ms, note=note)
        return f"Opened pair {pid}: {num}/{den} target=${abs(usd):.0f} (queued price backfill)."

    # ----------------------- THINKERS LIST -----------------------
    @R.at("thinkers")
    def _at_thinkers(eng: BotEngine, args: Dict[str, str]) -> str:
        """List thinkers stored in DB."""
        rows = eng.store.list_thinkers()
        if not rows:
            return "No thinkers."
        out = []
        for r in rows:
            out.append(f"#{r['id']} {r['kind']} enabled={r['enabled']} cfg={r['config_json']}")
        return "\n".join(out)

    # ----------------------- THINKER KINDS -----------------------
    @R.at("thinker-kinds")
    def _at_thinker_kinds(eng: BotEngine, args: Dict[str, str]) -> str:
        """List available thinker kinds (from factory auto-discovery)."""
        try:
            kinds = list(eng.thinkers.factory.kinds())
        except Exception:
            kinds = []
        return "Available thinker kinds:\n" + ("\n".join(sorted(kinds)) if kinds else "(none)")

    # ----------------------- THINKER ENABLE -----------------------
    @R.bang("thinker-enable", argspec=["id"])
    def _bang_thinker_enable(eng: BotEngine, args: Dict[str, str]) -> str:
        """Enable a thinker by ID. Usage: !thinker-enable <id>"""
        tid = args["id"].strip()
        if not tid.isdigit():
            return "Usage: !thinker-enable <id>"
        eng.store.update_thinker_enabled(int(tid), True)
        return f"Thinker #{tid} enabled."

    # ----------------------- THINKER DISABLE -----------------------
    @R.bang("thinker-disable", argspec=["id"])
    def _bang_thinker_disable(eng: BotEngine, args: Dict[str, str]) -> str:
        """Disable a thinker by ID. Usage: !thinker-disable <id>"""
        tid = args["id"].strip()
        if not tid.isdigit():
            return "Usage: !thinker-disable <id>"
        eng.store.update_thinker_enabled(int(tid), False)
        return f"Thinker #{tid} disabled."

    # ----------------------- THINKER REMOVE -----------------------
    @R.bang("thinker-rm", argspec=["id"])
    def _bang_thinker_rm(eng: BotEngine, args: Dict[str, str]) -> str:
        """Delete a thinker by ID. Usage: !thinker-rm <id>"""
        tid = args["id"].strip()
        if not tid.isdigit():
            return "Usage: !thinker-rm <id>"
        eng.store.delete_thinker(int(tid))
        return f"Thinker #{tid} deleted."

    # ----------------------- ALERT -----------------------
    @R.bang("alert", argspec=["symbol", "op", "price"], options=["msg"])
    def _bang_alert(eng: BotEngine, args: Dict[str, str]) -> str:
        """
        Create a simple threshold alert thinker.

        Usage:
          !alert <SYMBOL> (>=|<=) <PRICE> [msg:...]
        """
        sym = args["symbol"].upper()
        op = args["op"]
        pr = args["price"]
        if op not in (">=", "<="):
            return "Op must be >= or <="

        try:
            price = float(pr)
        except:
            return "Bad price."

        direction = "ABOVE" if op == ">=" else "BELOW"
        msg = args.get("msg", "")
        cfg = {"symbol": sym, "direction": direction, "price": price, "message": msg}
        tid = eng.store.insert_thinker("THRESHOLD_ALERT", cfg)
        return f"Thinker #{tid} THRESHOLD_ALERT set for {sym} {direction} {price}"

    # ----------------------- PSAR -----------------------
    @R.bang("psar", argspec=["position_id", "symbol", "direction"], options=["af", "max", "max_af", "win", "window", "window_min"])
    def _bang_psar(eng: BotEngine, args: Dict[str, str]) -> str:
        """
        Set a PSAR stop thinker.

        Usage:
          !psar <position_id> <symbol> <LONG|SHORT> [af:0.02] [max:0.2] [win:200]
        """
        pid = args["position_id"]
        sym = args["symbol"].upper()
        d = args["direction"].upper()
        if d not in ("LONG", "SHORT"):
            return "Direction must be LONG|SHORT"

        kv = {"af": 0.02, "max_af": 0.2, "window_min": 200}
        # Allow multiple option spellings
        if "af" in args: kv["af"] = float(args["af"])
        if "max_af" in args: kv["max_af"] = float(args["max_af"])
        if "max" in args: kv["max_af"] = float(args["max"])
        if "window_min" in args: kv["window_min"] = int(args["window_min"])
        if "window" in args: kv["window_min"] = int(args["window"])
        if "win" in args: kv["window_min"] = int(args["win"])

        cfg = {"position_id": pid, "symbol": sym, "direction": d, **kv}
        tid = eng.store.insert_thinker("PSAR_STOP", cfg)
        return f"Thinker #{tid} PSAR_STOP set for {pid}/{sym} dir={d} af={kv['af']} max={kv['max_af']} win={kv['window_min']}"

    # ----------------------- JOBS -----------------------
    @R.at("jobs", options=["state", "limit"])
    def _at_jobs(eng: BotEngine, args: Dict[str, str]) -> str:
        """
        List DB jobs.

        Usage:
          @jobs [state:PENDING|RUNNING|DONE|ERR] [limit:50]
        """
        state = args.get("state")
        limit = int(args.get("limit", "50"))

        rows = eng.store.list_jobs(state=state, limit=limit)
        if not rows:
            return "No jobs."

        def _fmt_ts(ms: int) -> str:
            return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat(timespec="seconds")

        lines = []
        for r in rows:
            err = (r["last_error"] or "")
            if len(err) > 120:
                err = err[:117] + "..."
            lines.append(
                f"{r['job_id']}  {r['state']}  {r['task']}  attempts={r['attempts']} "
                f"pos={r['position_id'] or '-'}  "
                f"created={_fmt_ts(r['created_ts'])}  updated={_fmt_ts(r['updated_ts'])}"
                + (f"\n  err: {err}" if r["state"] == "ERR" and err else "")
            )
        return "\n".join(lines)

    # ----------------------- RETRY_JOBS -----------------------
    @R.bang("retry_jobs", options=["id", "limit"])
    def _bang_retry_jobs(eng: BotEngine, args: Dict[str, str]) -> str:
        """
        Retry failed jobs.

        Usage:
          !retry_jobs [id:<job_id>] [limit:N]
            - With id: retries one job.
            - Without id: retries all ERR jobs (optionally limited).
        """
        jid = args.get("id")
        if jid:
            ok = eng.store.retry_job(jid)
            return f"{'Retried' if ok else 'Not found'}: {jid}"

        limit = args.get("limit")
        n = eng.store.retry_failed_jobs(limit=int(limit) if limit else None)
        return f"Retried {n} failed job(s)."

    # ----------------------- CHART -----------------------
    @R.at("chart", argspec=["symbol", "timeframe"])
    def _at_chart(eng: BotEngine, args: Dict[str, str]) -> str:
        """
        Render candlestick chart (with volume & a simple indicator) and send/show it.

        Usage:
          @chart <symbol> <timeframe>
        Example:
          @chart ETHUSDT 1h
        """
        symbol = args["symbol"].upper()
        tf = args["timeframe"]
        try:
            path = eh.render_chart(eng, symbol, tf)
            eng.send_photo(path, caption=f"{symbol} {tf}")
            return f"Chart generated for {symbol} {tf}"
        except Exception as e:
            log().exc(e, where="cmd.chart")
            return f"Error: {e}"

    @R.bang("position-rm", argspec=["position_id"])
    def _bang_position_rm(eng: BotEngine, args: Dict[str, str]) -> str:
        pid = int(args["position_id"])
        deleted = eng.store.delete_position_completely(pid)
        # TODO improve reporting (still shows this message even if the position id does not exist)
        return f"Deleted position {pid} ({deleted} row(s))."

    # ======================= POSITION EDIT =======================
    @R.bang("position-set",
            argspec=["position_id"],
            options=["num", "den", "dir_sign", "target_usd", "user_ts", "status", "note", "created_ts"])
    def _bang_position_set(eng: BotEngine, args: Dict[str, str]) -> str:
        """
        Update fields in a position row (by position_id).
        Usage:
          !position-set <position_id> [num:ETHUSDT] [den:STRKUSDT|-] [dir_sign:-1|+1]
                          [target_usd:5000] [user_ts:<when>] [status:OPEN|CLOSED]
                          [note:...] [created_ts:<when>]
        Notes:
          - user_ts/created_ts accept ISO or epoch via parse_when().
          - den:"-" (or empty/none/null) clears denominator (single-leg).
        """
        pid_s = args.get("position_id", "")
        if not pid_s.isdigit():
            return "Usage: !position-set <position_id> …options…"
        pid = int(pid_s)

        # capture only provided (recognized) option keys
        provided = {k: v for k, v in args.items()
                    if k in {"num", "den", "dir_sign", "target_usd", "user_ts", "status", "note",
                             "created_ts"} and v is not None}

        if not provided:
            return ("Nothing to update. Allowed keys: "
                    "num den dir_sign target_usd user_ts status note created_ts")

        try:
            fields = _coerce_position_fields(provided)
            n = _sql_update(eng.store.con, "positions", "position_id", pid, fields)
            if n == 0:
                return f"Position {pid} not found or unchanged."
            # brief echo of what changed
            changed = ", ".join(f"{k}={fields[k]!r}" for k in sorted(fields.keys()))
            return f"Position {pid} updated: {changed}"
        except Exception as e:
            log().exc(e, where="cmd.position-set")
            return f"Error updating position {pid}: {e}"

    # ======================= LEG EDIT =======================
    @R.bang("leg-set",
            argspec=["leg_id"],
            options=["position_id", "symbol", "qty", "entry_price", "entry_price_ts", "price_method", "need_backfill",
                     "note"])
    def _bang_leg_set(eng: BotEngine, args: Dict[str, str]) -> str:
        """
        Update fields in a leg row (by leg_id).
        Usage:
          !leg-set <leg_id> [position_id:123] [symbol:ETHUSDT] [qty:-0.75]
                           [entry_price:3521.4] [entry_price_ts:<when>]
                           [price_method:aggTrade|kline|mark_kline] [need_backfill:0|1]
                           [note:...]
        Notes:
          - entry_price_ts accepts ISO or epoch via parse_when().
          - Changing position_id/symbol must respect UNIQUE(position_id,symbol).
        """
        lid_s = args.get("leg_id", "")
        if not lid_s.isdigit():
            return "Usage: !leg-set <leg_id> …options…"
        lid = int(lid_s)

        provided = {k: v for k, v in args.items()
                    if
                    k in {"position_id", "symbol", "qty", "entry_price", "entry_price_ts", "price_method",
                          "need_backfill",
                          "note"} and v is not None}

        if not provided:
            return ("Nothing to update. Allowed keys: "
                    "position_id symbol qty entry_price entry_price_ts price_method need_backfill note")

        try:
            fields = _coerce_leg_fields(provided)
            n = _sql_update(eng.store.con, "legs", "leg_id", lid, fields)
            if n == 0:
                return f"Leg {lid} not found or unchanged."
            changed = ", ".join(f"{k}={fields[k]!r}" for k in sorted(fields.keys()))
            return f"Leg {lid} updated: {changed}"
        except Exception as e:
            # Likely UNIQUE(position_id,symbol) or FK violations, surface cleanly.
            log().exc(e, where="cmd.leg-set")
            return f"Error updating leg {lid}: {e}"

    return R

# === helpers (local to build_registry) ====================================
# TODO this is all so horrible, but if it works for the moment, it is fine

def _ts_human(ms: int | None) -> str:
    """Human timestamp from ms (UTC ISO seconds)."""
    if not ms:
        return "?"
    try:
        return datetime.fromtimestamp(int(ms)/1000, tz=timezone.utc).isoformat(timespec="seconds")
    except Exception:
        return "?"

# TODO make a klinescache facility last_prices(symbols) --> {symbol: {"price": price, "timestamp": ..., "timeframe": ..., "ok": ...}, ...}
def _last_cached_price(eng: BotEngine, symbol: str) -> Optional[float]:
    """Latest cached close from KlinesCache (first configured timeframe)."""
    kc = eng.kc
    if kc is None:
        return None
    tfs = getattr(eng.cfg, "KLINES_TIMEFRAMES", None) or ["1m"]
    for tf in tfs:
        try:
            r = kc.last_row(symbol, tf)
            if r and r["close"] is not None:
                return float(r["close"])
        except Exception:
            pass
    return None

def _fmt_num(x: Any, nd=2) -> str:
    if x is None:
        return "?"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "?"

def _fmt_qty(x: Any) -> str:
    # show up to 6 decimals but trim zeros
    if x is None: return "?"
    try:
        s = f"{float(x):.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    except Exception:
        return "?"

def _leg_pnl(entry: Optional[float], qty: Optional[float], mark: Optional[float]) -> Optional[float]:
    if entry is None or qty is None or mark is None:
        return None
    return (float(mark) - float(entry)) * float(qty)

def _position_signed_target(row) -> float:
    # Correct sign: dir_sign * target_usd
    ds = row["dir_sign"]
    tgt = row["target_usd"]
    return ds * tgt


# ===== helpers for field coercion / updates (local to build_registry) ====
def _blank_to_none(v: str | None):
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in ("", "none", "null", "-"):
        return None
    return v


def _boolish_to_int(v: str) -> int:
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"): return 1
    if s in ("0", "false", "no", "n", "off"): return 0
    # fall back to int (raises if bad)
    return 1 if int(float(s)) != 0 else 0


def _coerce_position_fields(raw: dict) -> dict:
    """Coerce incoming option strings to proper types for positions table."""
    out = {}
    if "num" in raw: out["num"] = str(raw["num"]).upper().strip()
    if "den" in raw:
        v = _blank_to_none(raw["den"])
        out["den"] = None if v is None else str(v).upper().strip()
    if "dir_sign" in raw:
        ds = int(raw["dir_sign"])
        if ds not in (-1, 1):
            raise ValueError("dir_sign must be -1 or +1")
        out["dir_sign"] = ds
    if "target_usd" in raw: out["target_usd"] = float(raw["target_usd"])
    if "user_ts" in raw: out["user_ts"] = parse_when(str(raw["user_ts"]))
    if "created_ts" in raw: out["created_ts"] = parse_when(str(raw["created_ts"]))
    if "status" in raw: out["status"] = str(raw["status"]).upper().strip()
    if "note" in raw: out["note"] = str(raw["note"])
    return out


def _coerce_leg_fields(raw: dict) -> dict:
    """Coerce incoming option strings to proper types for legs table."""
    out = {}
    if "position_id" in raw: out["position_id"] = int(raw["position_id"])
    if "symbol" in raw: out["symbol"] = str(raw["symbol"]).upper().strip()
    if "qty" in raw: out["qty"] = float(raw["qty"])
    if "entry_price" in raw: out["entry_price"] = float(raw["entry_price"])
    if "entry_price_ts" in raw: out["entry_price_ts"] = parse_when(str(raw["entry_price_ts"]))
    if "price_method" in raw: out["price_method"] = str(raw["price_method"])
    if "need_backfill" in raw: out["need_backfill"] = _boolish_to_int(raw["need_backfill"])
    if "note" in raw: out["note"] = str(raw["note"])
    return out

