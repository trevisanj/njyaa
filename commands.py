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
from bot_api import BotEngine, parse_when, parse_pair_or_single, exec_positions, render_chart


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
        line = f"Usage: {pre}{nm}" + (f" {pos}" if pos else "") + opt
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
                lines.append(f"  {usage}\n    – {summary}")
            else:
                lines.append(f"  {usage}")
        return "\n".join(lines)


def build_registry() -> CommandRegistry:
    R = CommandRegistry()

    # ----------------------- HELP -----------------------
    @R.at("help")
    def _help(eng: BotEngine, args: Dict[str, str]) -> str:
        """Show this help."""
        return R._help_text()

    # ----------------------- OPEN (alias) -----------------------
    @R.at("open")
    def _at_open(eng: BotEngine, args: Dict[str, str]) -> str:
        """List open RV positions (summary). Alias for: @positions status:open what:summary"""
        return R.dispatch(eng, "@positions status:open what:summary")

    # ----------------------- POSITIONS -----------------------
    @R.at("positions", options=["status", "what", "sort", "limit", "position_id", "pair"])
    def _at_positions(eng: BotEngine, args: Dict[str, str]) -> str:
        """
        Advanced positions view.

        Examples:
          @positions what:full limit:20
          @positions status:closed
          @positions pair:STRK/ETH
        """
        status = args.get("status", "open")
        what = args.get("what", "summary")
        sort_ = args.get("sort", "pnl")  # (kept for future; unused here)
        limit = int(args.get("limit", "100"))
        position_id = args.get("position_id")
        pair = args.get("pair")

        exec_args = {
            "status": status, "what": what, "sort": sort_, "limit": limit,
            "position_id": position_id, "pair": None
        }
        if pair:
            up = pair.upper()
            exec_args["pair"] = parse_pair_or_single(eng, up)

        return exec_positions(eng, exec_args)

    # ----------------------- !OPEN POSITION -----------------------
    @R.bang("open", argspec=["ts", "pair", "usd"], options=["note"])
    def _bang_open(eng: BotEngine, args: Dict[str, str]) -> str:
        """
        Open/add an RV position.

        Usage:
          !open <ISO|epoch_ms> <NUM/DEN|SYMBOL> <±usd> [note:...]
        Examples:
          !open 2025-11-10T13:44:05 STRK/ETH -5000 note:rv test
          !open 2025-11-10T13:44:05 ETHUSDT +3000
        """
        ts_ms = parse_when(args["ts"])
        num, den = parse_pair_or_single(eng, args["pair"])
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
            path = render_chart(eng, symbol, tf)
            eng.send_photo(path, caption=f"{symbol} {tf}")
            return f"Chart generated for {symbol} {tf}"
        except Exception as e:
            log().exc(e, where="cmd.chart")
            return f"Error: {e}"

    return R
