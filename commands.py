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
from klines_cache import KlinesCache
# inside bot_api.py (or a separate charts.py helper imported there)
import tempfile, os, io
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from bot_api import BotEngine, parse_when, parse_pair_or_single, exec_positions, render_chart

class CommandRegistry:
    def __init__(self):
        # keys are ('@'|'!', command_name)
        self._handlers: Dict[Tuple[str,str], Callable[[BotEngine, Dict[str,str]], str]] = {}

    def at(self, name: str):
        """Decorator for '@' (GET) commands."""
        def deco(fn):
            self._handlers[('@', name.lower())] = fn
            return fn
        return deco

    def bang(self, name: str):
        """Decorator for '!' (SET) commands."""
        def deco(fn):
            self._handlers[('!', name.lower())] = fn
            return fn
        return deco

    def dispatch(self, eng: BotEngine, msg) -> str:
        s = (msg or "").strip()
        log().debug("dispatch.enter", text=s)
        if not s or s[0] not in ("@", "!"):
            log().debug("dispatch.exit", reason="not-a-command")
            return "Unrecognized. Use @help for available commands."

        prefix = s[0]
        parts = s.split(None, 1)
        if not parts:
            log().debug("dispatch.exit", reason="empty-after-prefix")
            return "Empty command."

        head = parts[0][1:].lower()
        tail = parts[1].strip() if len(parts) > 1 else ""
        handler = self._handlers.get((prefix, head))

        if not handler:
            log().warn("dispatch.unknown", prefix=prefix, head=head)
            return f"Unknown {prefix}{head}. Try @help."

        args: Dict[str, str] = {}
        if prefix == '@':
            for tok in tail.split():
                if ":" in tok:
                    k, v = tok.split(":", 1)
                    args[k.strip().lower()] = v.strip()
                elif tok:
                    log().debug("dispatch.ignored-token", token=tok)
        else:
            args["_"] = tail

        log().debug("dispatch.call", cmd=head, prefix=prefix, args=args)
        try:
            out = handler(eng, args)
            log().debug("dispatch.ok", cmd=head)
            return out
        except Exception as e:
            log().exc(e, where="dispatch.handler", cmd=head)
            return f"Error: {e}"


def build_registry() -> CommandRegistry:
    R = CommandRegistry()

    @R.at("help")
    def _help(eng: BotEngine, args: Dict[str,str]) -> str:
        return (
            "Commands:\n"
            "  @open                 – list open RV positions (summary)\n"
            "  @positions [filters]  – advanced view; e.g. @positions what:full limit:20\n"
            "  !open <spec>          – open/add RV position (e.g. '!open 2025-11-10T13:44:05 STRK/ETH -5000 note: x')\n"
            "  !close <position_id>      – close a recorded position\n"
        )

    @R.at("open")
    def _at_open(eng: BotEngine, args: Dict[str,str]) -> str:
        # alias to @positions status:open summary
        text_alias = "@positions status:open what:summary"
        return R.dispatch(eng, text_alias)

    @R.at("positions")
    def _at_positions(eng: BotEngine, args: Dict[str, str]) -> str:
        # defaults
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

    @R.bang("open")
    def _bang_open(eng: BotEngine, args: Dict[str,str]) -> str:
        """
        !open <ISO|epoch_ms> <NUM/DEN | SYMBOL> <±usd> [note:...]
        Example:
          !open 2025-11-10T13:44:05 STRK/ETH -5000 note:rv test
          !open 2025-11-10T13:44:05 ETHUSDT +3000
        """
        tail = args.get("_","").strip()
        if not tail:
            return "Usage: !open <ts> <NUM/DEN|SYMBOL> <±usd> [note:...]"
        parts = tail.split()
        if len(parts) < 3:
            return "Usage: !open <ts> <NUM/DEN|SYMBOL> <±usd> [note:...]"
        ts_raw, pair_raw, usd_raw = parts[0], parts[1], parts[2]
        note = ""
        if "note:" in tail:
            note = tail.split("note:",1)[1].strip()

        ts_ms = parse_when(ts_raw)
        num, den = parse_pair_or_single(eng, pair_raw)

        usd = int(float(usd_raw))
        pid = eng.positionbook.open_position(num, den, usd, ts_ms, note=note)
        return f"Opened pair {pid}: {num}/{den} target=${abs(usd):.0f} (queued price backfill)."

    @R.at("thinkers")
    def _at_thinkers(eng: BotEngine, args: Dict[str,str]) -> str:
        rows = eng.store.list_thinkers()
        if not rows:
            return "No thinkers."
        out = []
        for r in rows:
            out.append(f"#{r['id']} {r['kind']} enabled={r['enabled']} cfg={r['config_json']}")
        return "\n".join(out)

    @R.at("thinker-kinds")
    def _at_thinker_kinds(ctx, args):
        kinds = list(ctx.eng.tm.factory.kinds())  # expose from factory
        return "Available thinker kinds:\n" + "\n".join(sorted(kinds))

    @R.bang("thinker-enable")
    def _bang_thinker_enable(eng: BotEngine, args: Dict[str,str]) -> str:
        tid = args.get("_","").strip()
        if not tid.isdigit():
            return "Usage: !thinker-enable <id>"
        eng.store.update_thinker_enabled(int(tid), True)
        return f"Thinker #{tid} enabled."

    @R.bang("thinker-disable")
    def _bang_thinker_disable(eng: BotEngine, args: Dict[str,str]) -> str:
        tid = args.get("_","").strip()
        if not tid.isdigit():
            return "Usage: !thinker-disable <id>"
        eng.store.update_thinker_enabled(int(tid), False)
        return f"Thinker #{tid} disabled."

    @R.bang("thinker-rm")
    def _bang_thinker_rm(eng: BotEngine, args: Dict[str,str]) -> str:
        tid = args.get("_","").strip()
        if not tid.isdigit():
            return "Usage: !thinker-rm <id>"
        eng.store.delete_thinker(int(tid))
        return f"Thinker #{tid} deleted."

    @R.bang("alert")
    def _bang_alert(eng: BotEngine, args: Dict[str,str]) -> str:
        """
        !alert <SYMBOL> >= <PRICE> [msg:...]
        !alert <SYMBOL> <= <PRICE> [msg:...]
        """
        tail = args.get("_","").strip()
        parts = tail.split()
        if len(parts) < 3:
            return "Usage: !alert <SYMBOL> (>=|<=) <PRICE> [msg:...]"
        sym, op, pr = parts[0].upper(), parts[1], parts[2]
        if op not in (">=", "<="):
            return "Op must be >= or <="
        direction = "ABOVE" if op == ">=" else "BELOW"
        try:
            price = float(pr)
        except:
            return "Bad price."
        msg = ""
        if "msg:" in tail:
            msg = tail.split("msg:",1)[1].strip()
        cfg = {"symbol": sym, "direction": direction, "price": price, "message": msg}
        tid = eng.store.insert_thinker("THRESHOLD_ALERT", cfg)
        return f"Thinker #{tid} THRESHOLD_ALERT set for {sym} {direction} {price}"

    @R.bang("psar")
    def _bang_psar(eng: BotEngine, args: Dict[str,str]) -> str:
        """
        !psar <position_id> <symbol> <LONG|SHORT> [af:0.02] [max:0.2] [win:200]
        """
        tail = args.get("_","").strip()
        parts = tail.split()
        if len(parts) < 3:
            return "Usage: !psar <position_id> <symbol> <LONG|SHORT> [af:0.02] [max:0.2] [win:200]"
        pid, sym, d = parts[0], parts[1].upper(), parts[2].upper()
        if d not in ("LONG","SHORT"):
            return "Direction must be LONG|SHORT"
        # optional kvs
        kv = {"af":0.02, "max_af":0.2, "window_min":200}
        for tok in parts[3:]:
            if ":" in tok:
                k,v = tok.split(":",1)
                k = k.lower().strip()
                v = v.strip()
                if k == "af": kv["af"] = float(v)
                elif k in ("max","max_af"): kv["max_af"] = float(v)
                elif k in ("win","window","window_min"): kv["window_min"] = int(v)
        cfg = {"position_id": pid, "symbol": sym, "direction": d, **kv}
        tid = eng.store.insert_thinker("PSAR_STOP", cfg)
        return f"Thinker #{tid} PSAR_STOP set for {pid}/{sym} dir={d} af={kv['af']} max={kv['max_af']} win={kv['window_min']}"

    # ---------- @jobs: list DB jobs ----------
    @R.at("jobs")
    def _at_jobs(eng: BotEngine, args: Dict[str, str]) -> str:
        """
        @jobs [state:PENDING|RUNNING|DONE|ERR] [limit:50]
        """
        state = args.get("state")
        limit = int(args.get("limit", "50"))

        rows = eng.store.list_jobs(state=state, limit=limit)
        if not rows:
            return "No jobs."

        def _fmt_ts(ms: int) -> str:
            # ISO UTC, seconds precision
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

    # ---------- !retry_jobs: retry failed jobs ----------
    @R.bang("retry_jobs")
    def _bang_retry_jobs(eng: BotEngine, args: Dict[str, str]) -> str:
        """
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

    @R.at("chart")
    def _at_chart(eng: BotEngine, args: dict) -> str:
        """
        @chart <symbol> <timeframe>
        Example: @chart ETHUSDT 1h
        """
        tail = args.get("_","").strip()
        parts = tail.split()
        if len(parts) < 2:
            return "Usage: @chart <symbol> <timeframe>"

        symbol, tf = parts[0].upper(), parts[1]
        try:
            path = render_chart(eng, symbol, tf)
            eng.send_photo(path)
        except Exception as e:
            log().exc(e, where="cmd.chart")
            return f"Error: {e}"

    return R
