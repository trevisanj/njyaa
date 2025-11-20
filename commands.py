# commands.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple, Optional, List, Sequence, Literal, TYPE_CHECKING

from common import *
from datetime import datetime, timezone, timedelta
import textwrap
from binance_um import BinanceUM
import tabulate
import enghelpers as eh
import re
from rich.console import Console
from rich.markdown import Markdown, Heading
from rich.theme import Theme
from rich.text import Text
from rich.rule import Rule
from common import Clock

if TYPE_CHECKING:
    from bot_api import BotEngine


__all__ = ["OC", "OCText", "OCMarkDown", "OCPhoto", "OCTable", "CO", "CommandRegistry", "build_registry",
           "RICH_MD_THEME", "RICH_MD_CONFIG", "RENDER_MARKDOWN"]

AT_KEY = "?"
BANG_KEY = "!"

# Rich Markdown theme/styles (tweak as needed)
T_NORMAL = "grey66"
RICH_MD_THEME = Theme({
    # "markdown.text": "plum2",
    # "markdown.normal": "plum2",
    "markdown.paragraph": T_NORMAL,
    "markdown.em": "italic light_sky_blue3",
    "markdown.strong": "bold grey93",
    "markdown.h1": "bold grey100",
    "markdown.h2": "bold grey93",
    "markdown.h3": "bold grey85",
    "markdown.item": T_NORMAL,
    "markdown.item.bullet": T_NORMAL,
    "markdown.block_quote": "steel_blue",
    "markdown.block_quote.text": "steel_blue",
    "markdown.code_block": "steel_blue",  # ineffective, since markdown.py:CodeBlock uses code_theme, defined below
})
RICH_MD_CONFIG = {
    "style": "markdown.text",
    # check pygments docs: https://pygments.org/styles/
    "code_theme": "lightbulb", # "monokai"/"lightbulb"/"github-dark"...
    "width": None,  # use terminal width
    "color_system": "truecolor",  # None|standard|256|truecolor
}

# Toggle pure markdown (no rich rendering) for debugging.
RENDER_MARKDOWN = True
# Whether to render command details as code or blockquote
BLOCKQUOTE = False

# Custom heading renderer to avoid boxed/centered titles.
class FlatHeading(Heading):
    def __rich_console__(self, console, options):
        text: Text = self.text
        text.justify = "left"
        yield text
        if self.tag in ("h1", "h2"):
            underline = "─" * max(len(text.plain), 1)
            underline_text = Text(underline, style=self.style_name, justify="left")
            yield underline_text

# Patch Markdown to use the flat heading renderer
Markdown.elements["heading_open"] = FlatHeading


# ============== UI OUTPUT MODEL ==============

# OutputKind = Literal["text", "markdown", "photo", "table"]

# ============== UI OUTPUT MODEL ==============

@dataclass
class OC:
    """Base command Output Component. Subclasses know how to render themselves."""

    def render_console(self, eng: BotEngine) -> str:
        raise NotImplementedError

    def render_telegram(self, eng: BotEngine) -> str:
        raise NotImplementedError


@dataclass
class OCText(OC):
    text: str

    def __post_init__(self):
        if self.text is None:
            raise ValueError("TextComponent requires 'text'")

    def render_console(self, eng: BotEngine) -> str:
        eng._send_text_console(self.text)
        return self.text

    def render_telegram(self, eng: BotEngine) -> str:
        eng._tsafe_text_telegram(self.text)
        return self.text


@dataclass
class OCMarkDown(OC):
    text: str

    def __post_init__(self):
        if self.text is None:
            raise ValueError("MarkdownComponent requires 'text'")

    def render_console(self, eng: BotEngine) -> str:
        if not RENDER_MARKDOWN:
            eng._send_text_console(self.text)
            return self.text
        console = Console(
            theme=RICH_MD_THEME,
            # color_system=RICH_MD_CONFIG.get("color_system"),
            width=RICH_MD_CONFIG.get("width"),
        )
        md = Markdown(self.text, style=RICH_MD_CONFIG["style"], code_theme=RICH_MD_CONFIG["code_theme"])
        with console.capture() as cap:
            console.print(md)
        rendered = cap.get()
        # Collapse excessive blank lines that can appear between markdown blocks
        rendered = re.sub(r"\n{2,}", "\n\n", rendered)
        eng._send_text_console(rendered)
        return rendered

    def render_telegram(self, eng: BotEngine) -> str:
        eng._tsafe_text_telegram(self.text)
        return self.text


@dataclass
class OCPhoto(OC):
    path: str
    caption: Optional[str] = None

    def __post_init__(self):
        if not self.path:
            raise ValueError("PhotoComponent requires 'path'")

    def render_console(self, eng: BotEngine) -> str:
        try:
            eng._send_photo_console(self.path, caption=self.caption)
        except Exception as e:
            log().exc(e, where="PhotoComponent.render_console", path=self.path)
            msg = f"[photo error: {e}]"
            eng._send_text_console(msg)
            return msg
        placeholder = f"[photo: {self.path}]"
        eng._send_text_console(placeholder)
        return placeholder

    def render_telegram(self, eng: BotEngine) -> str:
        try:
            eng._tsafe_photo_telegram(self.path, caption=self.caption)
        except Exception as e:
            log().exc(e, where="OCPhoto.render_telegram", path=self.path)
            msg = f"[photo error: {e}]"
            eng._tsafe_text_telegram(msg)
            return msg
        if self.caption:
            eng._tsafe_text_telegram(self.caption)
        return self.caption or ""


@dataclass
class OCTable(OC):
    headers: List[str]
    rows: List[Sequence[Any]]

    def __post_init__(self):
        if not self.headers:
            raise ValueError("TableComponent requires non-empty headers")
        if self.rows is None:
            raise ValueError("TableComponent requires 'rows' list")

    def render_console(self, eng: BotEngine) -> str:
        if not self.rows:
            eng._send_text_console("(empty)")
            return "(empty)"
        text = tabulate.tabulate(self.rows, headers=self.headers, tablefmt="github")
        eng._send_text_console(text)
        return text

    def render_telegram(self, eng: BotEngine) -> str:
        if not self.rows:
            msg = "(empty)"
            eng._tsafe_text_telegram(msg)
            return msg

        lines = []
        for row in self.rows:
            parts = [f"{h}: {v}" for h, v in zip(self.headers, row)]
            line = "; ".join(parts) if parts else "(empty row)"
            lines.append(line)
        body = "\n".join(lines)
        eng._tsafe_text_telegram(body)
        return body


class CO:
    """Collection of OutputComponent"""

    def __init__(self, *args):
        """Very tolerant init method for convenience"""
        cc = self.components = []
        for arg in args:
            argh = [arg] if not isinstance(arg, (List, Tuple)) else arg
            for aargh in argh:
                if isinstance(aargh, str):
                    cc.append(OCText(aargh))
                elif not (isinstance(aargh, OC) and type(OC) is not OC):
                    name = aargh.__class__.__name__
                    raise TypeError(f"Output item must be OC (output component) subclass or str, not {name}")
                else:
                    cc.append(aargh)


    def __iter__(self):
        return self.components.__iter__()

    components: List[OC] = field(default_factory=list)

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
        # keys are ('?'|'!', command_name)
        self._handlers: Dict[Tuple[str, str], Callable[[BotEngine, Dict[str, str]], Any]] = {}

        self._meta: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # ---- decorators ----
    def at(self, name: str, argspec: List[str] = None, options: List[str] = None, nreq: Optional[int] = None):
        return self._reg((AT_KEY, name.lower()), argspec, options, nreq)

    def bang(self, name: str, argspec: List[str] = None, options: List[str] = None, nreq: Optional[int] = None):
        return self._reg((BANG_KEY, name.lower()), argspec, options, nreq)

    def _reg(self, key: Tuple[str, str], argspec: Optional[List[str]], options: Optional[List[str]], nreq: Optional[int]):
        def deco(fn):
            self._handlers[key] = fn
            # Extract first docstring line as summary (if present)

            l_doc = _remove_indent(fn.__doc__ or "")
            summary = "" if not l_doc else l_doc[0]
            self._meta[key] = {
                "argspec": list(argspec or []),
                "options": set(options or []),
                "summary": summary,
                "name": key[1],
                "prefix": key[0],
                "doc": "\n".join(l_doc),
                "nreq": int(nreq) if nreq is not None else (len(argspec or [])),
            }
            return fn
        return deco

    # ---- dispatcher ----
    def dispatch(self, eng: BotEngine, msg: str) -> Any:
        s = (msg or "").strip()
        log().debug("dispatch.enter", text=s)
        if not s or s[0] not in (AT_KEY, BANG_KEY):
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
            return f"Unknown {prefix}{head}. Try {AT_KEY}help."

        # Parse args according to argspec/options
        args = self._parse_args(tail, meta)
        argspec_in_meta = list(meta["argspec"])
        nreq = meta.get("nreq", len(argspec_in_meta))
        # Required positionals: first nreq argspec entries
        required_names = argspec_in_meta[:nreq]
        missing = [a for a in required_names if a not in args]
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
            if not toks:
                break
            # If the next token looks like an option and the key is known, don't consume it as positional
            if ":" in toks[0]:
                k = toks[0].split(":", 1)[0].strip().lower()
                if not options or k in options:
                    break
            result[name] = toks.pop(0)

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
        argspec = list(meta["argspec"])
        nreq = meta.get("nreq", len(argspec))
        pos_parts: List[str] = []
        for i, a in enumerate(argspec):
            rendered = f"⟨{a}⟩"
            if i >= nreq:
                rendered = f"[{rendered}]"
            pos_parts.append(rendered)
        pos = " ".join(pos_parts)
        opt = ""
        if meta["options"]:
            opt_items = " ".join(f"[{o}:…]" for o in sorted(meta["options"]))
            opt = (" " + opt_items) if opt_items else ""
        line = f"**{pre}{nm}**" + (f" {pos}" if pos else "") + opt
        line = re.sub(r"\s{2,}", " ", line).strip()
        if reason:
            return f"{reason}\n{line}"
        return line

    def _help_text(self, detail: int = 1, command: Optional[str] = None) -> CO:
        """Build a help listing from registered commands."""
        metas = []
        cmd_filter = (command or "").strip().lower()
        want_prefix = None
        if cmd_filter.startswith(AT_KEY) or cmd_filter.startswith(BANG_KEY):
            want_prefix = cmd_filter[0]
            cmd_filter = cmd_filter[1:]

        for (prefix, name) in sorted(self._handlers.keys()):
            if cmd_filter and name != cmd_filter:
                continue
            if want_prefix and prefix != want_prefix:
                continue
            metas.append(self._meta[(prefix, name)])

        if detail == 1:
            parts = [f"**{m['prefix']}{m['name']}**" for m in metas]
            body = "  ".join(parts) if parts else "(none)"
            header = "Commands"
            return CO(OCMarkDown(f"{header}\n{'-' * len(header)}\n{body}"))

        if detail in (2, 3):
            header = "Commands"
            lines: List[str] = [header, "-" * len(header)]
            for m in metas:
                usage = self._usage_line(m)
                bullet = f"- {usage}"
                if detail == 3 and m["summary"]:
                    bullet += f": *{m['summary']}*"
                lines.append(bullet)
            return CO(OCMarkDown("\n".join(lines)))

        # detail 4: full docs for all matched commands
        if not metas:
            return CO(OCMarkDown("# Commands\n(no match)"))

        blocks: List[str] = []
        multi = len(metas) > 1
        if multi:
            header = "Commands"
            blocks.append(f"{header}\n{'-' * len(header)}")
        for m in metas:
            usage = self._usage_line(m)
            doc_raw = m["doc"].strip("\n")
            paras: List[str] = []
            if doc_raw:
                current: List[str] = []
                for line in doc_raw.splitlines():
                    if line.strip() == "":
                        if current:
                            paras.append("\n".join(current).strip("\n"))
                            current = []
                        continue
                    current.append(line.rstrip())
                if current:
                    paras.append("\n".join(current).strip("\n"))
            first_para_text = paras[0] if paras else ""

            if BLOCKQUOTE:
                DETAIL_PREFIX = "> "
                rest_text = "\n".join([DETAIL_PREFIX+x for x in ("\n\n".join(paras[1:])).split("\n")]) if len(paras) > 1 else ""
            else:
                # formats as code
                DETAIL_PREFIX = "  "
                rest_text = textwrap.indent("\n\n".join(paras[1:]), DETAIL_PREFIX) if len(paras) > 1 else ""

            block_parts = [usage]
            if first_para_text:
                block_parts.extend(["", f"*{first_para_text}*"])
            if rest_text:
                if BLOCKQUOTE:
                    block_parts.extend(["", rest_text, ""])
                else:
                    block_parts.extend(["", "```", rest_text, "```"])
            blocks.append("\n".join(block_parts))

        return CO(OCMarkDown("\n\n".join(blocks)))


def build_registry() -> CommandRegistry:
    R = CommandRegistry()
    # Local helpers to keep handlers concise
    def _md(text: str) -> CO:
        return CO(OCMarkDown(text))

    def _txt(text: str) -> CO:
        return CO(OCText(text))

    def _tbl(headers: List[str], rows: List[Sequence[Any]], intro: str | None = None) -> CO:
        comps: List[OC] = []
        if intro:
            comps.append(OCMarkDown(intro))
        comps.append(OCTable(headers=headers, rows=rows))
        return CO(comps)

    # ----------------------- HELP -----------------------
    # TODO improve help formatting
    @R.at("help", argspec=["command"], options=["detail"], nreq=0)
    def at_help(eng: BotEngine, args: Dict[str, str]) -> CO:
        f"""Show this help.

        Usage:
          {AT_KEY}help [command] [detail:1]
        """
        cmd = args.get("command")
        # default detail: 4 if a specific command was passed; otherwise 1
        detail = int(args.get("detail", "4" if cmd else "1"))
        return R._help_text(detail=detail, command=cmd)

    # ----------------------- CONFIG SET -----------------------
    @R.bang("config-set", options=["reference_balance", "leverage", "default_risk", "updated_by"])
    def _bang_config_set(eng: BotEngine, args: Dict[str, str]) -> CO:
        """
        Update global risk config.

        Usage:
          !config-set reference_balance:12000 leverage:2.0 default_risk:0.015
        """
        provided = {k: v for k, v in args.items()
                    if k in {"reference_balance", "leverage", "default_risk", "updated_by"} and v is not None}
        if not provided:
            return _txt("Usage: !config-set reference_balance:<usd> leverage:<mult> default_risk:<0-1> [updated_by:you]")
        try:
            fields = _coerce_config_fields(provided)
        except Exception as e:
            return _txt(f"Error: {e}")

        n = eng.store.update_config(fields)
        if n <= 0:
            return _txt("No config fields updated.")
        changed = ", ".join(f"{k}={fields[k]}" for k in sorted(fields.keys()))
        cfg = eng.store.get_config()
        summary = (f"Config updated ({n} field(s)). "
                   f"balance=${_fmt_num(cfg['reference_balance'],2)} "
                   f"leverage={_fmt_num(cfg['leverage'],2)} "
                   f"default_risk={_fmt_pct(cfg['default_risk'])}")
        return _txt(f"{summary} ({changed})")

    # ----------------------- OPEN (alias) -----------------------
    @R.at("open")
    def _at_open(eng: BotEngine, args: Dict[str, str]) -> CO:
        """List open RV positions (summary). Alias for: @positions status:open detail:2"""
        res = R.dispatch(eng, f"{AT_KEY}positions status:open detail:2")
        return res if isinstance(res, CO) else _txt(str(res))

    # ----------------------- POSITIONS (detail levels) -----------------------
    @R.at("positions", options=["status", "detail", "sort", "limit", "position_id", "pair"])
    def _at_positions(eng: BotEngine, args: Dict[str, str]) -> CO:
        """
        Positions report with detail levels.

        status: open/closed/all

        detail: 1 = summary
                2 = summary + per-position overview (with opened datetime & signed target)
                3 = level 2 + per-symbol leg summary (agg qty, WAP, last price, PnL)
                4 = level 2 + list every leg (timestamp, entry, last, qty, PnL)

        Examples:
          ?positions detail:1
          ?positions status:closed detail:2
          ?positions pair:STRK/ETH detail:3 limit:20
        """
        store = eng.store
        cfg = eng.store.get_config()
        ref_balance = cfg["reference_balance"]

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

        md_lines: List[str] = [f"# Positions ({status})"]
        lines: List[str] = []
        total_pnl_pct = _fmt_pct(_pct_of(total_pnl, ref_balance), show_sign=True)
        bal_str = _fmt_num(ref_balance, 2)
        lines.append(
            f"Positions: {len(rows)} | Target ≈ ${_fmt_num(total_target, 2)} | "
            f"PNL ≈ ${_fmt_num(total_pnl, 2)} ({total_pnl_pct} of balance ${bal_str})"
            + (f" (PnL incomplete for {pnl_missing_count} position(s))" if pnl_missing_count else "")
        )
        if detail == 1:
            return _md("\n".join(md_lines + [""] + lines))

        # ---------- detail 2+: per-position overview ----------
        count = 0
        for r in rows:
            if count >= limit:
                break
            pid = r["position_id"]
            opened_ms = r["user_ts"] or r["created_ts"]
            opened_str = ts_human(opened_ms)
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

            risk_pct = _fmt_pct(r["risk"], show_sign=False)
            pnl_pct = _fmt_pct(_pct_of(pos_pnl, ref_balance), show_sign=True)
            pnl_str = f"${_fmt_num(pos_pnl, 2)}" + (" (incomplete)" if any_missing else "")

            lines.append(
                f"{pid} {r['num']} / {r['den'] or '-'} "
                f"status={r['status']} opened={opened_str} "
                f"signed_target=${_fmt_num(signed_target, 2)} "
                f"risk={risk_pct} pnl={pnl_str} pnl%={pnl_pct}"
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
                    pnl_pct_leg = _fmt_pct(_pct_of(acc["pnl"], ref_balance), show_sign=True)
                    pnl_s = _fmt_num(acc["pnl"], 2)
                    if missing_map.get(s):
                        pnl_s += " (incomplete)"
                    lines.append(
                        f"  - {s}  qty={_fmt_qty(acc['qty'])}  entry≈{_fmt_num(wap, 6)}  "
                        f"last={_fmt_num(last, 6)}  pnl=${pnl_s} pnl%={pnl_pct_leg} risk={risk_pct}"
                    )

            # ---------- detail 4: list every leg ----------
            if detail == 4:
                for lg in store.get_legs(pid):
                    ts = lg["entry_price_ts"]
                    ts_h = ts_human(ts)
                    last = marks.get(lg["symbol"])
                    pnl = _leg_pnl(lg["entry_price"], lg["qty"], last)
                    pnl_pct_leg = _fmt_pct(_pct_of(pnl, ref_balance), show_sign=True)
                    pnl_str = _fmt_num(pnl, 2)
                    if pnl is None:
                        pnl_str = "? (missing price/entry/qty)"
                    lines.append(
                        f"  - {lg['leg_id']} {lg['symbol']}  t={ts_h}  qty={_fmt_qty(lg['qty'])}  "
                        f"entry={_fmt_num(lg['entry_price'], 6)}  last={_fmt_num(last, 6)}  "
                        f"pnl=${pnl_str} pnl%={pnl_pct_leg} risk={risk_pct}"
                    )

            count += 1

        return _md("\n".join(md_lines + [""] + lines))

    # ----------------------- RISK SNAPSHOT -----------------------
    @R.at("risk")
    def _at_risk(eng: BotEngine, args: Dict[str, str]) -> CO:
        """Show risk/exposure snapshot."""
        cfg = eng.store.get_config()
        ref_balance = cfg["reference_balance"]
        leverage = cfg["leverage"]

        rows = eng.store.list_open_positions()
        if not rows:
            txt = (
                f"# Risk\n"
                f"- Balance=${_fmt_num(ref_balance,2)} leverage={_fmt_num(leverage,2)} "
                f"default_risk={_fmt_pct(cfg['default_risk'])}\n"
                f"- No open positions."
            )
            return _md(txt)

        involved_syms = set()
        for r in rows:
            for lg in eng.store.get_legs(r["position_id"]):
                if lg["symbol"]:
                    involved_syms.add(lg["symbol"])
        marks: Dict[str, Optional[float]] = {s: _last_cached_price(eng, s) for s in involved_syms}

        total_exposure = 0.0
        exposure_missing = False
        total_pnl = 0.0
        table_rows: List[Sequence[Any]] = []

        for r in rows:
            pid = int(r["position_id"])
            legs = eng.store.get_legs(pid)
            risk_val = float(r["risk"])
            risk_budget = ref_balance * risk_val if ref_balance else None
            pos_pnl = 0.0
            pnl_missing = False
            notional = 0.0
            notional_missing = False

            for lg in legs:
                mk = marks.get(lg["symbol"])
                pnl = _leg_pnl(lg["entry_price"], lg["qty"], mk)
                if pnl is None:
                    pnl_missing = True
                else:
                    pos_pnl += pnl
                if lg["qty"] is None or mk is None:
                    notional_missing = True
                else:
                    notional += abs(float(lg["qty"])) * float(mk)

            if not notional_missing:
                total_exposure += notional
            else:
                exposure_missing = True
                notional = None

            if not pnl_missing:
                total_pnl += pos_pnl

            pnl_pct = _pct_of(pos_pnl, ref_balance)
            r_multiple = (pos_pnl / risk_budget) if risk_budget else None
            tp_2r = 2 * risk_budget if risk_budget is not None else None
            tp_3r = 3 * risk_budget if risk_budget is not None else None

            table_rows.append([
                pid,
                f"{r['num']}/{r['den'] or '-'}",
                _fmt_pct(risk_val),
                _fmt_num(risk_budget, 2),
                _fmt_num(notional, 2),
                _fmt_num(pos_pnl, 2) + (" (incomplete)" if pnl_missing else ""),
                _fmt_pct(pnl_pct, show_sign=True),
                _fmt_num(r_multiple, 2),
                f"{_fmt_num(tp_2r, 2)}/{_fmt_num(tp_3r, 2)}"
            ])

        max_exposure = ref_balance * leverage if ref_balance and leverage else None
        available = (max_exposure - total_exposure) if (max_exposure is not None) else None
        exposure_used = _pct_of(total_exposure, max_exposure) if max_exposure else None
        total_pnl_pct = _pct_of(total_pnl, ref_balance)

        md_lines = [
            "# Risk",
            f"- Balance=${_fmt_num(ref_balance,2)} leverage={_fmt_num(leverage,2)} "
            f"default_risk={_fmt_pct(cfg['default_risk'])}",
            f"- Exposure: ${_fmt_num(total_exposure,2)} / ${_fmt_num(max_exposure,2)} "
            f"(used {_fmt_pct(exposure_used)}; available=${_fmt_num(available,2)})"
            + (" (exposure incomplete)" if exposure_missing else ""),
            f"- PnL: ${_fmt_num(total_pnl,2)} ({_fmt_pct(total_pnl_pct, show_sign=True)})"
        ]

        headers = ["id", "pair", "risk%", "budget$", "notional$", "pnl$", "pnl%", "R", "2R/3R$"]
        return CO([OCMarkDown("\n".join(md_lines)), OCTable(headers=headers, rows=table_rows)])

    # ----------------------- !OPEN POSITION -----------------------
    @R.bang("open", argspec=["pair", "ts", "usd"], options=["note", "risk"])
    def _bang_open(eng: BotEngine, args: Dict[str, str]) -> CO:
        """
        Open/add an RV position.

        Usage:
          !open <ISO|epoch_ms> <NUM/DEN|SYMBOL> <±usd> [risk:0.02] [note:...]

        Examples:
          !open STRK/ETH 2025-11-10T13:44:05  -5000 note:rv test
          !open ETHUSDT  2025-11-10T13:44:05 +3000
        """
        ts_ms = parse_when(args["ts"])
        num, den = eh.parse_pair_or_single(eng, args["pair"])
        usd = int(float(args["usd"]))
        note = args.get("note", "")
        cfg = eng.store.get_config()
        risk_raw = args.get("risk")
        risk_val = cfg["default_risk"]
        if risk_raw is not None:
            risk_val = float(risk_raw)
        if risk_val <= 0:
            return _txt("risk must be > 0 (fraction, e.g., 0.02)")

        pid = eng.positionbook.open_position(num, den, usd, ts_ms, note=note, risk=risk_val)
        return _txt(
            f"Opened pair {pid}: {num}/{den} target=${abs(usd):.0f} "
            f"risk={_fmt_pct(risk_val)} (queued price backfill)."
        )

    # ----------------------- THINKERS LIST -----------------------
    @R.at("thinkers")
    def _at_thinkers(eng: BotEngine, args: Dict[str, str]) -> CO:
        """List thinkers stored in DB."""
        rows = eng.store.list_thinkers()
        if not rows:
            return _txt("No thinkers.")
        lines = ["# Thinkers", ""]
        for r in rows:
            lines.append(f"- `#{r['id']}` {r['kind']} enabled={r['enabled']} cfg={r['config_json']}")
        return _md("\n".join(lines))

    # ----------------------- THINKER KINDS -----------------------
    @R.at("thinker-kinds")
    def _at_thinker_kinds(eng: BotEngine, args: Dict[str, str]) -> CO:
        """List available thinker kinds (from factory auto-discovery)."""
        kinds = list(eng.tm.factory.kinds())
        ndigits = len(str(len(kinds)-1))
        body = "\n".join(f"- [{i:0{ndigits}d}] {k}" for i, k in enumerate(kinds))
        return _md("# Thinker kinds\n" + body + "\n\n*Use either index or kind name to create new thinkers")

    # ----------------------- THINKER ENABLE -----------------------
    @R.bang("thinker-enable", argspec=["id"])
    def _bang_thinker_enable(eng: BotEngine, args: Dict[str, str]) -> CO:
        """Enable a thinker by ID.

        Usage:: !thinker-enable <id>"""
        tid = args["id"].strip()
        if not tid.isdigit():
            return _txt("Usage: !thinker-enable <id>")
        eng.store.update_thinker_enabled(int(tid), True)
        return _txt(f"Thinker #{tid} enabled.")

    # ----------------------- THINKER DISABLE -----------------------
    @R.bang("thinker-disable", argspec=["id"])
    def _bang_thinker_disable(eng: BotEngine, args: Dict[str, str]) -> CO:
        """Disable a thinker by ID

        Usage: !thinker-disable <id>"""
        tid = args["id"].strip()
        if not tid.isdigit():
            return _txt("Usage: !thinker-disable <id>")
        eng.store.update_thinker_enabled(int(tid), False)
        return _txt(f"Thinker #{tid} disabled.")

    # ----------------------- THINKER REMOVE -----------------------
    @R.bang("thinker-rm", argspec=["id"])
    def _bang_thinker_rm(eng: BotEngine, args: Dict[str, str]) -> CO:
        """
        Delete a thinker by ID.

        Usage: !thinker-rm <id>"""
        tid = args["id"].strip()
        if not tid.isdigit():
            return _txt("Usage: !thinker-rm <id>")
        eng.store.delete_thinker(int(tid))
        return _txt(f"Thinker #{tid} deleted.")

    # ----------------------- THINKER NEW -----------------------
    @R.bang("thinker-new", argspec=["kind"], options=["enabled"])
    def _bang_thinker_new(eng: BotEngine, args: Dict[str, str]) -> CO:
        """
        Create a new thinker row.

        Usage:
          !thinker-new <kind|index> [enabled:0|1]
        """
        kind_arg = args["kind"].strip()
        factory = eng.tm.factory
        kinds = list(factory.kinds())
        kind: str
        if kind_arg.isdigit():
            idx = int(kind_arg)
            if idx < 0 or idx >= len(kinds):
                return _txt(f"Kind index out of range (0-{len(kinds)-1}).")
            kind = kinds[idx]
        else:
            kind = kind_arg.upper()
            if kind not in kinds:
                return _txt(f"Unknown thinker kind '{kind}'. Use @thinker-kinds for a list.")

        enabled_opt = args.get("enabled")
        enabled_val = 1
        if enabled_opt is not None:
            enabled_val = 1 if enabled_opt.strip() not in ("0", "false", "False") else 0

        tid = eng.store.insert_thinker(kind, config={})
        if not enabled_val:
            eng.store.update_thinker_enabled(tid, False)
        return _txt(f"Thinker #{tid} created (kind={kind}, enabled={bool(enabled_val)}).")

    # ----------------------- ALERT -----------------------
    @R.bang("alert", argspec=["symbol", "op", "price"], options=["msg"])
    def _bang_alert(eng: BotEngine, args: Dict[str, str]) -> CO:
        """
        Create a simple threshold alert thinker.

        Usage:
          !alert <SYMBOL> (>=|<=) <PRICE> [msg:...]
        """
        sym = args["symbol"].upper()
        op = args["op"]
        pr = args["price"]
        if op not in (">=", "<="):
            return _txt("Op must be >= or <=")

        try:
            price = float(pr)
        except:
            return _txt("Bad price.")

        direction = "ABOVE" if op == ">=" else "BELOW"
        msg = args.get("msg", "")
        cfg = {"symbol": sym, "direction": direction, "price": price, "message": msg}
        tid = eng.store.insert_thinker("THRESHOLD_ALERT", cfg)
        return _txt(f"Thinker #{tid} THRESHOLD_ALERT set for {sym} {direction} {price}")

    # ----------------------- PSAR -----------------------
    @R.bang("psar", argspec=["position_id", "symbol", "direction"], options=["af", "max", "max_af", "win", "window", "window_min"])
    def _bang_psar(eng: BotEngine, args: Dict[str, str]) -> CO:
        """
        Set a PSAR stop thinker.

        Usage:
          !psar <position_id> <symbol> <LONG|SHORT> [af:0.02] [max:0.2] [win:200]
        """
        pid = args["position_id"]
        sym = args["symbol"].upper()
        d = args["direction"].upper()
        if d not in ("LONG", "SHORT"):
            return _txt("Direction must be LONG|SHORT")

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
        return _txt(f"Thinker #{tid} PSAR_STOP set for {pid}/{sym} dir={d} af={kv['af']} max={kv['max_af']} win={kv['window_min']}")

    # ----------------------- JOBS -----------------------
    @R.at("jobs", options=["state", "limit"])
    def _at_jobs(eng: BotEngine, args: Dict[str, str]) -> CO:
        """
        List DB jobs.

        Usage:
          @jobs [state:PENDING|RUNNING|DONE|ERR] [limit:50]
        """
        state = args.get("state")
        limit = int(args.get("limit", "50"))

        rows = eng.store.list_jobs(state=state, limit=limit)
        if not rows:
            return _txt("No jobs.")

        def _fmt_ts(ms: int) -> str:
            return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat(timespec="seconds")

        headers = ["id", "state", "task", "attempts", "position", "created", "updated", "error"]
        rows_out: List[Sequence[Any]] = []
        for r in rows:
            err = (r["last_error"] or "")
            if len(err) > 120:
                err = err[:117] + "..."
            rows_out.append([
                r["job_id"],
                r["state"],
                r["task"],
                r["attempts"],
                r["position_id"] or "-",
                _fmt_ts(r["created_ts"]),
                _fmt_ts(r["updated_ts"]),
                err,
            ])
        return _tbl(headers, rows_out, intro="# Jobs")

    # ----------------------- RETRY_JOBS -----------------------
    @R.bang("retry_jobs", options=["id", "limit"])
    def _bang_retry_jobs(eng: BotEngine, args: Dict[str, str]) -> CO:
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
            return _txt(f"{'Retried' if ok else 'Not found'}: {jid}")

        limit = args.get("limit")
        n = eng.store.retry_failed_jobs(limit=int(limit) if limit else None)
        return _txt(f"Retried {n} failed job(s).")

    # ----------------------- CHART -----------------------
    @R.at("chart", argspec=["symbol", "timeframe"])
    def _at_chart(eng: BotEngine, args: Dict[str, str]) -> CO:
        """
        Render candlestick chart (with volume & a simple indicator).

        Usage:
          @chart <symbol> <timeframe>

        Example:
          @chart ETHUSDT 1m
        """
        symbol = args["symbol"].upper()
        tf = args["timeframe"]
        try:
            path = eh.render_chart(eng, symbol, tf)
            return CO(OCPhoto(path=path, caption=f"{symbol} {tf}"),
                      f"Chart generated for {symbol} {tf}")
        except Exception as e:
            log().exc(e, where="cmd.chart")
            return CO(f"Error: {e}")


    @R.bang("position-rm", argspec=["position_id"])
    def _bang_position_rm(eng: BotEngine, args: Dict[str, str]) -> CO:
        pid = int(args["position_id"])
        deleted = eng.store.delete_position_completely(pid)
        # TODO improve reporting (still shows this message even if the position id does not exist)
        return _txt(f"Deleted position {pid} ({deleted} row(s)).")

    # ======================= POSITION EDIT =======================
    @R.bang("position-set",
            argspec=["position_id"],
            options=["num", "den", "dir_sign", "target_usd", "risk", "user_ts", "status", "note", "created_ts"])
    def _bang_position_set(eng: BotEngine, args: Dict[str, str]) -> CO:
        """
        Update fields in a position row (by position_id).

        Usage:
          !position-set <position_id> [num:ETHUSDT] [den:STRKUSDT|-] [dir_sign:-1|+1]
                          [target_usd:5000] [risk:0.02] [user_ts:<when>] [status:OPEN|CLOSED]
                          [note:...] [created_ts:<when>]

        Notes:
          - user_ts/created_ts accept ISO or epoch via parse_when().
          - den:"-" (or empty/none/null) clears denominator (single-leg).
        """
        pid_s = args.get("position_id", "")
        if not pid_s.isdigit():
            return _txt("Usage: !position-set <position_id> …options…")
        pid = int(pid_s)

        # capture only provided (recognized) option keys
        provided = {k: v for k, v in args.items()
                    if k in {"num", "den", "dir_sign", "target_usd", "risk", "user_ts", "status", "note",
                             "created_ts"} and v is not None}

        if not provided:
            return _txt("Nothing to update. Allowed keys: num den dir_sign target_usd user_ts status note created_ts")

        try:
            fields = _coerce_position_fields(provided)
            n = eng.store.sql_update("positions", "position_id", pid, fields)
            if n == 0:
                return _txt(f"Position {pid} not found or unchanged.")
            # brief echo of what changed
            changed = ", ".join(f"{k}={fields[k]!r}" for k in sorted(fields.keys()))
            return _txt(f"Position {pid} updated: {changed}")
        except Exception as e:
            log().exc(e, where="cmd.position-set")
            return _txt(f"Error updating position {pid}: {e}")

    # ======================= LEG EDIT =======================
    @R.bang("leg-set",
            argspec=["leg_id"],
            options=["position_id", "symbol", "qty", "entry_price", "entry_price_ts", "price_method", "need_backfill",
                     "note"])
    def _bang_leg_set(eng: BotEngine, args: Dict[str, str]) -> CO:
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
            return _txt("Usage: !leg-set <leg_id> …options…")
        lid = int(lid_s)

        provided = {k: v for k, v in args.items()
                    if
                    k in {"position_id", "symbol", "qty", "entry_price", "entry_price_ts", "price_method",
                          "need_backfill",
                          "note"} and v is not None}

        if not provided:
            return _txt("Nothing to update. Allowed keys: position_id symbol qty entry_price entry_price_ts price_method need_backfill note")

        try:
            fields = _coerce_leg_fields(provided)
            n = eng.store.sql_update("legs", "leg_id", lid, fields)
            if n == 0:
                return _txt(f"Leg {lid} not found or unchanged.")
            changed = ", ".join(f"{k}={fields[k]!r}" for k in sorted(fields.keys()))
            return _txt(f"Leg {lid} updated: {changed}")
        except Exception as e:
            # Likely UNIQUE(position_id,symbol) or FK violations, surface cleanly.
            log().exc(e, where="cmd.leg-set")
            return _txt(f"Error updating leg {lid}: {e}")

    # ----------------------- LEG BACKFILL PRICE -----------------------
    @R.bang("leg-set-ebyp", argspec=["leg_id", "price"], options=["lookback_days"])
    def _bang_leg_set_ebyp(eng: BotEngine, args: Dict[str, str]) -> CO:
        """
        Entry-by-price set

        Set a leg's entry_price and entry_price_ts by locating the most recent candle
        whose [low, high] contains the given price, using a configurable coarse→fine path.

        Usage:
          !leg-backfill-price <leg_id> <price> [lookback_days:365]

        Behavior:
          - If the coarsest TF finds no engulfing candle over the lookback, it fails.
          - Otherwise it refines inside that candle's window; if a finer TF has no hit,
            it uses the last level that did.
        """
        try:
            leg_id = int(args["leg_id"])
            price = float(args["price"])
        except Exception:
            return _txt("Usage: !leg-backfill-price <leg_id> <price> [lookback_d:7] [path:1d,1h,1m]")

        lookback_days = int(args.get("lookback_d", "365"))
        path = ["1d", "1h", "1m"]

        leg = eng.store.con.execute("SELECT * FROM legs WHERE leg_id=?", (leg_id,)).fetchone()
        if not leg:
            return _txt(f"Leg #{leg_id} not found.")
        symbol = leg["symbol"]

        ts = _find_price_touch_ts(eng.api, symbol, price, lookback_days=lookback_days, path=path)
        if ts is None:
            return _txt(f"No {path[0]} candle for {symbol} contained {price} within ~{lookback_days}d.")

        eng.store.sql_update(
            table="legs",
            pk_field="leg_id",
            pk_value=leg_id,
            fields={
                "entry_price": price,
                "entry_price_ts": ts,
                "price_method": "manual_touch",
            },
        )

        return _txt(
            f"Leg #{leg_id} ({symbol}) updated: entry_price={price} "
            f"at {ts_human(ts)} (path={','.join(path)}, manual_touch)."
        )

    # ----------------------- PNL PER SYMBOL (LEGS) -----------------------
    @R.at("pnl-symbols", options=["status"])
    def _at_pnl_symbols(eng: BotEngine, args: Dict[str, str]) -> CO:
        """
        Aggregate PnL per symbol by traversing legs directly.

        Usage:
          @pnl-symbols [status:open|closed|all]

        Notes:
          - Uses last cached close from KlinesCache as mark price.
          - PnL per leg = (mark - entry_price) * qty.
          - If any of (entry_price, qty, mark) is missing, PnL is "?" and
            treated as 0 in aggregates, but missing counts are reported.
        """
        status_opt = (args.get("status") or "all").strip().lower()
        valid_status = {"open", "closed", "all"}
        if status_opt not in valid_status:
            return _txt("status must be one of: open|closed|all")

        # --- fetch legs joined with positions.status ---
        q = """
            SELECT l.*, p.status
            FROM legs l
            JOIN positions p ON p.position_id = l.position_id
        """
        params: list = []
        if status_opt in ("open", "closed"):
            q += " WHERE p.status = ?"
            params.append(status_opt.upper())
        q += " ORDER BY l.symbol, l.leg_id"

        rows = eng.store.con.execute(q, params).fetchall()
        if not rows:
            return _txt("No legs match.")

        # --- gather marks per symbol ---
        symbols = sorted({r["symbol"] for r in rows if r["symbol"]})
        marks: Dict[str, Optional[float]] = {s: _last_cached_price(eng, s) for s in symbols}

        # stats[symbol] = {
        #   "last_price": float|None,
        #   "open":  {"legs": int, "pnl_sum": float, "missing": int},
        #   "closed":{"legs": int, "pnl_sum": float, "missing": int},
        # }
        stats: Dict[str, Dict[str, Any]] = {}

        for r in rows:
            sym = r["symbol"]
            if not sym:
                continue
            pos_status = (r["status"] or "").upper()
            grp = "open" if pos_status == "OPEN" else "closed"

            if sym not in stats:
                stats[sym] = {
                    "last_price": marks.get(sym),
                    "open":  {"legs": 0, "pnl_sum": 0.0, "missing": 0},
                    "closed":{"legs": 0, "pnl_sum": 0.0, "missing": 0},
                }

            g = stats[sym][grp]
            g["legs"] += 1

            mark = stats[sym]["last_price"]
            pnl = _leg_pnl(r["entry_price"], r["qty"], mark)
            if pnl is None:
                g["missing"] += 1
            else:
                g["pnl_sum"] += float(pnl)

        # --- render ---
        total_open = total_closed = 0.0
        total_open_missing = total_closed_missing = 0
        headers = ["symbol", "last", "open pnl (legs/miss)", "closed pnl (legs/miss)", "total pnl"]
        table_rows: List[Sequence[Any]] = []

        for sym in sorted(stats.keys()):
            s = stats[sym]
            lp = s["last_price"]
            o = s["open"]
            c = s["closed"]

            total_sym = o["pnl_sum"] + c["pnl_sum"]
            total_open += o["pnl_sum"]
            total_closed += c["pnl_sum"]
            total_open_missing += o["missing"]
            total_closed_missing += c["missing"]

            table_rows.append([
                sym,
                _fmt_num(lp, nd=4),
                f"{_fmt_num(o['pnl_sum'])} ({o['legs']}/{o['missing']})",
                f"{_fmt_num(c['pnl_sum'])} ({c['legs']}/{c['missing']})",
                _fmt_num(total_sym),
            ])

        summary = (
            f"Totals: open={_fmt_num(total_open)} (missing legs={total_open_missing}), "
            f"closed={_fmt_num(total_closed)} (missing legs={total_closed_missing}), "
            f"grand_total={_fmt_num(total_open + total_closed)}"
        )

        return CO([
            OCMarkDown("# PnL per symbol\nMark source: last cached close."),
            OCTable(headers=headers, rows=table_rows),
            OCMarkDown(summary),
        ])


    return R

# === helpers (local to build_registry) ====================================
def _last_cached_price(eng: BotEngine, symbol: str) -> Optional[float]:
    """Latest cached close from KlinesCache (first configured timeframe)."""
    kc = eng.kc
    tfs = eng.cfg.KLINES_TIMEFRAMES or ["1m"]
    for tf in tfs:
        r = kc.last_row(symbol, tf)
        if r and r["close"] is not None:
            return float(r["close"])
    return None

def _fmt_num(x: Any, nd=2) -> str:
    if x is None:
        return "?"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "?"

def _pct_of(val: Any, base: Any) -> Optional[float]:
    try:
        if val is None or base is None:
            return None
        base_f = float(base)
        if base_f == 0.0:
            return None
        return float(val) / base_f
    except Exception:
        return None

def _fmt_pct(frac: Any, nd=2, show_sign: bool = False) -> str:
    if frac is None:
        return "?"
    try:
        sign = "+" if show_sign else ""
        return f"{float(frac)*100:{sign}.{nd}f}%"
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
    if "risk" in raw:
        rv = float(raw["risk"])
        if rv <= 0:
            raise ValueError("risk must be > 0")
        out["risk"] = rv
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

def _coerce_config_fields(raw: dict) -> dict:
    """Coerce incoming config fields."""
    out: Dict[str, Any] = {}
    if "reference_balance" in raw:
        rb = float(raw["reference_balance"])
        if rb <= 0:
            raise ValueError("reference_balance must be > 0")
        out["reference_balance"] = rb
    if "leverage" in raw:
        lv = float(raw["leverage"])
        if lv <= 0:
            raise ValueError("leverage must be > 0")
        out["leverage"] = lv
    if "default_risk" in raw:
        dr = float(raw["default_risk"])
        if dr <= 0 or dr >= 1:
            raise ValueError("default_risk must be between 0 and 1")
        out["default_risk"] = dr
    if "updated_by" in raw:
        out["updated_by"] = str(raw["updated_by"])
    return out


def _find_price_touch_ts(
    api: BinanceUM,
    symbol: str,
    price: float,
    *,
    lookback_days: int = 365,
    path: list[str] = None,        # e.g., ["1d", "1h", "1m"]
    now_ms: int | None = None,
    eps: float = 1e-12,
) -> int | None:
    """
    Find the timestamp (ms) of the most recent candle whose [low, high] contains `price`,
    using a coarse→fine 1-D grid search over kline intervals.

    Strategy
    --------
    1) At the coarsest interval (path[0]), fetch all klines over the last `lookback_days`.
    2) Scan **backwards** and pick the last candle whose [low, high] contains `price`.
    3) Restrict the time window to that candle's [open_ts, close_ts).
    4) Repeat for each finer interval in `path`.
    5) On the finest interval, return an interpolated timestamp in [open_ts, close_ts]
       using the distances from `price` to open/close:

           ts = open_ts + (close_ts - open_ts) *
                |price - open| / (|price - open| + |price - close|)

       (If the denominator is ~0, fall back to the midpoint.)

    Parameters
    ----------
    api : BinanceUM
        Live API client (no cache).
    symbol : str
        E.g. "ETHUSDT".
    price : float
        Target price to detect.
    lookback_days : int
        Window (in days) to scan on the coarsest interval.
    path : list[str]
        Timeframes coarse→fine. Default: ["1d", "1h", "1m"].
    prefer : str
        Kept only for backward-compatibility; currently ignored. Behavior is always
        "most recent hit" (scan backwards).
    now_ms : int | None
        Reference "now" in ms. Defaults to api.now_ms().
    eps : float
        Tolerance for float comparisons.

    Returns
    -------
    int | None
        Interpolated timestamp (ms) in [open_ts, close_ts] of the finest-matched candle,
        or None if no match at any level.
    """

    if path is None:
        path = ["1d", "1h", "1m"]

    if now_ms is None:
        now_ms = api.now_ms()

    # --- helpers ---
    def _hits(lo: float, hi: float) -> bool:
        return (lo - eps) <= price <= (hi + eps)

    # Coarsest window: full lookback
    start_ms = now_ms - int(lookback_days) * 86_400_000
    end_ms = now_ms
    window = (start_ms, end_ms)

    last_kline: list | None = None
    last_interval: str | None = None

    for interval in path:
        rows = api.klines(symbol, interval, window[0], window[1])
        if not (isinstance(rows, list) and rows):
            return None

        hit = None
        # Scan backwards: most recent candle first
        for r in reversed(rows):
            lo = float(r[3])  # low
            hi = float(r[2])  # high
            if _hits(lo, hi):
                hit = r
                break

        if not hit:
            return None

        o_ts = int(hit[0])
        c_ts = o_ts + tf_ms(interval)
        window = (o_ts, c_ts)
        last_kline = hit
        last_interval = interval

    if last_kline is None:
        return None

    # ---- interpolate inside the finest candle's [open_ts, close_ts] ----
    open_ts, close_ts = window
    o_price = float(last_kline[1])  # open
    c_price = float(last_kline[4])  # close

    dist_o = abs(price - o_price)
    dist_c = abs(price - c_price)
    denom = dist_o + dist_c

    if denom <= eps:
        # Degenerate case: open == close == price (or numerically very close)
        return int((open_ts + close_ts) // 2)

    #   ts = open_ts + (close_ts - open_ts) * |price-open| / (|price-open| + |price-close|)
    frac = dist_o / denom
    ts = open_ts + (close_ts - open_ts) * frac
    return int(round(ts))


def _remove_indent(docstring: str) -> List:
    if not docstring:
        return ""
    lines = docstring.splitlines()

    # compute common indent from lines after the first
    rest = [ln for ln in lines[1:] if ln.strip()]
    if rest:
        indents = [len(ln) - len(ln.lstrip(" ")) for ln in rest]
        common = min(indents)
    else:
        common = 0

    def strip_prefix(ln: str) -> str:
        if common <= 0:
            return ln
        # remove up to 'common' leading spaces if they are present
        return ln[common:] if ln.startswith(" " * common) else ln

    stripped = [strip_prefix(ln) for ln in lines]
    return stripped
