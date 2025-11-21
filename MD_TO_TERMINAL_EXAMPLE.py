from dataclasses import dataclass
from typing import Any, Dict, List

import mistune
from rich.console import Console, RenderableType
from rich.text import Text
from rich.rule import Rule
from rich.syntax import Syntax

# ==========================
# ===== STYLE CONFIG =======
# ==========================

@dataclass
class StyleConfig:
    heading_style: str = "bold underline"
    body_style: str = ""              # e.g. "white"
    bullet_style: str = "bold"
    strong_style: str = "bold"
    emphasis_style: str = "italic"
    inline_code_style: str = "reverse"
    code_block_theme: str = "monokai"
    code_block_default_lexer: str = "bash"  # fallback if no ```lang
    heading_separator: bool = False   # set True to print a Rule() after headings


STYLES = StyleConfig()

# Single global markdown parser -> AST
md_parser = mistune.create_markdown(renderer="ast")


# ==========================
# ===== RENDERER CORE ======
# ==========================

def _render_inlines(nodes: List[Dict[str, Any]], cfg: StyleConfig) -> Text:
    """Render inline nodes (text, strong, emphasis, codespan, breaks) to a single Rich Text."""
    out = Text()
    for node in nodes:
        t = node["type"]

        if t == "text":
            out.append(node["raw"])

        elif t in ("softbreak", "linebreak"):
            out.append("\n")

        elif t == "strong":
            inner = _render_inlines(node.get("children", []), cfg)
            inner.stylize(cfg.strong_style)
            out.append(inner)

        elif t == "emphasis":
            inner = _render_inlines(node.get("children", []), cfg)
            inner.stylize(cfg.emphasis_style)
            out.append(inner)

        elif t == "codespan":
            out.append(Text(node["raw"], style=cfg.inline_code_style))

        else:
            # Fallback: recurse on children, if any
            inner = _render_inlines(node.get("children", []), cfg)
            out.append(inner)

    return out


def _render_block(node: Dict[str, Any], cfg: StyleConfig):
    """Yield one or more Rich renderables for a single block-level node."""
    t = node["type"]

    # ---- headings (left-aligned by default) ----
    if t == "heading":
        txt = _render_inlines(node.get("children", []), cfg)
        if cfg.heading_style:
            txt.stylize(cfg.heading_style)
        yield txt
        if cfg.heading_separator:
            yield Rule()
        else:
            yield Text()  # blank line
        return

    # ---- paragraphs ----
    if t == "paragraph":
        txt = _render_inlines(node.get("children", []), cfg)
        if cfg.body_style:
            txt.stylize(cfg.body_style)
        yield txt
        yield Text()  # blank line
        return

    # ---- bullet / ordered lists ----
    if t == "list":
        for item in node.get("children", []):
            # list_item
            for r in _render_block(item, cfg):
                yield r
        yield Text()
        return

    if t == "list_item":
        # Treat children as a single inline paragraph for now
        bullet = Text("- ", style=cfg.bullet_style)
        inner = _render_inlines(node.get("children", []), cfg)
        bullet.append(inner)
        yield bullet
        return

    # ---- fenced / indented code blocks ----
    if t == "block_code":
        lang = (node.get("info") or "").strip() or cfg.code_block_default_lexer
        code_text = node.get("text", "")
        yield Syntax(code_text.rstrip("\n"), lang, theme=cfg.code_block_theme, line_numbers=False)
        yield Text()
        return

    # ---- horizontal rule ----
    if t == "thematic_break":
        yield Rule()
        return

    # ---- anything else: recurse into children if present ----
    for child in node.get("children", []):
        for r in _render_block(child, cfg):
            yield r


def render_markdown_to_rich(md_text: str, cfg: StyleConfig = STYLES) -> List[RenderableType]:
    """Parse markdown with mistune AST and convert to a list of Rich renderables."""
    ast = md_parser(md_text)
    renderables: List[RenderableType] = []
    for node in ast:
        for r in _render_block(node, cfg):
            renderables.append(r)
    return renderables


# ==========================
# ===== SAMPLE USAGE =======
# ==========================

SAMPLE = """
Commands
--------

**!alert** <symbol> <op> <price> [msg:…]

*Create a simple threshold alert thinker.*

```
        Usage:
          !alert <SYMBOL> (>=|<=) <PRICE> [msg:...]
```

**!config-set** [default_risk:…] [leverage:…] [reference_balance:…] [updated_by:…]

*Update global risk config.*

```
        Usage:
          !config-set reference_balance:12000 leverage:2.0 default_risk:0.015
```

**!leg-set** <leg_id> [entry_price:…] [entry_price_ts:…] [need_backfill:…] [note:…] [position_id:…] [price_method:…] [qty:…] [symbol:…]

*Update fields in a leg row (by leg_id).*

```
        Usage:
          !leg-set <leg_id> [position_id:123] [symbol:ETHUSDT] [qty:-0.75]
                           [entry_price:3521.4] [entry_price_ts:<when>]
                           [price_method:aggTrade|kline|mark_kline] [need_backfill:0|1]
                           [note:...]

        Notes:
          - entry_price_ts accepts ISO or epoch via parse_when().
          - Changing position_id/symbol must respect UNIQUE(position_id,symbol).
```

**!leg-set-ebyp** <leg_id> <price> [lookback_days:…]

*Entry-by-price set*

```
        Set a leg's entry_price and entry_price_ts by locating the most recent candle
        whose [low, high] contains the given price, using a configurable coarse→fine path.

        Usage:
          !leg-backfill-price <leg_id> <price> [lookback_days:365]

        Behavior:
          - If the coarsest TF finds no engulfing candle over the lookback, it fails.
          - Otherwise it refines inside that candle's window; if a finer TF has no hit,
            it uses the last level that did.
```

**!open** <pair> <ts> <usd> [note:…] [risk:…]

*Open/add an RV position.*

```
        Usage:
          !open <ISO|epoch_ms> <NUM/DEN|SYMBOL> <±usd> [risk:0.02] [note:...]

        Examples:
          !open STRK/ETH 2025-11-10T13:44:05  -5000 note:rv test
          !open ETHUSDT  2025-11-10T13:44:05 +3000
```

**!position-rm** <position_id>

**!position-set** <position_id> [created_ts:…] [den:…] [dir_sign:…] [note:…] [num:…] [risk:…] [status:…] [target_usd:…] [user_ts:…]

*Update fields in a position row (by position_id).*

```
        Usage:
          !position-set <position_id> [num:ETHUSDT] [den:STRKUSDT|-] [dir_sign:-1|+1]
                          [target_usd:5000] [risk:0.02] [user_ts:<when>] [status:OPEN|CLOSED]
                          [note:...] [created_ts:<when>]

        Notes:
          - user_ts/created_ts accept ISO or epoch via parse_when().
          - den:"-" (or empty/none/null) clears denominator (single-leg).
```

**!psar** <position_id> <symbol> <direction> [af:…] [max:…] [max_af:…] [win:…] [window:…] [window_min:…]

*Set a PSAR stop thinker.*

```
        Usage:
          !psar <position_id> <symbol> <LONG|SHORT> [af:0.02] [max:0.2] [win:200]
```

**!retry_jobs** [id:…] [limit:…]

*Retry failed jobs.*

```
        Usage:
          !retry_jobs [id:<job_id>] [limit:N]
            - With id: retries one job.
            - Without id: retries all ERR jobs (optionally limited).
```

**!thinker-disable** <id>

*Disable a thinker by ID*

```
        Usage: !thinker-disable <id>
```

**!thinker-enable** <id>

*Enable a thinker by ID. Usage: !thinker-enable <id>*

**!thinker-rm** <id>

*Delete a thinker by ID.*

```
        Usage: !thinker-rm <id>
```

**@chart-candle** <symbol> <timeframe> [<n=200>]

*Render candlestick chart (with volume & a simple indicator).*

```
        Usage:
          @chart-candle <symbol> <timeframe> [<n>]

        Example:
          @chart-candle ETHUSDT 1m 300
```

**@chart-rv** <pair_or_symbol> [<timeframe=1d>] [<n=200>]

*Render Close-price line chart for a symbol or ratio (with MA).*

```
        Usage:
          @chart-rv <pair_or_symbol> [<timeframe>] [<n>]

        Example:
          @chart-rv ETH/BTC 4h 500
```

**@klines-cache**

*Summarize cached klines (symbol, timeframe, n).*

```
        Usage:
          @klines-cache
```

**@help** <command> [detail:…]

*Show this help.*

```
        Usage:
          @help [command] [detail:1]
```

**@jobs** [limit:…] [state:…]

*List DB jobs.*

```
        Usage:
          @jobs [state:PENDING|RUNNING|DONE|ERR] [limit:50]
```

**@open**

*List open RV positions (summary). Alias for: @positions status:open detail:2*

**@pnl-symbols** [status:…]

*Aggregate PnL per symbol by traversing legs directly.*

```
        Usage:
          @pnl-symbols [status:open|closed|all]

        Notes:
          - Uses last cached close from KlinesCache as mark price.
          - PnL per leg = (mark - entry_price) * qty.
          - If any of (entry_price, qty, mark) is missing, PnL is "?" and
            treated as 0 in aggregates, but missing counts are reported.
```

**@positions** [detail:…] [limit:…] [pair:…] [position_id:…] [sort:…] [status:…]

*Positions report with detail levels.*

```
        detail: 1 = summary
                2 = summary + per-position overview (with opened datetime & signed target)
                3 = level 2 + per-symbol leg summary (agg qty, WAP, last price, PnL)
                4 = level 2 + list every leg (timestamp, entry, last, qty, PnL)

        Examples:
          @positions detail:1
          @positions status:closed detail:2
          @positions pair:STRK/ETH detail:3 limit:20
```

**@risk**

*Show risk/exposure snapshot.*

**@thinker-kinds**

*List available thinker kinds (from factory auto-discovery).*

**@thinkers**

*List thinkers stored in DB.*
"""

if __name__ == "__main__":
    console = Console()
    for item in render_markdown_to_rich(SAMPLE, STYLES):
        console.print(item)
