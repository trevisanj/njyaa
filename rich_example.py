from rich.console import Console
from rich.markdown import Markdown

console = Console()
markdown = Markdown("""# Commands

**!alert** <symbol> <op> <price> [msg:…]
*Create a simple threshold alert thinker.*
Usage:
          !alert <SYMBOL> (>=|<=) <PRICE> [msg:...]

**!config-set** [default_risk:…] [leverage:…] [reference_balance:…] [updated_by:…]
*Update global risk config.*
Usage:
          !config-set reference_balance:12000 leverage:2.0 default_risk:0.015

**!leg-set** <leg_id> [entry_price:…] [entry_price_ts:…] [need_backfill:…] [note:…] [position_id:…] [price_method:…] [qty:…] [symbol:…]
*Update fields in a leg row (by leg_id). Usage: !leg-set <leg_id> [position_id:123] [symbol:ETHUSDT] [qty:-0.75] [entry_price:3521.4] [entry_price_ts:<when>] [price_method:aggTrade|kline|mark_kline] [need_backfill:0|1] [note:...] Notes: - entry_price_ts accepts ISO or epoch via parse_when(). - Changing position_id/symbol must respect UNIQUE(position_id,symbol).*

**!leg-set-ebyp** <leg_id> <price> [lookback_days:…]
*Entry-by-price: set a leg's entry_price and entry_price_ts by locating the most recent candle whose [low, high] contains the given price, using a configurable coarse→fine path.*
Usage:
          !leg-backfill-price <leg_id> <price> [lookback_days:365]
        Behavior:
          - If the coarsest TF finds no engulfing candle over the lookback, it fails.
          - Otherwise it refines inside that candle's window; if a finer TF has no hit,
            it uses the last level that did.

**!open** <pair> <ts> <usd> [note:…] [risk:…]
*Open/add an RV position.*
Usage:
          !open <ISO|epoch_ms> <NUM/DEN|SYMBOL> <±usd> [risk:0.02] [note:...]
        Examples:
          !open STRK/ETH 2025-11-10T13:44:05  -5000 note:rv test
          !open ETHUSDT  2025-11-10T13:44:05 +3000

**!position-rm** <position_id>

**!position-set** <position_id> [created_ts:…] [den:…] [dir_sign:…] [note:…] [num:…] [risk:…] [status:…] [target_usd:…] [user_ts:…]
*Update fields in a position row (by position_id). Usage: !position-set <position_id> [num:ETHUSDT] [den:STRKUSDT|-] [dir_sign:-1|+1] [target_usd:5000] [risk:0.02] [user_ts:<when>] [status:OPEN|CLOSED] [note:...] [created_ts:<when>] Notes: - user_ts/created_ts accept ISO or epoch via parse_when(). - den:"-" (or empty/none/null) clears denominator (single-leg).*

**!psar** <position_id> <symbol> <direction> [af:…] [max:…] [max_af:…] [win:…] [window:…] [window_min:…]
*Set a PSAR stop thinker.*
Usage:
          !psar <position_id> <symbol> <LONG|SHORT> [af:0.02] [max:0.2] [win:200]

**!retry_jobs** [id:…] [limit:…]
*Retry failed jobs.*
Usage:
          !retry_jobs [id:<job_id>] [limit:N]
            - With id: retries one job.
            - Without id: retries all ERR jobs (optionally limited).

**!thinker-disable** <id>
*Disable a thinker by ID. Usage: !thinker-disable <id>*

**!thinker-enable** <id>
*Enable a thinker by ID. Usage: !thinker-enable <id>*

**!thinker-rm** <id>
*Delete a thinker by ID. Usage: !thinker-rm <id>*

**@chart** <symbol> <timeframe>
*Render candlestick chart (with volume & a simple indicator).*
Usage:
          @chart <symbol> <timeframe>
        Example:
          @chart ETHUSDT 1h

**@help** <command> [detail:…]
*Show this help.*

**@jobs** [limit:…] [state:…]
*List DB jobs.*
Usage:
          @jobs [state:PENDING|RUNNING|DONE|ERR] [limit:50]

**@open**
*List open RV positions (summary). Alias for: @positions status:open detail:2*

**@pnl-symbols** [status:…]
*Aggregate PnL per symbol by traversing legs directly.*
Usage:
          @pnl-symbols [status:open|closed|all]
        Notes:
          - Uses last cached close from KlinesCache as mark price.
          - PnL per leg = (mark - entry_price) * qty.
          - If any of (entry_price, qty, mark) is missing, PnL is "?" and
            treated as 0 in aggregates, but missing counts are reported.

**@positions** [detail:…] [limit:…] [pair:…] [position_id:…] [sort:…] [status:…]
*Positions report with detail levels.*
detail: 1 = summary
                2 = summary + per-position overview (with opened datetime & signed target)
                3 = level 2 + per-symbol leg summary (agg qty, WAP, last price, PnL)
                4 = level 2 + list every leg (timestamp, entry, last, qty, PnL)
        Examples:
          @positions detail:1
          @positions status:closed detail:2
          @positions pair:STRK/ETH detail:3 limit:20

**@risk**
*Show risk/exposure snapshot.*

**@thinker-kinds**
*List available thinker kinds (from factory auto-discovery).*

**@thinkers**
*List thinkers stored in DB.*
""")
console.print(markdown)