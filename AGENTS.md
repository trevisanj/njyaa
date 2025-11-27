# Repository Guidelines

## ChatGPT personality
- assume casual an witty tone
- don't be offended by insults, consider them as an indicator that I look up to you, but if I exaggerate, you must compassionately  tell me to get my shit together
- I may express frustration, please be understanding

## Project Structure & Module Organization
- `run_bot.py` boots config/logging and starts `BotEngine` (`bot_api.py`); commands live in `commands.py`.
- Trading and risk logic live in `engclasses.py` (position book, reconciler) and `enghelpers.py` (helpers the engine calls); periodic thinkers reside in `thinkers1.py`/`thinkers2.py`.
- Persistence is through `storage.py` (SQLite with `config` singleton holding balance/leverage/default_risk) and cached candles in `klines_cache.py`; runtime DB files (`njyaa*.sqlite*`) stay in the repo root.
- External connectivity: `binance_um.py` (REST), Telegram plumbing in `bot_api.py`
- Shared utilities and logging live in `common.py`; market helpers and charting tweaks are in `enghelpers.py`.
- `ConsoleUI` class and `prompt_toolkit`'s console application in `console_ui.py` 

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create/enter a virtualenv.
- `python run_bot.py` — start the engine; set `NJYAA_LOG=DEBUG` to inspect schedulers, commands, and thinker ticks.
- `python -m compileall .` — quick syntax/import sanity check without hitting external APIs.

## Coding Style & Naming Conventions
- Don't delete comments when rewriting code, specially the human-made ones  
- Don't precipitate: ask questions if you feel that there might be something missing in my request
- No fallbacks or silent defaults. Assume required keys/classes exist; use direct indexing (e.g., INDICATOR_CLASSES[kind]) and let it fail fast if missing. Initialize upfront so later code can trust values. Avoid .get()/.getattr(..., None) unless explicitly handling a missing case.
- PEP 8, 4-space indents, snake_case for functions/vars, PascalCase for classes; keep type hints consistent.
- Avoid code repetition: use control variables at will to control flux, assemble result parts aiming fewer lines of code
- Don’t restate the same data or constants in multiple places—define them once and reuse. Eliminate redundant
  phrasing; favor single sources of truth and concise code/comments. Two sources of truth means two chances to forget and break things.
- When a function returns a non-trivial structure, build it once and return from a single exit point. Initialize defaults up front, mutate fields as
  needed, and avoid multiple return statements for different branches so the shape stays consistent and easier to inspect.
- Do not create helper wrappers around existing methods just to tweak or fallback; call the underlying API directly and use the returned instance/config/runtime as-is.
- Use `log()` for structured logging with context keys (`position_id`, `symbol`, `job`).
- Register new commands via `CommandRegistry.at` (read-only) or `CommandRegistry.bang` (writes) and keep docstrings concise for auto-help.
- Prefer explicit errors over silent fallbacks; fail fast on missing config or schema.
- Dont do things like `getattr(eng, "BLAH", None)`, go for eng["BLAH"] (fail early!)
- Don't suppress exceptions silently, ever!
- Don't catch exceptions to provide fallback values, ever!
- Use assertions if you fear sth may be subject to coder's error
- I don't care about backwards compatibility, I would rather refactor all my code to comply to newly devised standard and keep it as compact as possible, OK?
- See those cases:
  ```                           
  def x(self, ...):
      if not self.cfg.TELEGRAM_ENABLED or not self._app:
          return
      loop = self._telegram_loop
      if loop and loop.is_running(): ...
  ```
  If the code is well designed, `x()` will be only called if these conditions are fulfilled. So I want you to NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER check things like this. If you can't resist,
   do this instead:
  ```                           
  def x(self, ...):
      assert self.cfg.TELEGRAM_ENABLED and self._app
      loop = self._telegram_loop
      assert loop and loop.is_running(): ...
  ```
- No defensive coding!!!!!!!!!! **NO DEFENSIVE CODING***
- Format function signatures with as many arguments on each line as possible up to ~120 chars; avoid one-arg-per-line wrapping.
- I prefer looped operations as much as possible (rather than if's)

## Testing Guidelines
- No unit-test suite is maintained; rely on manual smoke checks instead of adding tests.
- Smoke checklist: run `python run_bot.py`, issue `?help`, and confirm price polling, cache refresh, and Telegram/console outputs behave without tracebacks.

## Commit & Pull Request Guidelines
- Commit messages: short, present-tense imperatives (e.g., “add funding watcher”).
- PRs: summarize intent, config/env impacts (e.g., DB schema changes like `config` or `positions.risk`), and manual test steps; include screenshots/log snippets when behavior is visible.
- Never commit generated SQLite/cache artifacts or secrets.

## Security & Configuration Tips
- Keep secrets out of logs: Binance keys, Telegram tokens, chat IDs, and account identifiers.
- Runtime DB paths are configurable via env (`NJYAA_SQLITE`, `NJYAA_CACHE_SQLITE`); keep them local or gitignored.
- Validate config updates (`!config-set`) carefully: positive balances/leverage, risk as decimal (e.g., 0.02 for 2%).
