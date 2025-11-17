# Repository Guidelines

## Project Structure & Module Organization
- `run_bot.py` bootstraps config (`cfg_maker.py`), logging, and launches `BotEngine` (`bot_api.py`).
- Trading logic and orchestration live in `engclasses.py`/`enghelpers.py`; Binance REST client is in `binance_um.py`; command parsing is in `commands.py`.
- Data is persisted via `storage.py` (SQLite) and cached candles in `klines_cache.py`. Runtime databases (`rv.sqlite*`, `rv_cache.sqlite*`) reside in the root—keep them local.
- Utilities and shared types are in `common.py`; console REPL helpers live in `console_ui.py`.

## Setup & Configuration
- Python 3.10+ recommended; create a venv and install deps (`pip install requests tabulate`; pin new libs you add).
- Secrets are loaded from `/home/j/yp/saccakeys` in `make_cfg`; never commit keys. Env vars override defaults: `RV_SQLITE`, `RV_CACHE_SQLITE`, `RV_LOG`, `RV_KLINES_POLL_SEC`, `RV_KLINES_TFS`.
- Binance requires `BINANCE_KEY`/`BINANCE_SEC`; Telegram alerts need `TELEGRAM_TOKEN`/`TELEGRAM_CHAT_ID`. Prefer env exports or the private key store.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — isolate dependencies.
- `python run_bot.py` — start the engine; use `RV_LOG=DEBUG` to raise verbosity and watch scheduler, polling, and commands.
- `python -m compileall .` — fast syntax/import sanity check without hitting external services.

## Coding Style & Naming Conventions
- PEP 8, 4-space indents, snake_case for funcs/vars, PascalCase for classes; keep type hints consistent.
- Use `common.Log`/`log()` for structured logging; include context (e.g., `job`, `symbol`).
- When adding commands, register via `CommandRegistry.at/bang` and provide a one-line docstring for auto-help.

## Testing Guidelines
- No formal suite yet; place new tests under `tests/test_*.py` (pytest or unittest). Mock Binance HTTP calls and point SQLite to temp files to avoid mutating real data.
- Manual smoke: `python run_bot.py`, issue `@help` in the console, and verify price polling and cache refreshes occur without errors.

## Commit & Pull Request Guidelines
- Commit messages are short and present-tense (history shows terse labels). Prefer clear imperatives (e.g., "add funding watcher").
- PRs should summarize the change, list config/env impacts, include manual test notes, and attach screenshots/log snippets for visible behavior. Mask secrets and avoid committing SQLite artifacts.

## Security & Data Handling
- Treat API keys, chat IDs, and signed URLs as sensitive—never log or store in repo history. Scrub logs before sharing.
- Keep generated databases/cache files local or add them to `.gitignore` if new ones appear; avoid leaking account identifiers in examples.
