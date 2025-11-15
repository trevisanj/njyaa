#!/usr/bin/env python3
# FILE: console_ui.py

from __future__ import annotations
import sys, threading, signal, time
from typing import Optional
from cfg_maker import make_cfg
from common import log
# add near the top of console_ui.py
import os, atexit
import readline  # GNU readline on Linux/macOS


# --- import your project bits ---
from bot_api import (
    AppConfig, Storage, BinanceUM, MarketCatalog, PriceOracle,
    PositionBook, AlertsEngine, Reporter, Worker,
    TelegramBot, BotEngine, Clock, set_global_logger, Log,
    build_registry, CommandContext,
)

# ------------- small helpers -------------
def every(engine: BotEngine, interval_sec: float, fn):
    """Recursively reschedule a callable using the engine's internal scheduler."""
    def _wrap():
        try:
            fn()
        finally:
            engine.schedule_once(interval_sec, _wrap)
    engine.schedule_once(interval_sec, _wrap)

def start_engine(cfg: AppConfig) -> BotEngine:
    eng = BotEngine(cfg)
    # start worker thread; Telegram won't start when TELEGRAM_ENABLED=False
    eng.start()
    return eng


def stop_engine(eng: BotEngine):
    try:
        eng.stop()
    except Exception:
        pass

# ------------- console loop -------------
# replace your ConsoleUI with this one
class ConsoleUI:
    """
    Simple blocking REPL that dispatches the same @ / ! commands.
    - Up/Down history via readline.
    - TAB completes @/! commands; persists history to ~/.rv_console_history.
    """
    def __init__(self, eng: BotEngine):
        self.eng = eng
        self._alive = True
        self._printer_lock = threading.Lock()
        self._hist_file = os.path.expanduser("~/.rv_console_history")
        self._setup_readline()

    # ----- readline / completion -----
    def _setup_readline(self):
        if not readline:
            return
        # load/save history
        try:
            readline.read_history_file(self._hist_file)
        except FileNotFoundError:
            pass
        atexit.register(self._save_history)

        # tab completion
        readline.parse_and_bind("tab: complete")
        readline.set_completer(self._completer)

    def _save_history(self):
        if not readline:
            return
        try:
            readline.set_history_length(1000)
            readline.write_history_file(self._hist_file)
        except Exception:
            pass

    def _command_words(self):
        # pull command names from the registry (e.g., @open, @positions, !open, !psar, etc.)
        words = {":q", ":quit", ":exit"}
        try:
            for (prefix, name) in getattr(self.eng, "_registry")._handlers.keys():
                words.add(f"{prefix}{name}")
        except Exception:
            pass
        return sorted(words)

    def _completer(self, text, state):
        # very simple: complete first token only (the command head)
        head = (readline.get_line_buffer() or "").lstrip()
        if " " in head:
            # later you can add arg completion like 'status:'/'what:'; for now, noop
            return None
        matches = [w for w in self._command_words() if w.startswith(text)]
        return matches[state] if state < len(matches) else None

    # ----- I/O -----
    def _print(self, msg: str):
        with self._printer_lock:
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()

    # ----- main loop -----
    def run(self):
        self._print("RV console ready. Type @help, or :quit to exit. (TAB completes; ↑↓ recall history)")
        while self._alive:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line in (":q", ":quit", ":exit"):
                break
            try:
                out = self.eng.dispatch_command(line, chat_id=0)
                self._print(out or "⏳ thinking…")
            except Exception as e:
                log.exc(e, where="console.dispatch")
                self._print(f"Error: {e}")
        self._print("Shutting down…")
        self._alive = False

# ------------- main -------------
def main():
    # logging style: multi-line tracebacks; concise INFO by default
    lg = Log(level="INFO", name="rv", json_mode=False)
    set_global_logger(lg)

    cfg = make_cfg()

    cfg.TELEGRAM_ENABLED=False

    # reflect desired log level
    lg.set_level(cfg.LOG_LEVEL or "INFO")

    eng = start_engine(cfg)

    # ensure console also prints any engine.send_text() calls
    # BotEngine.send_text() already prints to stdout when Telegram is disabled.

    ui = ConsoleUI(eng)

    # graceful signals
    def _sigint(_sig, _frm):
        ui._alive = False
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    try:
        ui.run()
    finally:
        stop_engine(eng)

if __name__ == "__main__":
    import os
    main()
