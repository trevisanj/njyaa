#!/usr/bin/env python3
# FILE: bot_api.py

from __future__ import annotations

import sys
import json
import threading
import time
import os
import signal
import logging
import io
from queue import Queue, Empty
from typing import Callable, List, Optional, Dict, TYPE_CHECKING

from common import *  # Clock, log, AppConfig, tf_ms, etc.
from storage import Storage
from binance_um import BinanceUM
from klines_cache import KlinesCache
from engclasses import *  # MarketCatalog, PriceOracle, PositionBook, Worker, Reporter, ThinkerManager, etc.
import tabulate
import asyncio

from collections import deque
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import ANSI, merge_formatted_text
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, VSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from rich.console import Console


class _CommandCompleter(Completer):
    """prompt_toolkit completer that mirrors the registry command names."""

    def __init__(self, console: "ConsoleUI"):
        self.console = console

    def get_completions(self, document, complete_event):
        text = (document.text_before_cursor or "").lstrip()
        if " " in text:
            return
        for word in self.console._command_words():
            if word.startswith(text):
                yield Completion(word, start_position=-len(text))


class _EngineLogStream:
    """Bridge standard Log writes into BotEngine-managed sinks."""

    def __init__(self, engine: "BotEngine"):
        self.engine = engine

    def write(self, data: str):
        self.engine._handle_log_stream_write(data)

    def flush(self):
        self.engine._flush_log_stream()


if TYPE_CHECKING:
    from commands import CommandRegistry, CO, OCText  # only for type hints

# =======================
# Telegram logging tweaks
# =======================
# Rich PTB stacktraces on transient network hiccups spam the console.
# These switches let you quickly tone them down without hunting through PTB internals.
PTB_LOGGERS = [
    "telegram.ext._utils.networkloop",
    "telegram.ext._updater",
    "telegram.request",
]
PTB_LOG_LEVEL = logging.WARNING
PTB_STRIP_TRACEBACK = True  # keep the message but drop the multi-line traceback


CONSOLE_THEME = {
    "command_echo": "bold deep_sky_blue1",  # style for echoed commands in console log pane
}


class _PTBDropTraceback(logging.Filter):
    """Scrubs exc_info so PTB logs print a single line instead of full tracebacks."""

    def filter(self, record: logging.LogRecord) -> bool:
        if PTB_STRIP_TRACEBACK and record.exc_info:
            record.exc_info = None
            record.exc_text = None
        return True


_PTB_LOG_CONFIGURED = False


def _configure_ptb_logging():
    """Apply minimal formatting to PTB loggers (idempotent)."""
    global _PTB_LOG_CONFIGURED
    if _PTB_LOG_CONFIGURED:
        return
    filt = _PTBDropTraceback()
    for name in PTB_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(PTB_LOG_LEVEL)
        if PTB_STRIP_TRACEBACK:
            logger.addFilter(filt)
    _PTB_LOG_CONFIGURED = True


# =======================
# == INTERNAL SCHEDULER ==
# =======================

class InternalScheduler:
    def __init__(self):
        self._jobs: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    def schedule_every(
        self,
        name: str,
        interval_sec: int,
        fn: Callable[[], None],
        run_immediately: bool = False,
    ):
        nxt = time.time() if run_immediately else (time.time() + max(1, interval_sec))
        with self._lock:
            self._jobs[name] = {
                "interval": max(1, int(interval_sec)),
                "fn": fn,
                "next": nxt,
            }

    def cancel(self, name: str):
        with self._lock:
            self._jobs.pop(name, None)

    def start(self, thread_name: str = "rv-scheduler"):
        if self._thr and self._thr.is_alive():
            return
        self._stop.clear()
        self._thr = threading.Thread(
            target=self._run, name=thread_name, daemon=False
        )
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join()
            log().info("InternalScheduler: thread joined")

    def _run(self):
        while not self._stop.is_set():
            now = time.time()
            todo: list[tuple[str, Callable[[], None], int]] = []
            with self._lock:
                for name, j in self._jobs.items():
                    if now >= j["next"]:
                        todo.append((name, j["fn"], j["interval"]))
                        j["next"] = now + j["interval"]
            for name, fn, _itv in todo:
                try:
                    fn()
                except Exception as e:
                    log().exc(e, where="scheduler.job", job=name)
            time.sleep(0.5)


# =========================
# ====== CONSOLE UI =======
# =========================

class ConsoleUI:
    """
    prompt_toolkit-based console with a scrolling log pane above the prompt.
    All console output accumulates in the log so user input never gets clobbered.
    """

    def __init__(self, eng: "BotEngine"):
        self.eng = eng
        self._alive = True
        self._stop_notified = False
        self._hist_file = os.path.expanduser("~/.rv_console_history")
        self._ensure_history_file()
        self._history = FileHistory(self._hist_file)
        self._completer = _CommandCompleter(self)
        self._log_lines: deque[str] = deque(maxlen=1000)
        self._input_buffer = Buffer(
            history=self._history,
            completer=self._completer,
            complete_while_typing=True,
            multiline=False,
        )
        self._input_buffer.accept_handler = self._handle_submit
        self._log_control = FormattedTextControl(self._render_log, focusable=False)
        self._input_control = BufferControl(buffer=self._input_buffer, focus_on_click=True)
        self._app = self._build_application()

    def _ensure_history_file(self):
        hist_dir = os.path.dirname(self._hist_file)
        if hist_dir:
            try:
                os.makedirs(hist_dir, exist_ok=True)
            except Exception:
                pass
        try:
            with open(self._hist_file, "a", encoding="utf-8"):
                pass
        except Exception:
            pass

    def _command_words(self):
        words = {":q", ":quit", ":exit"}
        reg = self.eng._registry
        if reg:
            for (prefix, name) in reg._handlers.keys():
                words.add(f"{prefix}{name}")
        return sorted(words)

    def _render_log(self):
        if not self._log_lines:
            return ""
        fragments = []
        for line in self._log_lines:
            text = line if line.endswith("\n") else line + "\n"
            fragments.append(ANSI(text))
        return merge_formatted_text(fragments)

    def _build_application(self) -> Application:
        kb = KeyBindings()

        @kb.add("c-c")
        @kb.add("c-d")
        def _(event):
            self._on_exit_requested()

        prompt_row = VSplit(
            [
                Window(
                    content=FormattedTextControl(lambda: "> "),
                    width=4,
                    dont_extend_width=True,
                ),
                Window(content=self._input_control, height=1),
            ],
            height=1,
        )

        root = HSplit(
            [
                Window(
                    content=self._log_control,
                    wrap_lines=True,
                    height=Dimension(weight=1),
                ),
                Window(height=1, char="─"),
                prompt_row,
            ]
        )

        layout = Layout(root, focused_element=self._input_control)
        return Application(
            layout=layout,
            key_bindings=kb,
            full_screen=True,
            mouse_support=False,
            refresh_interval=0.1,
        )

    def _handle_submit(self, buff: Buffer) -> bool:
        line = buff.text.strip()
        buff.reset()
        if not line:
            return False
        if line in (":q", ":quit", ":exit"):
            self._on_exit_requested()
            return False
        self._history.append_string(line)
        try:
            self.eng.dispatch_command(line, chat_id=0, origin="console")
        except Exception as e:
            log().exc(e, where="console.dispatch")
            self.append_output(f"Error: {e}")
        return False

    def _on_exit_requested(self):
        if self._stop_notified:
            return
        self._stop_notified = True
        self._alive = False
        self.eng.request_stop()
        if self._app and self._app.is_running:
            self._app.exit()

    def append_output(self, text: str):
        if text is None:
            return
        lines = text.split("\n")
        if not lines:
            lines = [""]
        for idx, line in enumerate(lines):
            self._log_lines.append(line)
        if self._app:
            try:
                self._app.invalidate()
            except Exception:
                pass

    def run(self):
        self.append_output("RV console ready. Type @help, or :quit to exit. (TAB completes; ↑↓ recall history)")
        try:
            self._app.run()
        finally:
            if not self._stop_notified:
                self._stop_notified = True
                self.eng.request_stop()

    def stop(self):
        self._alive = False
        self._stop_notified = True
        if self._app and self._app.is_running:
            self._app.exit()


# =========================
# ===== TELEGRAM SHIM =====
# =========================

class TelegramBot:
    """
    Thin Telegram adapter. No internal enable flag.
    Currently mostly a placeholder to carry the registry reference,
    since handlers are bound directly to BotEngine methods.
    """

    def __init__(self, eng: "BotEngine"):
        self.engine = eng
        self._registry: Optional["CommandRegistry"] = None

    def set_registry(self, registry: "CommandRegistry"):
        self._registry = registry


# =========================
# ======= BOT ENGINE ======
# =========================

class BotEngine:
    """
    Core orchestrator for the RV system.

    Responsibilities:
      - Build and wire domain services (BinanceUM, Storage, MarketCatalog, etc.)
      - Run background Worker + InternalScheduler
      - Expose a command dispatcher used by all frontends (console, Telegram, etc.)
      - Provide send_text() / send_photo() helpers that fan out to all enabled frontends.
    """

    def __init__(self, cfg: AppConfig):
        _configure_ptb_logging()
        # keep imports local to avoid cycles when possible
        from thinkers1 import ThinkerManager
        from commands import CommandRegistry

        self.cfg = cfg

        # lazily built parts
        self.api: Optional[BinanceUM] = None
        self.store: Optional[Storage] = None
        self.mc: Optional[MarketCatalog] = None
        self.oracle: Optional[PriceOracle] = None
        self.positionbook: Optional[PositionBook] = None
        self.reporter: Optional[Reporter] = None
        self.worker: Optional[Worker] = None
        self.scheduler: Optional[InternalScheduler] = None
        self.tgbot: Optional[TelegramBot] = None
        self._registry: Optional[CommandRegistry] = None
        self.kc: Optional[KlinesCache] = None
        self.tm: Optional[ThinkerManager] = None

        # PTB application (only if Telegram is enabled)
        self._app = None
        self._tg_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._console_thread: Optional[threading.Thread] = None
        self._console_ui: Optional["ConsoleUI"] = None
        self._telegram_loop: Optional[asyncio.AbstractEventLoop] = None
        self._rich_console = Console(force_terminal=True, color_system="truecolor", soft_wrap=True)

        # rendering sinks
        self._sinks: List[str] = []
        self._console_lock = threading.Lock()
        self._log_stream_lock = threading.Lock()
        self._send_queue: Queue = Queue()
        self._log_stream_buffer = ""
        self._log_stream = _EngineLogStream(self)
        set_global_logger(
            Log(
                level=cfg.LOG_LEVEL or "INFO",
                stream=self._log_stream,
                name="rv",
                json_mode=False,
            )
        )

        # book-keeping
        self._excepthook_installed = False

        self._stopping = False

    # --------------------------------
    # Building + wiring (single place)
    # --------------------------------
    def _build_parts(self):
        assert self.api is None

        from thinkers1 import ThinkerManager
        from commands import build_registry

        # Clock / TZ once
        tz = self.cfg.TZ_LOCAL
        Clock.set_tz(tz)
        log().info("Clock TZ set", tz=str(tz))

        # Core deps
        self.api = BinanceUM(self.cfg)
        self.store = Storage(self.cfg.DB_PATH)
        self.mc = MarketCatalog(self.api)
        self.oracle = PriceOracle(self.cfg, self.api)
        self.positionbook = PositionBook(self.store, self.mc, self.oracle)
        self.reporter = Reporter()
        self.worker = Worker(self.cfg, self.store, self.api, self.mc, self.oracle)
        self.scheduler = InternalScheduler()
        self.kc = KlinesCache(self)
        self.tm = ThinkerManager(self)

        # Preload catalog to avoid first-use hiccups
        try:
            self.mc.load()
        except Exception as e:
            log().exc(e, where="engine.build.mc.load")

        # Registry (commands) — single canonical instance
        self._registry = build_registry()

        # reset sinks
        self._sinks = []

        # Telegram (optional)
        if self.cfg.TELEGRAM_ENABLED:
            try:
                from telegram.ext import Application, MessageHandler, CommandHandler, filters

                self.tgbot = TelegramBot(self)
                self.tgbot.set_registry(self._registry)

                self._app = Application.builder().token(self.cfg.TELEGRAM_TOKEN).build()
                self._app.bot_data["engine"] = self

                # Handlers bound directly to BotEngine methods
                self._app.add_handler(CommandHandler("start", self._cmd_start))
                self._app.add_handler(
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text)
                )

                log().info("Telegram: application prepared")

                # Register Telegram as a text sink for alerts/heartbeats/etc.
                self._sinks.append("telegram")
            except Exception as e:
                log().exc(e, where="engine.telegram.init")
                log().warn("Telegram disabled due to initialization error")
                self._app = None

        # Console sink (for alerts etc.), independent of whether ConsoleUI is actually run
        if self.cfg.CONSOLE_ENABLED and "console" not in self._sinks:
            self._sinks.append("console")

        # Schedule periodic jobs (always via our own scheduler)
        tick_sec = int(self.cfg.THINKING_SEC or 15)
        self.scheduler.schedule_every(
            "thinkers.tick", tick_sec, self._job_thinkers, run_immediately=True
        )
        self.scheduler.schedule_every(
            "heartbeat", 3600, self._job_heartbeat, run_immediately=False
        )

    def set_registry(self, registry: "CommandRegistry"):
        self._registry = registry

    # ---------- logging plumbing ----------
    def _handle_log_stream_write(self, data: str):
        if not data:
            return
        with self._log_stream_lock:
            self._log_stream_buffer += data
            while True:
                idx = self._log_stream_buffer.find("\n")
                if idx == -1:
                    break
                line = self._log_stream_buffer[:idx]
                self._log_stream_buffer = self._log_stream_buffer[idx + 1 :]
                self._emit_log_line(line)

    def _flush_log_stream(self):
        with self._log_stream_lock:
            if not self._log_stream_buffer:
                return
            line = self._log_stream_buffer
            self._log_stream_buffer = ""
            self._emit_log_line(line)

    def _emit_log_line(self, line: str):
        text = line or ""
        ui = getattr(self, "_console_ui", None)
        if ui is not None:
            try:
                ui.append_output(text)
                return
            except Exception:
                pass
        with self._console_lock:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()

    def _render_console_line(self, text: str, style: str) -> str:
        if not text:
            return ""
        with self._rich_console.capture() as capture:
            self._rich_console.print(text, style=style, highlight=False, markup=False)
        return capture.get().rstrip("\n")

    def _echo_console_command(self, text: str) -> None:
        if not text:
            return
        style = CONSOLE_THEME.get("command_echo", "bold cyan")
        rendered = self._render_console_line(text, style)
        if rendered:
            self._send_text_console(rendered)

    # ---------- text sinks ----------
    def _send_text_console(self, text: str) -> None:
        """Basic console sink for alerts/heartbeats."""
        if self._stopping:
            return
        ui = getattr(self, "_console_ui", None)
        if ui is not None:
            ui.append_output(text)
            return
        with self._console_lock:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()

    def _send_text_telegram(self, text: str) -> None:
        """Telegram sink for alerts/heartbeats."""
        if self._stopping: return
        loop = self._telegram_loop
        coro = self._app.bot.send_message(
            chat_id=int(self.cfg.TELEGRAM_CHAT_ID), text=text
        )
        asyncio.run_coroutine_threadsafe(coro, loop)

    # ---------- interface ----------
    def send_text(self, text: str) -> None:
        """
        Fan-out text to all configured sinks (console, Telegram, etc.).
        Used for alerts, heartbeats, thinker messages, etc.
        """
        if self._stopping: return
        self._render_co(text)

    # def send_photo(self, path: str, caption: str | None = None):
    #     """
    #     Send a photo through Telegram if enabled, else open locally.
    #     If both console and Telegram are enabled, Telegram is preferred for images.
    #     """
    #     if not os.path.exists(path):
    #         log().warn("engine.send_photo.missing", path=path)
    #         return
    #
    #     if self.cfg.TELEGRAM_ENABLED and self._app is not None:
    #         try:
    #             with open(path, "rb") as f:
    #                 self._app.create_task(
    #                     self._app.bot.send_photo(
    #                         chat_id=int(self.cfg.TELEGRAM_CHAT_ID),
    #                         photo=f,
    #                         caption=caption or "",
    #                     )
    #                 )
    #             log().info("photo.sent", path=path, caption=caption)
    #             return
    #         except Exception as e:
    #             log().exc(e, where="engine.send_photo", path=path)
    #
    #     # Fallback: local viewer
    #     log().info("photo.local", path=path)
    #     os.system(f"xdg-open '{path}' >/dev/null 2>&1 &")

    # ---------- photo sinks ----------

    def _send_photo_console(self, path: str, caption: str | None = None) -> None:
        """Open photo with local viewer (console path)."""
        if self._stopping: return
        log().info("photo.local", path=path, caption=caption)
        # best-effort; if it fails, we log the exception in caller
        os.system(f"xdg-open '{path}' >/dev/null 2>&1 &")

    def _send_photo_telegram(self, path: str, caption: str | None = None) -> None:
        """Telegram sink for photos."""
        if self._stopping: return
        loop = self._telegram_loop

        with open(path, "rb") as f:
            data = f.read()

        coro = self._app.bot.send_photo(
            chat_id=int(self.cfg.TELEGRAM_CHAT_ID),
            photo=data,
            caption=caption or "",
        )
        asyncio.run_coroutine_threadsafe(coro, loop)

    def refresh_klines_cache(self):
        """Pull recent klines for all relevant symbols."""
        open_positions = self.store.list_open_positions()
        syms = {
            lg["symbol"]
            for r in open_positions
            for lg in self.store.get_legs(r["position_id"])
            if lg["symbol"]
        }
        syms.update(self.cfg.AUX_SYMBOLS)
        log().debug(f"refresh_klines_cache(): symbols: {syms}")

        for s in syms:
            try:
                self._refresh_symbol(s)
                log().debug("engine.klines.refreshed", symbol=s)
            except Exception as e:
                log().warn(
                    "engine.klines.refresh.fail", symbol=s, err=str(e)
                )
                log().exc(e)

    def _refresh_symbol(self, symbol: str, timeframes=None):
        """Fetch new candles, upsert, then prune."""
        assert not isinstance(timeframes, str)
        timeframes = timeframes or self.cfg.KLINES_TIMEFRAMES

        for tf in timeframes:
            try:
                last_open = self.kc.latest_open_ts(symbol, tf)
                n = self._fetch_and_cache(symbol, tf, since_open_ts=last_open)
                log().debug(
                    "engine.klines.refresh.symbol.timeframe",
                    symbol=symbol,
                    tf=tf,
                    added=n,
                    last_open=last_open,
                )
            except Exception as e:
                log().warn(
                    "engine.klines.refresh.symbol.timeframe.fail",
                    symbol=symbol,
                    tf=tf,
                    err=str(e),
                )

        self.kc.prune_keep_last_n(keep_n=self.cfg.KLINES_CACHE_KEEP_BARS)

    def _fetch_and_cache(
        self, symbol: str, timeframe: str, since_open_ts: Optional[int]
    ) -> int:
        """Pull fresh klines from Binance and store them, safely overwriting the live candle."""
        now_ms = self.api.now_ms()
        start = max(0, since_open_ts - tf_ms(timeframe)) if since_open_ts else None
        klines = self.api.klines(
            symbol, timeframe, start, now_ms, self.cfg.KLINES_CACHE_KEEP_BARS
        )
        return self.kc.ingest_klines(symbol, timeframe, klines, now_ms=now_ms)

    # --------------------------------
    # Thread exception hook
    # --------------------------------
    def _install_thread_excepthook_once(self):
        """Ensure unhandled exceptions in ANY thread are logged via our logger."""
        if self._excepthook_installed:
            return

        def _hook(args):
            try:
                import traceback

                tb = "".join(
                    traceback.format_exception(
                        args.exc_type, args.exc_value, args.exc_traceback
                    )
                )
                log().error(
                    "thread.crash",
                    thread=args.thread.name,
                    exc=str(args.exc_value),
                )
                for line in tb.rstrip("\n").splitlines():
                    log().error(line)
            except Exception:
                sys.stderr.write(
                    f"[thread {args.thread.name}] {args.exc_type}: {args.exc_value}\n"
                )

        threading.excepthook = _hook
        self._excepthook_installed = True

    # --------------------------------
    # Telegram polling thread
    # --------------------------------
    def _telegram_thread_main(self):
        """
        Wrapper for PTB polling to:
          - avoid signal registration in non-main thread,
          - produce deterministic logs on exit/crash.
        """
        try:
            assert self._app is not None
            log().info("Telegram polling thread starting ...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._telegram_loop = loop
            self._app.run_polling(stop_signals=None, close_loop=True)
            log().info("Telegram polling stopped cleanly")
        except Exception as e:
            log().exc(e, where="telegram.thread")
        finally:
            self._telegram_loop = None
            log().warn("Telegram thread exiting")

    # --------------------------------
    # Lifecycle
    # --------------------------------
    def start(self):
        # Build everything lazily
        if self.api is None:
            self._build_parts()

        # Ensure thread exceptions are captured
        self._install_thread_excepthook_once()

        # Start worker
        self._worker_thread = threading.Thread(
            target=self.worker.run_forever, name="rv-worker", daemon=False
        )
        self._worker_thread.start()
        log().info("Worker thread started")

        # Start scheduler
        self.scheduler.start()

        # Telegram polling (in its own thread) if prepared
        if self._app is not None:
            log().info("Telegram polling…")
            self._tg_thread = threading.Thread(
                target=self._telegram_thread_main,
                name="rv-telegram",
                daemon=True,
            )
            self._tg_thread.start()

        else:
            log().info("Engine running without Telegram")

    def stop(self):
        log().info("Engine stopping …")
        # self._stopping = True
        self.request_stop()
        try:
            if self._console_ui and self._console_thread:
                log().info("Console: stopping UI thread")
                self._console_ui.stop()
                self._console_thread.join()
                log().info("Console: thread joined")
                self._console_ui = None
        except Exception as e:
            log().exc(e, where="engine.stop.console")

        try:
            if self.scheduler:
                log().info("Scheduler: stopping")
                self.scheduler.stop()
                log().info("Scheduler: stopped")
        except Exception as e:
            log().exc(e, where="engine.stop.scheduler")

        try:
            if self.worker:
                log().info("Worker: stopping thread")
                self.worker.stop()
                log().info("Worker: stopped")
                if self._worker_thread and self._worker_thread.is_alive():
                    self._worker_thread.join()
                    log().info("Worker: thread joined")
        except Exception as e:
            log().exc(e, where="engine.stop.worker")

        try:
            if self._app:
                log().info("Telegram: stopping application")
                loop = self._telegram_loop
                if loop and loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(self._app.stop(), loop)
                    fut.result()
                    log().info("Telegram: application stopped")
                    log().info("Telling self._telegram_loop to stop now")
                    self._telegram_loop.stop()
                else:
                    log().warn("Telegram loop missing while stopping")

                if self._tg_thread and self._tg_thread.is_alive():
                    log().info("Telegram: waiting for polling thread")
                    self._tg_thread.join()
                    log().info("Telegram: polling thread joined")
        except Exception as e:
            log().exc(e, where="engine.stop.telegram")


        self._print_threads_until_clear()

    def run(self):
        """
        Convenience entrypoint:
          - start engine (worker, scheduler, telegram)
          - if CONSOLE_ENABLED: run ConsoleUI
          - else: idle loop until Ctrl+C
        """
        self.start()
        console_enabled = bool(self.cfg.CONSOLE_ENABLED)

        # simple signal-aware loop
        def _sig_handler(sig, _frm):
            self._send_queue.put(('stop', None))
            raise KeyboardInterrupt()

        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        console_ui = None
        try:
            if console_enabled:
                console_ui = ConsoleUI(self)
                self._console_ui = console_ui
                self._console_thread = threading.Thread(
                    target=console_ui.run, name="rv-console", daemon=True
                )
                self._console_thread.start()
            else:
                log().info(
                    "BotEngine.run: no console; idle loop. Press Ctrl+C to exit."
                )
            while True:
                try:
                    target, payload = self._send_queue.get(timeout=0.5)
                except Empty:
                    continue
                if target == "stop":
                    log().info("Main loop: stop signal received")
                    break
        except KeyboardInterrupt:
            log().info("KeyboardInterrupt: shutting down…")
        finally:
            self.stop()
        log().info("Engine stopped cleanly")

    def _print_threads_until_clear(self, interval: float = 1., max_checks: int = 10):
        for _ in range(max_checks):
            threads = [(t.name, t.ident, t.daemon) for t in threading.enumerate()]
            print("Threads alive:", threads)
            if len(threads) <= 1:
                return
            time.sleep(interval)
        log().warn("Threads still alive after shutdown", threads=threads)

    # ---------- alerts / dispatch ----------
    def emit_alert(self, msg: str):
        """Emit alert to all configured sinks."""
        self.send_text(msg)

    def request_stop(self):
        self._send_queue.put(("stop", None))

    # --------------------------------
    # Scheduled jobs (driven by InternalScheduler)
    # --------------------------------
    def _job_heartbeat(self):
        try:
            txt = "👋 are you ok?\n(heartbeat ping)"
            self.send_text(txt)
        except Exception as e:
            log().exc(e, where="job.heartbeat")

    def _job_thinkers(self):
        try:
            log().debug("Gonna think ...")
            self.refresh_klines_cache()
            n_ok, n_fail = self.tm.run_once()
            log().debug("thinkers.cycle", n_ok=n_ok, n_fail=n_fail)
        except Exception as e:
            log().exc(e, where="job.thinkers")

    # ---- core output processor ----
    def _normalize_command_output(self, result: object) -> "CO":
        from commands import CO, OC
        result = CO(result) if isinstance(result, (str, OC)) else result
        assert isinstance(result, CO)
        return result

    def _render_co(self, result: object, sinks: Optional[List[str]] = None) -> None:
        co = self._normalize_command_output(result)
        targets = list(sinks) if sinks else list(self._sinks)
        if not targets:
            log().info("render_co.no_sinks", text="; ".join(str(c) for c in co.components))
            return
        for sink in targets:
            for comp in co.components:  # type: ignore[attr-defined]
                try:
                    if sink == "telegram":
                        comp.render_telegram(self)
                    elif sink == "console":
                        comp.render_console(self)
                    else:
                        log().warn("render_co.unknown_sink", sink=sink)
                except Exception as e:
                    log().exc(e, where="render_co", sink=sink, component=comp.__class__.__name__)

    def dispatch_command(self, text: str, chat_id: int = 0, origin: str = "console") -> str:
        """Build context and route through the registry, then render per-origin."""
        assert self._registry
        if origin == "console":
            self._echo_console_command(text)
        raw = self._registry.dispatch(self, text)
        self._render_co(raw, sinks=[origin])
        return ""

    # ---------- telegram handlers ----------
    async def _cmd_start(self, update, context):
        await update.message.reply_text("RV bot ready. Try: @help")

    async def _on_text(self, update, context):
        msg = (update.message.text or "").strip()
        chat_id = update.effective_chat.id
        log().debug("engine.on_text", chat_id=chat_id, text=msg)
        try:
            # IMPORTANT: engine will send messages/photos itself based on origin.
            _ = self.dispatch_command(msg, chat_id=chat_id, origin="telegram")
        except Exception as e:
            log().exc(e, where="engine.on_text.dispatch")
            await context.bot.send_message(chat_id=chat_id, text=f"Error: {e}")
