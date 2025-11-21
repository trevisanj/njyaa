from __future__ import annotations

import os
from typing import Optional
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.history import FileHistory
from common import log
import sys
from threading import Lock


class _CommandCompleter(Completer):
    """Simple completer listing @/! commands plus :q shortcuts."""

    def __init__(self, console: "ConsoleUI"):
        self.console = console

    def get_completions(self, document, complete_event):
        text = (document.text_before_cursor or "").lstrip()
        if " " in text:
            return
        for word in self.console._command_words():
            if word.startswith(text):
                yield Completion(word, start_position=-len(text))


class ConsoleUI:
    """
    Minimal console that keeps a blocking prompt in its own thread while the
    rest of the program continues printing logs to stdout.
    """

    class _History(FileHistory):
        """Persisted history that skips quit commands and empties."""
        def append_string(self, string: str) -> None:
            s = (string or "").strip()
            if not s or s in (":q", ":quit", ":exit"):
                return
            super().append_string(string)

    def __init__(self, eng: "BotEngine"):
        self.eng = eng
        self._alive = True
        self._stop_notified = False
        self._hist_file = os.path.expanduser("~/.NJYAA_console_history")
        self._history = ConsoleUI._History(self._hist_file)
        self._completer = _CommandCompleter(self)
        self._session = PromptSession(
            completer=self._completer,
            complete_while_typing=False,
            history=self._history,
        )
        self._complete_style = CompleteStyle.COLUMN

        self._print_lock = Lock()

    def _command_words(self):
        words = {":q", ":quit", ":exit"}
        reg = self.eng._registry
        if reg:
            for prefix, name in reg._handlers.keys():
                words.add(f"{prefix}{name}")
        return sorted(words)

    def run(self):
        log().info("Console ready; type @help or :q to exit")
        while self._alive:
            try:
                line = self._session.prompt(
                    "> ",
                    complete_style=self._complete_style,
                ).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line in (":q", ":quit", ":exit"):
                break
            try:
                self.eng.dispatch_command(line, chat_id=0, origin="console")
            except Exception as e:
                log().exc(e, where="console.dispatch")
                print(f"Error: {e}")
        if not self._stop_notified:
            self._stop_notified = True
            self.eng.request_stop()

    def stop(self):
        self._alive = False
        if not self._stop_notified:
            self._stop_notified = True
            self.eng.request_stop()
        app = getattr(self._session, "app", None)
        if app and app.is_running:
            app.exit("")

    def append_output(self, text: str):
        with self._print_lock:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
