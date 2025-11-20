from __future__ import annotations

import os
from collections import deque
from typing import Optional

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.filters import has_focus
from prompt_toolkit.formatted_text import ANSI, merge_formatted_text
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, VSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.containers import FloatContainer, Float
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.layout.menus import CompletionsMenu

from common import log


CONSOLE_THEME = {
    "command_echo": "bold deep_sky_blue1",
}


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
        self._follow_log = True
        self._log_control = FormattedTextControl(
            self._render_log, focusable=True, show_cursor=False
        )
        self._log_window = Window(
            content=self._log_control,
            wrap_lines=True,
            height=Dimension(weight=1),
            allow_scroll_beyond_bottom=True,
            always_hide_cursor=False,
            right_margins=[ScrollbarMargin(display_arrows=True)],
        )
        self._input_control = BufferControl(
            buffer=self._input_buffer, focus_on_click=True
        )
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

    def _render_status_bar(self):
        count = len(self._log_lines)
        return f" Logs: {count} lines "

    def _is_log_at_bottom(self) -> bool:
        info = self._log_window.render_info
        if not info:
            return True
        max_scroll = max(0, info.content_height - info.window_height)
        return self._log_window.vertical_scroll >= max_scroll

    def _scroll_log(self, delta: int):
        self._follow_log = False
        if delta > 0:
            for _ in range(delta):
                self._log_window._scroll_down()
        else:
            for _ in range(-delta):
                self._log_window._scroll_up()
        if self._app:
            self._app.invalidate()

    def _scroll_log_home(self):
        self._follow_log = False
        self._log_window.vertical_scroll = 0
        if self._app:
            self._app.invalidate()

    def _scroll_log_bottom(self):
        self._follow_log = True
        self._log_window.vertical_scroll = 10**9
        if self._app:
            # inva
            self._app.invalidate()

    def _build_application(self) -> Application:
        kb = KeyBindings()

        @kb.add("c-c")
        @kb.add("c-d")
        def _(event):
            self._on_exit_requested()

        @kb.add("f6")
        def _(event):
            event.app.layout.focus(self._log_control)

        @kb.add("f7")
        def _(event):
            event.app.layout.focus(self._input_control)

        @kb.add("up", filter=has_focus(self._log_control))
        def _(event):
            self._scroll_log(-1)

        @kb.add("down", filter=has_focus(self._log_control))
        def _(event):
            self._scroll_log(+1)

        @kb.add("pageup", filter=has_focus(self._log_control))
        def _(event):
            self._scroll_log(-10)

        @kb.add("pagedown", filter=has_focus(self._log_control))
        def _(event):
            self._scroll_log(+10)

        @kb.add("home", filter=has_focus(self._log_control))
        def _(event):
            self._scroll_log_home()

        @kb.add("end", filter=has_focus(self._log_control))
        def _(event):
            self._scroll_log_bottom()

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

        status_bar = Window(
            content=FormattedTextControl(self._render_status_bar, focusable=False),
            height=1,
            always_hide_cursor=True,
            style="class:log.status",
        )

        body = HSplit(
            [
                status_bar,
                self._log_window,
                Window(height=1, char="─"),
                prompt_row,
            ]
        )

        root = FloatContainer(
            content=body,
            floats=[
                Float(
                    content=CompletionsMenu(max_height=8, scroll_offset=1),
                    xcursor=True,
                    ycursor=True,
                )
            ],
        )

        layout = Layout(root, focused_element=self._input_control)
        return Application(
            layout=layout,
            key_bindings=kb,
            full_screen=True,
            mouse_support=True,
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
        auto_follow = self._follow_log and self._is_log_at_bottom()
        lines = text.split("\n")
        if not lines:
            lines = [""]
        for line in lines:
            self._log_lines.append(line)
        if auto_follow:
            self._scroll_log_bottom()
        if self._app:
            try:
                self._app.invalidate()
            except Exception:
                pass

    def run(self):
        self.append_output(
            "RV console ready. Type @help, or :quit to exit. (TAB completes; ↑↓ recall history)"
        )
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
