from __future__ import annotations

import curses
import curses.textpad
import threading
from typing import List, Dict, Any
from common import log


_COLOR_MAP = {
    "black": curses.COLOR_BLACK,
    "red": curses.COLOR_RED,
    "green": curses.COLOR_GREEN,
    "yellow": curses.COLOR_YELLOW,
    "blue": curses.COLOR_BLUE,
    "magenta": curses.COLOR_MAGENTA,
    "cyan": curses.COLOR_CYAN,
    "white": curses.COLOR_WHITE,
    "grey": curses.COLOR_BLACK,
    "gray": curses.COLOR_BLACK,
    "grey15": curses.COLOR_BLACK,
}


def _color_to_curses(name: str, default: int) -> int:
    if not name:
        return default
    return _COLOR_MAP.get(name.lower(), default)


class CursesConsoleUI:
    """
    Minimal curses-based console with scrollable log and prompt.
    Layout:
      - Log pad filling the top area (scrollable with arrows/pgup/pgdn/mouse wheel).
      - Bottom strip of 3 lines sharing the prompt background color, with the prompt on the middle line.
    """
    def __init__(self, eng: "BotEngine"):
        self.eng = eng
        self._alive = True
        self._stop_notified = False
        self._lines: List[str] = []
        self._scroll = 0  # 0 = bottom
        self._input = ""
        self._lock = threading.Lock()
        # local defaults; keep env-free to avoid config bloat
        self._prompt_text = "> "
        self._prompt_bg = _color_to_curses("grey15", curses.COLOR_BLACK)
        self._prompt_fg = _color_to_curses("white", curses.COLOR_WHITE)
        self._mouse_enabled = True
        self._pair_cache: Dict[tuple, int] = {}
        self._next_pair = 10  # leave low IDs for prompt/default
        import os
        self._hist_file = os.path.expanduser("~/.NJYAA_console_history")
        self._history: List[str] = []
        self._hist_idx: int = -1
        self._load_history()

    # ---------- public API ----------
    def run(self):
        curses.wrapper(self._run)

    def stop(self):
        self._alive = False
        if not self._stop_notified:
            self._stop_notified = True
            self.eng.request_stop()
        self._save_history()

    def append_output(self, text: str):
        with self._lock:
            for ln in (text or "").splitlines():
                self._lines.append(ln)
            if self._scroll == 0:
                self._scroll = 0  # stick to bottom

    # ---------- internals ----------
    def _run(self, stdscr):
        curses.curs_set(1)
        curses.start_color()
        curses.mousemask(curses.ALL_MOUSE_EVENTS)
        curses.use_default_colors()
        prompt_pair = 1
        curses.init_pair(prompt_pair, self._prompt_fg, self._prompt_bg)
        stdscr.nodelay(False)
        stdscr.keypad(True)

        while self._alive:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            prompt_height = 4
            log_height = max(1, h - prompt_height)
            log_width = max(1, w - 2)  # leave rightmost col for scrollbar

            # draw log area
            with self._lock:
                lines = list(self._lines)
            max_scroll = max(0, len(lines) - log_height)
            self._scroll = max(0, min(self._scroll, max_scroll))
            start = max(0, len(lines) - log_height - self._scroll)
            visible = lines[start:start + log_height]
            for i, ln in enumerate(visible):
                self._render_ansi_line(stdscr, i, ln, log_width)

            # draw scrollbar
            bar_col = w - 1
            if len(lines) > log_height:
                bar_height = max(1, int(log_height * log_height / len(lines)))
                top_frac = (len(lines) - log_height - self._scroll) / max(1, len(lines) - log_height)
                bar_start = int((log_height - bar_height) * top_frac)
                for i in range(log_height):
                    ch = " "
                    attr = curses.A_REVERSE if bar_start <= i < bar_start + bar_height else curses.A_DIM
                    try:
                        stdscr.addch(i, bar_col, ch, attr)
                    except Exception:
                        pass
            else:
                for i in range(log_height):
                    try:
                        stdscr.addch(i, bar_col, " ", curses.A_DIM)
                    except Exception:
                        pass

            # draw prompt area
            base_row = h - prompt_height
            for i in range(prompt_height):
                stdscr.addstr(base_row + i, 0, " " * (w - 1), curses.color_pair(prompt_pair))
            prompt_row = base_row + 1
            prompt_str = f"{self._prompt_text}{self._input}"
            try:
                stdscr.addnstr(prompt_row, 0, prompt_str, w - 1, curses.color_pair(prompt_pair))
            except Exception:
                pass
            cursor_x = min(len(prompt_str), w - 2)
            stdscr.move(prompt_row, cursor_x)

            # status/cheatsheet line (last line)
            cheatsheet = "F2: toggle mouse (select) | PgUp/PgDn: scroll | Arrows: history | Wheel: scroll"
            status = f"[mouse {'on' if self._mouse_enabled else 'off (select)'}]"
            status_line = f"{cheatsheet}  {status}"
            try:
                stdscr.addnstr(base_row + prompt_height - 1, 0, status_line, w - 1, curses.color_pair(prompt_pair) | curses.A_BOLD)
            except Exception:
                pass
            # ensure cursor sits on the prompt, not on the status line
            stdscr.move(prompt_row, cursor_x)

            stdscr.refresh()

            ch = stdscr.getch()
            if ch == curses.KEY_RESIZE:
                continue
            if ch == curses.KEY_F2:
                self._mouse_enabled = not self._mouse_enabled
                curses.mousemask(curses.ALL_MOUSE_EVENTS if self._mouse_enabled else 0)
                self.append_output(f"[mouse {'on' if self._mouse_enabled else 'off'} — selection {'disabled' if self._mouse_enabled else 'enabled'}]")
                continue
            if ch == curses.KEY_NPAGE:  # PgDn
                self._scroll = max(0, self._scroll - log_height // 2)
                continue
            if ch == curses.KEY_PPAGE:  # PgUp
                self._scroll = min(max_scroll, self._scroll + log_height // 2)
                continue
            if ch == curses.KEY_MOUSE:
                if not self._mouse_enabled:
                    continue
                try:
                    mx, my, _, _, bstate = curses.getmouse()
                    if bstate & curses.BUTTON4_PRESSED:  # wheel up
                        self._scroll = min(max_scroll, self._scroll + 3)
                    elif bstate & curses.BUTTON5_PRESSED:  # wheel down
                        self._scroll = max(0, self._scroll - 3)
                    elif mx == w - 1 and len(lines) > log_height:
                        # click on scrollbar sets scroll proportional to click position
                        click_ratio = my / max(1, log_height - 1)
                        self._scroll = int((len(lines) - log_height) * click_ratio)
                        self._scroll = max(0, min(self._scroll, max_scroll))
                except Exception:
                    pass
                continue
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                if self._input:
                    self._input = self._input[:-1]
                continue
            if ch in (10, 13):  # Enter
                line = self._input.strip()
                self._input = ""
                self._hist_idx = -1
                if not line:
                    continue
                if line in (":q", ":quit", ":exit"):
                    self.stop()
                    break
                if not self._history or self._history[-1] != line:
                    self._history.append(line)
                    self._save_history()
                try:
                    self.eng.dispatch_command(line, chat_id=0, origin="console")
                except Exception as e:
                    log().exc(e, where="console_curses.dispatch")
                    self.append_output(f"Error: {e}")
                continue
            if ch == 9:  # Tab ignore
                continue
            if ch == curses.KEY_UP:
                if self._history:
                    if self._hist_idx == -1:
                        self._hist_idx = len(self._history) - 1
                    elif self._hist_idx > 0:
                        self._hist_idx -= 1
                    self._input = self._history[self._hist_idx]
                continue
            if ch == curses.KEY_DOWN:
                if self._history:
                    if self._hist_idx == -1:
                        pass
                    elif self._hist_idx < len(self._history) - 1:
                        self._hist_idx += 1
                        self._input = self._history[self._hist_idx]
                    else:
                        self._hist_idx = -1
                        self._input = ""
                continue
            if ch in (3,):  # Ctrl+C
                self.stop()
                break
            if ch < 0 or ch > 255:
                continue
            self._input += chr(ch)

    def _render_ansi_line(self, win, y: int, text: str, max_width: int):
        """
        Minimal ANSI SGR renderer (bold + 8-color fg/bg) into the curses window.
        """
        fg = curses.COLOR_WHITE
        bg = -1
        bold = False
        x = 0
        i = 0
        while i < len(text) and x < max_width:
            ch = text[i]
            if ch == "\x1b":
                if text.startswith("\x1b[", i):
                    end = text.find("m", i)
                    if end == -1:
                        break
                    codes = text[i + 2:end].split(";")
                    if not codes or codes == [""]:
                        codes = ["0"]
                    for code in codes:
                        try:
                            c = int(code or "0")
                        except Exception:
                            continue
                        if c == 0:
                            fg, bg, bold = curses.COLOR_WHITE, -1, False
                        elif c == 1:
                            bold = True
                        elif 30 <= c <= 37:
                            fg = _COLOR_MAP_BASE[c - 30]
                        elif 90 <= c <= 97:
                            fg = _COLOR_MAP_BASE[c - 90]
                        elif 40 <= c <= 47:
                            bg = _COLOR_MAP_BASE[c - 40]
                        elif 100 <= c <= 107:
                            bg = _COLOR_MAP_BASE[c - 100]
                    i = end + 1
                    continue
            attr = self._attr_for(fg, bg, bold)
            try:
                win.addch(y, x, ch, attr)
            except Exception:
                pass
            x += 1
            i += 1

    def _attr_for(self, fg: int, bg: int, bold: bool) -> int:
        key = (fg, bg)
        pair_id = self._pair_cache.get(key)
        if pair_id is None:
            pair_id = self._next_pair
            self._next_pair += 1
            try:
                curses.init_pair(pair_id, fg, bg)
            except Exception:
                pair_id = 0
            self._pair_cache[key] = pair_id
        attr = curses.color_pair(pair_id)
        if bold:
            attr |= curses.A_BOLD
        return attr

    # ---------- history persistence ----------
    def _load_history(self):
        try:
            with open(self._hist_file, "r", encoding="utf-8") as f:
                for ln in f:
                    s = ln.rstrip("\n")
                    if s and s not in (":q", ":quit", ":exit"):
                        self._history.append(s)
        except Exception:
            self._history = []
        self._hist_idx = -1

    def _save_history(self):
        try:
            if not self._history:
                return
            with open(self._hist_file, "w", encoding="utf-8") as f:
                for ln in self._history[-1000:]:  # cap
                    f.write(ln + "\n")
        except Exception:
            pass


# base color lookup for SGR 30-37 / 40-47 / 90-97 / 100-107
_COLOR_MAP_BASE = [
    curses.COLOR_BLACK,
    curses.COLOR_RED,
    curses.COLOR_GREEN,
    curses.COLOR_YELLOW,
    curses.COLOR_BLUE,
    curses.COLOR_MAGENTA,
    curses.COLOR_CYAN,
    curses.COLOR_WHITE,
]
