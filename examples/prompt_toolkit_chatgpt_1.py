#!/usr/bin/env python3
from pathlib import Path

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window, HSplit, VSplit
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.styles import Style


# ====================================================
# CONFIG
# ====================================================
FILE_PATH = Path("STUPID_TEXT.txt")
SCROLLBAR_HEIGHT = 20   # symbolic scrollbar height


# ====================================================
# LOAD + PARSE ANSI -> REAL FRAGMENTS
# ====================================================
def load_ansi_fragments(path: Path):
    try:
        raw = path.read_text(encoding='utf-8')
    except Exception as e:
        raw = f"[ERROR] {e}"

    lines = raw.splitlines()

    parsed = []
    for line in lines:
        # FIRST convert ANSI(...) -> ANSI object
        a = ANSI(line)
        # THEN convert ANSI object -> fragment list
        flist = to_formatted_text(a)
        parsed.append(flist)

    return parsed


LINES = load_ansi_fragments(FILE_PATH)
TOP = 0


# ====================================================
# CONTENT
# ====================================================
def get_fragments():
    out = []
    for frag_list in LINES[TOP:]:
        out.extend(frag_list)
        out.append(("", "\n"))
    return out


content_control = FormattedTextControl(
    get_fragments,
    focusable=True,
    show_cursor=False,
)


# ====================================================
# MOUSE HANDLER
# ====================================================
def mouse_handler(event):
    global TOP
    if event.event_type == MouseEventType.SCROLL_UP:
        TOP = max(0, TOP - 3)
    elif event.event_type == MouseEventType.SCROLL_DOWN:
        TOP = min(len(LINES) - 1, TOP + 3)


# ====================================================
# SYMBOLIC SCROLLBAR
# ====================================================
def scrollbar_fragments():
    total = max(len(LINES), 1)
    h = SCROLLBAR_HEIGHT

    pos = TOP / (total - 1) if total > 1 else 0
    thumb = int(pos * (h - 1))

    frags = []
    for i in range(h):
        if i == thumb:
            frags.append(("class:sb.fg", " "))
        else:
            frags.append(("class:sb.bg", " "))
    return frags


scrollbar_control = FormattedTextControl(scrollbar_fragments)


# ====================================================
# KEY BINDINGS
# ====================================================
kb = KeyBindings()

@kb.add("q")
@kb.add("c-c")
def _(event):
    event.app.exit()

@kb.add("down")
@kb.add("j")
def _(event):
    global TOP
    TOP = min(len(LINES) - 1, TOP + 1)
    event.app.invalidate()

@kb.add("up")
@kb.add("k")
def _(event):
    global TOP
    TOP = max(0, TOP - 1)
    event.app.invalidate()

@kb.add("pagedown")
def _(event):
    global TOP
    TOP = min(len(LINES) - 1, TOP + 30)
    event.app.invalidate()

@kb.add("pageup")
def _(event):
    global TOP
    TOP = max(0, TOP - 30)
    event.app.invalidate()


# ====================================================
# APP
# ====================================================
def build_app():
    content_window = Window(
        content=content_control,
        always_hide_cursor=True,
    )
    content_window.mouse_handler = mouse_handler

    scrollbar_window = Window(
        content=scrollbar_control,
        width=1,
        dont_extend_width=True,
    )

    header = Window(
        height=1,
        dont_extend_height=True,
        content=FormattedTextControl([("bold", f" {FILE_PATH}   (q quits)")]),
    )

    root = HSplit([
        header,
        VSplit([content_window, scrollbar_window])
    ])

    return Application(
        layout=Layout(root),
        key_bindings=kb,
        mouse_support=True,
        full_screen=True,
        style=Style.from_dict({
            "sb.bg": "bg:#444444",
            "sb.fg": "bg:#aaaaaa",
        }),
    )


def main():
    build_app().run()


if __name__ == "__main__":
    main()
