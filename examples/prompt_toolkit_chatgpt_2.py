#!/usr/bin/env python3
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
import os

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_ansi_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path!r}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def make_scrollable_view(text: str) -> Window:
    """
    RETURNS a Window that displays ANSI text and supports scrolling.
    """
    control = FormattedTextControl(
        text=text,
        focusable=True,
        show_cursor=False,
    )
    return Window(
        content=control,
        wrap_lines=False,
        always_hide_cursor=True,
    )

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    text = load_ansi_file("SAMPLE.txt")
    window = make_scrollable_view(text)

    kb = KeyBindings()

    # ---- helpers to avoid duplicated logic ----
    def scroll(dy: int):
        window.vertical_scroll = max(0, window.vertical_scroll + dy)

    def scroll_page(sign: int, app):
        h = app.renderer.output.get_size().rows
        scroll(sign * (h - 2))

    # ---- keyboard ----
    @kb.add("up")
    def _(e): scroll(-1)

    @kb.add("down")
    def _(e): scroll(+1)

    @kb.add("pageup")
    def _(e): scroll_page(-1, e.app)

    @kb.add("pagedown")
    def _(e): scroll_page(+1, e.app)

    @kb.add("home")
    def _(e): window.vertical_scroll = 0

    @kb.add("end")
    def _(e): window.vertical_scroll = 10**9  # overshoot to max

    # ---- mouse ----
    @kb.add("<scroll-up>")
    def _(e): scroll(-3)

    @kb.add("<scroll-down>")
    def _(e): scroll(+3)

    # ---- quit ----
    @kb.add("q")
    @kb.add("c-c")
    def _(e): e.app.exit()

    app = Application(
        layout=Layout(window),
        key_bindings=kb,
        mouse_support=True,
        full_screen=True,
        style=Style.from_dict({"": ""}),
    )

    app.run()


if __name__ == "__main__":
    main()
