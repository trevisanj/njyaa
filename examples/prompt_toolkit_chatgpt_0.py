#!/usr/bin/env python3
from pathlib import Path

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.styles import Style


# ==============================
# Config
# ==============================

# Change this to the file you want to open:
FILE_PATH = Path("../AGENTS.md")


def load_file_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"[ERROR] File not found: {path}\n"
    except OSError as e:
        return f"[ERROR] Could not read {path}: {e}\n"


def build_app() -> Application:
    text = load_file_text(FILE_PATH)

    text_area = TextArea(
        text=text,
        read_only=True,
        focusable=True,
        scrollbar=True,         # draw scrollbar
        wrap_lines=False,       # horizontal scroll when needed
    )

    kb = KeyBindings()

    @kb.add("q")
    @kb.add("c-c")
    def _(event) -> None:
        "Quit app."
        event.app.exit()

    # Optional: vim-ish scrolling shortcuts
    @kb.add("j")
    def _(event) -> None:
        event.current_buffer.cursor_down()

    @kb.add("k")
    def _(event) -> None:
        event.current_buffer.cursor_up()

    style = Style.from_dict(
        {
            "textarea": "bg:#202020 #dddddd",
            "scrollbar.background": "bg:#404040",
            "scrollbar.button": "bg:#aaaaaa",
        }
    )

    return Application(
        layout=Layout(text_area),
        key_bindings=kb,
        mouse_support=True,     # enables mouse, including wheel + scrollbar clicks
        full_screen=True,
        style=style,
    )


def main() -> None:
    app = build_app()
    app.run()


if __name__ == "__main__":
    main()
