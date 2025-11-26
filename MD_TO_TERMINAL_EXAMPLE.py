from dataclasses import dataclass
from typing import Any, Dict, List

import mistune
from rich.console import Console, RenderableType
from rich.text import Text
from rich.rule import Rule
from rich.syntax import Syntax

# ==========================
# ===== STYLE CONFIG =======
# ==========================

@dataclass
class StyleConfig:
    heading_style: str = "bold underline"
    body_style: str = ""              # e.g. "white"
    bullet_style: str = "bold"
    strong_style: str = "bold"
    emphasis_style: str = "italic"
    inline_code_style: str = "reverse"
    code_block_theme: str = "monokai"
    code_block_default_lexer: str = "bash"  # fallback if no ```lang
    heading_separator: bool = False   # set True to print a Rule() after headings


STYLES = StyleConfig()

# Single global markdown parser -> AST
md_parser = mistune.create_markdown(renderer="ast")


# ==========================
# ===== RENDERER CORE ======
# ==========================

def _render_inlines(nodes: List[Dict[str, Any]], cfg: StyleConfig) -> Text:
    """Render inline nodes (text, strong, emphasis, codespan, breaks) to a single Rich Text."""
    out = Text()
    for node in nodes:
        t = node["type"]

        if t == "text":
            out.append(node["raw"])

        elif t in ("softbreak", "linebreak"):
            out.append("\n")

        elif t == "strong":
            inner = _render_inlines(node.get("children", []), cfg)
            inner.stylize(cfg.strong_style)
            out.append(inner)

        elif t == "emphasis":
            inner = _render_inlines(node.get("children", []), cfg)
            inner.stylize(cfg.emphasis_style)
            out.append(inner)

        elif t == "codespan":
            out.append(Text(node["raw"], style=cfg.inline_code_style))

        else:
            # Fallback: recurse on children, if any
            inner = _render_inlines(node.get("children", []), cfg)
            out.append(inner)

    return out


def _render_block(node: Dict[str, Any], cfg: StyleConfig):
    """Yield one or more Rich renderables for a single block-level node."""
    t = node["type"]

    # ---- headings (left-aligned by default) ----
    if t == "heading":
        txt = _render_inlines(node.get("children", []), cfg)
        if cfg.heading_style:
            txt.stylize(cfg.heading_style)
        yield txt
        if cfg.heading_separator:
            yield Rule()
        else:
            yield Text()  # blank line
        return

    # ---- paragraphs ----
    if t == "paragraph":
        txt = _render_inlines(node.get("children", []), cfg)
        if cfg.body_style:
            txt.stylize(cfg.body_style)
        yield txt
        yield Text()  # blank line
        return

    # ---- bullet / ordered lists ----
    if t == "list":
        for item in node.get("children", []):
            # list_item
            for r in _render_block(item, cfg):
                yield r
        yield Text()
        return

    if t == "list_item":
        # Treat children as a single inline paragraph for now
        bullet = Text("- ", style=cfg.bullet_style)
        inner = _render_inlines(node.get("children", []), cfg)
        bullet.append(inner)
        yield bullet
        return

    # ---- fenced / indented code blocks ----
    if t == "block_code":
        lang = (node.get("info") or "").strip() or cfg.code_block_default_lexer
        code_text = node.get("text", "")
        yield Syntax(code_text.rstrip("\n"), lang, theme=cfg.code_block_theme, line_numbers=False)
        yield Text()
        return

    # ---- horizontal rule ----
    if t == "thematic_break":
        yield Rule()
        return

    # ---- anything else: recurse into children if present ----
    for child in node.get("children", []):
        for r in _render_block(child, cfg):
            yield r


def render_markdown_to_rich(md_text: str, cfg: StyleConfig = STYLES) -> List[RenderableType]:
    """Parse markdown with mistune AST and convert to a list of Rich renderables."""
    ast = md_parser(md_text)
    renderables: List[RenderableType] = []
    for node in ast:
        for r in _render_block(node, cfg):
            renderables.append(r)
    return renderables


# ==========================
# ===== SAMPLE USAGE =======
# ==========================

SAMPLE = """
Replace sample with sth that does not pollute Ctrl+Shift+F"""

if __name__ == "__main__":
    console = Console()
    for item in render_markdown_to_rich(SAMPLE, STYLES):
        console.print(item)
