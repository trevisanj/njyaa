#!/usr/bin/env python3
import itertools
from rich.console import Console
from rich.table import Table

console = Console()

# Standard + bright color names Rich understands
color_names = [
    "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white",
    "bright_black", "bright_red", "bright_green", "bright_yellow", "bright_blue",
    "bright_magenta", "bright_cyan", "bright_white",
    # Some extras often supported by terminals
    "grey50", "grey60", "grey70", "grey80", "grey82", "grey85", "grey90",
    "silver", "dark_green", "dark_cyan", "dark_magenta", "dark_red",
    "orange1", "orange3", "gold1", "gold3", "deep_sky_blue1", "deep_sky_blue3",
    "turquoise2", "spring_green2", "spring_green3", "chartreuse2", "chartreuse3",
]

# Unique, keep order
seen = set(); colors = []
for c in color_names:
    if c not in seen:
        colors.append(c); seen.add(c)

# Build table
chunks = [colors[i:i+4] for i in range(0, len(colors), 4)]

table = Table(title="Rich color samples", show_lines=False)
table.add_column("Color", justify="center")
table.add_column("Sample", justify="center")
table.add_column("Color", justify="center")
table.add_column("Sample", justify="center")

for row in chunks:
    cells = []
    for c in row:
        cells.append(c)
        cells.append(f"[{c}]████ {c} ████[/]")
    # pad row to 4 colors
    while len(cells) < 8:
        cells.append("")
    table.add_row(*cells)

console.print(table)

print("\nRGB 0-255 quick sampler (every 32 steps):")
rgb_table = Table(show_header=True, header_style="bold", box=None)
rgb_table.add_column("R"); rgb_table.add_column("G"); rgb_table.add_column("B"); rgb_table.add_column("Sample")
for r in range(0, 256, 64):
    for g in range(0, 256, 64):
        for b in range(0, 256, 64):
            rgb_table.add_row(str(r), str(g), str(b), f"[rgb({r},{g},{b})]████ rgb({r},{g},{b}) ████[/]")
console.print(rgb_table)
