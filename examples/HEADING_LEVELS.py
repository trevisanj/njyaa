# ============================================================
# GLOBAL Egyptian underline patterns (L, M, R)
# ============================================================

EGY_UPATS = [
    ("██▓▒░", "◢◣", "░▒▓██"),   # h1
    ("▓░", "▲", "░▓"),         # h2
    ("▄▄", "◤◢", "▄▄"),        # h3
    ("═", "✦", "═"),           # h4
    ("·", "𓈖", "·"),           # h5
]

# EGY_UPATS = [
#     ("◢", "■", "◣"),      # h1 solid pyramid cap
#     ("◤", "▹", "◥"),      # h2 airy directional geometry
#     ("◧", "●", "◨"),      # h3 circle-in-square aesthetic
#     ("⌜", "∙", "⌝"),      # h4 minimalist sand glyphs
#     ("˹", "·", "˺"),       # h5 soft dust brackets
# ]


def gen_uline_for(title: str, level: int) -> str:
    if not title:
        raise ValueError("Title must not be empty")

    try:
        left, mid, right = EGY_UPATS[level]
    except IndexError:
        raise ValueError(f"Invalid level {level}. Must be 0..{len(EGY_UPATS)-1}")

    reps = len(title)
    middle = (mid * reps)[:reps]
    return f"{left}{middle}{right}"


# ============================================================
#
# ============================================================

def fmt_uline(title: str, level: int) -> tuple[str, str]:
    """Formatted underline (returns tuple)"""
    uline = gen_uline_for(title, level)
    return " "*len(EGY_UPATS[level][0]) + title, uline


# ============================================================
# Example / test: print all levels
# ============================================================

if __name__ == "__main__":
    for lvl in range(len(EGY_UPATS)):
        title = f"LEVEL {lvl}"
        t, u = fmt_uline(title, lvl)
        print(t)
        print(u)
        print()  # spacing
