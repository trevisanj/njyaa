#!/usr/bin/env python3
# FILE: run_bot.py
import fcntl
import os
import sys
from pathlib import Path

from cfg_maker import make_cfg
from bot_api import BotEngine

_LOCK_FD = None


def acquire_singleton_lock():
    global _LOCK_FD
    lock_path = Path(__file__).resolve().parent / "njyaa.lock"
    _LOCK_FD = open(lock_path, "a+")
    try:
        fcntl.flock(_LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("Another njyaa instance is already running here.")
    _LOCK_FD.seek(0)
    _LOCK_FD.truncate()
    _LOCK_FD.write(f"{os.getpid()}\n")
    _LOCK_FD.flush()

def main():
    acquire_singleton_lock()
    cfg = make_cfg()
    inject_cmd = ""
    # inject_cmd = "?chart-ind 4 3" # os.getenv("NJYAA_INJECT_CMD")
    # inject_cmd = "!exit-attach 8 4"
    # inject_cmd = "!thinker-enable 12"

    mode = 4

    match mode:
        case 0:  # console only
            cfg.TELEGRAM_ENABLED = False
            cfg.CONSOLE_ENABLED = True

        case 1:  # telegram only
            cfg.TELEGRAM_ENABLED = True
            cfg.CONSOLE_ENABLED = False

        case 2:  # both
            cfg.TELEGRAM_ENABLED = True
            cfg.CONSOLE_ENABLED = True

        case 3:  # none
            cfg.TELEGRAM_ENABLED = False
            cfg.CONSOLE_ENABLED = False

        case 4:
            pass

        case _:
            raise ValueError(f"Invalid mode: {mode}")

    eng = BotEngine(cfg)
    eng.run(inject_command=inject_cmd)

if __name__ == "__main__":
    main()
