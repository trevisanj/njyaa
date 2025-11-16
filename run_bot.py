#!/usr/bin/env python3
# FILE: run_bot_bk02.py

import signal
from zoneinfo import ZoneInfo
import os, sys

from bot_api import log, BotEngine
from cfg_maker import make_cfg

def main():
    # run_bot.py (main)

    cfg = make_cfg()
    eng = BotEngine(cfg)

    # 5) Graceful shutdown hooks
    def _graceful_exit(signum, frame):
        log().info("Signal received, stopping...", sign=signum)
        try:
            eng.stop()
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    # 6) Run (blocks)
    eng.start()

if __name__ == "__main__":
    main()
