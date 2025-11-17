#!/usr/bin/env python3
# FILE: run_bot.py
from cfg_maker import make_cfg
from common import Log, set_global_logger
from bot_api import BotEngine

def main():
    cfg = make_cfg()
    cfg.TELEGRAM_ENABLED = True
    cfg.CONSOLE_ENABLED = False

    lg = Log(level=cfg.LOG_LEVEL, name="rv", json_mode=False)
    set_global_logger(lg)

    eng = BotEngine(cfg)
    eng.run()

if __name__ == "__main__":
    main()
