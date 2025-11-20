#!/usr/bin/env python3
# FILE: run_bot.py
from cfg_maker import make_cfg
from bot_api import BotEngine

def main():
    cfg = make_cfg()

    mode = 2  # 0..3

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

        case _:
            raise ValueError(f"Invalid mode: {mode}")

    eng = BotEngine(cfg)
    eng.run()

if __name__ == "__main__":
    main()
