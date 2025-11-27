#!/usr/bin/env python3
# FILE: sstrats.py
from __future__ import annotations
from typing import Dict, Optional, Sequence
from collections import OrderedDict
import numpy as np
import pandas as pd
from common import LAST_TS, log, WINDOW_SIZE, tf_ms, TooFewDataPoints, ts_human, IND_STATES, STATE_TS
from indicators import BaseIndicator, INDICATOR_CLASSES


class StopStrategy:
    """
    Base stop strategy:
      - holds indicators (instantiated once)
      - owns runtime slice (states, cfg)
      - runs indicators, saves their state/history, and returns suggested stop + history rows
    """
    kind: str = "BASE_STRAT"

    def __init__(self, eng, thinker, ctx: dict, pos, name: str, attached_at):
        self.eng = eng
        self.thinker = thinker
        self.ctx = ctx
        self.pos = pos
        self.name = name
        self.timeframe = thinker.get_timeframe()
        self.attached_at = attached_at
        self.inds: Dict[str, BaseIndicator] = {}
        self.ind_states = ctx.setdefault(IND_STATES, OrderedDict())
        self._stopper: Optional["StopperIndicator"] = None
        self._stop_info: Optional[Dict[str, Sequence]] = None
        self.cfg = ctx.setdefault("cfg", {})
        self.history_rows: list[dict] = []

        ctx.setdefault(STATE_TS, -1)

        self.setup()

    ind_map = None

    def on_conf_ind(self):
        """Hook to tweak indicator configs after auto-build."""

    def on_get_default_cfg(self, ind_kind: str) -> dict:
        cls = INDICATOR_CLASSES[ind_kind]
        return cls.default_cfg(position=self.pos)

    def setup(self):
        self._build_indicators()
        self.on_conf_ind()

    def _build_indicators(self):
        items = [(k, v) for k, v in self.ind_map.items()] if isinstance(self.ind_map, dict) else \
                [(k, k) for k in self.ind_map]
        for kind, name in items:
            cfg = self.on_get_default_cfg(kind)
            ind = self.inds[name] = BaseIndicator.from_kind(kind, name, cfg, self)
            if kind == "stopper":
                self._stopper = ind

        assert self._stopper, f"sstrat.init.fail.no_stopper: sstrat_kind={self.kind}"

    def on_run(self, bars: pd.DataFrame):
        raise NotImplementedError

    def get_stop_info(self):
        return self._stopper._stop_info

    def get_window_size(self) -> int:
        if not WINDOW_SIZE in self.ctx:
            self.ctx[WINDOW_SIZE] = max(ind.__class__.window_size(ind.cfg) for ind in self.inds.values())
        return self.ctx[WINDOW_SIZE]

    def run(self):
        last_ts = self.ctx.get(LAST_TS)
        window_size = self.get_window_size()
        tfms = tf_ms(self.timeframe)

        anchor_ts = last_ts or self.attached_at
        start_ts = int(anchor_ts) - (window_size + 1) * tfms

        self.last_bars = bars = self.eng.kc.pair_bars(self.pos.num, self.pos.den, self.timeframe, start_ts)
        n_bars = len(bars)
        log().debug("sstrat.bars", sstrat_kind=self.kind, n_bars=n_bars, window_size=window_size,
                    anchor_ts=ts_human(anchor_ts), position_id=self.pos.id)
        v_ts = bars.index.astype("int64") // 1_000_000

        start_idx = int(v_ts.searchsorted(anchor_ts, side="right")-1)
        if start_idx < window_size-1:
            raise TooFewDataPoints(f"[trail] Not enough klines (window_size={window_size}, start_idx={start_idx}), can't run strategy!")

        if start_idx < n_bars-1:
            self._run_pass(bars[:-1], start_idx, True)

        self._run_pass(bars[-window_size:], window_size-1, False)

        ts_ms = int(v_ts[-1])
        self.ctx[LAST_TS] = ts_ms

        return None

    def _run_pass(self, bars, start_idx: int, save_state: bool):
        self._start_idx = start_idx
        self._end_idx = len(bars)
        self._v_ts = bars.index.astype("int64") // 1_000_000

        self.on_run(bars)

        if save_state:
            ts_ms_str = str(self._v_ts[-1])
            self.ind_states[ts_ms_str] = {name: ind._temp_state for name, ind in self.inds.items()}
            d = self.ind_states
            while len(d) > 2:
                d.pop(next(iter(d)))

        ts_ms = self._v_ts[start_idx:]
        for ind_name, ind in self.inds.items():
            if not ind.outputs:
                continue
            for out_name, arr in ind.outputs.items():
                vals = arr[start_idx:]
                log().debug(f"sstrat.history.write", strat=self.kind, ind=ind_name, out=out_name,
                            position_id=self.pos.id, points=len(vals))
                self.eng.ih.insert_history2(self.thinker._thinker_id, self.pos.id, f"{ind_name}-{out_name}", ts_ms,
                                            vals)

    @classmethod
    def from_kind(cls, kind: str, eng, thinker, ctx: dict, pos, name, attached_at: int):
        if kind not in SSTRAT_CLASSES:
            raise ValueError(f"Unknown strategy kind: {kind}")
        strat_cls = SSTRAT_CLASSES[kind]
        return strat_cls(eng, thinker, ctx, pos, name, attached_at)

    def get_ind_state(self, ind_name: str):
        idx = self._start_idx
        ts_ms = self._v_ts[idx - 1]
        ts_ms_key = str(ts_ms) if idx > 0 else "OUT_OF_BOUNDS"

        states = self.ind_states.get(ts_ms_key)
        if states:
            return states[ind_name]

        log().debug("sstrat.state.miss", ind=ind_name, idx_minus_1=idx-1, ts=ts_human(ts_ms), have_ts=list(self.ind_states.keys()))
        return None


class SSPSAR(StopStrategy):
    """
    Stop strategy: PSAR feeding Stopper (protective, one-sided).
    """
    kind = "SSPSAR"
    ind_map = ["psar", "stopper"]

    def on_run(self, bars: pd.DataFrame):
        psar = self.inds["psar"]
        stopper = self.inds["stopper"]

        psar_out = psar.run(bars)
        _ = stopper.run(bars, psar_out["value"])


class SSATR(StopStrategy):
    """
    ATR-based trail: stop = close - k*ATR (long) / close + k*ATR (short), then ratchet.
    """
    kind = "SSATR"
    ind_map = ["atr", "stopper"]

    def on_conf_ind(self):
        if "atr_period" not in self.cfg:
            self.cfg["atr_period"] = 14
        if "atr_k" not in self.cfg:
            self.cfg["atr_k"] = 2.0
        atr = self.inds["atr"]
        atr.cfg["period"] = int(self.cfg["atr_period"])
        stopper = self.inds["stopper"]
        stopper.cfg["side"] = self.pos.side

    def on_run(self, bars: pd.DataFrame):
        atr_out = self.inds["atr"].run(bars)
        stopper = self.inds["stopper"]
        closes = bars["Close"].values
        atr_vals = atr_out["value"]
        side = self.pos.side
        k = float(self.cfg["atr_k"])
        candidates = np.full(len(closes), np.nan, dtype=float)
        mask = ~np.isnan(atr_vals)
        if side > 0:
            candidates[mask] = closes[mask] - k * atr_vals[mask]
        else:
            candidates[mask] = closes[mask] + k * atr_vals[mask]
        stopper.run(bars, candidates)


SSTRAT_CLASSES = {cls.kind: cls for cls in StopStrategy.__subclasses__()}
