#!/usr/bin/env python3
# FILE: indicator_engines.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, Sequence
import numpy as np
import pandas as pd
from common import LAST_TS, log
import engclasses as ec


class BaseIndicator:
    """Minimal base for indicator steppers."""
    kind: str = "BASE"

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.state: Optional[dict] = None
        self.outputs: Optional[Dict[str, np.ndarray]] = None
        self.on_init()

    @classmethod
    def default_cfg(cls, *, position: ec.Position) -> dict:
        return {}

    @classmethod
    def lookback_bars(cls, cfg: dict) -> int:
        return 1

    @classmethod
    def from_kind(cls, kind: str, cfg: dict):
        if kind not in INDICATOR_CLASSES:
            raise ValueError(f"Unknown indicator kind: {kind}")
        return INDICATOR_CLASSES[kind](cfg)

    def on_init(self):
        """Inherit to create new instance variables etc."""
        pass

    def run(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Execute indicator, storing outputs for later inspection."""
        self.outputs = self.on_run(*args, **kwargs)
        return self.outputs

    def on_run(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class PSARIndicator(BaseIndicator):
    """Parabolic SAR indicator with configurable initial trend."""
    kind = "psar"

    @classmethod
    def lookback_bars(cls, cfg: dict) -> int:
        return 5

    @classmethod
    def default_cfg(cls, *, position: ec.Position) -> dict:
        trend = "UP" if position.side > 1 else "DOWN"
        return {"af": 0.02, "max_af": 0.2, "initial_trend": trend}

    def on_run(self, df: pd.DataFrame, start_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        Compute PSAR; returns (new_state, outputs).
        outputs: {"value", "ep", "af", "trend"} aligned to df.index (NaNs where unchanged).
        """
        if df is None or df.empty:
            raise ValueError("bars required")
        if start_idx >= len(df):
            return {"value": np.full(len(df), np.nan)}

        af0 = self.cfg["af"]
        afmax = self.cfg["max_af"]
        init_up = True if str(self.cfg["initial_trend"]).upper() == "UP" else False

        o_arr = df["Open"].values
        h_arr = df["High"].values
        l_arr = df["Low"].values
        c_arr = df["Close"].values

        n_total = len(df)
        val_arr = np.full(n_total, np.nan, dtype=float)

        stopped = False
        stop_val: Optional[float] = None
        stopped_count = int(self.state.get("stopped_count", 0)) if self.state else 0

        if self.state is None:
            if start_idx != 0:
                start_idx = 0
            need = self.lookback_bars(self.cfg)
            if len(df) < need:
                raise ValueError(f"Need at least {need} bars to bootstrap PSAR")
            h0 = float(np.max(h_arr[:need]))
            l0 = float(np.min(l_arr[:need]))
            up = init_up
            ep = h0 if up else l0
            af = af0
            psar = l0 if up else h0
            idx = need
        else:
            psar = float(self.state["psar"])
            ep = float(self.state["ep"])
            af = float(self.state["af"])
            trend = self.state.get("trend")
            stopped = bool(self.state.get("stopped", False)) or trend == "STOPPED"
            up = True if trend == "UP" else False
            stop_val = psar if stopped else None
            stopped_count = int(self.state.get("stopped_count", 0))
            idx = max(start_idx, 0)

        if stopped and stop_val is not None:
            for i in range(idx, n_total):
                val_arr[i] = stop_val
            stopped_count += max(0, n_total - idx)
            return {"value": val_arr}

        for i in range(idx, n_total):
            o, h, l, c = o_arr[i], h_arr[i], l_arr[i], c_arr[i]
            prev_psar = psar
            prev_ep = ep
            prev_up = up

            psar = prev_psar + af * (prev_ep - prev_psar)
            if up:
                if i >= 1:
                    psar = min(psar, l_arr[i - 1])
                if i >= 2:
                    psar = min(psar, l_arr[i - 2])
            else:
                if i >= 1:
                    psar = max(psar, h_arr[i - 1])
                if i >= 2:
                    psar = max(psar, h_arr[i - 2])

            if up:
                if l < psar:
                    stop_val = psar
                    stopped = True
                else:
                    if h > prev_ep:
                        ep = h
                        af = min(af + af0, afmax)
                    else:
                        ep = prev_ep
            else:
                if h > psar:
                    stop_val = psar
                    stopped = True
                else:
                    if l < prev_ep:
                        ep = l
                        af = min(af + af0, afmax)
                    else:
                        ep = prev_ep

            val_arr[i] = psar

            if stopped and stop_val is not None:
                stopped_count += (n_total - i)
                for j in range(i, n_total):
                    val_arr[j] = stop_val
                psar = stop_val
                up = prev_up
                break

        new_state = {"psar": psar, "ep": ep, "af": af, "trend": ("STOPPED" if stopped else ("UP" if up else "DOWN")), "stopped": stopped, "stopped_count": stopped_count}
        outputs = {"value": val_arr}
        self.state = new_state
        return outputs


class ATRIndicator(BaseIndicator):
    """Wilder ATR indicator."""
    kind = "atr"

    @classmethod
    def lookback_bars(cls, cfg: dict) -> int:
        return max(cfg["period"] + 1, 2)

    @classmethod
    def default_cfg(cls, *, position: ec.Position) -> dict:
        return {"period": 14}

    def on_run(self, df: pd.DataFrame, start_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        Compute ATR; returns (new_state, outputs).
        outputs: {"value", "tr"} aligned to df.index (NaNs where unchanged).
        """
        if df is None or df.empty:
            raise ValueError("bars required")
        if start_idx >= len(df):
            return {"value": np.full(len(df), np.nan)}

        period = int(self.cfg["period"])
        need = self.lookback_bars(self.cfg)

        h_arr = df["High"].values
        l_arr = df["Low"].values
        c_arr = df["Close"].values
        o_arr = df["Open"].values

        n_total = len(df)
        val_arr = np.full(n_total, np.nan, dtype=float)
        idx = max(start_idx, 0)

        if self.state is None:
            if len(df) < need:
                raise ValueError(f"Need at least {need} bars to bootstrap ATR(p={period})")
            prev_close = c_arr[0]
            trs: list[float] = []
            for i in range(1, need):
                h = h_arr[i]; l = l_arr[i]; c = c_arr[i]
                tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
                trs.append(tr)
                prev_close = c
            atr = sum(trs) / float(period)
            idx = need
        else:
            atr = float(self.state["atr"])
            prev_close = float(self.state["prev_close"])

        for i in range(idx, n_total):
            h = h_arr[i]; l = l_arr[i]; c = c_arr[i]
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            atr = ((atr * (period - 1)) + tr) / period
            prev_close = c
            val_arr[i] = atr

        new_state = {"atr": atr, "prev_close": prev_close}
        outputs = {"value": val_arr}
        self.state = new_state
        return outputs


class StopperIndicator(BaseIndicator):
    """
    Protective stop ratchet based on upstream proposed levels.
    cfg:
      side: +1 for long, -1 for short
    run inputs:
      bars: DataFrame with Close
      values: np.ndarray of proposed stop levels (same length as bars)
    outputs:
      value: ratcheted stop series
      flag: 1.0 when stop is hit, NaN otherwise
    """
    kind = "stopper"

    @classmethod
    def default_cfg(cls, *, position: ec.Position) -> dict:
        return {"side": position.side}

    def on_init(self):
        self._stop_info = None

    def on_run(self, df: pd.DataFrame, values: np.ndarray, start_idx: int = 0) -> Dict[str, np.ndarray]:
        if df is None or df.empty:
            raise ValueError("bars required")
        side = int(self.cfg.get("side") or 0)
        if side == 0:
            raise ValueError("Stopper side must be +1 or -1")
        n_total = len(df)
        val_arr = np.full(n_total, np.nan, dtype=float)
        flag_arr = np.full(n_total, np.nan, dtype=float)
        stop = None if self.state is None else self.state.get("stop")
        closes = df["Close"].values

        for i in range(start_idx, n_total):
            candidate = values[i] if values is not None and i < len(values) else np.nan
            if not np.isnan(candidate):
                if stop is None:
                    stop = candidate
                else:
                    stop = max(stop, candidate) if side > 0 else min(stop, candidate)
            val_arr[i] = stop if stop is not None else np.nan
            if stop is None:
                continue
            hit = closes[i] <= stop if side > 0 else closes[i] >= stop
            if hit:
                flag_arr[i] = 1.0

        self.state = {"stop": stop}
        # makes it retrievable to the sstrat later
        self._stop_info = {"value": val_arr, "flag": flag_arr}
        return self._stop_info


class StopStrategy:
    """
    Base stop strategy:
      - holds indicators (instantiated once)
      - owns runtime slice (states, cfg)
      - runs indicators, saves their state/history, and returns suggested stop + history rows
    """
    kind: str = "BASE_STRAT"
    def __init__(self, eng, thinker, ctx: dict, pos, name: str):
        self.eng = eng
        self.thinker = thinker
        self.ctx = ctx
        self.pos = pos
        self.name = name
        self.inds: Dict[str, BaseIndicator] = {}
        self._stopper: Optional["StopperIndicator"] = None
        self._stop_info: Optional[Dict[str, Sequence]] = None
        ctx.setdefault("cfg", {})
        ctx.setdefault("states", {})
        self.cfg: dict = ctx["cfg"]
        self.history_rows: list[dict] = []
        self.setup()

    ind_map = None  # subclass can set to list of kinds or dict {kind: name}

    def on_conf_ind(self):
        """Hook to tweak indicator configs after auto-build."""

    def on_get_default_cfg(self, ind_kind: str) -> dict:
        """
        Return default cfg for a given indicator kind.
        Override if strategy needs custom defaults.
        """
        cls = INDICATOR_CLASSES[ind_kind]
        return cls.default_cfg(position=self.pos)

    def setup(self):
        """Convenience wrapper to configure indicators and perform extra setup."""
        self._build_indicators()
        self.on_conf_ind()

    def _build_indicators(self):
        items = [(k, v) for k, v in self.ind_map.items()] if isinstance(self.ind_map, dict) else \
                [(k, k) for k in self.ind_map]
        states = self.ctx["states"]
        for kind, name in items:
            cfg = self.on_get_default_cfg(kind)
            ind = BaseIndicator.from_kind(kind, cfg)
            if name in states:
                ind.state = states[name]
            self.inds[name] = ind

            if kind == "stopper":
                # Grabs last stopper
                self._stopper = ind

        assert self._stopper, f"sstrat.init.fail.no_stopper: sstrat_kind={self.kind}"

    def on_run(self, bars: pd.DataFrame, start_idx: int):
        raise NotImplementedError

    def get_stop_info(self):
        """Return dict with value/flag keys from last run."""
        return self._stopper._stop_info

    def get_lookback_bars(self) -> int:
        """Return minimum bars needed based on contained indicators."""
        if not self.inds:
            raise ValueError("Indicators not initialized for strategy")
        return max(ind.__class__.lookback_bars(ind.cfg) for ind in self.inds.values())

    def run(self, bars: pd.DataFrame):
        """
        Execute strategy: compute start_idx, run indicators/stop logic, persist state/history.
        """
        log().debug("sstrat.run.start", strat=self.kind, position_id=self.pos.id, bars=len(bars))
        last_ts = self.ctx.get(LAST_TS)
        start_idx = 0
        if last_ts is not None:
            dt = pd.to_datetime(int(last_ts), unit="ms")
            start_idx = int(bars.index.searchsorted(dt, side="left"))
            if start_idx >= len(bars):
                start_idx = max(0, len(bars) - 1)
        log().debug("sstrat.run.window", strat=self.kind, position_id=self.pos.id, start_idx=start_idx, last_ts=last_ts)

        self.on_run(bars, start_idx=start_idx)

        self.ctx["states"] = {name: ind.state for name, ind in self.inds.items()}
        if not bars.empty:
            self.ctx[LAST_TS] = int(bars.index[-1].value // 1_000_000)

        # auto-record indicator outputs from start_idx onward
        ts_slice = bars.index[start_idx:]
        ts_ms = (ts_slice.astype("datetime64[ns]").astype("int64") // 1_000_000)
        for ind_name, ind in self.inds.items():
            if not ind.outputs: continue
            for out_name, arr in ind.outputs.items():
                vals = arr[start_idx:]
                log().debug("sstrat.hist.write", strat=self.kind, ind=ind_name, out=out_name,
                            position_id=self.pos.id, points=len(vals))
                self.eng.ih.insert_history2(self.thinker._thinker_id, self.pos.id, f"{ind_name}-{out_name}",
                                            ts_ms, vals,)
        return None

    @classmethod
    def from_kind(cls, kind: str, eng, thinker, ctx: dict, pos, name: str = ""):
        if kind not in SSTRAT_CLASSES:
            raise ValueError(f"Unknown strategy kind: {kind}")
        strat_cls = SSTRAT_CLASSES[kind]
        return strat_cls(eng, thinker, ctx, pos, name)


class SSPSAR(StopStrategy):
    """
    Stop strategy: PSAR feeding Stopper (protective, one-sided).
    """
    kind = "SSPSAR"
    ind_map = ["psar", "stopper"]

    def on_run(self, bars: pd.DataFrame, start_idx: int):
        psar = self.inds["psar"]
        stopper = self.inds["stopper"]

        psar_out = psar.run(bars, start_idx=start_idx)
        _ = stopper.run(bars, psar_out["value"], start_idx=start_idx)



SSTRAT_CLASSES = {cls.kind: cls for cls in StopStrategy.__subclasses__()}


# registry
INDICATOR_CLASSES = {cls.kind: cls for cls in BaseIndicator.__subclasses__()}
