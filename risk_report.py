from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from common import Clock, pct_of, leg_pnl
import math


@dataclass
class RiskThresholds:
    warn_exposure_ratio: float = 1.0
    warn_loss_mult: float = 1.0


default_thresholds = RiskThresholds()


@dataclass
class RiskAlert:
    kind: str
    message: str
    position_id: Optional[int] = None


@dataclass
class PositionRiskReport:
    position_id: int
    pair: str
    risk: Optional[float]
    risk_budget: Optional[float]
    pnl: Optional[float]
    pnl_missing: bool
    notional: Optional[float]
    notional_missing: bool
    r_multiple: Optional[float]
    tp_2r: Optional[float]
    tp_3r: Optional[float]
    alert_tags: List[str]


@dataclass
class RiskReport:
    ref_balance: Optional[float]
    leverage: Optional[float]
    max_exposure: Optional[float]
    total_exposure: float
    exposure_missing: bool
    total_pnl: float
    total_pnl_pct: Optional[float]
    exposure_used_pct: Optional[float]
    available_exposure: Optional[float]
    positions: List[PositionRiskReport]
    alerts: List[RiskAlert]
    ts_ms: int


# ---------- formatting helpers (isolated to avoid deps) ----------

def _fmt_num(x: Any, nd=2) -> str:
    if x is None:
        return "?"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "?"


def _fmt_pct(frac: Any, nd=2, show_sign: bool = False) -> str:
    if frac is None:
        return "?"
    try:
        sign = "+" if show_sign else ""
        return f"{float(frac)*100:{sign}.{nd}f}%"
    except Exception:
        return "?"


def build_risk_report(eng, thresholds: RiskThresholds = default_thresholds) -> RiskReport:
    cfg = eng.store.get_config()
    ref_balance = cfg["reference_balance"]
    leverage = cfg["leverage"]

    rows = eng.store.list_open_positions()
    involved_syms = set()
    for r in rows:
        for lg in eng.store.get_legs(r["position_id"]):
            if lg["symbol"]:
                involved_syms.add(lg["symbol"])
    marks: Dict[str, Optional[float]] = {s: eng.kc.last_cached_price(s) for s in involved_syms}

    positions: List[PositionRiskReport] = []
    total_exposure = 0.0
    exposure_missing = False
    total_pnl = 0.0
    alerts: List[RiskAlert] = []

    for r in rows:
        pid = int(r["position_id"])
        legs = eng.store.get_legs(pid)
        risk_val = float(r["risk"])
        risk_budget = ref_balance * risk_val if ref_balance else None
        pos_pnl = 0.0
        pnl_missing = False
        notional = 0.0
        notional_missing = False

        for lg in legs:
            mk = marks.get(lg["symbol"])
            pnl = leg_pnl(lg["entry_price"], lg["qty"], mk)
            if pnl is None:
                pnl_missing = True
            else:
                pos_pnl += pnl
            if lg["qty"] is None or mk is None:
                notional_missing = True
            else:
                notional += abs(float(lg["qty"])) * float(mk)

        if not notional_missing:
            total_exposure += notional
        else:
            exposure_missing = True
            notional = None

        if not pnl_missing:
            total_pnl += pos_pnl

        r_multiple = (pos_pnl / risk_budget) if risk_budget else None
        tp_2r = 2 * risk_budget if risk_budget is not None else None
        tp_3r = 3 * risk_budget if risk_budget is not None else None

        pos_alerts: List[str] = []
        if risk_budget and not pnl_missing and pos_pnl <= -thresholds.warn_loss_mult * risk_budget:
            pos_alerts.append("⚠️ LOSS")
            alerts.append(
                RiskAlert(
                    kind="loss",
                    position_id=pid,
                    message=(
                        f"#{pid} drawdown {pos_pnl:.2f} <= -{thresholds.warn_loss_mult}R "
                        f"(risk={risk_val:.3f}, budget={risk_budget:.2f})"
                    ),
                )
            )

        positions.append(PositionRiskReport(
            position_id=pid,
            pair=f"{r['num']}/{r['den'] or '-'}",
            risk=risk_val,
            risk_budget=risk_budget,
            pnl=pos_pnl if not pnl_missing else None,
            pnl_missing=pnl_missing,
            notional=notional,
            notional_missing=notional_missing,
            r_multiple=r_multiple,
            tp_2r=tp_2r,
            tp_3r=tp_3r,
            alert_tags=pos_alerts,
        ))

    max_exposure = ref_balance * leverage if ref_balance and leverage else None
    available = (max_exposure - total_exposure) if (max_exposure is not None) else None
    exposure_used = pct_of(total_exposure, max_exposure) if max_exposure else None
    total_pnl_pct = pct_of(total_pnl, ref_balance)

    if max_exposure and total_exposure > thresholds.warn_exposure_ratio * max_exposure:
        alerts.append(
            RiskAlert(
                kind="exposure",
                position_id=None,
                message=(
                    f"⚠️ Exposure {total_exposure:.2f} exceeds "
                    f"{thresholds.warn_exposure_ratio*100:.0f}% of max {max_exposure:.2f}"
                    + (" (incomplete)" if exposure_missing else "")
                ),
            )
        )

    return RiskReport(
        ref_balance=ref_balance,
        leverage=leverage,
        max_exposure=max_exposure,
        total_exposure=total_exposure,
        exposure_missing=exposure_missing,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        exposure_used_pct=exposure_used,
        available_exposure=available,
        positions=positions,
        alerts=alerts,
        ts_ms=Clock.now_utc_ms(),
    )


def format_risk_report(report: RiskReport) -> Dict[str, Any]:
    """Render RiskReport into markdown text and table data."""
    md_lines = [
        "# Risk",
        f"- Balance=${_fmt_num(report.ref_balance,2)} leverage={_fmt_num(report.leverage,2)}",
        f"- Exposure: ${_fmt_num(report.total_exposure,2)} / ${_fmt_num(report.max_exposure,2)} "
        f"(used {_fmt_pct(report.exposure_used_pct)}; available=${_fmt_num(report.available_exposure,2)})"
        + (" (incomplete)" if report.exposure_missing else ""),
        f"- PnL: ${_fmt_num(report.total_pnl,2)} ({_fmt_pct(report.total_pnl_pct, show_sign=True)})",
    ]
    if report.alerts:
        md_lines.append("- Alerts:")
        for a in report.alerts:
            md_lines.append(f"  - {a.message}")

    headers = ["id", "pair", "risk%", "budget$", "notional$", "pnl$", "pnl%", "R", "2R/3R$", "alerts"]
    rows: List[List[Any]] = []
    for p in report.positions:
        alerts_tag = " ".join(p.alert_tags)
        notional = _fmt_num(p.notional, 2)
        if p.notional_missing:
            notional += " (incomplete)"
        pnl_s = _fmt_num(p.pnl, 2)
        if p.pnl_missing:
            pnl_s += " (incomplete)"
        rows.append([
            p.position_id,
            p.pair,
            _fmt_pct(p.risk),
            _fmt_num(p.risk_budget, 2),
            notional,
            pnl_s,
            _fmt_pct(pct_of(p.pnl, report.ref_balance), show_sign=True),
            _fmt_num(p.r_multiple, 2),
            f"{_fmt_num(p.tp_2r,2)}/{_fmt_num(p.tp_3r,2)}",
            alerts_tag,
        ])

    if not rows:
        md_lines.append("- No open positions.")

    return {"markdown": "\n".join(md_lines), "headers": headers, "rows": rows}
