#!/usr/bin/env python3
# FILE: stop_report.py
from __future__ import annotations
import math
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from common import Clock, tf_ms, ts_human, PP_CTX, TRAILING_SNAPSHOT
from risk_report import _fmt_num, _fmt_pct


@dataclass
class StopRow:
    thinker_id: int
    position_id: int
    pair: str
    side: int
    sstrat_kind: str
    timeframe: str
    price: float
    stop: float
    gap_pct: float
    hit: bool
    last_stop_ts: int
    stale: bool
    delta_pct_last_n: Optional[float]
    hits_last_n: int
    last_hit_ts: Optional[int]
    attached_at: int
    last_alert_ts: Optional[int]


@dataclass
class StopReport:
    rows: List[StopRow]
    generated_ts: int
    window_n: int
    freshness_k: float


def _clean_series(ts_seq: List[int], val_seq: List[Any]) -> List[tuple[int, float]]:
    items: List[tuple[int, float]] = []
    for ts, val in zip(ts_seq, val_seq):
        if val is None:
            continue
        fval = float(val)
        if math.isnan(fval):
            continue
        items.append((int(ts), fval))
    return items


def _history_stats(eng, thinker_id: int, position_id: int, window_n: int) -> dict:
    vals = eng.ih.last_n(thinker_id, position_id, "stopper-value", n=window_n, asc=True)
    flags = eng.ih.last_n(thinker_id, position_id, "stopper-flag", n=window_n, asc=True)

    clean_vals = _clean_series(vals["open_ts"], vals["value"]) if vals else []
    clean_flags = _clean_series(flags["open_ts"], flags["value"]) if flags else []

    delta_pct_last_n: Optional[float] = None
    last_stop_ts: Optional[int] = None
    if clean_vals:
        first_ts, first_stop = clean_vals[0]
        last_ts, last_stop = clean_vals[-1]
        assert first_stop != 0.0, "first stop is zero"
        delta_pct_last_n = (last_stop - first_stop) * 100 / first_stop
        last_stop_ts = last_ts

    hits_last_n = 0
    last_hit_ts: Optional[int] = None
    for ts, val in clean_flags:
        if val >= 1.0:
            hits_last_n += 1
            last_hit_ts = ts

    return {
        "delta_pct_last_n": delta_pct_last_n,
        "hits_last_n": hits_last_n,
        "last_hit_ts": last_hit_ts,
        "last_stop_ts": last_stop_ts,
    }


def build_stop_report(eng, window_n: int = 50, freshness_k: float = 2.0, thinker_id: Optional[int] = None) -> StopReport:
    now_ms = Clock.now_utc_ms()
    rows: List[StopRow] = []

    thinkers = eng.store.list_thinkers()
    tid_list = [int(thinker_id)] if thinker_id is not None else [
        int(r["id"]) for r in thinkers if r["kind"] == "TRAILING_STOP" and int(r["enabled"]) == 1
    ]
    if not tid_list:
        raise ValueError("No TRAILING_STOP thinkers found")

    for tid in tid_list:
        inst = eng.tm.get_in_carbonite(tid, expected_kind="TRAILING_STOP")
        rt = inst.runtime()
        pp_ctx = rt[PP_CTX]
        tf_cfg = inst._cfg["timeframe"]
        tf_ms_val = tf_ms(tf_cfg)
        freshness_ms = int(freshness_k * tf_ms_val)
        for pid_str, ctx in pp_ctx.items():
            if "invalid" in ctx and ctx["invalid"]:
                continue
            pid = int(pid_str)
            pos = eng.store.get_position(pid)
            if not pos:
                raise ValueError(f"Position {pid} not found for thinker {tid}")

            snapshot = ctx[TRAILING_SNAPSHOT]
            stop = float(snapshot["stop"])
            price = float(snapshot["price"])
            gap_pct = float(snapshot["gap_pct"])
            hit = bool(snapshot["hit"])
            snap_ts = int(snapshot["ts_ms"])
            sstrat_kind = snapshot["sstrat_kind"]
            timeframe = snapshot["timeframe"]
            last_alert_ts = snapshot["last_alert_ts"]
            attached_at = int(snapshot["attached_at"])

            stats = _history_stats(eng, tid, pid, window_n)
            last_stop_ts = stats["last_stop_ts"] or snap_ts
            stale = (now_ms - last_stop_ts) > freshness_ms

            row = StopRow(
                thinker_id=tid,
                position_id=pid,
                pair=pos.get_pair(),
                side=pos.side,
                sstrat_kind=sstrat_kind,
                timeframe=timeframe,
                price=price,
                stop=stop,
                gap_pct=gap_pct,
                hit=hit,
                last_stop_ts=last_stop_ts,
                stale=stale,
                delta_pct_last_n=stats["delta_pct_last_n"],
                hits_last_n=stats["hits_last_n"],
                last_hit_ts=stats["last_hit_ts"],
                attached_at=attached_at,
                last_alert_ts=last_alert_ts,
            )
            rows.append(row)

    return StopReport(rows=rows, generated_ts=now_ms, window_n=window_n, freshness_k=freshness_k)


def format_stop_report_md(report: StopReport) -> Dict[str, Any]:
    md_lines = [
        "# Stops",
        f"- generated: {ts_human(report.generated_ts)}",
        f"- window_n={report.window_n} freshness_k={_fmt_num(report.freshness_k, 2)}",
    ]
    headers = ["thinker", "pos", "pair", "side", "sstrat", "tf", "price", "stop", "gap%", "hit", "stale", "Δstop%", "hits", "last_hit", "last_stop", "attached", "last_alert"]
    rows_tbl: List[List[Any]] = []
    if not report.rows:
        md_lines.append("- No trailing attachments.")
    for r in report.rows:
        rows_tbl.append([
            r.thinker_id,
            r.position_id,
            r.pair,
            "LONG" if r.side > 0 else "SHORT",
            r.sstrat_kind,
            r.timeframe,
            _fmt_num(r.price, 4),
            _fmt_num(r.stop, 4),
            _fmt_pct(r.gap_pct / 100, nd=3, show_sign=True),
            "hit" if r.hit else "",
            "STALE" if r.stale else "",
            _fmt_pct(r.delta_pct_last_n / 100, nd=3, show_sign=True) if r.delta_pct_last_n is not None else "?",
            r.hits_last_n,
            ts_human(r.last_hit_ts) if r.last_hit_ts else "-",
            ts_human(r.last_stop_ts),
            ts_human(r.attached_at),
            ts_human(r.last_alert_ts) if r.last_alert_ts else "-",
        ])
    md_body = "\n".join(md_lines)
    return {"markdown": md_body, "headers": headers, "rows": rows_tbl}


def format_stop_report_html(report: StopReport) -> Dict[str, Any]:
    rows_html = []
    headers = ["Thinker", "Pos", "Pair", "Side", "Sstrat", "TF", "Price", "Stop", "Gap %", "Hit", "Stale", "Δstop %", "Hits", "Last hit", "Last stop", "Attached", "Last alert"]
    for r in report.rows:
        rows_html.append(
            "<tr>" +
            "".join([
                f"<td>{r.thinker_id}</td>",
                f"<td>{r.position_id}</td>",
                f"<td>{r.pair}</td>",
                f"<td>{'LONG' if r.side > 0 else 'SHORT'}</td>",
                f"<td>{r.sstrat_kind}</td>",
                f"<td>{r.timeframe}</td>",
                f"<td>{_fmt_num(r.price,4)}</td>",
                f"<td>{_fmt_num(r.stop,4)}</td>",
                f"<td>{_fmt_pct(r.gap_pct/100, nd=3, show_sign=True)}</td>",
                f"<td>{'hit' if r.hit else ''}</td>",
                f"<td>{'STALE' if r.stale else ''}</td>",
                f"<td>{_fmt_pct(r.delta_pct_last_n/100, nd=3, show_sign=True) if r.delta_pct_last_n is not None else '?'}</td>",
                f"<td>{r.hits_last_n}</td>",
                f"<td>{ts_human(r.last_hit_ts) if r.last_hit_ts else '-'}</td>",
                f"<td>{ts_human(r.last_stop_ts)}</td>",
                f"<td>{ts_human(r.attached_at)}</td>",
                f"<td>{ts_human(r.last_alert_ts) if r.last_alert_ts else '-'}</td>",
            ]) +
            "</tr>"
        )

    table_html = "".join([
        "<table>",
        "<thead><tr>",
        "".join(f"<th>{h}</th>" for h in headers),
        "</tr></thead>",
        "<tbody>",
        "".join(rows_html) if rows_html else "<tr><td colspan='17'>No trailing attachments.</td></tr>",
        "</tbody>",
        "</table>",
    ])
    style = """
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      h1 { margin-bottom: 0.2em; }
      .meta { color: #444; margin-bottom: 1em; }
      table { border-collapse: collapse; width: 100%; font-size: 13px; }
      th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }
      th { background: #f2f2f2; }
      tr:nth-child(even) { background: #fafafa; }
    </style>
    """
    html = "".join([
        "<html><head><meta charset='utf-8'>",
        style,
        "</head><body>",
        "<h1>Stop Report</h1>",
        f"<div class='meta'>Generated: {ts_human(report.generated_ts)} · window_n={report.window_n} · freshness_k={_fmt_num(report.freshness_k,2)}</div>",
        table_html,
        "</body></html>",
    ])
    fd, path = tempfile.mkstemp(prefix="stop_report_", suffix=".html")
    with os.fdopen(fd, "w") as f:
        f.write(html)
    return {"html": html, "path": path, "headers": headers}
