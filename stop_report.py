#!/usr/bin/env python3
# FILE: stop_report.py
from __future__ import annotations
import math
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from common import Clock, tf_ms, ts_human, PP_CTX, THOUGHT, NOW_MS, LAST_MOVE_ALERT_TS, LAST_HIT_ALERT_TS, float2str, log, is_sane
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
    has_stop: bool


@dataclass
class StopReport:
    rows: List[StopRow]
    generated_ts: int
    window_n: int
    freshness_k: float
    thinker_ticks: Dict[int, Optional[int]]
    history_counts: Dict[tuple[int, int], int]
    thinker_status: Dict[int, str]


STOP_REPORT_GUIDE = """
**Parameters**
- `window_n`: how many recent indicator-history rows to summarize (hit counts, Δstop%). Bigger = longer lookback. Default `full` (all).
- `freshness_k`: multiple of timeframe used to mark `stale` (last stop older than `freshness_k * tf_ms`). Raise to be more tolerant, lower to be stricter. Default 2.

**Summary fields**
- `n_rows`: trailing attachments included.
- `n_stale`: rows whose last stop is older than `freshness_k * timeframe`.
- `n_hits_now`: rows whose latest stopper flag is 1.
- `pending`: rows with no stopper history yet.
- `gap%` / `Δstop%`: min/avg/max across rows with finite values.

**Position columns**
- `gap%`: `(price - stop) / price`; positive = price above stop (long headroom), negative = stop above price (shorts); `?` if undefined.
- `hit`: latest stopper flag only; past hits clear if the most recent flag is 0.
- `stale`: last stop older than `freshness_k * timeframe`.
- `Δstop%`: stop drift from oldest to newest in the last `window_n` records; `?` if missing.
- `hits`: number of hit flags in the last `window_n`; `last_hit`: most recent hit ts or `PENDING` if no history; `last_alert`: when a hit alert was emitted (blank if none).
- `hist_n`: stopper-value records pulled for that position.

**Thinker table**
- Per thinker: `Status`, `sstrat`, `tf`, `n_rows`, `n_hits_now`, `n_stale`, `pending`, `gap% avg/min/max`, `Δstop% avg/min/max`, `Last_tick`.
""".strip()


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


def _history_stats(eng, thinker_id: int, position_id: int, window_n: Optional[int]) -> dict:
    vals = eng.ih.last_n(thinker_id, position_id, "stopper-value", n=window_n, asc=True) if window_n \
        else eng.ih.range_by_ts(thinker_id, position_id, "stopper-value", fmt="columnar")
    flags = eng.ih.last_n(thinker_id, position_id, "stopper-flag", n=window_n, asc=True) if window_n \
        else eng.ih.range_by_ts(thinker_id, position_id, "stopper-flag", fmt="columnar")

    clean_vals = _clean_series(vals["open_ts"], vals["value"]) if vals else []
    clean_flags = _clean_series(flags["open_ts"], flags["value"]) if flags else []

    delta_pct_last_n: Optional[float] = None
    last_stop_ts: Optional[int] = None
    last_stop_val: Optional[float] = None
    if clean_vals:
        first_ts, first_stop = clean_vals[0]
        last_ts, last_stop = clean_vals[-1]
        assert first_stop != 0.0, "first stop is zero"
        delta_pct_last_n = (last_stop - first_stop) * 100 / first_stop
        last_stop_ts = last_ts
        last_stop_val = last_stop

    hits_last_n = 0
    last_hit_ts: Optional[int] = None
    hit_now = False
    for ts, val in clean_flags:
        if val >= 1.0:
            hits_last_n += 1
            last_hit_ts = ts
    if flags and flags.get("value"):
        last_val = flags["value"][-1]
        if last_val is not None:
            try:
                fval = float(last_val)
                if not math.isnan(fval) and fval >= 1.0:
                    hit_now = True
            except Exception:
                pass

    return {
        "delta_pct_last_n": delta_pct_last_n,
        "hits_last_n": hits_last_n,
        "last_hit_ts": last_hit_ts,
        "last_stop_ts": last_stop_ts,
        "hit_now": hit_now,
        "last_stop_val": last_stop_val,
        "count_vals": len(clean_vals),
    }


def _latest_price_for_position(eng, pos, timeframe: str) -> float:
    now_ms = Clock.now_utc_ms()
    tfms = tf_ms(timeframe)
    start_ts = max(0, now_ms - 3 * tfms)
    df = eng.kc.pair_bars(pos.num, pos.den, timeframe, start_ts, None)
    if df.empty:
        raise RuntimeError(f"No klines for {pos.get_pair()} {timeframe}")
    return float(df["Close"].iloc[-1])


def build_stop_report(eng, window_n: Optional[int] = None, freshness_k: float = 2.0, thinker_id: Optional[int] = None) -> StopReport:
    now_ms = Clock.now_utc_ms()
    rows: List[StopRow] = []
    thinker_ticks: Dict[int, Optional[int]] = {}
    hist_counts: Dict[tuple[int, int], int] = {}
    thinker_status: Dict[int, str] = {}

    thinkers = eng.store.list_thinkers()
    tid_list = [int(thinker_id)] if thinker_id is not None else [
        int(r["id"]) for r in thinkers if r["kind"] == "TRAILING_STOP"
    ]
    if not tid_list:
        raise ValueError("No TRAILING_STOP thinkers found")

    for tid in tid_list:
        inst = eng.tm.get_in_carbonite(tid, expected_kind="TRAILING_STOP")
        rt = inst.runtime()
        pp_ctx = rt[PP_CTX]
        t_thought = rt.get("THOUGHT") or rt.get(THOUGHT) or {}
        thinker_ticks[tid] = t_thought.get(NOW_MS) or t_thought.get("now_ms")
        tr = next((r for r in thinkers if int(r["id"]) == tid), None)
        thinker_status[tid] = "ENABLED" if tr and int(tr["enabled"]) == 1 else "disabled"
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

            timeframe = inst._cfg["timeframe"]
            price = _latest_price_for_position(eng, pos, timeframe)
            sstrat_kind = ctx["sstrat_kind"]
            attached_at = int(ctx["attached_at"])

            stats = _history_stats(eng, tid, pid, window_n)
            hist_counts[(tid, pid)] = stats.get("count_vals") or 0
            if is_sane(stats["last_stop_val"]) and stats["last_stop_ts"] is not None:
                stop = stats["last_stop_val"]
                last_stop_ts = stats["last_stop_ts"]
                gap_pct = (price - stop) * 100 / price
                stale = (now_ms - last_stop_ts) > freshness_ms
                hit = bool(stats["hit_now"])
                has_stop = True
            else:
                stop = math.nan
                last_stop_ts = None
                gap_pct = math.nan
                stale = False
                hit = False
                has_stop = False
            thought = ctx.get(THOUGHT) or {}
            last_alert_ts = thought.get(LAST_HIT_ALERT_TS)

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
                has_stop=has_stop,
            )
            rows.append(row)

    return StopReport(rows=rows, generated_ts=now_ms, window_n=window_n or 0, freshness_k=freshness_k, thinker_ticks=thinker_ticks, history_counts=hist_counts, thinker_status=thinker_status)


def format_stop_report_md(report: StopReport):
    from commands import OCMarkDown, OCTable, CO  # lazy import to avoid circular
    from common import NOW_MS

    def _md(lines):
        return OCMarkDown("\n".join(lines))

    elements: List[Any] = [
        _md([
            "# Stop Report",
            f"- generated: {ts_human(report.generated_ts)}",
            f"- `window_n`={'full' if report.window_n == 0 else f'{report.window_n}'} `freshness_k`={_fmt_num(report.freshness_k, 2)}",
        ]),
    ]

    rows_tbl: List[List[Any]] = []
    thinker_tbl: List[List[Any]] = []
    if not report.rows:
        elements.append(_md(["## Summary", "- Nothing to summarize"]))
    else:
        n = len(report.rows)
        stale_n = sum(1 for r in report.rows if r.stale)
        hit_n = sum(1 for r in report.rows if r.hit)
        pending_n = sum(1 for r in report.rows if not r.has_stop)
        gaps = [r.gap_pct for r in report.rows if is_sane(r.gap_pct)]
        deltas = [r.delta_pct_last_n for r in report.rows if is_sane(r.delta_pct_last_n)]
        summary_lines = ["## Summary",
                         f"- `n_rows`={n} `n_stale`={stale_n} `n_hits_now`={hit_n} `pending`={pending_n}"]
        if gaps:
            summary_lines.append(
                f"- `gap%` avg={_fmt_pct(sum(gaps)/len(gaps)/100, nd=3, show_sign=True)} "
                f"min={_fmt_pct(min(gaps)/100, nd=3, show_sign=True)} "
                f"max={_fmt_pct(max(gaps)/100, nd=3, show_sign=True)}"
            )
        if deltas:
            summary_lines.append(
                f"- `Δstop%` avg={_fmt_pct(sum(deltas)/len(deltas)/100, nd=3, show_sign=True)} "
                f"min={_fmt_pct(min(deltas)/100, nd=3, show_sign=True)} "
                f"max={_fmt_pct(max(deltas)/100, nd=3, show_sign=True)}"
            )
        elements.append(_md(summary_lines))

        for r in report.rows:
            gap_disp = "?" if not is_sane(r.gap_pct) else _fmt_pct(r.gap_pct / 100, nd=3, show_sign=True)
            hist_n = report.history_counts.get((r.thinker_id, r.position_id), 0)
            rows_tbl.append([
                r.thinker_id,
                r.position_id,
                r.pair,
                "LONG" if r.side > 0 else "SHORT",
                "?" if not is_sane(r.price) else float2str(r.price),
                "?" if not is_sane(r.stop) else float2str(r.stop),
                gap_disp,
                "hit" if r.hit else "",
                "STALE" if r.stale else "",
                _fmt_pct(r.delta_pct_last_n / 100, nd=3, show_sign=True) if is_sane(r.delta_pct_last_n) else "?",
                r.hits_last_n,
                "PENDING" if not r.has_stop else (ts_human(r.last_hit_ts) if r.last_hit_ts else "-"),
                ts_human(r.last_alert_ts) if r.last_alert_ts else "-",
                hist_n,
            ])

        thinker_meta: Dict[int, dict] = {}
        for r in report.rows:
            meta = thinker_meta.setdefault(r.thinker_id, {
                "sstrat": r.sstrat_kind,
                "tf": r.timeframe,
                "rows": 0,
                "stale": 0,
                "hit": 0,
                "pending": 0,
                "gaps": [],
                "deltas": [],
            })
            meta["rows"] += 1
            meta["stale"] += 1 if r.stale else 0
            meta["hit"] += 1 if r.hit else 0
            meta["pending"] += 1 if not r.has_stop else 0
            if is_sane(r.gap_pct):
                meta["gaps"].append(r.gap_pct)
            if is_sane(r.delta_pct_last_n):
                meta["deltas"].append(r.delta_pct_last_n)
        ticks = report.thinker_ticks or {}
        status_map = report.thinker_status or {}
        for tid, meta in sorted(thinker_meta.items()):
            tick_val = ticks.get(tid)
            status = status_map.get(tid, "?")
            gap_str = "n/a"
            delta_str = "n/a"
            if meta["gaps"]:
                gap_str = f"{_fmt_pct(sum(meta['gaps'])/len(meta['gaps'])/100, nd=3, show_sign=True)} / {_fmt_pct(min(meta['gaps'])/100, nd=3, show_sign=True)} / {_fmt_pct(max(meta['gaps'])/100, nd=3, show_sign=True)}"
            if meta["deltas"]:
                delta_str = f"{_fmt_pct(sum(meta['deltas'])/len(meta['deltas'])/100, nd=3, show_sign=True)} / {_fmt_pct(min(meta['deltas'])/100, nd=3, show_sign=True)} / {_fmt_pct(max(meta['deltas'])/100, nd=3, show_sign=True)}"
            thinker_tbl.append([tid, status, meta["sstrat"], meta["tf"], meta["rows"], meta["hit"], meta["stale"], meta["pending"], gap_str, delta_str, ts_human(tick_val) if tick_val else "-"])

    headers_positions = ["Thinker", "Pos", "Pair", "Side", "Price", "Stop", "Gap %", "Hit", "Stale", "Δstop %", "Hits", "Last hit", "Last alert", "Hist_n"]
    headers_thinkers = ["Thinker", "Status", "Sstrat", "TF", "n_rows", "n_hits_now", "n_stale", "pending", "gap% avg/min/max", "Δstop% avg/min/max", "Last_tick"]

    if rows_tbl:
        elements.append(_md(["## Positions"]))
        elements.append(OCTable(headers=headers_positions, rows=rows_tbl))
    if thinker_tbl:
        elements.append(_md(["## Thinkers"]))
        elements.append(OCTable(headers=headers_thinkers, rows=thinker_tbl))

    elements.append(_md(["## Guide", STOP_REPORT_GUIDE]))

    return CO(elements)


def format_stop_report_html(report: StopReport):
    from commands import OCHTML, CO  # lazy import to avoid circular
    rows_html = []
    headers = ["Thinker", "Pos", "Pair", "Side", "Price", "Stop", "Gap %", "Hit", "Stale", "Δstop %", "Hits", "Last hit", "Last alert", "Hist_n"]
    n = len(report.rows)
    stale_n = sum(1 for r in report.rows if r.stale)
    hit_n = sum(1 for r in report.rows if r.hit)
    pending_n = sum(1 for r in report.rows if not r.has_stop)
    gaps = [r.gap_pct for r in report.rows if is_sane(r.gap_pct)]
    deltas = [r.delta_pct_last_n for r in report.rows if is_sane(r.delta_pct_last_n)]

    thinker_tbl: List[List[Any]] = []
    thinker_meta: Dict[int, dict] = {}
    ticks = report.thinker_ticks or {}
    status_map = report.thinker_status or {}

    for r in report.rows:
        rows_html.append(
            "<tr>" +
            "".join([
                f"<td>{r.thinker_id}</td>",
                f"<td>{r.position_id}</td>",
                f"<td>{r.pair}</td>",
                f"<td>{'LONG' if r.side > 0 else 'SHORT'}</td>",
                f"<td>{float2str(r.price)}</td>",
                f"<td>{float2str(r.stop)}</td>",
                f"<td>{_fmt_pct(r.gap_pct/100, nd=3, show_sign=True) if is_sane(r.gap_pct) else '?'}</td>",
                f"<td>{'hit' if r.hit else ''}</td>",
                f"<td>{'STALE' if r.stale else ''}</td>",
                f"<td>{_fmt_pct(r.delta_pct_last_n/100, nd=3, show_sign=True) if is_sane(r.delta_pct_last_n) else '?'}</td>",
                f"<td>{r.hits_last_n}</td>",
                f"<td>{'PENDING' if not r.has_stop else (ts_human(r.last_hit_ts) if r.last_hit_ts else '-')}</td>",
                f"<td>{ts_human(r.last_alert_ts) if r.last_alert_ts else '-'}</td>",
                f"<td>{report.history_counts.get((r.thinker_id, r.position_id), 0)}</td>",
            ]) +
            "</tr>"
        )
        meta = thinker_meta.setdefault(r.thinker_id, {
            "sstrat": r.sstrat_kind,
            "tf": r.timeframe,
            "rows": 0,
            "stale": 0,
            "hit": 0,
            "pending": 0,
            "gaps": [],
            "deltas": [],
        })
        meta["rows"] += 1
        meta["stale"] += 1 if r.stale else 0
        meta["hit"] += 1 if r.hit else 0
        meta["pending"] += 1 if not r.has_stop else 0
        if is_sane(r.gap_pct):
            meta["gaps"].append(r.gap_pct)
        if is_sane(r.delta_pct_last_n):
            meta["deltas"].append(r.delta_pct_last_n)
    for tid, meta in sorted(thinker_meta.items()):
        tick_val = ticks.get(tid)
        status = status_map.get(tid, "?")
        gap_str = "n/a"
        delta_str = "n/a"
        if meta["gaps"]:
            gap_str = f"{_fmt_pct(sum(meta['gaps'])/len(meta['gaps'])/100, nd=3, show_sign=True)} / {_fmt_pct(min(meta['gaps'])/100, nd=3, show_sign=True)} / {_fmt_pct(max(meta['gaps'])/100, nd=3, show_sign=True)}"
        if meta["deltas"]:
            delta_str = f"{_fmt_pct(sum(meta['deltas'])/len(meta['deltas'])/100, nd=3, show_sign=True)} / {_fmt_pct(min(meta['deltas'])/100, nd=3, show_sign=True)} / {_fmt_pct(max(meta['deltas'])/100, nd=3, show_sign=True)}"
        thinker_tbl.append([tid, status, meta["sstrat"], meta["tf"], meta["rows"], meta["hit"], meta["stale"], meta["pending"], gap_str, delta_str, ts_human(tick_val) if tick_val else "-"])

    table_html = "".join([
        "<table>",
        "<thead><tr>",
        "".join(f"<th>{h}</th>" for h in headers),
        "</tr></thead>",
        "<tbody>",
        "".join(rows_html) if rows_html else "<tr><td colspan='14'>Empty positions table.</td></tr>",
        "</tbody>",
        "</table>",
    ])
    thinker_headers_html = ["Thinker", "Status", "Sstrat", "TF", "n_rows", "n_hits_now", "n_stale", "pending", "gap% avg/min/max", "Δstop% avg/min/max", "Last_tick"]
    thinker_html = "".join([
        "<table>",
        "<thead><tr>",
        "".join(f"<th>{h}</th>" for h in thinker_headers_html),
        "</tr></thead>",
        "<tbody>",
        "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in thinker_tbl) if thinker_tbl else f"<tr><td colspan='{len(thinker_headers_html)}'>Empty thinkers table.</td></tr>",
        "</tbody>",
        "</table>",
    ])
    style = """
    <style>
      :root {
        color-scheme: dark;
        --bg: #0d1117;
        --panel: #161b22;
        --fg: #d1d5db;
        --muted: #9ca3af;
        --accent: #58a6ff;
        --border: #30363d;
      }
      body { font-family: "JetBrains Mono", "Fira Code", Menlo, monospace; margin: 20px; background: var(--bg); color: var(--fg); font-size: 14px; }
      h1 { margin-bottom: 0.2em; color: var(--accent); }
      h2 { margin: 1.2em 0 0.5em; color: var(--accent); }
      .meta { color: var(--muted); margin-bottom: 0.4em; }
      ul.meta-list { color: var(--muted); margin-top: 0.2em; }
      table { border-collapse: collapse; width: 100%; font-size: 13px; background: var(--panel); }
      th, td { border: 1px solid var(--border); padding: 6px 8px; text-align: left; }
      th { background: #111827; color: var(--fg); }
      tr:nth-child(even) { background: #11151d; }
      tr:nth-child(odd) { background: #0f141c; }
    </style>
    """
    summary_items = [
        f"n_rows={n} n_stale={stale_n} n_hits_now={hit_n} pending={pending_n}",
    ]
    if gaps:
        summary_items.append(
            f"gap% avg={_fmt_pct(sum(gaps)/len(gaps)/100, nd=3, show_sign=True)} "
            f"min={_fmt_pct(min(gaps)/100, nd=3, show_sign=True)} "
            f"max={_fmt_pct(max(gaps)/100, nd=3, show_sign=True)}"
        )
    if deltas:
        summary_items.append(
            f"Δstop% avg={_fmt_pct(sum(deltas)/len(deltas)/100, nd=3, show_sign=True)} "
            f"min={_fmt_pct(min(deltas)/100, nd=3, show_sign=True)} "
            f"max={_fmt_pct(max(deltas)/100, nd=3, show_sign=True)}"
        )
    summary_html = "".join(f"<li>{s}</li>" for s in summary_items)
    guide_html = "<br>".join(STOP_REPORT_GUIDE.splitlines())
    html = "".join([
        "<html><head><meta charset='utf-8'>",
        style,
        "</head><body>",
        "<h1>Stop Report</h1>",
        f"<div class='meta'>Generated: {ts_human(report.generated_ts)} · window_n={report.window_n} · freshness_k={_fmt_num(report.freshness_k,2)}</div>",
        f"<ul class='meta-list'>{summary_html}</ul>",
        "<h2>Positions</h2>",
        table_html,
        "<h2>Thinkers</h2>",
        thinker_html,
        "<h2>Guide</h2>",
        f"<div class='meta' style='white-space:pre-wrap'>{guide_html}</div>",
        "</body></html>",
    ])
    fd, path = tempfile.mkstemp(prefix="stop_report_", suffix=".html")
    with os.fdopen(fd, "w") as f:
        f.write(html)
    caption = f"Stop report @ {ts_human(report.generated_ts)}"
    return CO(OCHTML(path, caption=caption, open_local=True))
