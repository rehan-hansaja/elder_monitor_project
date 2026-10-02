"""Evaluation against manual ground truth: state accuracy, confusion, durations, bed-event P/R, failures."""
from __future__ import annotations
import csv, json
from collections import Counter
import numpy as np
from .core import *
from .events import detect_bed_events


def load_gt_segments(path):
    segs = []
    with open(path) as f:
        for r in csv.DictReader(f):
            st = r["state"].strip().upper()
            segs.append(Segment(st, parse_ts(r["start"]), parse_ts(r["end"]), 1.0))
    return sorted(segs, key=lambda s: s.start)


def load_gt_events(path):
    with open(path) as f:
        return [dict(event=r["event"].strip().lower(), t=parse_ts(r["time"])) for r in csv.DictReader(f)]


def to_grid(segs, T, dt=0.5):
    n = int(T / dt)
    arr = [UNKNOWN] * n
    for s in segs:
        for i in range(int(s.start / dt), min(n, int(round(s.end / dt)))):
            arr[i] = s.state
    return arr


def match(pred_t, gt_t, tol):
    pred_t, gt_t = sorted(pred_t), sorted(gt_t)
    used, tp = set(), 0
    for p in pred_t:
        cands = [(abs(p - g), gi) for gi, g in enumerate(gt_t) if gi not in used and abs(p - g) <= tol]
        if cands:
            used.add(min(cands)[1]); tp += 1
    fp, fn = len(pred_t) - tp, len(gt_t) - tp
    return dict(tp=tp, fp=fp, fn=fn, precision=round(tp / (tp + fp), 3) if tp + fp else None,
                recall=round(tp / (tp + fn), 3) if tp + fn else None)


def evaluate(pred_segs, gt_segs, pred_events, T, cfg, gt_events=None, tol=10.0, dt=0.5, topk=5):
    g, p = to_grid(gt_segs, T, dt), to_grid(pred_segs, T, dt)
    labels = [s for s in STATES if s in set(g) | set(p)]
    li = {s: i for i, s in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), int)
    for a, b in zip(g, p):
        cm[li[a], li[b]] += 1
    acc = float(np.mean([a == b for a, b in zip(g, p)]))
    per = {}
    for s in labels:
        tp = cm[li[s], li[s]]
        pr, rc = tp / max(cm[:, li[s]].sum(), 1), tp / max(cm[li[s], :].sum(), 1)
        per[s] = dict(precision=round(pr, 3), recall=round(rc, 3), f1=round(2 * pr * rc / max(pr + rc, 1e-9), 3),
                      support_sec=round(cm[li[s], :].sum() * dt, 1))
    confusions = sorted([(labels[i], labels[j], int(cm[i, j] * dt)) for i in range(len(labels))
                         for j in range(len(labels)) if i != j and cm[i, j] > 0], key=lambda x: -x[2])[:8]
    # durations
    gd, pd = Counter(), Counter()
    for a in g: gd[a] += dt
    for b in p: pd[b] += dt
    dur = {s: dict(gt=round(gd[s], 1), pred=round(pd[s], 1), abs_error=round(abs(gd[s] - pd[s]), 1))
           for s in labels}
    mae = float(np.mean([v["abs_error"] for v in dur.values()]))
    # events
    if gt_events is None:
        gt_events = [dict(event=e["event"], t=e["start_sec"]) for e in detect_bed_events(gt_segs, cfg)]
    ev = {}
    for kind in ("bed_exit", "bed_return"):
        ev[kind] = match([e["start_sec"] for e in pred_events if e["event"] == kind],
                         [e["t"] for e in gt_events if e["event"] == kind], tol)
    # failure runs
    runs, i = [], 0
    while i < len(g):
        if g[i] != p[i]:
            j = i
            while j < len(g) and g[j] == g[i] and p[j] == p[i]:
                j += 1
            runs.append(dict(start=i * dt, end=j * dt, gt=g[i], pred=p[i], length=(j - i) * dt))
            i = j
        else:
            i += 1
    runs = sorted(runs, key=lambda r: -r["length"])[:topk]
    return dict(frame_accuracy=round(acc, 4), per_class=per, labels=labels, confusion_matrix=cm.tolist(),
                top_confusions=[dict(gt=a, pred=b, seconds=s) for a, b, s in confusions],
                durations=dur, duration_mae_sec=round(mae, 2), event_tolerance_sec=tol,
                bed_events=ev, failure_cases=runs)


def format_report(res):
    L = [f"Frame-level state accuracy: {res['frame_accuracy'] * 100:.1f}%", "", "Per-class:"]
    for s, v in res["per_class"].items():
        L.append(f"  {s:<22} P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f} ({v['support_sec']:.0f}s GT)")
    L += ["", "Most frequent confusions (GT -> predicted, seconds):"]
    L += [f"  {c['gt']} -> {c['pred']}: {c['seconds']}s" for c in res["top_confusions"]]
    L += ["", f"{'State':<22}{'GT':>8}{'Pred':>8}{'AbsErr':>8}"]
    for s, v in res["durations"].items():
        L.append(f"{s:<22}{fmt_mmss(v['gt']):>8}{fmt_mmss(v['pred']):>8}{v['abs_error']:>7.1f}s")
    L.append(f"Mean absolute duration error: {res['duration_mae_sec']}s")
    L += ["", f"Bed events (match tolerance ±{res['event_tolerance_sec']}s):"]
    for k, v in res["bed_events"].items():
        L.append(f"  {k}: TP={v['tp']} FP={v['fp']} FN={v['fn']} precision={v['precision']} recall={v['recall']}")
    L += ["", "Failure cases (longest mismatched runs):"]
    for r in res["failure_cases"]:
        L.append(f"  {fmt_mmss(r['start'])}-{fmt_mmss(r['end'])}  GT={r['gt']}  PRED={r['pred']}  ({r['length']:.0f}s)")
    return "\n".join(L)
