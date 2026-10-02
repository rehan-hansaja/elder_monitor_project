"""Rule-based contextual alerting: NORMAL / MONITOR / ALERT with an explanation per rule."""
from __future__ import annotations
from .core import *

LEVELS = ["NORMAL", "MONITOR", "ALERT"]


def apply_event_decisions(events, cfg):
    """Each exit gets `decision` (at confirmation: MONITOR) and `final_decision` once the outcome is known."""
    for e in events:
        if e["event"] != "bed_exit":
            continue
        d = e["out_of_bed_duration_sec"]
        if d >= cfg.alert_out_sec:
            e["final_decision"], e["decision_reason"] = "ALERT", f"out of bed {d:.0f}s >= {cfg.alert_out_sec:.0f}s"
        elif d >= cfg.monitor_out_sec:
            e["final_decision"], e["decision_reason"] = "MONITOR", f"out of bed {d:.0f}s >= {cfg.monitor_out_sec:.0f}s"
        elif not e["returned"]:
            e["final_decision"], e["decision_reason"] = "MONITOR", "still out of bed at end of video"
        else:
            e["final_decision"], e["decision_reason"] = "NORMAL", f"returned after {d:.0f}s (short trip)"
    return events


def _caregiver_intervals(frame_states, gap=2.0):
    out, cur = [], None
    for f in frame_states:
        if f.persons >= 2:
            if cur and f.t - cur[1] <= gap:
                cur[1] = f.t
            else:
                cur = [f.t, f.t]; out.append(cur)
    return out


def evaluate_alerts(segs, events, frame_states, cfg):
    alerts = []
    add = lambda t, lvl, rule, msg: alerts.append(dict(time=fmt_ts(t), time_sec=round(t, 1), level=lvl, rule=rule, message=msg))

    for e in events:                                                  # R1: prolonged absence
        if e["event"] == "bed_exit":
            d, t0 = e["out_of_bed_duration_sec"], e["start_sec"]
            if d >= cfg.alert_out_sec:
                add(t0 + cfg.alert_out_sec, "ALERT", "prolonged_absence", f"Out of bed >{cfg.alert_out_sec:.0f}s")
            elif d >= cfg.monitor_out_sec:
                add(t0 + cfg.monitor_out_sec, "MONITOR", "long_absence", f"Out of bed >{cfg.monitor_out_sec:.0f}s")

    for s in segs:
        if s.state == LYING_OUTSIDE_BED and s.duration >= cfg.alert_floor_sec:   # R2: possible fall
            add(s.start + cfg.alert_floor_sec, "ALERT", "possible_fall", "Person lying outside the bed")
        if s.state == SITTING_ON_BED:                                            # R3: long edge sitting
            flags = [f.feats.get("feet_off_bed") for f in frame_states if s.start <= f.t < s.end]
            flags = [x for x in flags if x is not None]
            edge = bool(flags) and sum(flags) / len(flags) >= 0.5
            lim = cfg.monitor_edge_sec if edge else 3 * cfg.monitor_edge_sec
            if s.duration >= lim:
                add(s.start + lim, "MONITOR", "long_sitting_on_bed",
                    "Sitting on bed edge unusually long" if edge else "Sitting up in bed unusually long")
        if s.state == UNKNOWN and s.duration >= cfg.monitor_unknown_sec:         # R4: cannot determine
            add(s.start + cfg.monitor_unknown_sec, "MONITOR", "uncertain_activity",
                f"Activity undetermined for >{cfg.monitor_unknown_sec:.0f}s")

    cg = _caregiver_intervals(frame_states)                                       # R5: caregiver present
    for a in alerts:
        if any(lo - cfg.caregiver_grace_sec <= a["time_sec"] <= hi + cfg.caregiver_grace_sec for lo, hi in cg):
            lvl = LEVELS[max(0, SEVERITY[a["level"]] - 1)]
            a["message"] += f" [downgraded {a['level']}->{lvl}: caregiver present]"
            a["level"] = lvl
    alerts.sort(key=lambda a: a["time_sec"])

    levels = [a["level"] for a in alerts] + [e.get("final_decision", "NORMAL") for e in events]
    overall = max(levels, key=lambda l: SEVERITY[l]) if levels else "NORMAL"
    return alerts, overall, cg
