"""Summary, timeline text/plot."""
from __future__ import annotations
import json, os
from .core import *


def state_durations(segs):
    d = {s: 0.0 for s in STATES}
    for s in segs:
        d[s.state] += s.duration
    return d


def out_periods(events, duration):
    per, cur = [], None
    for e in sorted(events, key=lambda e: e["start_sec"]):
        if e["event"] == "bed_exit":
            cur = e["start_sec"]
        elif e["event"] == "bed_return" and cur is not None:
            per.append((cur, e["start_sec"])); cur = None
    if cur is not None:
        per.append((cur, duration))
    return per


def build_summary(segs, events, duration, overall, alerts):
    dur = state_durations(segs)
    periods = out_periods(events, duration)
    out_t = sum(b - a for a, b in periods)
    in_t = dur[LYING_IN_BED] + dur[SITTING_ON_BED]
    unclassified = max(0.0, duration - in_t - out_t)
    r = lambda x: int(round(x))
    exits = [e for e in events if e["event"] == "bed_exit"]
    rets = [e for e in events if e["event"] == "bed_return"]
    machine = {
        "observation_duration_sec": r(duration),
        "activity_duration_sec": {s.lower(): r(v) for s, v in dur.items() if v > 0 or s != LYING_OUTSIDE_BED},
        "bed_exit_count": len(exits), "bed_return_count": len(rets),
        "total_in_bed_sec": r(in_t), "total_out_of_bed_sec": r(out_t),
        "bedside_or_unknown_sec": r(unclassified),
        "longest_out_of_bed_period_sec": r(max([b - a for a, b in periods], default=0)),
        "final_state": segs[-1].state.lower() if segs else "unknown",
        "overall_decision": overall, "alerts": alerts,
    }
    human = {
        "total_observation_time": fmt_dur(duration),
        "activity_summary": {s.lower(): fmt_dur(v) for s, v in dur.items() if v > 0 or s != LYING_OUTSIDE_BED},
        "bed_summary": {"time_in_bed": fmt_dur(in_t), "time_out_of_bed": fmt_dur(out_t),
                        "bed_exit_count": len(exits)},
    }
    return machine, human


def timeline_text(segs):
    return "\n".join(f"{fmt_mmss(s.start)} – {fmt_mmss(s.end)}  {s.state}" +
                     (f"   ({s.note})" if s.note else "") for s in segs)


COLORS = {LYING_IN_BED: "#4c78a8", SITTING_ON_BED: "#9ecae9", SITTING_OUTSIDE_BED: "#f58518",
          STANDING: "#54a24b", WALKING: "#88d27a", OUT_OF_BED: "#b279a2", LYING_OUTSIDE_BED: "#e45756",
          UNKNOWN: "#bab0ac"}


def plot_timeline(segs, events, path, title="Activity timeline"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 2.6))
    for s in segs:
        ax.barh(0, s.duration, left=s.start, color=COLORS[s.state], edgecolor="white", height=0.6)
    for e in events:
        c = "red" if e["event"] == "bed_exit" else "green"
        ax.axvline(e["start_sec"], color=c, ls="--", lw=1.2)
        ax.text(e["start_sec"], 0.45, e["event"], rotation=90, va="bottom", ha="right", fontsize=7, color=c)
    ax.set_yticks([]); ax.set_xlabel("seconds"); ax.set_title(title); ax.set_ylim(-0.5, 1.4)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()],
              labels=list(COLORS), ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def save_all(out_dir, segs, events, machine, human, trace, frame_states=None):
    os.makedirs(out_dir, exist_ok=True)
    w = lambda name, txt: open(os.path.join(out_dir, name), "w").write(txt)
    w("timeline.txt", timeline_text(segs) + "\n")
    w("summary.json", json.dumps(machine, indent=2))
    w("summary_readable.json", json.dumps(human, indent=2))
    w("events.json", json.dumps([{k: v for k, v in e.items() if not k.endswith("_sec")} for e in events], indent=2))
    w("segments.json", json.dumps([dict(state=s.state, start=round(s.start, 1), end=round(s.end, 1),
                                        conf=round(s.conf, 3), note=s.note) for s in segs], indent=2))
    w("agent_trace.txt", trace.render() + "\n")
    if frame_states:
        w("frame_states.json", json.dumps([dict(t=round(f.t, 2), state=f.state, conf=round(f.conf, 3),
                                               reason=f.reason, persons=f.persons, **{"feat_" + k: v for k, v in f.feats.items()})
                                          for f in frame_states]))
    plot_timeline(segs, events, os.path.join(out_dir, "timeline.png"))
