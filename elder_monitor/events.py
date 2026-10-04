"""Bed exit / return detection on the state timeline, with agent-style evidence gathering."""
from __future__ import annotations
from .core import *


def _stretch_end(segs, j):
    k = j
    while k + 1 < len(segs) and segs[k + 1].state in IN_BED_STATES:
        k += 1
    return k


def _find_return(segs, i, cfg):
    """First in-bed stretch after index i that lasts >= min_in_sec (or runs to the end of the video)."""
    j = i + 1
    while j < len(segs):
        if segs[j].state in IN_BED_STATES:
            k = _stretch_end(segs, j)
            if segs[k].end - segs[j].start >= cfg.min_in_sec or k == len(segs) - 1:
                return j, k
            j = k + 1
        else:
            j += 1
    return None, None


def detect_bed_events(segs, cfg, evidence_fn=None, trace: Trace = None):
    ev, n = [], len(segs)
    loc, last_in, i = "unknown", None, 0
    T = lambda **kw: trace.add(**kw) if trace else None

    while i < n:
        s = segs[i]
        # in bed
        if s.state in IN_BED_STATES:
            if loc == "out":                                   # candidate RETURN
                k = _stretch_end(segs, i)
                chain = segs[i:k + 1]
                if segs[k].end - s.start >= cfg.min_in_sec or k == n - 1:
                    lie = next((x for x in chain if x.state == LYING_IN_BED), None)
                    conf_t = lie.start if lie else s.start + min(cfg.min_in_sec, segs[k].end - s.start)
                    prev = segs[i - 1].state if i > 0 else UNKNOWN
                    conf = sum(x.conf for x in chain) / len(chain)
                    ev.append(dict(event="bed_return", start_time=fmt_ts(s.start), confirmed_time=fmt_ts(conf_t),
                                   previous_state=prev.lower(), current_state=(lie or s).state.lower(),
                                   confidence=round(min(conf, 0.99), 2), decision="NORMAL",
                                   start_sec=round(s.start, 1), confirmed_sec=round(conf_t, 1)))
                    T(t=s.start, observation=f"Person now {s.state} after being out of bed.",
                      thought="Was out of bed; need to check the in-bed state is sustained, not a pass-by.",
                      action=f"look ahead {cfg.min_in_sec:.0f}s", result=f"in-bed for {segs[k].end - s.start:.0f}s"
                      + (f", lies down at {fmt_ts(conf_t)}" if lie else ", sitting only"),
                      conclusion="BED_RETURN confirmed")
                    loc = "in"
                last_in, i = k, k + 1
                continue
            loc, last_in, i = "in", i, i + 1
            continue

        # out of bed
        if s.state in OUT_STATES:
            if loc == "unknown":
                loc = "out"; i += 1; continue
            if loc == "in":                                    # candidate EXIT
                li = segs[last_in]
                start_t = li.end
                j, _ = _find_return(segs, i, cfg)
                stop = j if j is not None else n
                contents = segs[i:stop]
                out_end = segs[j].start if j is not None else segs[-1].end
                out_dur = out_end - start_t
                moved = [x for x in contents if x.state in MOVED_AWAY_STATES]
                T(t=start_t, observation=f"{li.state} -> {s.state}: person is beside/away from the bed.",
                  thought="One transition is not enough to call a bed exit; need what happened before and after.",
                  action=f"inspect previous segment", result=f"{li.state} until {fmt_ts(li.end)} (conf {li.conf:.2f})")
                T(action="inspect following segments up to the next in-bed stretch",
                  result=" -> ".join(f"{x.state}({x.duration:.0f}s)" for x in contents)
                  + (f" -> back {segs[j].state} at {fmt_ts(segs[j].start)}" if j is not None else " (until end of video)"))

                if j is not None and out_dur < cfg.min_out_sec:
                    T(conclusion=f"REJECT: only {out_dur:.0f}s off the bed then back (<{cfg.min_out_sec:.0f}s) "
                                 "-> stood up and sat back down, not an exit")
                    last_in, i = j, j
                    continue

                reduced = False
                if not moved:
                    away = None
                    if evidence_fn:
                        away = evidence_fn(start_t, out_end)
                        T(action="check location evidence (bed overlap of the person's box)",
                          result=f"mean bed overlap {away['mean_bed_frac']:.2f}")
                    if (out_dur >= cfg.standing_only_exit_sec and away is not None
                            and away["mean_bed_frac"] < cfg.away_bed_frac):
                        reduced = True
                    else:
                        T(conclusion="REJECT: stayed standing at the bedside with no movement away")
                        if j is None:
                            break
                        last_in, i = j, j
                        continue

                first = moved[0] if moved else contents[0]
                confirmed = (first.start if first.start > start_t + 1e-6
                             else start_t + min(cfg.exit_confirm_lag_sec, first.duration))
                cur = next((x for x in contents if x.start <= confirmed < x.end), first)
                pool = [li, contents[0], first]
                conf = sum(x.conf for x in pool) / len(pool) * (0.85 if reduced else 1.0)
                ev.append(dict(event="bed_exit", start_time=fmt_ts(start_t), confirmed_time=fmt_ts(confirmed),
                               previous_state=li.state.lower(), current_state=cur.state.lower(),
                               confidence=round(min(conf, 0.99), 2), decision="MONITOR",
                               start_sec=round(start_t, 1), confirmed_sec=round(confirmed, 1),
                               out_of_bed_duration_sec=round(out_dur, 1), returned=j is not None))
                T(conclusion=f"BED_EXIT confirmed at {fmt_ts(confirmed)} (started {fmt_ts(start_t)}); "
                             f"off bed for {out_dur:.0f}s" + ("" if j is not None else ", still out at end of video"))
                loc = "out"
                i = j if j is not None else n
                continue
            i += 1                                           # loc == "out": stay out
            continue
        i += 1                                               # UNKNOWN: ignore for location logic
    return ev
