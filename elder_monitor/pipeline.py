"""Glue: per-frame states -> Viterbi -> segments -> agent -> events -> alerts -> summary."""
from __future__ import annotations
from .core import *
from .temporal import viterbi_decode, build_segments, merge_short
from .agent import TemporalAgent
from .alerts import apply_event_decisions, evaluate_alerts
from .report import build_summary


def analyze_states(frame_states, duration, cfg, video=None, bed=None, vlm=None):
    trace = Trace()
    path = viterbi_decode(frame_states, cfg)
    segs = merge_short(build_segments(frame_states, path, duration), cfg)
    agent = TemporalAgent(cfg, frame_states, video, bed, vlm, trace)
    segs = merge_short(agent.resolve_uncertain(segs), cfg)
    events = agent.detect_events(segs)
    apply_event_decisions(events, cfg)
    alerts, overall, caregiver = evaluate_alerts(segs, events, frame_states, cfg)
    machine, human = build_summary(segs, events, duration, overall, alerts)
    return dict(segments=segs, events=events, alerts=alerts, overall=overall, summary=machine,
                readable=human, trace=trace, caregiver=caregiver)
