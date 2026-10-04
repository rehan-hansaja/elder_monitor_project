"""Agentic layer: decides WHEN more temporal context is needed and which tool to call.
Tools: previous/next segments, location evidence, VLM look at a segment."""
from __future__ import annotations
import numpy as np
from .core import *
from .temporal import coalesce
from .events import detect_bed_events

class TemporalAgent:
    def __init__(self, cfg, frame_states, video=None, bed=None, trace=None):
        self.cfg, self.fs, self.video, self.bed = cfg, frame_states, video, bed
        self.trace = trace or Trace()
        self.ts = np.array([f.t for f in frame_states]) if frame_states else np.array([])

    # Tools
    def evidence(self, t0, t1):
        """Tool: is the person on the bed during [t0,t1)? (uses visible frames only)"""
        sel = [f for f in self.fs if t0 <= f.t < t1 and "bed_frac" in f.feats]
        if not sel:
            return {"mean_bed_frac": 0.0, "n": 0}
        return {"mean_bed_frac": float(np.mean([f.feats["bed_frac"] for f in sel])), "n": len(sel)}

    # uncertain segments
    def resolve_uncertain(self, segs):
        cfg, out = self.cfg, [s.copy() for s in segs]
        for i, s in enumerate(out):
            if not (s.state == UNKNOWN or s.conf < cfg.low_conf):
                continue
            prev = out[i - 1] if i > 0 else None
            nxt = out[i + 1] if i + 1 < len(out) else None
            self.trace.add(t=s.start, observation=f"{s.state} for {s.duration:.0f}s (conf {s.conf:.2f})",
                           thought="Evidence too weak to trust; check the neighbouring segments.",
                           action="look at previous and next segment",
                           result=f"prev={prev.state if prev else None}, next={nxt.state if nxt else None}")
            # 1) same state on both sides + short => temporary occlusion / noise
            if prev and nxt and prev.state == nxt.state and prev.state != UNKNOWN and s.duration <= cfg.unknown_bridge_sec:
                self.trace.add(conclusion=f"Temporary occlusion: bridge to {prev.state}")
                s.state, s.note, s.conf = prev.state, "bridged-occlusion", 0.6
                continue
            # 2) hidden between two in-bed states => still in bed under the blanket
            if prev and nxt and prev.state in IN_BED_STATES and nxt.state in IN_BED_STATES \
                    and s.duration <= cfg.unknown_bridge_sec:
                new = LYING_IN_BED if LYING_IN_BED in (prev.state, nxt.state) else SITTING_ON_BED
                self.trace.add(conclusion=f"Hidden between two in-bed states: label {new}")
                s.state, s.note, s.conf = new, "hidden-in-bed", 0.5
                continue

            self.trace.add(conclusion="Keep UNKNOWN (insufficient evidence) rather than guess")
            if s.state != UNKNOWN and s.conf < cfg.low_conf:
                s.state, s.note = UNKNOWN, "low-confidence"
        return coalesce(out)

    # events
    def detect_events(self, segs):
        return detect_bed_events(segs, self.cfg, evidence_fn=self.evidence, trace=self.trace)
