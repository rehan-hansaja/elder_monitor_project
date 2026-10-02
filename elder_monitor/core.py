"""Shared types, state constants and small helpers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ---- states --------------------------------------------------------------
LYING_IN_BED = "LYING_IN_BED"
SITTING_ON_BED = "SITTING_ON_BED"
SITTING_OUTSIDE_BED = "SITTING_OUTSIDE_BED"
STANDING = "STANDING"
WALKING = "WALKING"
OUT_OF_BED = "OUT_OF_BED"            # away from bed / out of camera view
LYING_OUTSIDE_BED = "LYING_OUTSIDE_BED"  # extra state: lying on floor/sofa (fall cue)
UNKNOWN = "UNKNOWN"

STATES = [LYING_IN_BED, SITTING_ON_BED, SITTING_OUTSIDE_BED, STANDING,
          WALKING, OUT_OF_BED, LYING_OUTSIDE_BED, UNKNOWN]
IN_BED_STATES = frozenset({LYING_IN_BED, SITTING_ON_BED})
OUT_STATES = frozenset({STANDING, WALKING, SITTING_OUTSIDE_BED, OUT_OF_BED, LYING_OUTSIDE_BED})
MOVED_AWAY_STATES = frozenset({WALKING, OUT_OF_BED, SITTING_OUTSIDE_BED, LYING_OUTSIDE_BED})
SEVERITY = {"NORMAL": 0, "MONITOR": 1, "ALERT": 2}


# ---- time helpers --------------------------------------------------------
def fmt_ts(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def fmt_mmss(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def fmt_dur(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m}m {s:02d}s"


def parse_ts(s) -> float:
    """'90' | '1:30' | '01:01:30' -> seconds."""
    parts = [float(p) for p in str(s).strip().split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


# ---- data classes --------------------------------------------------------
@dataclass
class FrameObs:
    """Raw perception output for one sampled frame (patient only)."""
    t: float
    persons: int = 0                       # all persons detected (caregiver flag)
    bbox: Optional[tuple] = None           # x1,y1,x2,y2 (px)
    kpts: Optional[np.ndarray] = None      # (17,3) x,y,conf
    det_conf: float = 0.0


@dataclass
class FrameState:
    t: float
    state: str
    conf: float
    scores: dict
    reason: str = ""
    persons: int = 0
    feats: dict = field(default_factory=dict)


@dataclass
class Segment:
    state: str
    start: float
    end: float
    conf: float = 1.0
    note: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def copy(self) -> "Segment":
        return Segment(self.state, self.start, self.end, self.conf, self.note)


def mk_scores(primary: str, conf: float, secondary: Optional[str] = None,
              sec_share: float = 0.6) -> dict:
    """Turn 'I think it is X with confidence c' into a distribution over states."""
    conf = float(min(max(conf, 0.05), 0.97))
    p = {s: 0.0 for s in STATES}
    p[primary] = conf
    rest = 1.0 - conf
    if secondary and secondary != primary:
        p[secondary] += rest * sec_share
        rest *= (1 - sec_share)
    others = [s for s in STATES if s != primary]
    for s in others:
        p[s] += rest / len(others)
    return p


class Trace:
    """Human-readable log of the agent's observe -> think -> act -> conclude steps."""
    def __init__(self):
        self.steps = []

    def add(self, t=None, observation=None, thought=None, action=None, result=None, conclusion=None):
        self.steps.append({k: v for k, v in dict(
            step=len(self.steps) + 1, t=None if t is None else fmt_ts(t), observation=observation,
            thought=thought, action=action, result=result, conclusion=conclusion).items() if v is not None})

    def render(self) -> str:
        out = []
        for s in self.steps:
            out.append(f"[{s['step']:03d}]" + (f" @{s['t']}" if 't' in s else ""))
            for k in ("observation", "thought", "action", "result", "conclusion"):
                if k in s:
                    out.append(f"    {k.capitalize():<11}: {s[k]}")
        return "\n".join(out)
