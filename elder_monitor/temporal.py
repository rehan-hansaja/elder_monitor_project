"""Temporal state tracking: Viterbi decoding over a state-transition graph, then segment building."""
from __future__ import annotations
import numpy as np
from .core import *

# physically plausible direct transitions (symmetric);
_EDGES = [
    (LYING_IN_BED, SITTING_ON_BED),
    (SITTING_ON_BED, STANDING), (SITTING_ON_BED, SITTING_OUTSIDE_BED),
    (SITTING_ON_BED, WALKING), (SITTING_ON_BED, OUT_OF_BED),
    (STANDING, WALKING), (STANDING, SITTING_OUTSIDE_BED), (STANDING, OUT_OF_BED),
    (STANDING, LYING_OUTSIDE_BED),
    (WALKING, SITTING_OUTSIDE_BED), (WALKING, OUT_OF_BED), (WALKING, LYING_OUTSIDE_BED),
    (SITTING_OUTSIDE_BED, OUT_OF_BED), (SITTING_OUTSIDE_BED, LYING_OUTSIDE_BED),
    (OUT_OF_BED, LYING_OUTSIDE_BED),
]


def transition_costs(cfg) -> np.ndarray:
    idx = {s: i for i, s in enumerate(STATES)}
    K = len(STATES)
    C = np.full((K, K), cfg.bad_trans_cost)
    for a, b in _EDGES:
        C[idx[a], idx[b]] = C[idx[b], idx[a]] = cfg.trans_cost
    u = idx[UNKNOWN]
    C[u, :] = C[:, u] = cfg.unk_trans_cost
    np.fill_diagonal(C, 0.0)
    return C


def viterbi_decode(frames, cfg):
    """Most likely state sequence given per-frame scores (emissions) and the transition graph."""
    T, K = len(frames), len(STATES)
    if T == 0:
        return []
    E = np.array([[-np.log(max(f.scores.get(s, 0.0), 1e-3)) for s in STATES] for f in frames])
    C = transition_costs(cfg)
    dp = np.zeros((T, K)); bp = np.zeros((T, K), dtype=int)
    dp[0] = E[0]
    for t in range(1, T):
        cand = dp[t - 1][:, None] + C
        bp[t] = cand.argmin(0)
        dp[t] = cand.min(0) + E[t]
    path = [int(dp[-1].argmin())]
    for t in range(T - 1, 0, -1):
        path.append(int(bp[t][path[-1]]))
    return [STATES[i] for i in reversed(path)]


def build_segments(frames, path, duration):
    """Collapse the per-frame path into segments. Boundaries fall on frame timestamps."""
    segs = []
    for i, (f, s) in enumerate(zip(frames, path)):
        start = 0.0 if i == 0 else f.t
        end = frames[i + 1].t if i + 1 < len(frames) else max(duration, f.t)
        c = f.scores.get(s, 0.0)
        if segs and segs[-1].state == s:
            n_prev = getattr(segs[-1], "_n", 1)
            segs[-1].conf = (segs[-1].conf * n_prev + c) / (n_prev + 1)
            segs[-1]._n = n_prev + 1
            segs[-1].end = end
        else:
            sg = Segment(s, start, end, c)
            sg._n = 1
            segs.append(sg)
    return segs


def coalesce(segs):
    out = []
    for s in segs:
        if out and out[-1].state == s.state:
            a = out[-1]
            tot = a.duration + s.duration
            a.conf = (a.conf * a.duration + s.conf * s.duration) / tot if tot > 0 else a.conf
            a.end = s.end
            a.note = a.note or s.note
        else:
            out.append(s.copy())
    return out


def merge_short(segs, cfg):
    """Absorb segments shorter than min_segment_sec into a neighbour."""
    segs = coalesce(segs)
    while len(segs) > 1:
        short = [i for i, s in enumerate(segs) if s.duration < cfg.min_segment_sec]
        if not short:
            break
        i = min(short, key=lambda k: segs[k].duration)
        if i == 0:
            tgt = 1
        elif i == len(segs) - 1:
            tgt = i - 1
        elif segs[i - 1].state == segs[i + 1].state:
            tgt = i - 1
        else:
            tgt = i - 1 if segs[i - 1].duration >= segs[i + 1].duration else i + 1
        t = segs[tgt]
        t.start, t.end = min(t.start, segs[i].start), max(t.end, segs[i].end)
        segs.pop(i)
        segs = coalesce(segs)
    return segs
