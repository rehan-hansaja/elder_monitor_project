"""Per-frame posture reasoning: pose keypoints + bed region + short motion history -> state scores.
This is deliberately *soft*: it outputs a distribution; the temporal model (Viterbi) decides."""
from __future__ import annotations
import math
from collections import deque
import numpy as np
from .core import *

L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 5, 6, 11, 12, 13, 14, 15, 16


def _mid(k, idx, kc):
    pts = [k[i][:2] for i in idx if k[i][2] >= kc]
    return np.mean(pts, axis=0) if pts else None


def compute_features(obs: FrameObs, bed, kc: float) -> dict:
    x1, y1, x2, y2 = obs.bbox
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    f = dict(bbox_w=w, bbox_h=h, aspect=w / h, cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
             body_h=max(w, h), bed_frac=bed.frac_bbox(obs.bbox), torso_angle=None,
             knee_ratio=None, hip_in_bed=None, feet_off_bed=None, kpt_vis=0.0)
    k = obs.kpts
    if k is None:
        return f
    f["kpt_vis"] = float(np.mean(k[[0, 5, 6, 11, 12, 13, 14, 15, 16], 2]))
    sh, hip = _mid(k, [L_SH, R_SH], kc), _mid(k, [L_HIP, R_HIP], kc)
    knee, ank = _mid(k, [L_KNEE, R_KNEE], kc), _mid(k, [L_ANK, R_ANK], kc)
    if sh is not None and hip is not None:
        d = sh - hip
        L = float(np.hypot(*d)) + 1e-6
        f["torso_angle"] = math.degrees(math.atan2(abs(d[0]), abs(d[1]) + 1e-6))   # 0 = upright, 90 = flat
        f["torso_len"] = L
        if knee is not None:
            f["knee_ratio"] = float((knee[1] - hip[1]) / L)      # ~1 standing, ~0 sitting
    if hip is not None:
        f["hip_in_bed"] = bool(bed.contains(hip))
        f["cx"], f["cy"] = float(hip[0]), float(hip[1])
    ank_pts = [k[i][:2] for i in (L_ANK, R_ANK) if k[i][2] >= kc]
    if ank_pts:
        f["feet_off_bed"] = bool(np.mean([bed.contains(p) for p in ank_pts]) < 0.5)
    return f


class FrameClassifier:
    def __init__(self, cfg, bed):
        self.cfg, self.bed = cfg, bed
        self.hist = deque()
        self.last_seen_t = None
        self.last_state = UNKNOWN
        self.last_bed_frac = 0.0
        self.ref_h = None                 # running estimate of standing bbox height

    # public
    def run(self, observations):
        return [self.step(o) for o in observations]

    def step(self, obs: FrameObs) -> FrameState:
        if obs.bbox is None:
            return self._no_person(obs)
        f = compute_features(obs, self.bed, self.cfg.kpt_conf)
        f["speed"] = self._speed(obs.t, f)
        state, scores, reason = self._classify(f)
        if state in (WALKING, STANDING) and f["knee_ratio"] is not None and scores[state] > 0.6:
            self.ref_h = f["bbox_h"] if self.ref_h is None else 0.95 * self.ref_h + 0.05 * f["bbox_h"]
        self.last_seen_t, self.last_state, self.last_bed_frac = obs.t, state, f["bed_frac"]
        keep = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()
                if k in ("bed_frac", "torso_angle", "knee_ratio", "speed", "feet_off_bed", "hip_in_bed", "kpt_vis")}
        return FrameState(obs.t, state, scores[state], scores, reason, obs.persons, keep)

    # helpers
    def _speed(self, t, f):
        self.hist.append((t, f["cx"], f["cy"], f["body_h"]))
        while self.hist and self.hist[0][0] < t - self.cfg.speed_window_sec:
            self.hist.popleft()
        if len(self.hist) < 2:
            return 0.0
        t0, x0, y0, h0 = self.hist[0]
        dt = t - t0
        if dt < 0.5:
            return 0.0
        hm = (h0 + f["body_h"]) / 2
        return math.hypot(f["cx"] - x0, f["cy"] - y0) / hm / dt + 0.5 * abs(f["body_h"] - h0) / hm / dt

    def _fs(self, obs, state, conf, reason, secondary=None):
        sc = mk_scores(state, conf, secondary)
        return FrameState(obs.t, max(sc, key=sc.get), sc[max(sc, key=sc.get)], sc, reason, obs.persons)

    def _no_person(self, obs):
        c = self.cfg
        if self.last_seen_t is None:
            return self._fs(obs, UNKNOWN, 0.5, "no person seen yet")
        miss, ls = obs.t - self.last_seen_t, self.last_state
        if ls in IN_BED_STATES and self.last_bed_frac >= 0.4 and miss <= c.bed_occlusion_carry_sec:
            conf = max(0.35, 0.7 - 0.3 * miss / c.bed_occlusion_carry_sec)
            return self._fs(obs, ls, conf, f"hidden {miss:.0f}s; last seen on bed -> blanket/occlusion")
        if ls in OUT_STATES and miss <= c.out_of_view_after_sec:
            return self._fs(obs, ls, 0.5, "brief occlusion, carried forward")
        if ls in OUT_STATES:
            return self._fs(obs, OUT_OF_BED, 0.65, f"not visible for {miss:.0f}s after being off-bed -> left view")
        return self._fs(obs, UNKNOWN, 0.5, f"not visible for {miss:.0f}s, cannot infer")

    # rules
    def _classify(self, f):
        c = self.cfg
        vis, bed, speed, ang, kr = f["kpt_vis"], f["bed_frac"], f["speed"], f["torso_angle"], f["knee_ratio"]
        S = lambda st, cf, why, sec=None: (st, mk_scores(st, cf, sec), why)

        if f["torso_angle"] is not None and vis < c.min_vis:
            return S(UNKNOWN, 0.5, f"keypoints too weak (vis={vis:.2f})")

        # 1) posture from torso orientation (fallback: bbox aspect)
        if ang is not None:
            if ang >= c.torso_horizontal_deg:
                posture, pc = "horizontal", min(1.0, 0.6 + (ang - c.torso_horizontal_deg) / 60)
            elif ang <= c.torso_upright_deg:
                posture, pc = "upright", min(1.0, 0.6 + (c.torso_upright_deg - ang) / 60)
            else:
                posture, pc = "ambiguous", 0.45
            pc *= 0.6 + 0.4 * min(1.0, vis / 0.6)
        else:
            a = f["aspect"]
            posture, pc = (("horizontal", 0.5) if a > 1.4 else ("upright", 0.5) if a < 0.8 else ("ambiguous", 0.35))

        on_bed = bed >= c.bed_frac_in or bool(f["hip_in_bed"])

        # 2) horizontal -> lying in bed vs. lying elsewhere (fall cue)
        if posture == "horizontal":
            if bed >= c.bed_frac_in:
                return S(LYING_IN_BED, pc * (0.6 + 0.4 * bed), f"horizontal torso {ang and round(ang)}deg, bed overlap {bed:.2f}")
            if bed <= c.bed_frac_out:
                return S(LYING_OUTSIDE_BED, pc * 0.85, f"horizontal but only {bed:.2f} on bed -> floor/sofa")
            if f["hip_in_bed"] is True:      # partial overlap: the hip position breaks the tie
                return S(LYING_IN_BED, 0.55, f"horizontal, partial overlap ({bed:.2f}) but hips on bed", LYING_OUTSIDE_BED)
            if f["hip_in_bed"] is False:
                return S(LYING_OUTSIDE_BED, 0.55, f"horizontal, partial overlap ({bed:.2f}) and hips off bed", LYING_IN_BED)
            return S(LYING_IN_BED, 0.4, f"horizontal, partially on bed ({bed:.2f})", LYING_OUTSIDE_BED)

        # 3) upright
        if posture == "upright":
            if kr is not None:
                if kr >= c.standing_knee_ratio:
                    if speed >= c.walk_speed_bh_s:
                        return S(WALKING, pc * min(1.0, 0.7 + speed / 4), f"upright, legs extended, speed {speed:.2f} bh/s", STANDING)
                    return S(STANDING, pc * 0.85, f"upright, legs extended, speed {speed:.2f}",
                             WALKING if speed >= 0.6 * c.walk_speed_bh_s else None)
                if on_bed:
                    return S(SITTING_ON_BED, pc * 0.85, f"upright, knees bent (kr={kr:.2f}), hips on bed")
                if bed < 0.25:
                    return S(SITTING_OUTSIDE_BED, pc * 0.85, f"upright, knees bent (kr={kr:.2f}), off bed")
                return S(SITTING_ON_BED, 0.5, "sitting near bed edge", SITTING_OUTSIDE_BED)
            # knees hidden (blanket / furniture)
            if on_bed:
                return S(SITTING_ON_BED, 0.6 * pc + 0.1, "upright, legs hidden, on bed -> sitting up")
            if speed >= c.walk_speed_bh_s:
                return S(WALKING, 0.6 * pc, "upright, legs hidden, moving", STANDING)
            if self.ref_h:
                r = f["bbox_h"] / self.ref_h
                if r < 0.75:
                    return S(SITTING_OUTSIDE_BED, 0.5, f"height {r:.2f} of standing ref", STANDING)
                return S(STANDING, 0.5, f"height {r:.2f} of standing ref", SITTING_OUTSIDE_BED)
            return S(UNKNOWN, 0.5, "upright but legs hidden, no reference")

        # 4) ambiguous angle (propped up / leaning / bending)
        if on_bed:
            return S(SITTING_ON_BED, 0.5, "torso tilted on bed -> propped up", LYING_IN_BED)
        return S(UNKNOWN, 0.5, f"ambiguous posture (angle {ang and round(ang)}deg)")
