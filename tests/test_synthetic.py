"""Offline test of the temporal/agent/event/alert/eval logic (no video, no models needed).
Run:  python tests/test_synthetic.py"""

import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from elder_monitor.config import Config
from elder_monitor.core import *
from elder_monitor.pipeline import analyze_states
from elder_monitor.evaluate import evaluate, format_report
from elder_monitor.classifier import FrameClassifier, compute_features
from elder_monitor.perception import BedRegion

random.seed(0)
cfg = Config()
DT = 0.5

# ground truth: brief stand (no exit), real exit+return, sitting-up-only, occlusion blip, unknown blip
GT = [(0, 60, LYING_IN_BED), (60, 75, SITTING_ON_BED), (75, 81, STANDING), (81, 95, SITTING_ON_BED),
      (95, 140, LYING_IN_BED), (140, 150, SITTING_ON_BED), (150, 154, STANDING), (154, 200, WALKING),
      (200, 260, SITTING_OUTSIDE_BED), (260, 275, WALKING), (275, 285, SITTING_ON_BED), (285, 360, LYING_IN_BED)]
T = GT[-1][1]
gt_segs = [Segment(s, a, b) for a, b, s in GT]
truth = lambda t: next(s for a, b, s in GT if a <= t < b)

frames = []
for i in range(int(T / DT)):
    t = i * DT
    st = truth(t)
    r = random.random()
    if 110 <= t < 125:                      # blanket occlusion: weak UNKNOWN evidence in the middle of lying
        sc = mk_scores(UNKNOWN, 0.5)
    elif r < 0.08:                          # random wrong frame
        wrong = random.choice([s for s in STATES if s != st])
        sc = mk_scores(wrong, 0.6)
    else:
        sc = mk_scores(st, random.uniform(0.6, 0.92))
    m = max(sc, key=sc.get)
    persons = 2 if 170 <= t < 180 else 1    # caregiver passes by while walking
    frames.append(FrameState(t, m, sc[m], sc, "", persons, {"bed_frac": 0.9 if st in IN_BED_STATES else 0.02,
                                                            "feet_off_bed": st == SITTING_ON_BED}))

res = analyze_states(frames, T, cfg)
print("TIMELINE"); print("\n".join(f"  {fmt_mmss(s.start)}-{fmt_mmss(s.end)} {s.state} {('('+s.note+')') if s.note else ''}" for s in res["segments"]))
print("EVENTS"); [print("  ", {k: v for k, v in e.items() if not k.endswith('_sec')}) for e in res["events"]]
print("OVERALL:", res["overall"], res["alerts"])
print(res["summary"]["activity_duration_sec"])

ev = evaluate(res["segments"], gt_segs, res["events"], T, cfg)
print(format_report(ev))

exits = [e for e in res["events"] if e["event"] == "bed_exit"]
rets = [e for e in res["events"] if e["event"] == "bed_return"]
assert len(exits) == 1, f"expected exactly 1 exit (brief stand must be rejected), got {len(exits)}"
assert len(rets) == 1, f"expected 1 return, got {len(rets)}"
assert abs(exits[0]["start_sec"] - 150) <= 6, exits[0]
assert ev["frame_accuracy"] > 0.9, ev["frame_accuracy"]
assert ev["bed_events"]["bed_exit"]["precision"] == 1.0 and ev["bed_events"]["bed_exit"]["recall"] == 1.0
print("\nSYNTHETIC PIPELINE TEST PASSED")

# Per-frame classifier on hand-made keypoints
bed = BedRegion([[100, 300], [500, 300], [500, 450], [100, 450]], (640, 480))


def kp(sh, hip, knee, ank, c=0.9):
    k = np.zeros((17, 3)); k[:, 2] = c
    for i in (5, 6): k[i, :2] = sh
    for i in (11, 12): k[i, :2] = hip
    for i in (13, 14): k[i, :2] = knee
    for i in (15, 16): k[i, :2] = ank
    k[0, :2] = sh
    return k


def classify(kpts, bbox):
    fc = FrameClassifier(cfg, bed)
    return fc.step(FrameObs(t=0.0, persons=1, bbox=bbox, kpts=kpts))


lying = classify(kp((150, 370), (260, 375), (360, 378), (450, 380)), (140, 340, 460, 410))
sit_bed = classify(kp((300, 250), (300, 340), (380, 345), (385, 430)), (250, 230, 420, 440))
sit_chair = classify(kp((560, 200), (560, 290), (600, 295), (605, 380)), (540, 180, 630, 390))
stand = classify(kp((560, 100), (560, 200), (562, 300), (562, 400)), (530, 80, 600, 410))
floor = classify(kp((20, 460), (120, 465), (210, 468), (300, 470)), (10, 440, 320, 478))
for name, r, exp in [("lying", lying, LYING_IN_BED), ("sit_bed", sit_bed, SITTING_ON_BED),
                     ("sit_chair", sit_chair, SITTING_OUTSIDE_BED), ("stand", stand, STANDING),
                     ("floor", floor, LYING_OUTSIDE_BED)]:
    print(f"  {name:<10} -> {r.state:<22} conf={r.conf:.2f}  ({r.reason})")
    assert r.state == exp, (name, r.state, exp)
print("CLASSIFIER UNIT TESTS PASSED")
