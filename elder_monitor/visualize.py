"""Annotated video + failure-case frames (for the README / interview)."""
from __future__ import annotations
import os
import cv2, numpy as np
from .core import *
from .perception import iter_sampled_frames, read_frame_at

SK = [(5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (5, 7), (7, 9), (6, 8), (8, 10)]


def _draw(frame, bed, ob, label):
    if bed is not None:
        bed.draw(frame)
    if ob is not None and ob.bbox is not None:
        x1, y1, x2, y2 = map(int, ob.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if ob.kpts is not None:
            for a, b in SK:
                if ob.kpts[a][2] > 0.3 and ob.kpts[b][2] > 0.3:
                    cv2.line(frame, tuple(ob.kpts[a][:2].astype(int)), tuple(ob.kpts[b][:2].astype(int)), (255, 0, 0), 2)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30 + 22 * (label.count("\n"))), (0, 0, 0), -1)
    for i, line in enumerate(label.split("\n")):
        cv2.putText(frame, line, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def state_at(segs, t):
    for s in segs:
        if s.start <= t < s.end:
            return s.state
    return segs[-1].state if segs else UNKNOWN


def write_annotated_video(video, cfg, obs, segs, bed, path):
    w = None
    for (t, frame), ob in zip(iter_sampled_frames(video, cfg.sample_fps), obs):
        fr = _draw(frame, bed, ob, f"{fmt_mmss(t)}  {state_at(segs, t)}")
        if w is None:
            h, ww = fr.shape[:2]
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), cfg.sample_fps, (ww, h))
        w.write(fr)
    if w:
        w.release()


def save_failure_frames(video, failures, obs, bed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ts = np.array([o.t for o in obs])
    for i, f in enumerate(failures, 1):
        t = (f["start"] + f["end"]) / 2
        fr = read_frame_at(video, t)
        if fr is None:
            continue
        ob = obs[int(np.abs(ts - t).argmin())]
        fr = _draw(fr, bed, ob, f"FAIL {i} @ {fmt_mmss(t)}\nGT={f['gt']}  PRED={f['pred']}")
        cv2.imwrite(os.path.join(out_dir, f"failure_{i}_{int(t)}s.jpg"), fr)
