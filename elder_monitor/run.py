"""CLI:  python -m elder_monitor.run --video v.mp4 --out outputs/run1 [--gt gt.csv] [--annotated]"""
from __future__ import annotations
import argparse, json, os
from .config import load_config
from .core import *
from .perception import video_meta, load_bed, extract_observations
from .classifier import FrameClassifier
from .pipeline import analyze_states
from .report import save_all
from .vlm import make_vlm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="outputs/run")
    ap.add_argument("--gt", help="ground-truth states CSV (start,end,state)")
    ap.add_argument("--gt-events", help="optional ground-truth events CSV (time,event)")
    ap.add_argument("--annotated", action="store_true", help="write annotated mp4")
    a = ap.parse_args()

    cfg = load_config(a.config if os.path.exists(a.config) else None)
    os.makedirs(a.out, exist_ok=True)
    meta = video_meta(a.video)
    print(f"[video] {meta['duration']:.0f}s, {meta['width']}x{meta['height']}")
    bed = load_bed(cfg, a.video, a.out)
    obs = extract_observations(cfg, a.video, bed, os.path.join(a.out, "obs_cache.pkl"))
    fstates = FrameClassifier(cfg, bed).run(obs)
    res = analyze_states(fstates, meta["duration"], cfg, video=a.video, bed=bed, vlm=make_vlm(cfg))
    save_all(a.out, res["segments"], res["events"], res["summary"], res["readable"], res["trace"], fstates)

    print("\n=== TIMELINE ===\n" + "\n".join(f"{fmt_mmss(s.start)} - {fmt_mmss(s.end)} {s.state}" for s in res["segments"]))
    print("\n=== SUMMARY ===\n" + json.dumps(res["readable"], indent=2))
    print("\n=== EVENTS ===\n" + json.dumps([{k: v for k, v in e.items() if not k.endswith('_sec')} for e in res["events"]], indent=2))
    print(f"\n=== DECISION: {res['overall']} ===")
    for al in res["alerts"]:
        print(f"  {al['time']} {al['level']:<7} {al['rule']}: {al['message']}")

    if a.gt:
        from .evaluate import load_gt_segments, load_gt_events, evaluate, format_report
        from .visualize import save_failure_frames
        gt = load_gt_segments(a.gt)
        gte = load_gt_events(a.gt_events) if a.gt_events else None
        ev = evaluate(res["segments"], gt, res["events"], meta["duration"], cfg, gte)
        txt = format_report(ev)
        print("\n=== EVALUATION ===\n" + txt)
        open(os.path.join(a.out, "evaluation.txt"), "w").write(txt + "\n")
        json.dump(ev, open(os.path.join(a.out, "evaluation.json"), "w"), indent=2)
        save_failure_frames(a.video, ev["failure_cases"], obs, bed, os.path.join(a.out, "failures"))
    if a.annotated:
        from .visualize import write_annotated_video
        write_annotated_video(a.video, cfg, obs, res["segments"], bed, os.path.join(a.out, "annotated.mp4"))
    print(f"\nAll outputs in {a.out}/")


if __name__ == "__main__":
    main()
