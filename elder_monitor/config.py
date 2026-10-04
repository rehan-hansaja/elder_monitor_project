"""All tunable thresholds in one place (override via config.yaml)."""
from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class Config:
    # --- sampling / models
    sample_fps: float = 2.0
    yolo_pose_model: str = "yolov8m-pose.pt"
    yolo_det_model: str = "yolov8m.pt"      # only used to auto-find the bed
    det_conf: float = 0.25
    imgsz: int = 640
    kpt_conf: float = 0.3
    reacquire_sec: float = 10.0             # re-pick the patient after this long without a match
    # --- bed region: list of normalised [x,y] points (0-1). None => auto-detect
    bed_polygon: Optional[list] = None
    # --- per-frame rules
    torso_horizontal_deg: float = 55.0
    torso_upright_deg: float = 40.0
    standing_knee_ratio: float = 0.55       # (knee_y-hip_y)/torso_len; >= this => legs extended down
    walk_speed_bh_s: float = 0.15           # body-heights per second
    speed_window_sec: float = 2.0
    bed_frac_in: float = 0.5                # bbox overlap with bed => "on bed"
    bed_frac_out: float = 0.2
    min_vis: float = 0.25                   # mean keypoint conf below this => UNKNOWN
    bed_occlusion_carry_sec: float = 60.0   # keep in-bed state while person is hidden (blanket)
    out_of_view_after_sec: float = 3.0      # no detection this long after being outside => OUT_OF_BED
    # --- temporal model
    trans_cost: float = 4.0
    unk_trans_cost: float = 5.0
    bad_trans_cost: float = 12.0
    min_segment_sec: float = 2.0
    low_conf: float = 0.45
    unknown_bridge_sec: float = 60.0
    # --- bed events
    min_out_sec: float = 10.0               # out shorter than this => "stood up and sat back down"
    min_in_sec: float = 8.0                 # in-bed stretch needed to count as a return
    standing_only_exit_sec: float = 30.0
    away_bed_frac: float = 0.15
    exit_confirm_lag_sec: float = 2.0
    # --- alert rules (demo values; scale up for real 8-hour night videos)
    monitor_out_sec: float = 120.0
    alert_out_sec: float = 300.0
    monitor_edge_sec: float = 120.0
    monitor_unknown_sec: float = 30.0
    alert_floor_sec: float = 5.0
    caregiver_grace_sec: float = 5.0

def load_config(path=None) -> Config:
    cfg = Config()
    if path:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(Config)}
        for k, v in data.items():
            if k not in valid:
                raise KeyError(f"Unknown config key: {k}")
            setattr(cfg, k, v)
    return cfg
