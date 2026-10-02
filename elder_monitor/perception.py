"""Video sampling, bed region, YOLO-pose person detection and patient selection."""
from __future__ import annotations
import json, math, os, pickle
import numpy as np
from .core import FrameObs


# ---------------------------------------------------------------- video
def video_meta(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": int(n), "duration": n / fps, "width": w, "height": h}


def iter_sampled_frames(path, sample_fps):
    """Yield (t_seconds, BGR frame) at ~sample_fps."""
    import cv2
    cap = cv2.VideoCapture(path)
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(native / sample_fps)))
    idx = 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                yield idx / native, frame
        idx += 1
    cap.release()


def read_frame_at(path, t):
    import cv2
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ---------------------------------------------------------------- bed
class BedRegion:
    def __init__(self, poly_px, wh):
        import cv2
        self.w, self.h = wh
        self.poly = np.asarray(poly_px, np.float32)
        self.mask = np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(self.mask, [self.poly.astype(np.int32)], 1)

    def contains(self, pt) -> bool:
        x, y = int(round(pt[0])), int(round(pt[1]))
        return 0 <= x < self.w and 0 <= y < self.h and self.mask[y, x] == 1

    def frac_bbox(self, bbox) -> float:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, x2 = max(0, x1), min(self.w, x2)
        y1, y2 = max(0, y1), min(self.h, y2)
        area = (x2 - x1) * (y2 - y1)
        return float(self.mask[y1:y2, x1:x2].sum() / area) if area > 0 else 0.0

    def draw(self, frame, color=(0, 200, 255)):
        import cv2
        cv2.polylines(frame, [self.poly.astype(np.int32)], True, color, 2)
        return frame


def load_bed(cfg, video, out_dir=None) -> BedRegion:
    """Bed from config polygon (normalised coords) or auto-detected with YOLO (COCO 'bed')."""
    import cv2
    meta = video_meta(video)
    w, h = meta["width"], meta["height"]
    if cfg.bed_polygon:
        poly = [[x * w, y * h] for x, y in cfg.bed_polygon]
    else:
        poly = _auto_detect_bed(cfg, video, meta)
    bed = BedRegion(poly, (w, h))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "bed.json"), "w") as f:
            json.dump({"bed_polygon_normalised": [[round(x / w, 4), round(y / h, 4)] for x, y in poly]}, f)
        fr = read_frame_at(video, 0.0)
        if fr is not None:
            cv2.imwrite(os.path.join(out_dir, "bed_preview.jpg"), bed.draw(fr))
    return bed


def _auto_detect_bed(cfg, video, meta):
    import cv2
    from ultralytics import YOLO
    det = YOLO(cfg.yolo_det_model)
    cap = cv2.VideoCapture(video)
    boxes = []
    for frac in np.linspace(0.05, 0.95, 8):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frac * meta["frames"]))
        ok, fr = cap.read()
        if not ok:
            continue
        r = det.predict(fr, conf=0.15, classes=[59], verbose=False)[0]   # 59 = bed
        if r.boxes is not None and len(r.boxes):
            i = int(r.boxes.conf.argmax())
            boxes.append(r.boxes.xyxy[i].cpu().numpy())
    cap.release()
    if not boxes:
        raise RuntimeError("No bed detected. Run `python tools/select_bed.py <video>` and paste the "
                           "polygon into config.yaml (bed_polygon).")
    x1, y1, x2, y2 = np.median(np.stack(boxes), axis=0)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# ---------------------------------------------------------------- detection
class Perception:
    def __init__(self, cfg):
        from ultralytics import YOLO
        self.cfg = cfg
        self.model = YOLO(cfg.yolo_pose_model)

    def detect(self, frame):
        r = self.model.predict(frame, conf=self.cfg.det_conf, imgsz=self.cfg.imgsz, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        kp = r.keypoints.data.cpu().numpy()          # (N,17,3)
        return [dict(bbox=tuple(map(float, xyxy[i])), conf=float(conf[i]), kpts=kp[i]) for i in range(len(conf))]


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class PatientSelector:
    """Keeps following the patient; other people are treated as caregivers/visitors.
    First lock = person overlapping the bed most. Afterwards: nearest box to the last one."""
    def __init__(self, bed, cfg):
        self.bed, self.cfg = bed, cfg
        self.last, self.last_t = None, None

    def pick(self, t, dets):
        n = len(dets)
        if n == 0:
            return None, 0
        gap = None if self.last_t is None else t - self.last_t
        if self.last is None or gap > self.cfg.reacquire_sec:
            best = max(dets, key=lambda d: (self.bed.frac_bbox(d["bbox"]),
                                            (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])))
        else:
            lx1, ly1, lx2, ly2 = self.last
            lc = ((lx1 + lx2) / 2, (ly1 + ly2) / 2)
            diag = math.hypot(lx2 - lx1, ly2 - ly1) + 1e-6
            best, best_cost, best_dist = None, 1e9, 0
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                dist = math.hypot((x1 + x2) / 2 - lc[0], (y1 + y2) / 2 - lc[1]) / diag
                cost = dist + (1 - _iou(self.last, d["bbox"]))
                if cost < best_cost:
                    best, best_cost, best_dist = d, cost, dist
            if best_dist > 1.0 + 0.5 * gap:      # too far from where the patient was -> someone else
                return None, n
        self.last, self.last_t = best["bbox"], t
        return best, n


def extract_observations(cfg, video, bed, cache_path=None):
    key = dict(v=os.path.basename(video), size=os.path.getsize(video), fps=cfg.sample_fps,
               m=cfg.yolo_pose_model, c=cfg.det_conf, sz=cfg.imgsz, bed=bed.poly.round(0).tolist())
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            blob = pickle.load(f)
        if blob.get("key") == key:
            print(f"[perception] using cached observations: {cache_path}")
            return blob["obs"]
    per, sel, obs = Perception(cfg), PatientSelector(bed, cfg), []
    for t, frame in iter_sampled_frames(video, cfg.sample_fps):
        d, n = sel.pick(t, per.detect(frame))
        obs.append(FrameObs(t=t, persons=n, bbox=d["bbox"] if d else None,
                            kpts=d["kpts"] if d else None, det_conf=d["conf"] if d else 0.0))
        if len(obs) % 100 == 0:
            print(f"[perception] {len(obs)} frames (t={t:.0f}s)")
    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump({"key": key, "obs": obs}, f)
    return obs
