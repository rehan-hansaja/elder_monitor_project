"""Optional Vision-Language-Model tool, used ONLY by the agent for ambiguous segments."""
from __future__ import annotations
import base64, json, os, re
from .core import STATES, fmt_ts

PROMPT = """You are analysing frames from a fixed indoor camera watching an elderly person. The yellow polygon marks the bed.
Frames are in chronological order at these times (seconds): {times}.
Choose the person's state for this period. Allowed states:
LYING_IN_BED, SITTING_ON_BED, SITTING_OUTSIDE_BED, STANDING, WALKING, OUT_OF_BED (not in view / away from bed),
LYING_OUTSIDE_BED, UNKNOWN (insufficient evidence - prefer this over guessing).
Reply with ONLY a JSON object: {{"state": "...", "on_bed": true/false, "confidence": 0-1, "reason": "short"}}"""


class NullVLM:
    available = False

    def classify(self, video, times, bed):
        return None


class AnthropicVLM:
    available = True

    def __init__(self, model):
        import anthropic
        self.client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
        self.model = model

    def classify(self, video, times, bed):
        import cv2
        from .perception import read_frame_at
        content = []
        for t in times:
            fr = read_frame_at(video, t)
            if fr is None:
                continue
            if bed is not None:
                fr = bed.draw(fr.copy())
            h, w = fr.shape[:2]
            if w > 768:
                fr = cv2.resize(fr, (768, int(h * 768 / w)))
            ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": base64.b64encode(buf).decode()}})
        if not content:
            return None
        content.append({"type": "text", "text": PROMPT.format(times=", ".join(f"{t:.0f}" for t in times))})
        msg = self.client.messages.create(model=self.model, max_tokens=300,
                                          messages=[{"role": "user", "content": content}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        m = re.search(r"\{.*\}", text, re.S)
        try:
            out = json.loads(m.group(0))
        except Exception:
            return None
        out["state"] = str(out.get("state", "UNKNOWN")).upper()
        if out["state"] not in STATES:
            out["state"] = "UNKNOWN"
        return out


def make_vlm(cfg):
    if cfg.vlm_backend == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicVLM(cfg.vlm_model)
    if cfg.vlm_backend != "none":
        print("[vlm] backend requested but unavailable (missing key?) -> running without VLM")
    return NullVLM()
