# Frame selection for one video: shot detection, windows of at most 30 s, evenly spaced candidate frames, a quality
# filter and perceptual-hash deduplication. Writes candidate frames and candidates.jsonl.
"""
PASS 0  shot proposals via PySceneDetect (region proposals only); long shots
        (>MAX_SHOT_S) force-windowed into <=WINDOW_S sub-windows.
PASS 1  adaptive candidate sampling (~1 per CAND_EVERY_S, cap CAND_CAP) decoded
        directly via cv2 seek; CV prefilter: sharpness (var-of-Laplacian),
        exposure (mean), blank floor (fraction high-contrast), soft-transition
        (neighbour mean-abs-diff).
PASS 1.5 perceptual-hash dedup, cap KEEP_PER_SHOT survivors per shot.

Output: <out_dir>/<video_id>/  candidate jpgs + candidates.jsonl (one row per
surviving candidate, with shot/window indices, timestamp, CV metrics).
"""
from tc3_vqa.paths import WORK
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import imagehash
from PIL import Image

FFMPEG = "ffmpeg"

# ---- tunables ----
SHOT_THRESHOLD = 27.0      # PySceneDetect ContentDetector
MIN_SHOT_S = 1.0           # drop shots shorter than this
MAX_SHOT_S = 60.0          # shots longer than this are force-windowed
WINDOW_S = 30.0            # force-window size for long shots
CAND_EVERY_S = 4.0         # ~1 candidate per N seconds of (sub)window
CAND_CAP = 12              # max candidates per (sub)window before prefilter
KEEP_PER_SHOT = 6          # max survivors per (sub)window after dedup

SHARPNESS_MIN = 40.0       # var-of-Laplacian below this = too blurry
EXPOSURE_MIN = 15.0        # mean below = near-black
EXPOSURE_MAX = 240.0       # mean above = blown out
HIGH_CONTRAST_MIN = 0.02   # fraction of pixels far from mean colour (blank floor)
PHASH_HAMMING = 6          # dedup threshold (<=this = near-duplicate)


def shot_list(video_path: str, downscale: int, frame_skip: int):
    """PASS 0: PySceneDetect shots as (start_s, end_s). Uses downscale +
    frame_skip so multi-hour videos detect in reasonable time."""
    from scenedetect import open_video, SceneManager, ContentDetector
    video = open_video(video_path)
    if downscale and downscale > 1:
        video.downscale = downscale
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=SHOT_THRESHOLD))
    sm.detect_scenes(video, show_progress=False, frame_skip=frame_skip)
    scenes = sm.get_scene_list()
    out = []
    if not scenes:
        # whole video is one shot — return its full span
        dur = video.duration.get_seconds() if video.duration else 0.0
        return [(0.0, dur)] if dur > 0 else []
    for s, e in scenes:
        out.append((s.seconds, e.seconds))
    return out


def force_window(shots):
    """Split shots longer than MAX_SHOT_S into <=WINDOW_S sub-windows.
    Returns list of (shot_idx, win_idx, start_s, end_s, was_split)."""
    windows = []
    for si, (s, e) in enumerate(shots):
        dur = e - s
        if dur < MIN_SHOT_S:
            continue
        if dur <= MAX_SHOT_S:
            windows.append((si, 0, s, e, False))
        else:
            n = int(np.ceil(dur / WINDOW_S))
            for wi in range(n):
                ws = s + wi * WINDOW_S
                we = min(e, ws + WINDOW_S)
                if we - ws >= MIN_SHOT_S:
                    windows.append((si, wi, ws, we, True))
    return windows


def ffmpeg_grab(video_path: str, t: float):
    """Decode a single frame at time t via ffmpeg (codec-agnostic SW decode,
    e.g. AV1/dav1d that OpenCV can't read). Returns BGR ndarray or None."""
    try:
        proc = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}",
             "-i", video_path, "-frames:v", "1", "-f", "image2pipe",
             "-vcodec", "mjpeg", "pipe:1"],
            capture_output=True, timeout=60,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        arr = np.frombuffer(proc.stdout, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


class FrameGrabber:
    """Grab frames by timestamp; uses OpenCV seek, falls back to ffmpeg per-frame
    when the codec is unreadable by OpenCV (AV1)."""
    def __init__(self, video_path: str):
        self.path = video_path
        self.cap = cv2.VideoCapture(video_path)
        ok, _ = self.cap.read()
        self.use_ffmpeg = not ok
        if self.use_ffmpeg:
            print(f"[grabber] OpenCV cannot decode {Path(video_path).name} -> ffmpeg fallback", flush=True)

    def grab(self, t: float):
        if not self.use_ffmpeg:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return frame
            # transient cv2 failure -> try ffmpeg once
            return ffmpeg_grab(self.path, t)
        return ffmpeg_grab(self.path, t)

    def release(self):
        self.cap.release()


def cv_metrics(bgr):
    """Sharpness, exposure, blank-floor metrics on a frame (BGR)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
    mean = float(small.mean())
    # blank floor: fraction of pixels far from frame mean colour
    rgb = cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)
    mean_rgb = rgb.mean(axis=(0, 1))
    dist = np.linalg.norm(rgb - mean_rgb, axis=2)
    frac_hc = float((dist > 40).mean())
    return sharp, mean, frac_hc


def prefilter(sharp, mean, frac_hc):
    """Return reject reason or '' if passes PASS 1 CV gate."""
    if mean < EXPOSURE_MIN:
        return "near_black"
    if mean > EXPOSURE_MAX:
        return "blown_out"
    if frac_hc < HIGH_CONTRAST_MIN:
        return "blank_uniform"
    if sharp < SHARPNESS_MIN:
        return "blurry"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--out-dir", default=WORK + "/smoke_frames")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--frame-skip", type=int, default=0,
                    help="0 = every frame; >0 skips frames during shot detection (for long videos)")
    args = ap.parse_args()

    out = Path(args.out_dir) / args.video_id
    out.mkdir(parents=True, exist_ok=True)

    # Auto frame-skip for very long videos so PySceneDetect finishes in reasonable time.
    cap0 = cv2.VideoCapture(args.video)
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap0.release()
    vdur = nframes / fps if fps else 0
    frame_skip = args.frame_skip
    if frame_skip == 0 and vdur > 1800:   # >30 min
        frame_skip = max(1, int(fps))      # ~1 sampled frame per second for detection
        print(f"[select] long video ({vdur:.0f}s) -> auto frame_skip={frame_skip}", flush=True)

    print(f"[select] {args.video_id} — PASS 0 shot detection (downscale={args.downscale}, frame_skip={frame_skip})", flush=True)
    shots = shot_list(args.video, args.downscale, frame_skip)
    print(f"[select] {len(shots)} raw shots", flush=True)
    windows = force_window(shots)
    n_split = sum(1 for w in windows if w[4])
    print(f"[select] {len(windows)} windows after force-windowing ({n_split} from long-shot splits)", flush=True)

    grabber = FrameGrabber(args.video)

    candidates = []
    counts = {"sampled": 0, "near_black": 0, "blown_out": 0, "blank_uniform": 0,
              "blurry": 0, "passed": 0, "deduped": 0, "kept": 0}

    for (si, wi, ws, we, was_split) in windows:
        dur = we - ws
        n_cand = min(CAND_CAP, max(1, int(dur / CAND_EVERY_S)))
        # evenly spaced timestamps inside the window, avoiding the exact boundaries
        ts = np.linspace(ws, we, n_cand + 2)[1:-1] if n_cand > 1 else np.array([(ws + we) / 2])

        survivors = []  # (ts, metrics, phash, bgr)
        for t in ts:
            counts["sampled"] += 1
            frame = grabber.grab(t)
            if frame is None:
                continue
            sharp, mean, frac_hc = cv_metrics(frame)
            reason = prefilter(sharp, mean, frac_hc)
            if reason:
                counts[reason] += 1
                continue
            counts["passed"] += 1
            ph = imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            survivors.append({"ts": float(t), "sharp": sharp, "mean": mean,
                              "frac_hc": frac_hc, "phash": ph, "bgr": frame})

        # PASS 1.5 dedup (greedy by sharpness desc)
        survivors.sort(key=lambda d: -d["sharp"])
        kept = []
        for s in survivors:
            if any((s["phash"] - k["phash"]) <= PHASH_HAMMING for k in kept):
                counts["deduped"] += 1
                continue
            kept.append(s)
            if len(kept) >= KEEP_PER_SHOT:
                break

        for ki, s in enumerate(kept):
            ms = int(s["ts"] * 1000)
            fname = f"shot{si:04d}_w{wi:02d}_c{ki:02d}_t{ms:08d}.jpg"
            fpath = out / fname
            cv2.imwrite(str(fpath), s["bgr"], [cv2.IMWRITE_JPEG_QUALITY, 90])
            counts["kept"] += 1
            candidates.append({
                "video_id": args.video_id,
                "shot_idx": si, "window_idx": wi, "cand_idx": ki,
                "was_long_split": was_split,
                "window_start_s": round(ws, 2), "window_end_s": round(we, 2),
                "ts_s": round(s["ts"], 2),
                "sharpness": round(s["sharp"], 1), "exposure_mean": round(s["mean"], 1),
                "frac_high_contrast": round(s["frac_hc"], 4),
                "frame_path": str(fpath),
            })

    grabber.release()

    with open(out / "candidates.jsonl", "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

    print(f"[select] {args.video_id} DONE", flush=True)
    for k, v in counts.items():
        print(f"    {k:14s} {v}", flush=True)
    print(f"[select] wrote {len(candidates)} candidates -> {out}/candidates.jsonl", flush=True)


if __name__ == "__main__":
    main()
