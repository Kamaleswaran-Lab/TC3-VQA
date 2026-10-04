# Regenerates frames from their source videos: downloads each video with yt-dlp, grabs the frame at the recorded
# timestamp with ffmpeg and applies the caption-mask rectangles. Age-restricted sources need --cookies.
import argparse, json, os, subprocess, sys, tempfile
from PIL import Image, ImageDraw

def download(video_id, workdir, cookies=None):
    out = os.path.join(workdir, f"{video_id}.mp4")
    if os.path.exists(out): return out
    cmd = ["yt-dlp", "-f", "bv*[height<=1080]+ba/b", "--merge-output-format", "mp4", "-o", out]
    if cookies: cmd += ["--cookies", cookies]
    cmd.append(f"https://www.youtube.com/watch?v={video_id}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out

def grab(video_path, t_s, out_path):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t_s:.3f}", "-i", video_path,
                    "-frames:v", "1", "-q:v", "2", out_path], check=True)

def apply_masks(path, rects):
    if not rects: return
    im = Image.open(path).convert("RGB"); W, H = im.size; d = ImageDraw.Draw(im)
    for x0, y0, x1, y1 in rects:
        d.rectangle([x0 * W, y0 * H, x1 * W, y1 * H], fill=(0, 0, 0))
    im.save(path, quality=92)

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default=os.path.join(here, "..", "data", "items.jsonl"))
    ap.add_argument("--out", default=os.path.join(here, "..", "frames"))
    ap.add_argument("--workdir", default=None, help="cache for downloaded videos (default: temp dir)")
    ap.add_argument("--cookies", default=None, help="cookies.txt for age-restricted sources")
    ap.add_argument("--video", default=None, help="only this video_id")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    workdir = a.workdir or tempfile.mkdtemp(prefix="tc3_vqa_")
    todo = {}
    for line in open(a.core):
        r = json.loads(line)
        if a.video and r["video_id"] != a.video: continue
        for ref in r["frame_refs"]:
            if ref.get("local_path"): continue  # pixels already shipped (public-domain / CC sources)
            todo.setdefault(r["video_id"], {})[ref["frame_id"]] = (ref["timestamp_s"], ref.get("mask_rects", []), r.get("requires_login", False))
    n_ok = n_fail = 0
    for vid, frames in todo.items():
        try:
            vp = download(vid, workdir, a.cookies)
        except subprocess.CalledProcessError:
            need = any(v[2] for v in frames.values())
            print(f"[skip] {vid}: download failed{' (age-restricted: pass --cookies)' if need else ''}", file=sys.stderr)
            n_fail += len(frames); continue
        for fid, (t, rects, _) in frames.items():
            outp = os.path.join(a.out, fid)
            if os.path.exists(outp): n_ok += 1; continue
            try:
                grab(vp, t, outp); apply_masks(outp, rects); n_ok += 1
            except subprocess.CalledProcessError:
                print(f"[fail] {fid}", file=sys.stderr); n_fail += 1
    print(f"frames written: {n_ok}, failed: {n_fail}")

if __name__ == "__main__":
    main()
