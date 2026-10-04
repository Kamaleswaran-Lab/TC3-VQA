# Scores the candidate frames of each window jointly with a VLM served by vLLM: action phase and usefulness per
# frame, and a keep or drop verdict per window.
"""
Reads <video_id>/candidates.jsonl (from select_frames.py), groups by
(shot_idx, window_idx), scores each group in one multi-image call, writes
<video_id>/scored.jsonl and a window-level <video_id>/windows.jsonl with the
DROP / keep verdict.

A window is DROPPED when no frame scores >= KEEP_SCORE with an informative,
medically-relevant action_phase.
"""
from tc3_vqa.paths import WORK
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

KEEP_SCORE = 3            # frame score (0-5) threshold to keep
MAX_WINDOW_K = 4          # max frames kept per window for the QA unit
GROUP_IMG_CAP = 6         # max images per multi-image VLM call
IMG_MAX_SIDE = 896        # downscale frames before VLM (cap image tokens; raw is up to 2560x1440)


def load_resized(path):
    """Open + downscale so the long side <= IMG_MAX_SIDE (keeps VLM token count bounded)."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = IMG_MAX_SIDE / max(w, h)
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return im

SYSTEM_PROMPT = """You are curating frames for a Tactical Combat Casualty Care (TCCC) visual-QA
dataset. You are shown several frames sampled from ONE short window of a video, in time order. For
EACH frame, judge whether it is a useful, unambiguous still for asking a TCCC question.

Be strict. A frame is only "informative" if a medical/casualty element is clearly visible IN THAT
FRAME (a wound, a tourniquet/CAT, a chest seal, an NPA/airway, IV/IO, packing, a casualty being
treated). Soldiers standing in gear, talking heads, slides, animations/diagrams, logos, transitions,
and crowd/landscape shots are NOT informative.

For each frame report its action_phase:
  active_procedure    — a TCCC intervention is visibly being performed right now
  post_procedure_result — the result of an intervention is visible (e.g. applied tourniquet) but no action ongoing
  assessment_only     — a casualty/medic is present but no specific intervention is visible
  no_medical_action   — nothing medically relevant (gear, talking, slide, animation, scenery)

Output ONLY a JSON array, one object per frame in the order shown:
[{"idx":0,"informative":true,"visible_evidence":"<short literal noun phrase of the medical element visible, or 'none'>","action_phase":"active_procedure|post_procedure_result|assessment_only|no_medical_action","occlusion":"none|partial|heavy","score":0-5}, ...]
score: 5 = textbook-clear active procedure; 0 = nothing usable."""


def parse_json_array(text: str):
    if not text:
        return None
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default=WORK + "/smoke_frames")
    ap.add_argument("--video-ids", nargs="+", default=None)
    ap.add_argument("--video-list", default=None, help="file with one video_id per line (avoids argparse hyphen issue)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()
    if args.video_list:
        args.video_ids = [x.strip() for x in open(args.video_list) if x.strip()]

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.88,
              trust_remote_code=True, dtype="bfloat16",
              limit_mm_per_prompt={"image": GROUP_IMG_CAP})
    print(f"[score] vLLM loaded in {time.time()-t0:.0f}s", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Build one request per (video, shot, window) group
    requests = []   # (video_id, key, [cand rows], [PIL images])
    for vid in args.video_ids:
        cpath = Path(args.frames_dir) / vid / "candidates.jsonl"
        if not cpath.exists():
            print(f"[score] no candidates for {vid}", flush=True)
            continue
        rows = [json.loads(l) for l in open(cpath)]
        groups = defaultdict(list)
        for r in rows:
            groups[(r["shot_idx"], r["window_idx"])].append(r)
        for key, cands in groups.items():
            cands = sorted(cands, key=lambda c: c["cand_idx"])[:GROUP_IMG_CAP]
            imgs = []
            ok = []
            for c in cands:
                try:
                    imgs.append(load_resized(c["frame_path"]))
                    ok.append(c)
                except Exception:
                    pass
            if imgs:
                requests.append((vid, key, ok, imgs))

    print(f"[score] {len(requests)} window-groups to score", flush=True)

    # Build vLLM prompts
    vllm_inputs = []
    for (vid, key, cands, imgs) in requests:
        content = [{"type": "image"} for _ in imgs]
        content.append({"type": "text",
                        "text": f"Here are {len(imgs)} frames from one window, in time order. Score each."})
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}]
        prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": imgs}})

    print(f"[score] generating ...", flush=True)
    t0 = time.time()
    outs = llm.generate(vllm_inputs, sampling_params=sp)
    print(f"[score] done in {time.time()-t0:.0f}s ({(time.time()-t0)/max(1,len(outs)):.2f}s/group)", flush=True)

    # Collect per-frame scores + assemble windows
    scored_by_vid = defaultdict(list)
    windows_by_vid = defaultdict(list)
    parse_fail = 0
    for (vid, key, cands, imgs), out in zip(requests, outs):
        arr = parse_json_array(out.outputs[0].text)
        if arr is None or len(arr) < len(cands):
            parse_fail += 1
            arr = [{"idx": i, "informative": False, "visible_evidence": "none",
                    "action_phase": "no_medical_action", "occlusion": "none", "score": 0}
                   for i in range(len(cands))]
        kept = []
        for c, s in zip(cands, arr):
            rec = {**c, **{k: s.get(k) for k in
                   ("informative", "visible_evidence", "action_phase", "occlusion", "score")}}
            scored_by_vid[vid].append(rec)
            phase_ok = s.get("action_phase") in ("active_procedure", "post_procedure_result")
            if s.get("informative") and phase_ok and (s.get("score") or 0) >= KEEP_SCORE:
                kept.append(rec)
        # window verdict
        kept.sort(key=lambda r: (-(r.get("score") or 0),
                                 0 if r.get("action_phase") == "active_procedure" else 1))
        window = kept[:MAX_WINDOW_K]
        windows_by_vid[vid].append({
            "video_id": vid, "shot_idx": key[0], "window_idx": key[1],
            "verdict": "keep" if window else "drop",
            "n_candidates": len(cands), "n_kept": len(window),
            "window_frames": [w["frame_path"] for w in window],
            "max_score": max([(r.get("score") or 0) for r in (kept or [{"score": 0}])]),
        })

    for vid in args.video_ids:
        d = Path(args.frames_dir) / vid
        if not d.exists():
            continue
        with open(d / "scored.jsonl", "w") as f:
            for r in scored_by_vid[vid]:
                f.write(json.dumps(r) + "\n")
        with open(d / "windows.jsonl", "w") as f:
            for w in windows_by_vid[vid]:
                f.write(json.dumps(w) + "\n")
        kept_w = sum(1 for w in windows_by_vid[vid] if w["verdict"] == "keep")
        print(f"[score] {vid}: {len(windows_by_vid[vid])} windows -> {kept_w} keep, "
              f"{len(windows_by_vid[vid])-kept_w} drop; {len(scored_by_vid[vid])} frames scored", flush=True)

    print(f"[score] parse failures: {parse_fail}/{len(requests)}", flush=True)


if __name__ == "__main__":
    main()
