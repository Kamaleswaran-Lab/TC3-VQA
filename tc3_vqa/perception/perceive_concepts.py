# Concept assignment: the perception model labels each window with one of the twelve concepts, or none, with a
# factual description and a confidence. The doctrine answer is templated from the concept inventory, not written.
"""
Input:  frame-selector windows (windows.jsonl per video) + concept_inventory.json
Per kept window:
  Qwen2.5-VL-72B sees the window's frames + the concept list (id + visual_triggers) and returns
  {visual_observation, visible_evidence, matched_concept_id ('none' allowed), match_confidence, why}.
Assembly:
  - matched concept (conf != low)  -> ANSWERABLE QA. doctrine_answer = inventory gold_quote (verbatim),
    source = inventory citation (char-offset provenance), question = inventory question_template.
  - else                            -> REFUSE QA (refusal_reason).
Refusal-gold is additionally seeded separately from rejected scene buckets (not here).

Output: pilot_qa.jsonl
"""
from tc3_vqa.paths import WORK
import argparse
import json
import re
import time
from pathlib import Path
from PIL import Image

IMG_MAX_SIDE = 896


SYSTEM_PROMPT = """You label frames for a Tactical Combat Casualty Care (TCCC) visual-QA dataset.
You are shown 1-4 frames from ONE moment of a video (time order) and a list of TCCC concepts, each with
its visual signature. Your ONLY job is to report what is VISIBLE and which ONE concept (if any) the
frames clearly depict.

You DO NOT give medical advice, treatment steps, doses, or doctrine. You only describe what you see.

Output ONE JSON object:
{
  "visual_observation": "1-2 sentence factual description of what is visible (casualty, body region, what the responder is doing)",
  "visible_evidence": "the single most specific medical element you can point to IN THE FRAME (e.g. 'CAT windlass being twisted on a bare thigh'), or 'none'",
  "matched_concept_id": "<exactly one concept_id from the provided list, or 'none'>",
  "match_confidence": "high|medium|low",
  "why": "<=15 words naming the visual signature you saw"
}

Rules:
- Choose a concept ONLY if its visual signature is clearly present. If the frames are ambiguous, show no
  clear procedure, or match no listed concept, answer matched_concept_id='none'.
- 'none' is the CORRECT answer for ambiguous or non-procedural frames. Do not guess.
- Describe only what is visible. Never state what should be done."""


def parse_json(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def load_resized(p):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    s = IMG_MAX_SIDE / max(w, h)
    if s < 1.0:
        im = im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.LANCZOS)
    return im


def build_concept_block(inv):
    lines = ["TCCC concepts (pick at most one that is clearly visible):"]
    for c in inv:
        trig = "; ".join(c["visual_triggers"][:4])
        lines.append(f"- {c['concept_id']}: {trig}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default=WORK + "/concept_inventory.json")
    ap.add_argument("--frames-dir", default=WORK + "/smoke_frames")
    ap.add_argument("--video-ids", nargs="+", default=None)
    ap.add_argument("--video-list", default=None, help="file with one video_id per line (avoids argparse hyphen issue)")
    ap.add_argument("--out", default=WORK + "/pilot_qa.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--max-model-len", type=int, default=16384)
    args = ap.parse_args()
    if args.video_list:
        args.video_ids = [x.strip() for x in open(args.video_list) if x.strip()]

    inv = json.load(open(args.inventory))
    inv_by_id = {c["concept_id"]: c for c in inv}
    concept_block = build_concept_block(inv)

    # collect kept windows
    windows = []
    for vid in args.video_ids:
        wpath = Path(args.frames_dir)/vid/"windows.jsonl"
        if not wpath.exists():
            continue
        for l in open(wpath):
            w = json.loads(l)
            if w["verdict"] == "keep" and w.get("window_frames"):
                windows.append(w)
    print(f"[gen] {len(windows)} kept windows to process", flush=True)

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.88,
              trust_remote_code=True, dtype="bfloat16", limit_mm_per_prompt={"image": 4})
    print(f"[gen] vLLM loaded in {time.time()-t0:.0f}s", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    vinputs = []
    for w in windows:
        imgs = [load_resized(p) for p in w["window_frames"][:4] if Path(p).exists()]
        content = [{"type": "image"} for _ in imgs]
        content.append({"type": "text",
                        "text": f"{concept_block}\n\nHere are {len(imgs)} frame(s) from one window, in time order. "
                                f"Report visible content and pick at most one concept."})
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vinputs.append({"prompt": prompt, "multi_modal_data": {"image": imgs}})

    print(f"[gen] perception calls ...", flush=True)
    t0 = time.time()
    outs = llm.generate(vinputs, sampling_params=sp)
    print(f"[gen] done in {time.time()-t0:.0f}s", flush=True)

    records = []
    n_ans = n_ref = n_parsefail = 0
    for w, out in zip(windows, outs):
        p = parse_json(out.outputs[0].text)
        if p is None:
            n_parsefail += 1
            continue
        cid = p.get("matched_concept_id", "none")
        conf = p.get("match_confidence", "low")
        base = {
            "video_id": w["video_id"], "shot_idx": w["shot_idx"], "window_idx": w["window_idx"],
            "frame_paths": w["window_frames"],
            "visual_observation": p.get("visual_observation"),
            "visible_evidence": p.get("visible_evidence"),
            "perception_raw": p,
        }
        if cid in inv_by_id and cid != "none" and conf in ("high", "medium"):
            c = inv_by_id[cid]
            rec = {**base,
                   "scene_eligible_for_qa": True, "refusal_reason": None,
                   "question": c["question_template"],
                   "matched_concept_id": cid,
                   "clinical_state": {"tccc_stage": c["expected_tccc_stage"]},
                   "doctrine_answer": c["gold_quote"],            # VERBATIM from corpus, NOT VLM
                   "source": {"citation_id": c["citation_id"], "source_id": c["source_id"],
                              "char_start": c["char_start"], "char_end": c["char_end"]},
                   "safety_critical": c["safety_critical"]}
            n_ans += 1
        else:
            rec = {**base,
                   "scene_eligible_for_qa": False,
                   "refusal_reason": "no_clear_concept" if cid == "none" else "low_confidence_match",
                   "question": None, "matched_concept_id": None,
                   "clinical_state": None, "doctrine_answer": None, "source": None,
                   "safety_critical": None}
            n_ref += 1
        records.append(rec)

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[gen] answerable={n_ans} refuse={n_ref} parse_fail={n_parsefail}", flush=True)
    print(f"[gen] wrote {len(records)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
