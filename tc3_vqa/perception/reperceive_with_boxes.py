# Re-perception with the verified equipment boxes as evidence, same prompt and concept inventory. Writes the new
# concept and observation per item; later steps rerun only where the concept changed.
from tc3_vqa.paths import WORK
import argparse, json, re, time
from pathlib import Path
from collections import defaultdict
from PIL import Image

IMG_MAX_SIDE = 896

SYSTEM_PROMPT = """You label frames for a Tactical Combat Casualty Care (TCCC) visual-QA dataset.
You are shown 1-4 frames from ONE moment of a video (time order), a list of TCCC concepts with their
visual signatures, and an independently-detected list of equipment verified to be present in these frames.
Your ONLY job is to report what is VISIBLE and which ONE concept (if any) the frames clearly depict.

You DO NOT give medical advice, treatment steps, doses, or doctrine. You only describe what you see.

Use the detected-equipment list as supporting visual evidence, but rely on what you actually see; if the
detected equipment and the scene disagree, trust the frame.

Output ONE JSON object:
{
  "visual_observation": "1-2 sentence factual description of what is visible",
  "visible_evidence": "the single most specific medical element you can point to IN THE FRAME, or 'none'",
  "matched_concept_id": "<exactly one concept_id from the provided list, or 'none'>",
  "match_confidence": "high|medium|low",
  "why": "<=15 words naming the visual signature you saw"
}

Rules:
- Choose a concept ONLY if its visual signature is clearly present. If ambiguous or matching no concept, answer 'none'.
- 'none' is the CORRECT answer for ambiguous or non-procedural frames. Do not guess.
- Describe only what is visible. Never state what should be done."""


def parse_json(t):
    if not t: return None
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None


def load_resized(p):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = IMG_MAX_SIDE/max(w, h)
    if s < 1.0: im = im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.LANCZOS)
    return im


def concept_block(inv):
    lines = ["TCCC concepts (pick at most one that is clearly visible):"]
    for c in inv:
        lines.append(f"- {c['concept_id']}: {'; '.join(c['visual_triggers'][:4])}")
    return "\n".join(lines)


def frame_id(p):
    a = p.split("/"); return f"{a[-2]}/{Path(a[-1]).stem}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default=WORK + "/concept_inventory.json")
    ap.add_argument("--items", default=WORK + "/clinician_review_queue_hybrid.jsonl")
    ap.add_argument("--detect", default=WORK + "/detect_final.jsonl")
    ap.add_argument("--out", default=WORK + "/perceive.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=16384)
    args = ap.parse_args()

    inv = json.load(open(args.inventory))
    cblock = concept_block(inv)
    detlab = defaultdict(set)
    for r in (json.loads(l) for l in open(args.detect)):
        if r["label"] != "gloved_hands":
            detlab[r["frame_id"]].add(r["label"])
    items = [x for x in (json.loads(l) for l in open(args.items)) if x.get("item_id", "").startswith("ans")]
    print(f"[reperceive] {len(items)} answerable windows", flush=True)

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.88, trust_remote_code=True, dtype="bfloat16", limit_mm_per_prompt={"image": 4})
    print(f"[reperceive] loaded {time.time()-t0:.0f}s", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=400)

    vin, meta = [], []
    for it in items:
        fps = [p for p in it.get("frame_paths", [])[:4] if Path(p).exists()]
        if not fps: continue
        imgs = [load_resized(p) for p in fps]
        det = sorted(set().union(*[detlab.get(frame_id(p), set()) for p in fps])) or ["(none detected)"]
        content = [{"type": "image"} for _ in imgs]
        content.append({"type": "text", "text":
            f"{cblock}\n\nIndependently detected equipment verified present in these frame(s): "
            f"[{', '.join(det)}].\n\nHere are {len(imgs)} frame(s) from one window, in time order. "
            f"Report visible content and pick at most one concept."})
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vin.append({"prompt": prompt, "multi_modal_data": {"image": imgs}})
        meta.append((it, det))

    print("[reperceive] perception ...", flush=True)
    outs = llm.generate(vin, sampling_params=sp)
    n_change = n_none = 0
    with open(args.out, "w") as f:
        for (it, det), o in zip(meta, outs):
            p = parse_json(o.outputs[0].text) or {}
            cid = p.get("matched_concept_id", "none"); conf = p.get("match_confidence", "low")
            prev = it.get("matched_concept_id")
            new = cid if (cid != "none" and conf in ("high", "medium")) else "none"
            n_change += (new != prev); n_none += (new == "none")
            f.write(json.dumps({"item_id": it["item_id"], "video_id": it["video_id"],
                "frame_paths": it["frame_paths"], "detected": det, "previous_concept": prev,
                "concept": new, "confidence": conf, "observation": p.get("visual_observation"),
                "evidence": p.get("visible_evidence"), "raw": o.outputs[0].text[:200]}, ensure_ascii=False) + "\n")
    print(f"[reperceive] wrote {args.out} | concept changed={n_change} | none (to refusal)={n_none}", flush=True)


if __name__ == "__main__":
    main()
