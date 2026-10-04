# Crop verification with InternVL3-8B: each proposed box is drawn on the frame and the model answers whether the named
# equipment is inside it.
from tc3_vqa.paths import WORK
import argparse, base64, io, json, time
from pathlib import Path
from PIL import Image, ImageDraw

ONT = json.load(open(Path(__file__).with_name("equipment_ontology.json")))
DESC = {c["id"]: c["desc"] for c in ONT["classes"]}
IMG_MAX_SIDE = 896

def boxed_url(path, bbox_norm):
    im = Image.open(path).convert("RGB"); W, H = im.size
    s = IMG_MAX_SIDE / max(W, H)
    if s < 1.0:
        im = im.resize((max(1, int(W*s)), max(1, int(H*s))), Image.LANCZOS); W, H = im.size
    d = ImageDraw.Draw(im)
    x1, y1, x2, y2 = bbox_norm
    d.rectangle([x1*W, y1*H, x2*W, y2*H], outline=(255, 0, 0), width=max(2, int(W/200)))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def parse_yn(t):
    t = (t or "").strip().lower()
    if t.startswith("yes"): return "yes"
    if t.startswith("no"): return "no"
    if "yes" in t[:12] and "no" not in t[:12]: return "yes"
    if "no" in t[:12] and "yes" not in t[:12]: return "no"
    return "parse"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenGVLab/InternVL3-8B")
    ap.add_argument("--qwen-out", default=WORK + "/detect_qwen25vl72b.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--include-hands", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.qwen_out)]
    tasks = []
    for r in rows:
        for bi, b in enumerate(r["boxes"]):
            if b["label"] == "gloved_hands" and not args.include_hands:
                continue
            if Path(r["image_path"]).exists():
                tasks.append((r["frame_id"], r["image_path"], bi, b["label"], b["bbox_norm"]))
    print(f"[verify] {args.model} on {len(tasks)} boxes", flush=True)

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.9,
              trust_remote_code=True, dtype="bfloat16", limit_mm_per_prompt={"image": 1})
    print(f"[verify] loaded in {time.time()-t0:.0f}s", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=4)

    convs = []
    for fid, path, bi, label, bbox in tasks:
        q = (f"An object in this image is marked with a red rectangle. "
             f"Does the object inside the red rectangle show {DESC[label]}? "
             f"Answer with only 'yes' or 'no'.")
        content = [{"type": "text", "text": q},
                   {"type": "image_url", "image_url": {"url": boxed_url(path, bbox)}}]
        convs.append([{"role": "user", "content": content}])

    print("[verify] generating ...", flush=True)
    outs = llm.chat(convs, sampling_params=sp)

    n_yes = 0
    with open(args.out, "w") as f:
        for (fid, path, bi, label, bbox), o in zip(tasks, outs):
            v = parse_yn(o.outputs[0].text)
            n_yes += (v == "yes")
            f.write(json.dumps({"frame_id": fid, "box_idx": bi, "label": label,
                                "bbox_norm": bbox, "verify": v,
                                "raw": o.outputs[0].text[:20]}, ensure_ascii=False) + "\n")
    print(f"[verify] wrote {args.out} | boxes {len(tasks)} | yes {n_yes} ({n_yes/len(tasks):.1%})", flush=True)

if __name__ == "__main__":
    main()
