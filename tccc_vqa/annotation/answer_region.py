# Answer-region boxes: the VLM returns one box around the evidence for the recognition answer, given the question,
# the answer and the visible evidence. Output uses the detection box schema.
from tccc_vqa.paths import WORK
import argparse, base64, io, json, math, re, time
from pathlib import Path
from PIL import Image

def smart_resize(h, w, factor=28, min_pixels=4*28*28, max_pixels=1280*28*28):
    h_bar = max(factor, round(h/factor)*factor); w_bar = max(factor, round(w/factor)*factor)
    if h_bar*w_bar > max_pixels:
        beta = math.sqrt((h*w)/max_pixels)
        h_bar = max(factor, math.floor(h/beta/factor)*factor); w_bar = max(factor, math.floor(w/beta/factor)*factor)
    elif h_bar*w_bar < min_pixels:
        beta = math.sqrt(min_pixels/(h*w))
        h_bar = math.ceil(h*beta/factor)*factor; w_bar = math.ceil(w*beta/factor)*factor
    return h_bar, w_bar

def prep(path):
    im = Image.open(path).convert("RGB"); w, h = im.size
    rh, rw = smart_resize(h, w); im = im.resize((rw, rh), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), rw, rh

def build_prompt(it):
    ev = it.get("visible_evidence") or it.get("visual_observation") or it.get("answer")
    return (f"This is a single frame from a Tactical Combat Casualty Care (TCCC) video.\n"
            f"Question: {it['question']}\nCorrect answer: {it['answer']}\n"
            f"Visual evidence for this answer: {ev}\n\n"
            f"Draw ONE tight bounding box around the region of THIS image that shows the visual evidence "
            f"for the answer (the body part, equipment, and hands directly involved in the intervention). "
            f'Output ONLY JSON: {{"bbox_2d": [x1, y1, x2, y2]}} with integer pixel coordinates in THIS image '
            f"(top-left origin). If the evidence is not visible in the frame, output {{}}.")

def parse_box(text, rw, rh):
    m = re.search(r"\{[^{}]*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    bb = d.get("bbox_2d") or d.get("bbox")
    if not (isinstance(bb, list) and len(bb) == 4):
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bb]
    except Exception:
        return None
    x1, x2 = sorted((max(0, min(x1, rw)), max(0, min(x2, rw))))
    y1, y2 = sorted((max(0, min(y1, rh)), max(0, min(y2, rh))))
    if x2-x1 < 2 or y2-y1 < 2:
        return None
    return [round(x1/rw, 4), round(y1/rh, 4), round(x2/rw, 4), round(y2/rh, 4)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--grounding-input", default=WORK + "/grounding_input.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = json.load(open(args.grounding_input))["items"]
    if args.limit:
        items = items[:args.limit]
    print(f"[ground] {args.model} on {len(items)} items", flush=True)

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.9,
              trust_remote_code=True, dtype="bfloat16", limit_mm_per_prompt={"image": 1})
    print(f"[ground] loaded in {time.time()-t0:.0f}s", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=128)

    convs, meta = [], []
    for it in items:
        if not Path(it["image_path"]).exists():
            continue
        url, rw, rh = prep(it["image_path"])
        convs.append([{"role": "user", "content": [{"type": "text", "text": build_prompt(it)},
                                                   {"type": "image_url", "image_url": {"url": url}}]}])
        meta.append((it, rw, rh))

    print("[ground] generating ...", flush=True)
    outs = llm.chat(convs, sampling_params=sp)

    n_box = 0
    with open(args.out, "w") as f:
        for (it, rw, rh), o in zip(meta, outs):
            txt = o.outputs[0].text
            bb = parse_box(txt, rw, rh)
            boxes = [{"label": "answer_region", "bbox_norm": bb}] if bb else []
            n_box += len(boxes)
            f.write(json.dumps({"frame_id": it["item_id"], "image_path": it["image_path"],
                                "concepts": [it["concept"]], "boxes": boxes, "raw": txt[:120]},
                               ensure_ascii=False) + "\n")
    print(f"[ground] wrote {args.out} | items {len(meta)} | regions {n_box}", flush=True)

if __name__ == "__main__":
    main()
