# Equipment detection with Qwen2.5-VL: the model returns boxes for the ontology classes as JSON. Images are resized to
# the model's input size first, so the returned pixel coordinates refer to the sent image.
from tc3_vqa.paths import WORK
import argparse, base64, io, json, math, re, time
from pathlib import Path
from PIL import Image

ONT = json.load(open(Path(__file__).with_name("equipment_ontology.json")))
LABELS = {c["id"] for c in ONT["classes"]}

def smart_resize(h, w, factor=28, min_pixels=4*28*28, max_pixels=1280*28*28):
    h_bar = max(factor, round(h/factor)*factor)
    w_bar = max(factor, round(w/factor)*factor)
    if h_bar*w_bar > max_pixels:
        beta = math.sqrt((h*w)/max_pixels)
        h_bar = max(factor, math.floor(h/beta/factor)*factor)
        w_bar = max(factor, math.floor(w/beta/factor)*factor)
    elif h_bar*w_bar < min_pixels:
        beta = math.sqrt(min_pixels/(h*w))
        h_bar = math.ceil(h*beta/factor)*factor
        w_bar = math.ceil(w*beta/factor)*factor
    return h_bar, w_bar

def prep(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    rh, rw = smart_resize(h, w)
    im = im.resize((rw, rh), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return url, rw, rh

def class_block():
    return "\n".join(f"- {c['id']}: {c['desc']}" for c in ONT["classes"])

def build_prompt(concepts):
    ctx = (", ".join(concepts.replace("_", " ").split(",")) if isinstance(concepts, str)
           else ", ".join(c.replace("_", " ") for c in concepts))
    rules = " ".join(ONT["rules"])
    return (f"This is a single frame from a Tactical Combat Casualty Care (TCCC) procedure "
            f"(likely context: {ctx}). Detect every piece of equipment/tool from the list below that is "
            f"CLEARLY and unambiguously visible in the frame.\n\nLABELS:\n{class_block()}\n\n"
            f"Rules: {rules}\n\nOutput ONLY a JSON array. Each element: "
            f'{{"label": <one label id from the list>, "bbox_2d": [x1, y1, x2, y2]}} '
            f"with integer pixel coordinates in THIS image (top-left origin). "
            f"Only include clearly visible items. If none are visible, output [].")

def parse_boxes(text, rw, rh):
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for e in arr if isinstance(arr, list) else []:
        if not isinstance(e, dict):
            continue
        lab = e.get("label")
        bb = e.get("bbox_2d") or e.get("bbox")
        if lab not in LABELS or not (isinstance(bb, list) and len(bb) == 4):
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bb]
        except Exception:
            continue
        x1, x2 = sorted((max(0, min(x1, rw)), max(0, min(x2, rw))))
        y1, y2 = sorted((max(0, min(y1, rh)), max(0, min(y2, rh))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append({"label": lab, "bbox_norm": [round(x1/rw, 4), round(y1/rh, 4),
                                                round(x2/rw, 4), round(y2/rh, 4)]})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--detect-input", default=WORK + "/detect_input.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N frames")
    args = ap.parse_args()

    items = json.load(open(args.detect_input))["items"]
    if args.limit:
        items = items[:args.limit]
    print(f"[detect] {args.model} on {len(items)} frames", flush=True)

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.9,
              trust_remote_code=True, dtype="bfloat16", limit_mm_per_prompt={"image": 1})
    print(f"[detect] loaded in {time.time()-t0:.0f}s", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1024)

    convs, meta = [], []
    for it in items:
        p = it["image_path"]
        if not Path(p).exists():
            continue
        url, rw, rh = prep(p)
        content = [{"type": "text", "text": build_prompt(it["concepts"])},
                   {"type": "image_url", "image_url": {"url": url}}]
        convs.append([{"role": "user", "content": content}])
        meta.append((it, rw, rh))

    print("[detect] generating ...", flush=True)
    outs = llm.chat(convs, sampling_params=sp)

    n_box = 0
    with open(args.out, "w") as f:
        for (it, rw, rh), o in zip(meta, outs):
            txt = o.outputs[0].text
            boxes = parse_boxes(txt, rw, rh)
            n_box += len(boxes)
            f.write(json.dumps({"frame_id": it["frame_id"], "image_path": it["image_path"],
                                "concepts": it["concepts"], "resize_wh": [rw, rh],
                                "boxes": boxes, "raw": txt[:400]}, ensure_ascii=False) + "\n")
    print(f"[detect] wrote {args.out} | frames {len(meta)} | boxes {n_box}", flush=True)

if __name__ == "__main__":
    main()
