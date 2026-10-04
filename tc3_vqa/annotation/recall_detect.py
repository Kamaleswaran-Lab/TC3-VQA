# Re-detection for classes flagged as missed in box review: the VLM looks for that class only on that frame and adds a
# box when the object is visible.
from tc3_vqa.paths import WORK
import argparse, base64, io, json, math, re, time
from pathlib import Path
from PIL import Image

ONT = json.load(open(Path(__file__).with_name("equipment_ontology.json")))
DESC = {c["id"]: c["desc"] for c in ONT["classes"]}

def smart_resize(h, w, factor=28, min_pixels=4*28*28, max_pixels=1280*28*28):
    hb = max(factor, round(h/factor)*factor); wb = max(factor, round(w/factor)*factor)
    if hb*wb > max_pixels:
        b = math.sqrt((h*w)/max_pixels); hb = max(factor, math.floor(h/b/factor)*factor); wb = max(factor, math.floor(w/b/factor)*factor)
    elif hb*wb < min_pixels:
        b = math.sqrt(min_pixels/(h*w)); hb = math.ceil(h*b/factor)*factor; wb = math.ceil(w*b/factor)*factor
    return hb, wb

def prep(path):
    im = Image.open(path).convert("RGB"); w, h = im.size; rh, rw = smart_resize(h, w)
    im = im.resize((rw, rh), Image.LANCZOS); buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), rw, rh

def prompt(labels):
    block = "\n".join(f"- {l}: {DESC[l]}" for l in labels)
    return (f"This TCCC frame may contain specific equipment that was previously left unboxed. "
            f"Detect ONLY the following item(s), and ONLY if each is CLEARLY and unambiguously visible:\n{block}\n\n"
            f'Output ONLY a JSON array; each element {{"label": <one of the listed ids>, "bbox_2d": [x1,y1,x2,y2]}} '
            f"with integer pixel coordinates in THIS image. If an item is not actually visible, omit it. "
            f"If none are visible, output [].")

def parse(text, labels, rw, rh):
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m: return []
    try: arr = json.loads(m.group(0))
    except Exception: return []
    out = []
    for e in arr if isinstance(arr, list) else []:
        if not isinstance(e, dict): continue
        lab = e.get("label"); bb = e.get("bbox_2d") or e.get("bbox")
        if lab not in labels or not (isinstance(bb, list) and len(bb) == 4): continue
        try: x1, y1, x2, y2 = [float(v) for v in bb]
        except Exception: continue
        x1, x2 = sorted((max(0, min(x1, rw)), max(0, min(x2, rw)))); y1, y2 = sorted((max(0, min(y1, rh)), max(0, min(y2, rh))))
        if x2-x1 < 2 or y2-y1 < 2: continue
        out.append({"label": lab, "bbox_norm": [round(x1/rw, 4), round(y1/rh, 4), round(x2/rw, 4), round(y2/rh, 4)]})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--input", default=WORK + "/recall_input.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()
    items = json.load(open(args.input))["items"]
    print(f"[recall] {args.model} on {len(items)} frames", flush=True)
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.9, trust_remote_code=True, dtype="bfloat16", limit_mm_per_prompt={"image": 1})
    print(f"[recall] loaded in {time.time()-t0:.0f}s", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=512)
    convs, meta = [], []
    for it in items:
        if not Path(it["image_path"]).exists(): continue
        url, rw, rh = prep(it["image_path"])
        convs.append([{"role": "user", "content": [{"type": "text", "text": prompt(it["missed_labels"])},
                                                   {"type": "image_url", "image_url": {"url": url}}]}])
        meta.append((it, set(it["missed_labels"]), rw, rh))
    print("[recall] generating ...", flush=True)
    outs = llm.chat(convs, sampling_params=sp)
    nb = 0
    with open(args.out, "w") as f:
        for (it, labs, rw, rh), o in zip(meta, outs):
            boxes = parse(o.outputs[0].text, labs, rw, rh); nb += len(boxes)
            f.write(json.dumps({"frame_id": it["frame_id"], "image_path": it["image_path"],
                                "requested": sorted(labs), "boxes": boxes, "raw": o.outputs[0].text[:200]}, ensure_ascii=False) + "\n")
    print(f"[recall] wrote {args.out} | frames {len(meta)} | recovered boxes {nb}", flush=True)

if __name__ == "__main__":
    main()
