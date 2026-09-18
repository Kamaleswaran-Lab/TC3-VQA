# Second detector: GroundingDINO proposes equipment boxes on the same frames so the VLM boxes can be cross-checked.
# Returned phrases are mapped to ontology classes.
from tccc_vqa.paths import WORK
import argparse, json, time
from pathlib import Path
from PIL import Image
import torch

# (class_id, lowercase query phrase); kept short and non-overlapping
QUERIES = [
    ("limb_tourniquet", "limb tourniquet"),
    ("junctional_tourniquet", "junctional tourniquet"),
    ("hemostatic_dressing", "hemostatic gauze dressing"),
    ("pressure_bandage", "pressure bandage"),
    ("nasopharyngeal_airway", "nasopharyngeal airway tube"),
    ("surgical_airway_device", "cricothyroidotomy tube"),
    ("scalpel", "scalpel"),
    ("chest_seal", "chest seal"),
    ("decompression_needle", "decompression needle"),
    ("iv_io_catheter", "intravenous catheter"),
    ("iv_fluid_bag", "iv fluid bag"),
    ("syringe", "syringe"),
    ("hypothermia_wrap", "hypothermia blanket"),
    ("casualty_card", "casualty card"),
    ("marker_pen", "marker pen"),
    ("trauma_shears", "trauma shears"),
    ("gloved_hands", "gloved hand"),
]
# Plain-word variants, used to check whether the vocabulary rather than the detector is the limit; mapping back is lossy.
SIMPLE_QUERIES = [
    ("limb_tourniquet", "tourniquet"),
    ("hemostatic_dressing", "gauze"),
    ("pressure_bandage", "bandage"),
    ("nasopharyngeal_airway", "tube"),
    ("scalpel", "knife"),
    ("chest_seal", "patch"),
    ("decompression_needle", "needle"),
    ("syringe", "syringe"),
    ("hypothermia_wrap", "blanket"),
    ("casualty_card", "card"),
    ("marker_pen", "pen"),
    ("trauma_shears", "scissors"),
    ("gloved_hands", "hand"),
]
PHRASE2CLS = {p: c for c, p in QUERIES}
PROMPT = " . ".join(p for _, p in QUERIES) + " ."

def map_label(text_label):
    t = (text_label or "").strip().lower()
    if t in PHRASE2CLS:
        return PHRASE2CLS[t]
    # otherwise map to the class with the largest token overlap
    tt = set(t.split())
    best, bj = None, 0.0
    for c, p in QUERIES:
        pt = set(p.split())
        j = len(tt & pt) / len(tt | pt) if (tt | pt) else 0
        if j > bj:
            bj, best = j, c
    return best if bj > 0 else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--detect-input", default=WORK + "/detect_input.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--simple", action="store_true", help="generic-noun queries (vocabulary test)")
    args = ap.parse_args()

    global QUERIES, PHRASE2CLS, PROMPT
    if args.simple:
        QUERIES = SIMPLE_QUERIES
        PHRASE2CLS = {p: c for c, p in QUERIES}
        PROMPT = " . ".join(p for _, p in QUERIES) + " ."

    items = json.load(open(args.detect_input))["items"]
    if args.limit:
        items = items[:args.limit]
    print(f"[gdino] {args.model} on {len(items)} frames", flush=True)

    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    dev = "cuda"
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(dev).eval()
    print("[gdino] loaded", flush=True)

    t0 = time.time(); n_box = 0
    with open(args.out, "w") as f:
        for i, it in enumerate(items):
            p = it["image_path"]
            if not Path(p).exists():
                continue
            im = Image.open(p).convert("RGB"); W, H = im.size
            inputs = proc(images=im, text=PROMPT, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**inputs)
            try:
                res = proc.post_process_grounded_object_detection(
                    out, inputs.input_ids, threshold=args.box_threshold,
                    text_threshold=args.text_threshold, target_sizes=[(H, W)])[0]
            except TypeError:
                res = proc.post_process_grounded_object_detection(
                    out, inputs.input_ids, box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold, target_sizes=[(H, W)])[0]
            labels = res.get("text_labels", res.get("labels"))
            boxes = []
            for box, score, lab in zip(res["boxes"], res["scores"], labels):
                cls = map_label(lab if isinstance(lab, str) else str(lab))
                if cls is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                x1, x2 = sorted((max(0, min(x1, W)), max(0, min(x2, W))))
                y1, y2 = sorted((max(0, min(y1, H)), max(0, min(y2, H))))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                boxes.append({"label": cls, "raw_phrase": str(lab),
                              "score": round(float(score), 3),
                              "bbox_norm": [round(x1/W, 4), round(y1/H, 4), round(x2/W, 4), round(y2/H, 4)]})
            n_box += len(boxes)
            f.write(json.dumps({"frame_id": it["frame_id"], "image_path": p,
                                "concepts": it["concepts"], "boxes": boxes}, ensure_ascii=False) + "\n")
            if (i+1) % 200 == 0:
                print(f"[gdino] {i+1}/{len(items)} | boxes {n_box} | {time.time()-t0:.0f}s", flush=True)
    print(f"[gdino] wrote {args.out} | frames {len(items)} | boxes {n_box} | {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
