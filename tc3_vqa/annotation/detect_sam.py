# Box refinement with SAM 2. Each box prompts a mask; the tight box of the mask replaces the original when the mask
# is plausible, otherwise the original box is kept.
from tc3_vqa.paths import WORK
import argparse, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import torch

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1); inter = iw*ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua > 0 else 0.0

def mask_bbox(m):
    ys, xs = np.where(m)
    if len(xs) < 10:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--qwen-out", default=WORK + "/detect_qwen25vl72b.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.qwen_out)]
    if args.limit:
        rows = rows[:args.limit]
    n_box = sum(len(r["boxes"]) for r in rows)
    print(f"[sam] {args.model} | {len(rows)} frames / {n_box} boxes", flush=True)

    from transformers import Sam2Processor, Sam2Model
    dev = "cuda"
    proc = Sam2Processor.from_pretrained(args.model)
    model = Sam2Model.from_pretrained(args.model).to(dev).eval()
    print("[sam] loaded", flush=True)

    t0 = time.time(); n_ref = 0; n_done = 0
    fout = open(args.out, "w")
    for ri, r in enumerate(rows):
        boxes = r["boxes"]
        if not boxes or not Path(r["image_path"]).exists():
            for bi, b in enumerate(boxes):
                fout.write(json.dumps({"frame_id": r["frame_id"], "box_idx": bi, "label": b["label"],
                                       "bbox_norm": b["bbox_norm"], "orig_bbox_norm": b["bbox_norm"],
                                       "refined": False}, ensure_ascii=False) + "\n")
            continue
        im = Image.open(r["image_path"]).convert("RGB"); W, H = im.size
        px = [[float(b["bbox_norm"][0]*W), float(b["bbox_norm"][1]*H),
               float(b["bbox_norm"][2]*W), float(b["bbox_norm"][3]*H)] for b in boxes]
        inputs = proc(images=im, input_boxes=[px], return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inputs, multimask_output=False)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]  # (nbox, nmask, H, W)
        masks = np.asarray(masks).astype(bool)
        for bi, b in enumerate(boxes):
            orig = b["bbox_norm"]
            m = masks[bi]
            if m.ndim == 3:
                m = m[0]
            mb = mask_bbox(m)
            refined = False; new = orig
            if mb is not None:
                cand = [mb[0]/W, mb[1]/H, mb[2]/W, mb[3]/H]
                oa = (orig[2]-orig[0])*(orig[3]-orig[1]); ca = (cand[2]-cand[0])*(cand[3]-cand[1])
                # accept the refined box when it overlaps the original (IoU >= 0.1) and does not grow past 1.2x its area
                if oa > 0 and iou(cand, orig) >= 0.1 and ca <= oa*1.2:
                    new = [round(v, 4) for v in cand]; refined = True; n_ref += 1
            fout.write(json.dumps({"frame_id": r["frame_id"], "box_idx": bi, "label": b["label"],
                                   "bbox_norm": new, "orig_bbox_norm": orig,
                                   "iou_orig": round(iou(new, orig), 3), "refined": refined},
                                  ensure_ascii=False) + "\n")
            n_done += 1
        if (ri+1) % 200 == 0:
            print(f"[sam] {ri+1}/{len(rows)} frames | refined {n_ref}/{n_done} | {time.time()-t0:.0f}s", flush=True)
    fout.close()
    print(f"[sam] wrote {args.out} | boxes {n_done} | refined {n_ref} ({n_ref/n_done:.1%}) | {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
