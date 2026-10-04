# Agreement between the VLM boxes and the GroundingDINO boxes on the same frame, by class and IoU. Boxes confirmed
# by both detectors are marked as consensus.
from tc3_vqa.paths import WORK
import json, sys
from collections import defaultdict

P = WORK
IOU_THR = 0.5

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua > 0 else 0.0

def load(path):
    d = {}
    for l in open(path):
        r = json.loads(l)
        d[r["frame_id"]] = r
    return d

def main():
    qwen = load(f"{P}/detect_qwen25vl72b.jsonl")
    gd = load(f"{P}/detect_gdino.jsonl")
    frames = sorted(set(qwen) | set(gd))

    consensus, q_only, g_only = [], [], []
    per_cls = defaultdict(lambda: {"q": 0, "g": 0, "match": 0})
    for fid in frames:
        qb = qwen.get(fid, {}).get("boxes", [])
        gb = gd.get(fid, {}).get("boxes", [])
        for b in qb: per_cls[b["label"]]["q"] += 1
        for b in gb: per_cls[b["label"]]["g"] += 1
        used_g = set()
        for qi, q in enumerate(qb):
            best, bi = IOU_THR, -1
            for gi, g in enumerate(gb):
                if gi in used_g or g["label"] != q["label"]:
                    continue
                v = iou(q["bbox_norm"], g["bbox_norm"])
                if v >= best:
                    best, bi = v, gi
            if bi >= 0:
                used_g.add(bi)
                per_cls[q["label"]]["match"] += 1
                # consensus box = Qwen geometry (refined later by SAM), tag both scores
                consensus.append({"frame_id": fid, "image_path": qwen[fid]["image_path"],
                                  "label": q["label"], "bbox_norm": q["bbox_norm"],
                                  "iou": round(best, 3), "gdino_score": gb[bi].get("score"),
                                  "source": "consensus"})
            else:
                q_only.append({"frame_id": fid, "label": q["label"], "bbox_norm": q["bbox_norm"], "source": "qwen_only"})
        for gi, g in enumerate(gb):
            if gi not in used_g:
                g_only.append({"frame_id": fid, "label": g["label"], "bbox_norm": g["bbox_norm"],
                               "score": g.get("score"), "source": "gdino_only"})

    nq = sum(c["q"] for c in per_cls.values())
    ng = sum(c["g"] for c in per_cls.values())
    nm = sum(c["match"] for c in per_cls.values())
    print(f"Qwen boxes={nq} | GDINO boxes={ng} | consensus(matched pairs)={nm}")
    print(f"  Qwen confirmed-by-GDINO = {nm}/{nq} = {nm/nq:.2%}")
    print(f"  GDINO confirmed-by-Qwen = {nm}/{ng} = {nm/ng:.2%}")
    print(f"  qwen_only={len(q_only)}  gdino_only={len(g_only)}")
    print(f"\n{'class':24s} {'qwen':>5} {'gdino':>6} {'match':>6} {'q→g%':>6}")
    for c in sorted(per_cls, key=lambda k: -per_cls[k]["q"]):
        s = per_cls[c]
        print(f"{c:24s} {s['q']:5d} {s['g']:6d} {s['match']:6d} {(s['match']/s['q']*100 if s['q'] else 0):6.0f}")

    with open(f"{P}/detect_consensus.jsonl", "w") as f:
        for b in consensus: f.write(json.dumps(b, ensure_ascii=False) + "\n")
    json.dump({"qwen_only": q_only, "gdino_only": g_only},
              open(f"{P}/detect_singlesource.json", "w"))
    print(f"\nwrote {P}/detect_consensus.jsonl ({len(consensus)}) + detect_singlesource.json")

if __name__ == "__main__":
    main()
