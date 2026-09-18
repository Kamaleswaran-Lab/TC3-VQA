# Checks each answer-region box against the equipment boxes on its frame and writes the merged answer-region layer
# with a consistency flag.
from tccc_vqa.paths import WORK
import json
from collections import defaultdict
from pathlib import Path

P = WORK
ONT = json.load(open(Path(__file__).with_name("equipment_ontology.json")))

def contain(inner, outer):
    ix1, iy1, ix2, iy2 = inner; ox1, oy1, ox2, oy2 = outer
    x1, y1, x2, y2 = max(ix1, ox1), max(iy1, oy1), min(ix2, ox2), min(iy2, oy2)
    inter = max(0, x2-x1)*max(0, y2-y1); ia = (ix2-ix1)*(iy2-iy1)
    return inter/ia if ia > 0 else 0.0

def main():
    c2cls = defaultdict(set)
    for c in ONT["classes"]:
        for cc in c["concepts"]:
            if cc != "all":
                c2cls[cc].add(c["id"])
    img2fid = {r["image_path"]: r["frame_id"]
               for r in (json.loads(l) for l in open(f"{P}/detect_qwen25vl72b.jsonl"))}
    detf = defaultdict(list)
    for r in (json.loads(l) for l in open(f"{P}/detect_sam.jsonl")):
        detf[r["frame_id"]].append(r)

    gr = [json.loads(l) for l in open(f"{P}/grounding_qwen25vl72b.jsonl")]
    n_region = n_eval = n_contain = 0
    out = []
    for g in gr:
        ar = g["boxes"][0]["bbox_norm"] if g["boxes"] else None
        concept = g["concepts"][0]
        fid = img2fid.get(g["image_path"])
        rel = [b for b in detf.get(fid, []) if b["label"] in c2cls.get(concept, set())]
        best = max((contain(b["bbox_norm"], ar) for b in rel), default=None) if (ar and rel) else None
        contains = (best is not None and best >= 0.5)
        if ar:
            n_region += 1
            if rel:
                n_eval += 1; n_contain += contains
        out.append({"item_id": g["frame_id"], "image_path": g["image_path"], "concept": concept,
                    "answer_region": ar, "n_concept_equip": len(rel),
                    "containment": round(best, 3) if best is not None else None,
                    "contains_concept_equip": contains if (ar and rel) else None})
    with open(f"{P}/grounding_final.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"items={len(gr)} | answer-regions={n_region} ({n_region/len(gr):.0%}) | "
          f"evaluable={n_eval} | contains concept-equip>=50% = {n_contain} ({n_contain/n_eval:.0%})")
    print(f"wrote {P}/grounding_final.jsonl")

if __name__ == "__main__":
    main()
