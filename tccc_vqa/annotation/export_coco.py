# Merges the refined boxes and their verification tiers into the final detection layer and exports it as COCO JSON.
from tccc_vqa.paths import WORK
import json
from pathlib import Path
from PIL import Image

P = WORK
ONT = json.load(open(Path(__file__).with_name("equipment_ontology.json")))
CLASSES = [c["id"] for c in ONT["classes"]]
CID = {c: i+1 for i, c in enumerate(CLASSES)}

def main():
    sam = [json.loads(l) for l in open(f"{P}/detect_sam.jsonl")]
    verify = {(r["frame_id"], r["box_idx"]): r["verify"]
              for r in (json.loads(l) for l in open(f"{P}/detect_verify_internvl3.jsonl"))}
    img_path = {r["frame_id"]: r["image_path"]
                for r in (json.loads(l) for l in open(f"{P}/detect_qwen25vl72b.jsonl"))}

    images, anns = [], []
    fid2img = {}
    wh_cache = {}
    for r in sam:
        fid = r["frame_id"]
        if fid not in fid2img:
            path = img_path[fid]
            if path not in wh_cache:
                im = Image.open(path); wh_cache[path] = im.size
            W, H = wh_cache[path]
            img_id = len(images) + 1
            fid2img[fid] = (img_id, W, H)
            images.append({"id": img_id, "file_name": path, "width": W, "height": H, "frame_id": fid})
        img_id, W, H = fid2img[fid]
        x1, y1, x2, y2 = r["bbox_norm"]
        bx, by, bw, bh = x1*W, y1*H, (x2-x1)*W, (y2-y1)*H
        v = verify.get((fid, r["box_idx"]))
        tier = "hand" if r["label"] == "gloved_hands" else ("verified" if v == "yes"
               else "disputed" if v == "no" else "unchecked")
        anns.append({"id": len(anns)+1, "image_id": img_id, "category_id": CID[r["label"]],
                     "bbox": [round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1)],
                     "area": round(bw*bh, 1), "iscrowd": 0,
                     "tier": tier, "sam_refined": r["refined"],
                     "verify_internvl3": v})
    coco = {"info": {"description": "TCCC-VQA equipment detection layer (auto-annotated: Qwen2.5-VL-72B "
                     "proposer + SAM2 refine + InternVL3 cross-verify). tier/verify per annotation.",
                     "ontology_version": ONT["version"]},
            "images": images,
            "annotations": anns,
            "categories": [{"id": CID[c], "name": c} for c in CLASSES]}
    json.dump(coco, open(f"{P}/detect_coco.json", "w"))
    from collections import Counter
    t = Counter(a["tier"] for a in anns)
    print(f"COCO: images={len(images)} annotations={len(anns)} categories={len(CLASSES)}")
    print("tiers:", dict(t))
    print(f"wrote {P}/detect_coco.json")

if __name__ == "__main__":
    main()
