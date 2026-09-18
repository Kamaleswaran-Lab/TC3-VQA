# Assembles the full release layout from the reviewed items: frames, item records, the doctrine multiple-choice files,
# the evaluation inputs and the viewer.
from tccc_vqa.paths import WORK
import json, os, shutil
from PIL import Image

P = WORK
QA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval")
REL = f"{P}/release_full"
SRC = f"{P}/clinician_review_queue_hybrid.jsonl"

def released(p):
    m = p.replace("/full_frames/", "/masked_frames/")
    return m if os.path.exists(m) else p

def fid(p):
    a = p.split("/")
    return f"{a[-2]}__{os.path.splitext(a[-1])[0]}.jpg"

def main():
    if os.path.exists(REL):
        shutil.rmtree(REL)
    for d in ["frames", "core", "mcq", "eval/scripts", "viewer"]:
        os.makedirs(f"{REL}/{d}", exist_ok=True)

    items = [json.loads(l) for l in open(SRC)]
    # detection + answer-region layers (embedded per record so the release matches the documented schema)
    from collections import defaultdict
    def frame_key(p):
        a = p.split("/"); return f"{a[-2]}/{os.path.splitext(a[-1])[0]}"
    detf = defaultdict(list)
    if os.path.exists(f"{P}/detect_final.jsonl"):
        for r in (json.loads(l) for l in open(f"{P}/detect_final.jsonl")):
            r["box_idx"] = len(detf[r["frame_id"]]); detf[r["frame_id"]].append(r)
    tier = {}
    if os.path.exists(f"{P}/detect_verify_internvl3.jsonl"):
        tier = {(r["frame_id"], r["box_idx"]): r.get("verify") for r in (json.loads(l) for l in open(f"{P}/detect_verify_internvl3.jsonl"))}
    ground = {}
    if os.path.exists(f"{P}/grounding_final.jsonl"):
        ground = {r["item_id"]: r for r in (json.loads(l) for l in open(f"{P}/grounding_final.jsonl"))}
    def src_of(fid_, bi, label):
        if label == "gloved_hands": return "hand"
        v = tier.get((fid_, bi)); return "verified" if v == "yes" else "disputed" if v == "no" else "unchecked"

    # 1) bundle frames (released form, original res) + build path map
    pmap = {}
    for it in items:
        for p in it.get("frame_paths", []):
            rp = released(p)
            if not os.path.exists(rp):
                continue
            fn = fid(p)
            if fn not in pmap:
                shutil.copy(rp, f"{REL}/frames/{fn}")
                pmap[p] = f"frames/{fn}"
    # 2) core records: relative frame paths + embedded detection / answer_region
    with open(f"{REL}/core/tccc_vqa.jsonl", "w") as f:
        for it in items:
            it = dict(it)
            det = []
            for p in it.get("frame_paths", []):
                fk = frame_key(p); rfp = pmap.get(p, p)
                for b in detf.get(fk, []):
                    det.append({"frame": rfp, "label": b["label"], "bbox_norm": b["bbox_norm"],
                                "source": src_of(fk, b["box_idx"], b["label"])})
            it["frame_paths"] = [pmap.get(p, p) for p in it.get("frame_paths", [])]
            if it.get("task_type") == "answerable":
                it["detection"] = det
                it["answer_region"] = (ground.get(it["item_id"]) or {}).get("answer_region")
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    # detection in COCO form, file_name remapped to the release's relative frames, filtered to shipped frames
    if os.path.exists(f"{P}/detect_coco.json"):
        coco = json.load(open(f"{P}/detect_coco.json"))
        relmap = {}  # old image_id -> kept
        new_imgs = []
        for im in coco.get("images", []):
            rel = pmap.get(im["file_name"])
            if rel:
                im = dict(im); im["file_name"] = rel; new_imgs.append(im); relmap[im["id"]] = True
        coco["images"] = new_imgs
        coco["annotations"] = [a for a in coco.get("annotations", []) if relmap.get(a["image_id"])]
        json.dump(coco, open(f"{REL}/core/detection_coco.json", "w"))
        print(f"  COCO: {len(new_imgs)} images / {len(coco['annotations'])} boxes (file_names -> relative)")

    # 3) mcq extension (+ generator)
    def remap_frames(infile, outfile):
        d = json.load(open(f"{P}/{infile}"))
        its = d["items"] if isinstance(d, dict) else d
        for x in its:
            if "frames" in x:
                x["frames"] = [pmap.get(p, p) for p in x["frames"]]
        json.dump(d, open(outfile, "w"), ensure_ascii=False, indent=1)
    remap_frames("doctrine_mcq_easy.json", f"{REL}/mcq/doctrine_mcq_easy.json")
    remap_frames("doctrine_mcq_hard.json", f"{REL}/mcq/doctrine_mcq_hard.json")
    shutil.copy(f"{QA}/build_doctrine_mcq.py", f"{REL}/mcq/build_doctrine_mcq.py")

    # 4) eval harness (inputs with relative frames + scripts)
    remap_frames("eval_input.json", f"{REL}/eval/recognition_input.json")
    remap_frames("doctrine_input.json", f"{REL}/eval/doctrine_open_input.json")
    remap_frames("how_input.json", f"{REL}/eval/how_input.json")
    remap_frames("refusal_input.json", f"{REL}/eval/refusal_input.json")
    shutil.copy(f"{P}/ablation_freegen.jsonl", f"{REL}/eval/ablation_freegen.jsonl")
    for s in ["eval_vlm.py", "eval_doctrine.py", "eval_refusal.py", "score_eval.py", "score_doctrine.py"]:
        if os.path.exists(f"{QA}/{s}"):
            shutil.copy(f"{QA}/{s}", f"{REL}/eval/scripts/{s}")

    # 5) viewer (self-contained browse tool)
    if os.path.exists(f"{P}/full_viewer"):
        shutil.copytree(f"{P}/full_viewer", f"{REL}/viewer", dirs_exist_ok=True)

    # report
    na = sum(1 for i in items if i.get("task_type") == "answerable")
    print(f"release -> {REL}")
    print(f"  frames bundled: {len(pmap)} (released form)")
    print(f"  core: {len(items)} items ({na} answerable + {len(items)-na} refusal)")
    os.system(f"du -sh {REL} 2>/dev/null; echo '--- tree ---'; ls -R {REL} | head -40")



if __name__ == "__main__":
    main()
