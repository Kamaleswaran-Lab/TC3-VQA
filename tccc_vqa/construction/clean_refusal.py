# Keeps the refusal items that have a scene-grounded refusal reason and a usable frame; drops black or blank frames and
# demoted doctrine questions.
from tccc_vqa.paths import WORK
import json
import numpy as np
from PIL import Image
from collections import Counter

P = WORK
SRC = f"{P}/clinician_review_queue_hybrid.jsonl"
OUT = f"{P}/clinician_review_queue_hybrid.jsonl"

def frame_ok(it):
    fps = it.get("frame_paths") or []
    if not fps:
        return False
    try:
        a = np.asarray(Image.open(fps[0]).convert("L"), dtype=np.float32)
    except Exception:
        return False
    return a.mean() >= 20 and a.std() >= 10           # not near-black / not blank

def main():
    items = [json.loads(l) for l in open(SRC)]
    ans = [i for i in items if i.get("task_type") == "answerable"]
    ref = [i for i in items if i.get("task_type") == "refusal"]

    keep_ref, drop_ref = [], []
    for it in ref:
        if it.get("refusal_reason") and frame_ok(it):
            keep_ref.append(it)
        else:
            drop_ref.append(it)

    out = ans + keep_ref
    with open(OUT, "w") as f:
        for it in out:
            f.write(json.dumps(it) + "\n")

    # record what was dropped and why (auditability)
    drop_rec = []
    for it in drop_ref:
        reason = "no_refusal_reason" if not it.get("refusal_reason") else "black_or_blank_frame"
        drop_rec.append({"item_id": it["item_id"], "why": reason,
                         "had_reason": it.get("refusal_reason"),
                         "moved": bool(it.get("moved_to_refusal"))})
    json.dump({"kept_refusal": len(keep_ref), "dropped": len(drop_ref),
               "drop_detail": drop_rec,
               "kept_ids": sorted(i["item_id"] for i in keep_ref)},
              open(f"{P}/refusal_audit/clean.json", "w"), indent=1)

    print(f"answerable: {len(ans)} (unchanged)")
    print(f"refusal: {len(ref)} -> kept {len(keep_ref)} / dropped {len(drop_ref)}")
    print(f"  drop reasons: {dict(Counter(d['why'] for d in drop_rec))}")
    print(f"total 1: {len(out)}  -> {OUT}")
    print(f"kept reason dist: {dict(Counter(i.get('refusal_reason') for i in keep_ref))}")

    # regenerate refusal eval input filtered to the kept 187
    rin = f"{P}/refusal_input.json"
    try:
        ri = json.load(open(rin))
        keep_ids = {i["item_id"] for i in keep_ref}
        ritems = ri["items"] if isinstance(ri, dict) else ri
        filt = [x for x in ritems if x.get("id") in keep_ids or x.get("item_id") in keep_ids]
        obj = ({**ri, "items": filt} if isinstance(ri, dict) else filt)
        json.dump(obj, open(f"{P}/refusal_input.json", "w"), ensure_ascii=False, indent=1)
        print(f"refusal eval input: {len(ritems)} -> {len(filt)}  -> refusal_input.json")
    except FileNotFoundError:
        print("(refusal_input.json not found; skip eval-input filter)")

if __name__ == "__main__":
    main()
