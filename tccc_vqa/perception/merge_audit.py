# Merges the visual-audit verdicts into the candidate pool and assigns review tiers. The audit outranks the
# same-family cross-check, which outranks the generation confidence.
from tccc_vqa.paths import WORK
import json
from collections import Counter, defaultdict

VER = WORK + "/candidate_verified.json"
REF = WORK + "/refusal_gold.json"
import os, sys
AUDITS = sys.argv[1:]                      # JSON files with result.audit_verdicts, one per audit batch
OUT_Q = WORK + "/clinician_review_queue.jsonl"

items = json.load(open(VER))
refusals = json.load(open(REF))

# merge audits by id
av = {}
for f in AUDITS:
    if not os.path.exists(f):
        print(f"NOTE: {f} not present yet, skipping")
        continue
    for a in json.load(open(f))["result"]["audit_verdicts"]:
        av[a["id"]] = a
print(f"merged audits: {len(av)} (expected {len(items)})")
missing = [i for i in range(len(items)) if i not in av]
if missing:
    print(f"WARNING missing audit for {len(missing)} ids: {missing[:10]}")

PRIORITY = {"yes": "primary", "partial": "review", "no": "salvage"}

records = []
for i, it in enumerate(items):
    a = av.get(i)
    cv = a["concept_visible"] if a else None
    priority = PRIORITY.get(cv, "review")   # no-audit fallback -> review
    records.append({
        "item_id": f"ans_{i:05d}", "task_type": "answerable",
        "video_id": it["video_id"], "shot_idx": it["shot_idx"], "window_idx": it["window_idx"],
        "frame_paths": it["frame_paths"], "question": it["question"],
        "matched_concept_id": it["matched_concept_id"], "safety_critical": it["safety_critical"],
        "candidate_doctrine_answer": it["doctrine_answer"], "provenance": it["source"],
        "visual_observation": it["visual_observation"], "visible_evidence": it["visible_evidence"],
        "ablation_free_gen_answer": it["free_gen_doctrine"],
        "checks": {
            "citation_offset_ok": True, "visible_evidence_present": True,
            "cross_check_visible": it["cross_check"]["visible"],
            "audit_concept_visible": cv,                    # verdict of the independent auditor on the frames
            "audit_answerable": (a["answerable"] if a else None),
        },
        "independent_audit": ({"concept_visible": a["concept_visible"], "answerable": a["answerable"],
                               "what_you_see": a["what_you_see"]} if a else None),
        "review_priority": priority,                        # primary(yes) | review(partial) | salvage(no)
        "clinician_decision": None, "clinician_notes": None,
    })

for j, rf in enumerate(refusals):
    records.append({
        "item_id": f"ref_{j:05d}", "task_type": "refusal",
        "video_id": rf["video_id"], "scene_id": rf["scene_id"], "frame_paths": rf["frame_paths"],
        "question": rf["question"], "gold_label": "refuse", "refusal_reason": rf["refusal_reason"],
        "scene_category": rf["scene_category"], "tier": "REFUSAL_GOLD", "review_priority": "primary",
        "clinician_decision": None, "clinician_notes": None,
    })

with open(OUT_Q, "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---- stats on FULL pool (precision computed ONLY over audited items, no fallback distortion) ----
ans = [r for r in records if r["task_type"] == "answerable"]
n = len(ans)
audited = [r for r in ans if r["checks"]["audit_concept_visible"] is not None]
na = len(audited)
pri = Counter(r["review_priority"] for r in ans)
audit_dist = Counter(r["checks"]["audit_concept_visible"] for r in audited)
gen_prec = sum(1 for r in audited if r["checks"]["audit_concept_visible"] in ("yes", "partial")) / max(1, na)
gen_clear = sum(1 for r in audited if r["checks"]["audit_concept_visible"] == "yes") / max(1, na)
print(f"audit coverage: {na}/{n}")

# cross-check vs audit confusion (full pool)
cc_no = [r for r in ans if r["checks"]["cross_check_visible"] == "no"]
cc_no_just = sum(1 for r in cc_no if r["checks"]["audit_concept_visible"] != "yes") / max(1, len(cc_no))
cc_yes = [r for r in ans if r["checks"]["cross_check_visible"] in ("yes", "partial")]
cc_yes_conf = sum(1 for r in cc_yes if r["checks"]["audit_concept_visible"] == "yes") / max(1, len(cc_yes))

# per-family + per-concept precision
print(f"answerable {len(ans)} | audited {len(audited)} | audit verdicts {dict(audit_dist)}")
