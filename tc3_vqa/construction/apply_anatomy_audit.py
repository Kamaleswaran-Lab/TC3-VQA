# Applies the anatomy audit: corrected body regions are written to the item, and concept disagreements are flagged for
# the next review pass.
from tc3_vqa.paths import WORK
import json, glob
P=WORK
v=( {} )
for fp in glob.glob(f"{P}/e_audit_out/batch_*.json"):
    for it in json.load(open(fp)).get("items",[]): v[it["item_id"]]=it
# update metadata anatomy for fix_anatomy/fix_both
meta=[json.loads(l) for l in open(f"{P}/metadata.jsonl")]
for m in meta:
    a=v.get(m["item_id"])
    if a and a["verdict"] in ("fix_anatomy","fix_both") and a.get("true_anatomy"):
        m["anatomy_region"]=a["true_anatomy"]; m["anatomy_e_corrected"]=True
open(f"{P}/metadata.jsonl","w").write("\n".join(json.dumps(m,ensure_ascii=False) for m in meta))
# flag items in the queue
items=[json.loads(l) for l in open(f"{P}/clinician_review_queue_hybrid.jsonl")]
from collections import Counter
acts=Counter()
for x in items:
    a=v.get(x.get("item_id"))
    if not a: continue
    acts[a["verdict"]]+=1
    x["e_audit"]={"verdict":a["verdict"],"true_concept":a.get("true_concept"),"true_anatomy":a.get("true_anatomy"),"reason":a.get("reason")}
    if a["verdict"] in ("fix_anatomy","fix_both") and a.get("true_anatomy"): x["anatomy"]=a["true_anatomy"]
    if a["verdict"] in ("fix_concept","fix_both","refuse","ambiguous"): x["review_priority2"]="e_concept_mismatch"
open(f"{P}/clinician_review_queue_hybrid.jsonl","w").write("\n".join(json.dumps(x,ensure_ascii=False) for x in items))
print("E applied. verdicts:",dict(acts))
for iid,a in v.items(): print(f"  {iid}: {a['verdict']} | true_concept={a.get('true_concept')} true_anatomy={a.get('true_anatomy')} | {a.get('reason','')[:45]}")
