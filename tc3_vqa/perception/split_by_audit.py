# Splits the candidate pool by audit verdict: visible or partially visible concepts stay, the rest form the
# false-positive reservoir. Nothing is deleted.
from tc3_vqa.paths import WORK
import json
from collections import Counter

Q = WORK + "/clinician_review_queue.jsonl"
OUT_CLIN = WORK + "/clinician_review_queue.jsonl"      # overwrite -> high-yield only
OUT_REJ = WORK + "/audit_rejected.jsonl"               # audit=no reservoir

recs = [json.loads(l) for l in open(Q)]
ans = [r for r in recs if r["task_type"] == "answerable"]
ref = [r for r in recs if r["task_type"] == "refusal"]

keep = [r for r in ans if r["checks"]["audit_concept_visible"] in ("yes", "partial")]
rej = [r for r in ans if r["checks"]["audit_concept_visible"] == "no"]
# tag the rejected with why
for r in rej:
    r["tier"] = "AUDIT_REJECTED"
    r["review_priority"] = "deprioritized"

clinician = keep + ref   # answerable (yes/partial) + refusal-gold
with open(OUT_CLIN, "w") as f:
    for r in clinician:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUT_REJ, "w") as f:
    for r in rej:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# stats
print(f"clinician_review_queue.jsonl : {len(clinician)} "
      f"({len(keep)} answerable[yes/partial] + {len(ref)} refusal-gold)")
print(f"audit_rejected.jsonl         : {len(rej)} (audit=no, reservoir)")
print("\nclinician answerable concept distribution (post-audit-clean):")
cd = Counter(r["matched_concept_id"] for r in keep)
for c, n in cd.most_common():
    print(f"  {c:24s} {n:4d} ({100*n/len(keep):.0f}%)")
print("\nremaining priority split:", dict(Counter(r["review_priority"] for r in keep)))
