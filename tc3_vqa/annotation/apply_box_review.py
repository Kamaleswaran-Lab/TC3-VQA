# Applies the box-review verdicts (keep, relabel, delete, missed) to the refined detection layer and reports the
# correction rate by class and verification tier.
from tc3_vqa.paths import WORK
import json, glob, os
from collections import defaultdict, Counter

P = WORK
ONT = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "equipment_ontology.json")))
VALID = {c["id"] for c in ONT["classes"]}

def main():
    # merge verdicts
    verdict = {}   # (frame_id, idx) -> {verdict,new_label}
    missed = []
    for fp in glob.glob(f"{P}/review_corrections/batch_*.json"):
        d = json.load(open(fp))
        for c in d.get("corrections", []):
            fid = c["frame_id"]
            for b in c.get("boxes", []):
                verdict[(fid, b["idx"])] = (b["verdict"], (b.get("new_label") or "").strip())
            for m in c.get("missed", []):
                missed.append((fid, m.get("label")))

    sam = [json.loads(l) for l in open(f"{P}/detect_sam.jsonl")]
    tier = {(r["frame_id"], r["box_idx"]): r["verify"]
            for r in (json.loads(l) for l in open(f"{P}/detect_verify_internvl3.jsonl"))}

    per = defaultdict(lambda: Counter())          # label -> verdict counts
    del_by_tier = Counter(); kept_by_tier = Counter()
    corrected = []; n_no_verdict = 0
    for r in sam:
        key = (r["frame_id"], r["box_idx"]); lab = r["label"]
        v = verdict.get(key)
        if v is None:
            n_no_verdict += 1; act, nl = "keep", ""      # boxes without a verdict are kept
        else:
            act, nl = v
        per[lab][act] += 1
        itier = "hand" if lab == "gloved_hands" else tier.get(key, "unchecked")
        if act == "delete":
            del_by_tier[itier] += 1; continue            # drop
        if act == "relabel" and nl in VALID:
            r = {**r, "label": nl, "relabelled_from": lab}
        if act != "delete":
            kept_by_tier[itier] += 1
        r["review_verdict"] = act
        corrected.append(r)

    with open(f"{P}/detect_sam_corrected.jsonl", "w") as f:
        for r in corrected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tot = sum(sum(c.values()) for c in per.values())
    kept = sum(c["keep"] for c in per.values()); rel = sum(c["relabel"] for c in per.values()); dele = sum(c["delete"] for c in per.values())
    print(f"boxes judged={tot} (no-verdict default-keep={n_no_verdict}) | kept={kept} relabel={rel} delete={dele}")
    print(f"CORRECTION RATE = {(rel+dele)/tot:.1%} (relabel {rel/tot:.1%} + delete {dele/tot:.1%}) | corrected set boxes={len(corrected)}")
    print(f"\ndelete by InternVL3 tier: {dict(del_by_tier)}")
    print(f"kept   by InternVL3 tier: {dict(kept_by_tier)}")
    print(f"\nper-class (delete rate desc):")
    rows = sorted(per.items(), key=lambda kv: -(kv[1]['delete']/max(1,sum(kv[1].values()))))
    for lab, c in rows:
        n = sum(c.values())
        print(f"  {lab:24s} n={n:4d} keep={c['keep']:4d} relabel={c['relabel']:3d} delete={c['delete']:3d}  del%={c['delete']/n*100:4.0f}")
    mc = Counter(l for _, l in missed)
    print(f"\nmissed flagged={len(missed)} by label: {dict(mc.most_common())}")
    json.dump({"total": tot, "kept": kept, "relabel": rel, "delete": dele,
               "correction_rate": round((rel+dele)/tot, 4),
               "delete_by_tier": dict(del_by_tier), "kept_by_tier": dict(kept_by_tier),
               "missed_by_label": dict(mc)}, open(f"{P}/review_summary.json", "w"), indent=1)
    print(f"\nwrote {P}/detect_sam_corrected.jsonl + review_summary.json")

if __name__ == "__main__":
    main()
