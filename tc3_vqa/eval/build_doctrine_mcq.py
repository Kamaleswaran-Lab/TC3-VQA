# Builds the closed-set doctrine questions. Easy distractors are doctrine of another concept; hard distractors are
# doctrine of the same concept on another facet.
from tc3_vqa.paths import WORK
import argparse, json, hashlib, re
from collections import defaultdict, Counter

P = WORK

# collapse long-tailed raw facets into 6 coarse buckets
FACET_RULES = [
    ("indication",  ["indicat", "purpose", "when"]),
    ("technique",   ["techniqu", "placement", "site", "lubric", "manual_pressure", "sizing", "packing", "insertion", "application", "catheter", "route"]),
    ("sequence",    ["sequence", "next_step", "next step", "timing", "conversion", "escalat", "subsequent", "prerequisite"]),
    ("verification",["effectiveness", "verif", "reassess", "monitor", "endpoint", "confirm"]),
    ("caution",     ["caution", "contraindicat", "precaution", "precaut", "safety", "complication", "consequence", "blind"]),
]
def coarse_facet(f):
    f = (f or "").lower()
    for name, keys in FACET_RULES:
        if any(k in f for k in keys):
            return name
    return "general"

def tokens(s): return set(re.findall(r"[a-z]{3,}", (s or "").lower()))
def jaccard(a, b):
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if (ta and tb) else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qtype", default="doctrine_scene", choices=["doctrine_scene", "reasoning"])
    ap.add_argument("--n-distract", type=int, default=3)
    args = ap.parse_args()

    ans = [json.loads(l) for l in open(f"{P}/clinician_review_queue_hybrid.jsonl")
           if json.loads(l).get("task_type") == "answerable"]
    meta = {it["id"]: it for it in json.load(open(f"{P}/eval_input.json"))["items"]}

    def q_of(it):
        for q in it["questions"]:
            if q["type"] == args.qtype:
                return q
        return None

    records = []
    facet_pool = defaultdict(list)    # coarse_facet -> [(concept, answer)]
    concept_pool = defaultdict(list)  # concept -> [(coarse_facet, answer)]
    for it in ans:
        q = q_of(it)
        if not q:
            continue
        gold = (q.get("answer_normalized") or q.get("answer") or "").strip()
        if len(gold) < 15:
            continue
        cf = coarse_facet(q.get("facet") or q.get("reason_type"))
        rec = {"iid": it["item_id"], "q": q, "gold": gold, "concept": it["matched_concept_id"], "cf": cf}
        records.append(rec)
        facet_pool[cf].append((it["matched_concept_id"], gold))
        concept_pool[it["matched_concept_id"]].append((cf, gold))
    allpool = [(c, a) for lst in facet_pool.values() for (c, a) in lst]

    def cross_concept_cands(rec):
        same = [(c, a) for (c, a) in facet_pool[rec["cf"]] if c != rec["concept"] and jaccard(a, rec["gold"]) <= 0.6]
        other = [(c, a) for (c, a) in allpool if c != rec["concept"] and jaccard(a, rec["gold"]) <= 0.6]
        return same, other

    def pick(rec, mode):
        seed = int(hashlib.md5(rec["iid"].encode()).hexdigest(), 16)
        chosen, used_txt, used_facets, used_c = [], set(), set(), set()
        def order(lst):
            return sorted(range(len(lst)), key=lambda i: hashlib.md5(f"{seed}_{i}".encode()).hexdigest())
        fellback = False
        if mode == "hard":
            # same concept, DIFFERENT coarse facet, distinct facets/text
            cands = [(cf, a) for (cf, a) in concept_pool[rec["concept"]]
                     if cf != rec["cf"] and a != rec["gold"] and jaccard(a, rec["gold"]) <= 0.6]
            for i in order(cands):
                cf, a = cands[i]
                if len(chosen) >= args.n_distract: break
                if cf in used_facets or any(jaccard(a, t) > 0.6 for t in used_txt): continue
                chosen.append(a); used_txt.add(a); used_facets.add(cf)
            # second pass: same concept, allow repeated facet (different text)
            if len(chosen) < args.n_distract:
                for i in order(cands):
                    cf, a = cands[i]
                    if len(chosen) >= args.n_distract: break
                    if any(jaccard(a, t) > 0.6 for t in used_txt): continue
                    chosen.append(a); used_txt.add(a)
            # fallback: cross-concept fill for thin concepts
            if len(chosen) < args.n_distract:
                fellback = True
                same, other = cross_concept_cands(rec)
                for src in (same, other):
                    for i in order(src):
                        c, a = src[i]
                        if len(chosen) >= args.n_distract: break
                        if c in used_c or any(jaccard(a, t) > 0.6 for t in used_txt): continue
                        chosen.append(a); used_txt.add(a); used_c.add(c)
        else:  # easy = cross-concept
            same, other = cross_concept_cands(rec)
            for src in (same, other):
                for i in order(src):
                    c, a = src[i]
                    if len(chosen) >= args.n_distract: break
                    if c in used_c or any(jaccard(a, t) > 0.6 for t in used_txt): continue
                    chosen.append(a); used_txt.add(a); used_c.add(c)
        return chosen, fellback

    for mode, suffix in (("easy", "easy"), ("hard", "hard")):
        items, dropped, fb = [], 0, 0
        for rec in records:
            ds, fellback = pick(rec, mode)
            if len(ds) < args.n_distract:
                dropped += 1; continue
            fb += int(fellback)
            opts = [rec["gold"]] + ds
            s = int(hashlib.md5((rec["iid"] + "opt").encode()).hexdigest(), 16)
            opts = [opts[i] for i in sorted(range(len(opts)), key=lambda i: hashlib.md5(f"{s}_{i}".encode()).hexdigest())]
            m = meta.get(rec["iid"], {})
            items.append({
                "id": rec["iid"], "kind": f"doctrine_mcq_{mode}", "question": rec["q"]["question"],
                "frames": m.get("frames", []), "options": opts, "gold": rec["gold"],
                "concept": rec["concept"], "facet": rec["cf"], "march": m.get("march"),
                "safety_critical": m.get("safety_critical"), "severity": m.get("severity"),
            })
        out = f"{P}/doctrine_mcq_{suffix}.json"
        json.dump({"abstain": False, "items": items}, open(out, "w"), ensure_ascii=False, indent=1)
        gp = Counter(i["options"].index(i["gold"]) for i in items)
        print(f"[{mode}] {len(items)} MCQs (dropped {dropped}, cross-concept fallback {fb}) -> {out}")
        print(f"      gold-position balance: {dict(gp)}")

if __name__ == "__main__":
    main()
