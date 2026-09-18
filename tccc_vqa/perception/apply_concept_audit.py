# Applies the concept-audit verdicts after re-perception: confirmed items are kept, corrected items get the new concept
# with a re-anchored doctrine answer and rebuilt options, and items with no visible intervention become refusal items.
from tccc_vqa.paths import CORPUS, WORK
import json, glob, random
from collections import Counter, defaultdict

P = WORK
CHUNKS = {c["citation_id"]: c["text"] for c in (json.loads(l) for l in open(CORPUS + "/index/chunks.jsonl"))}
INV = {c["concept_id"]: c for c in json.load(open(f"{P}/concept_inventory.json"))}

def main():
    # audit verdicts
    verdict = {}
    for fp in glob.glob(f"{P}/audit_out/batch_*.json"):
        for a in json.load(open(fp)).get("audits", []):
            verdict[a["item_id"]] = a
    items = [json.loads(l) for l in open(f"{P}/clinician_review_queue_hybrid.jsonl")]

    # readable label per concept, taken from the recognition answers
    readable = {}
    for x in items:
        for q in x.get("questions", []):
            if q.get("type") == "recognition_mcq" and x.get("matched_concept_id"):
                readable.setdefault(x["matched_concept_id"], Counter())[q["answer"]] += 1
    readable = {k: v.most_common(1)[0][0] for k, v in readable.items()}
    all_labels = list(readable.values())

    def prov(cid, cs, ce):
        t = CHUNKS.get(cid, "")
        return {"citation_id": cid, "char_start": cs, "char_end": ce, "source_quote": t[cs:ce]}

    def rebuild(item, concept, rng):
        c = INV[concept]
        corr = readable.get(concept, concept.replace("_", " ").title())
        distract = [l for l in all_labels if l != corr]
        rng.shuffle(distract)
        opts = [corr] + distract[:3]; rng.shuffle(opts)
        recog = {"qid": f"{item['item_id']}#0", "type": "recognition_mcq",
                 "question": "Which TCCC intervention is being performed in this frame?",
                 "options": opts, "answer": corr}
        doc = {"qid": f"{item['item_id']}#1", "type": "doctrine_scene",
               "question": c["question_template"], "answer": c["gold_quote"],
               "source": c["citation_id"], "provenance": prov(c["citation_id"], c["char_start"], c["char_end"]),
               "faithful": 1.0}
        r = dict(item); r["matched_concept_id"] = concept
        r["questions"] = [recog, doc]
        r["action"] = "concept_corrected"; r["corrected_from"] = item.get("matched_concept_id")
        r["candidate_doctrine_answer"] = c["gold_quote"]
        r["provenance"] = {"citation_id": c["citation_id"], "char_start": c["char_start"], "char_end": c["char_end"]}
        return r

    out = []
    stats = Counter()
    for x in items:
        iid = x["item_id"]
        if x.get("task_type") == "refusal" or not iid.startswith("ans"):
            out.append(x); stats["refusal_kept"] += 1; continue
        a = verdict.get(iid)
        if a is None:
            out.append(x); stats["unchanged_agree"] += 1; continue   # not audited: the two perception passes agreed
        v = a["verdict"]; correct = a.get("correct")
        if v == "original":
            out.append(x); stats["unchanged_confirmed"] += 1
        elif v in ("reperceived", "other") and correct in INV:
            out.append(rebuild(x, correct, random.Random(hash(iid) & 0xffff))); stats["concept_corrected"] += 1
        else:  # none -> refusal
            r = dict(x); r["task_type"] = "refusal"; r["action"] = "moved_to_refusal"
            r["matched_concept_id"] = None
            r["questions"] = [{"qid": f"{iid}#r", "type": "refusal",
                               "question": next((q["question"] for q in x.get("questions", []) if q.get("type") == "recognition_mcq"),
                                                "Which TCCC intervention is being performed in this frame?"),
                               "answer": "Cannot be determined from the image",
                               "rationale": "no clearly depicted TCCC intervention (pixel audit)"}]
            out.append(r); stats["moved_to_refusal"] += 1

    with open(f"{P}/clinician_review_queue_hybrid.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # provenance verification on corrected items
    bad = 0
    for r in out:
        if r.get("action") == "concept_corrected":
            for q in r["questions"]:
                if q["type"] == "doctrine_scene":
                    pv = q["provenance"]
                    if CHUNKS[pv["citation_id"]][pv["char_start"]:pv["char_end"]] != q["answer"][:len(pv["source_quote"])] and pv["source_quote"] != q["answer"]:
                        # gold_quote should equal source_quote span
                        if CHUNKS[pv["citation_id"]][pv["char_start"]:pv["char_end"]] != q["answer"]:
                            bad += 1
    ans = sum(1 for r in out if r.get("task_type") != "refusal" and r["item_id"].startswith("ans"))
    ref = sum(1 for r in out if r.get("task_type") == "refusal")
    print("concept audit applied:", dict(stats))
    print(f"total={len(out)} | answerable={ans} | refusal={ref}")
    print(f"corrected-item doctrine provenance mismatches: {bad}")
    print("concept distribution (answerable):",
          dict(Counter(r.get("matched_concept_id") for r in out if r.get("task_type") != "refusal" and r["item_id"].startswith("ans")).most_common()))

if __name__ == "__main__":
    main()
