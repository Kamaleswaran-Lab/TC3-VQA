# Assembles the generated recognition, doctrine and reasoning questions into items. Each doctrine and reasoning answer
# is matched to its exact corpus span and offsets are recorded; answers that cannot be matched drop their question.
from tccc_vqa.paths import CORPUS, WORK
import json, glob, re, sys

P = WORK
CH = {c["citation_id"]: c["text"] for c in (json.loads(l) for l in open(CORPUS + "/index/chunks.jsonl"))}

def snap(ans, cid):
    """return (exact_substring, char_start, char_end) or None."""
    t = CH.get(cid, "")
    if not ans or not t:
        return None
    i = t.find(ans)
    if i >= 0:
        return ans, i, i + len(ans)
    m = re.compile(r"\s+".join(re.escape(w) for w in ans.split())).search(t)
    if m:
        return t[m.start():m.end()], m.start(), m.end()
    return None

def main():
    gen_dir = sys.argv[1] if len(sys.argv) > 1 else f"{P}/gen_out"
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"{P}/qa_built.json"
    gen = {}
    for fp in glob.glob(f"{gen_dir}/batch_*.json"):
        for it in json.load(open(fp)).get("items", []):
            gen[it["item_id"]] = it
    built = {}
    stats = {"items": 0, "doctrine": 0, "reasoning": 0, "doctrine_drop": 0, "reasoning_drop": 0}
    for iid, g in gen.items():
        stats["items"] += 1
        qs = []
        m = g.get("mcq", {})
        if m.get("options") and m.get("answer"):
            qs.append({"qid": f"{iid}#0", "type": "recognition_mcq",
                       "question": m.get("question", "Which TCCC intervention is being performed in this frame?"),
                       "options": m["options"], "answer": m["answer"]})
        for kind, qtype in (("doctrine", "doctrine_scene"), ("reasoning", "reasoning")):
            q = g.get(kind, {})
            sp = snap(q.get("answer", ""), q.get("cid", ""))
            if sp:
                exact, cs, ce = sp
                qobj = {"qid": f"{iid}#{1 if kind=='doctrine' else 2}", "type": qtype,
                        "question": q.get("question"), "answer": exact,
                        "source": q.get("cid"), "source_id": q.get("source_id"),
                        "provenance": {"citation_id": q.get("cid"), "char_start": cs, "char_end": ce, "source_quote": exact},
                        "facet": q.get("facet") if kind == "doctrine" else None,
                        "reason_type": q.get("reason_type") if kind == "reasoning" else None}
                qs.append(qobj); stats[kind] += 1
            else:
                stats[f"{kind}_drop"] += 1
        built[iid] = {"item_id": iid, "questions": qs}
    json.dump(built, open(out_path, "w"), ensure_ascii=False, indent=1)
    # provenance integrity
    bad = sum(1 for b in built.values() for q in b["questions"]
              if q["type"] in ("doctrine_scene", "reasoning")
              and CH[q["provenance"]["citation_id"]][q["provenance"]["char_start"]:q["provenance"]["char_end"]] != q["answer"])
    print(f"built {stats['items']} items | doctrine {stats['doctrine']} (drop {stats['doctrine_drop']}) | "
          f"reasoning {stats['reasoning']} (drop {stats['reasoning_drop']}) | provenance bad={bad}")
    from collections import Counter
    qd = Counter(len(b["questions"]) for b in built.values())
    print(f"questions/item: {dict(qd)} | wrote {out_path}")

if __name__ == "__main__":
    main()
