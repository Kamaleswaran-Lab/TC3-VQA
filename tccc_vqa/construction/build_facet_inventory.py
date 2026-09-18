# Verifies mined doctrine facets against the corpus text and writes the facet inventory; facets whose quote is not
# found in the source are discarded.
from tccc_vqa.paths import CORPUS, WORK
import json
import re


def find_span(text, quote):
    """Locate quote in text and return (start, end): exact match first, then a whitespace-tolerant match."""
    p = text.find(quote)
    if p >= 0:
        return p, p + len(quote)
    toks = quote.split()
    if not toks:
        return None
    pat = re.compile(r"\s+".join(re.escape(t) for t in toks))
    m = pat.search(text)
    return (m.start(), m.end()) if m else None

P = WORK
import sys
MINED = sys.argv[1]                       # facet-mining output: {"result": {"mined": [...]}}
OUT = f"{P}/concept_inventory_facets.json"

chunks = {}
for l in open(CORPUS + "/index/chunks.jsonl"):
    c = json.loads(l)
    chunks[c["citation_id"]] = c["text"]

inv = {c["concept_id"]: c for c in json.load(open(f"{P}/concept_inventory.json"))}
mined = json.load(open(MINED))["result"]["mined"]

out = {}
kept = dropped = 0
for m in mined:
    cid = m["concept_id"]
    base = inv[cid]
    facets = []
    # 1) keep the existing, pre-verified technique facet
    facets.append({"facet": "technique_how", "question": base["question_template"],
                   "gold_quote": base["gold_quote"], "citation_id": base["citation_id"],
                   "source_id": base["source_id"], "char_start": base["char_start"],
                   "char_end": base["char_end"], "safety_critical": base["safety_critical"],
                   "expected_tccc_stage": base["expected_tccc_stage"]})
    seen_q = {base["question_template"]}
    for f in m["facets"]:
        q = f["quote_verbatim"]; ccid = f["citation_id"]
        text = chunks.get(ccid)
        if not text:
            dropped += 1; continue
        span = find_span(text, q)
        if span is None:
            dropped += 1; continue          # not in the source text, even allowing whitespace differences
        if f["question"] in seen_q:
            continue
        s, e = span
        kept += 1
        seen_q.add(f["question"])
        facets.append({"facet": f["facet"], "question": f["question"],
                       "gold_quote": text[s:e], "citation_id": ccid,    # the span as it appears in the source
                       "source_id": ccid.split(":")[0], "char_start": s, "char_end": e,
                       "safety_critical": base["safety_critical"],
                       "expected_tccc_stage": base["expected_tccc_stage"]})
    out[cid] = facets[:4]   # coarse: cap 4 facets/concept

json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)

# verify all offsets exact (re-check)
bad = 0
for cid, fs in out.items():
    for f in fs:
        if chunks[f["citation_id"]][f["char_start"]:f["char_end"]] != f["gold_quote"]:
            bad += 1
print(f"mined facets kept(verified)={kept} dropped(not verbatim)={dropped}")
print(f"offset re-check mismatches: {bad}")
print(f"\nfacets per concept:")
for cid, fs in out.items():
    print(f"  {cid:24s} {len(fs)}  [{', '.join(f['facet'] for f in fs)}]")
print(f"\ntotal facet Q&A: {sum(len(v) for v in out.values())} (was 12)")
print(f"wrote {OUT}")
