# Selects the doctrine passages of each scene: exact inner-product search over the corpus index, a prose filter, a
# concept-keyword rerank and the core anchor of the concept. Writes the top passages with a quality report.
from tccc_vqa.paths import CORPUS, WORK
import json, re
import numpy as np

IDX = CORPUS + "/index"
P = WORK
EXCLUDE_SRC = {"arxiv_tccc_v1", "pubmed_tccc_v1", "tccc_glossary_v1"}
TOPK = 6
KW_BOOST = 0.04
MIN_ONTOPIC, MIN_SRC = 4, 3   # acceptance gate

KW = {
 "tourniquet_application": ["tourniquet","windlass","extremity","limb","hemorrhage","bleeding","proximal","amputation"],
 "tourniquet_conversion": ["tourniquet","conversion","convert","reassess","replace","two hours","dressing"],
 "wound_packing": ["pack","gauze","hemostatic","wound","pressure","bleeding","hemorrhage"],
 "junctional_hemorrhage": ["junctional","groin","inguinal","axilla","pelvic","junction"],
 "chest_seal": ["chest seal","occlusive","sucking","penetrating","pneumothorax","open chest","vented"],
 "needle_decompression": ["needle","decompression","tension","pneumothorax","intercostal","chest","catheter"],
 "nasopharyngeal_airway": ["nasopharyngeal","airway","nasal","unconscious","recovery position","trumpet"],
 "surgical_airway_cric": ["cricothyroidotomy","surgical airway","cricothyroid","neck","incision","membrane"],
 "hypothermia_prevention": ["hypothermia","thermal","blanket","warming","temperature","cold"],
 "txa_administration": ["txa","tranexamic","antifibrinolytic","hemorrhage","bleeding"],
 "iv_io_access": ["intravenous","intraosseous","vascular access","catheter","fluid","saline"],
 "tccc_documentation": ["documentation","dd 1380","casualty card","record","marker","tccc card"],
}

# CORE intervention-specific terms — a non-anchor passage must contain >=1 to enter the pool (kills tangents like TBI/NPWT/DCR)
CORE = {
 "tourniquet_application": ["tourniquet","windlass"],
 "tourniquet_conversion": ["tourniquet","windlass"],
 "wound_packing": ["pack","gauze","hemostatic"],
 "junctional_hemorrhage": ["junctional","groin","inguinal","axilla"],
 "chest_seal": ["chest seal","occlusive","sucking chest","vented seal"],
 "needle_decompression": ["needle decompression","tension pneumothorax","intercostal","decompress","needle thoracostomy"],
 "nasopharyngeal_airway": ["nasopharyngeal","nasal airway","npa","nasal trumpet"],
 "surgical_airway_cric": ["cricothyro","surgical airway","cricothyroidotomy"],
 "hypothermia_prevention": ["hypothermia","thermal","warming","hypothermic"],
 "txa_administration": ["txa","tranexamic"],
 "iv_io_access": ["intravenous","intraosseous","iv access","io access","vascular access","saline lock"],
 "tccc_documentation": ["dd 1380","casualty card","tccc card","documentation"],
}
def has_core(text, concept):
    tl = text.lower()
    return any(k in tl for k in CORE.get(concept, []))

def is_doctrine_prose(t):
    """positive gate: keep only clean prose doctrine; reject headers/legends/caps-titles/ref-lists/tables."""
    t = (t or "").strip()
    if len(t) < 140: return False
    if re.search(r"RED text indicates|BLUE text indicates|Guidelines\s+\d{1,2}\s+\w+\s+20\d\d", t[:170]): return False
    if re.match(r"\d+\s+Tactical Combat Casualty Care", t): return False
    if sum(c.isdigit() for c in t) / len(t) > 0.12: return False
    if re.search(r"Epub|PMID|doi:|et al\.|;\s?\d{2,4}:", t): return False
    head = t[:120]; HL = [c for c in head if c.isalpha()]
    if HL and sum(c.isupper() for c in HL) / len(HL) > 0.5: return False     # caps title
    if t.count(".") < 2: return False
    L = [c for c in t if c.isalpha()]
    if L and sum(c.islower() for c in L) / len(L) < 0.66: return False        # not prose
    return True

def is_junk(t):
    return not is_doctrine_prose(t)

def main():
    chunks = [json.loads(l) for l in open(f"{IDX}/chunks.jsonl")]
    emb = np.load(f"{IDX}/embeddings.npy").astype(np.float32)
    import sys
    scenes_file = sys.argv[1] if len(sys.argv) > 1 else f"{P}/pilot_scenes.json"
    emb_file = sys.argv[2] if len(sys.argv) > 2 else f"{P}/query_emb.npy"
    out_file = sys.argv[3] if len(sys.argv) > 3 else f"{P}/pilot_retrieved.json"
    inv = {c["concept_id"]: c for c in json.load(open(f"{P}/concept_inventory.json"))}
    scenes = json.load(open(scenes_file))["scenes"]
    qe = np.load(emb_file)

    # clean gold rows
    clean = [i for i, c in enumerate(chunks) if c.get("source_id") not in EXCLUDE_SRC and not is_junk(c["text"])]
    junk_removed = sum(1 for c in chunks if c.get("source_id") not in EXCLUDE_SRC and is_junk(c["text"]))
    clean = np.array(clean)
    cemb = emb[clean]
    cid_by_chunkidx = {c["citation_id"]: i for i, c in enumerate(chunks)}
    print(f"[sel] clean gold chunks={len(clean)} | junk removed={junk_removed}")

    def ontopic(text, concept):
        tl = text.lower()
        return sum(1 for k in KW.get(concept, []) if k in tl)

    out = []; ot_counts = []; src_div = []
    for s, q in zip(scenes, qe):
        sims = cemb @ q                      # cosine over clean chunks
        kwb = np.array([min(3, ontopic(chunks[clean[j]]["text"], s["concept"])) for j in range(len(clean))]) * KW_BOOST
        score = sims + kwb
        order = np.argsort(-score)[:60]
        # anchor: inventory gold_quote chunk (guaranteed on-topic)
        anc_ci = cid_by_chunkidx.get(inv[s["concept"]]["citation_id"])
        picked = []; seen = set()
        if anc_ci is not None:
            picked.append((anc_ci, float(score[np.where(clean == anc_ci)[0][0]]) if anc_ci in clean else 1.0, True))
            seen.add(chunks[anc_ci]["source_id"])
        # diversity-first then score
        core_ok = [j for j in order if has_core(chunks[int(clean[j])]["text"], s["concept"])]
        use = core_ok if core_ok else list(order)   # require CORE; fall back to anchor-only context if none
        for j in use:
            ci = int(clean[j]); src = chunks[ci]["source_id"]
            if ci == anc_ci: continue
            if src not in seen:
                picked.append((ci, float(score[j]), False)); seen.add(src)
            if len(picked) >= TOPK: break
        for j in use:
            if len(picked) >= TOPK: break
            ci = int(clean[j])
            if ci != anc_ci and all(ci != p[0] for p in picked):
                picked.append((ci, float(score[j]), False))
        retrieved = []
        for ci, sc, anc in picked[:TOPK]:
            txt = inv[s["concept"]]["gold_quote"] if anc else chunks[ci]["text"]
            retrieved.append({"cid": chunks[ci]["citation_id"], "source_id": chunks[ci]["source_id"],
                              "anchor": anc, "score": round(sc, 3), "ontopic": ontopic(txt, s["concept"]),
                              "section": chunks[ci].get("section_path"), "text": txt})
        ot = sum(1 for r in retrieved if r["ontopic"] > 0); ot_counts.append(ot)
        src_div.append(len(set(r["source_id"] for r in retrieved)))
        out.append({**{k: s[k] for k in ("item_id","concept","concept_readable","visible_evidence","visual_observation","image_path","detected","anatomy")}, "retrieved": retrieved})

    json.dump({"scenes": out}, open(out_file, "w"), indent=1)
    from collections import Counter
    allsrc = Counter(r["source_id"] for o in out for r in o["retrieved"])
    n = len(out)
    pass_ot = sum(1 for x in ot_counts if x >= MIN_ONTOPIC); pass_sr = sum(1 for x in src_div if x >= MIN_SRC)
    print(f"[sel] scenes={n} | mean on-topic/scene={sum(ot_counts)/n:.1f} (>= {MIN_ONTOPIC}: {pass_ot}/{n}) | "
          f"mean sources/scene={sum(src_div)/n:.1f} (>= {MIN_SRC}: {pass_sr}/{n})")
    print(f"[sel] GATE on-topic {'PASS' if pass_ot>=0.9*n else 'FAIL'} | diversity {'PASS' if pass_sr>=0.9*n else 'FAIL'}")
    print(f"[sel] source usage: {dict(allsrc)}")
    worst = sorted(range(n), key=lambda i: ot_counts[i])[:6]
    print("[sel] lowest on-topic scenes:", [(out[i]["item_id"], out[i]["concept"], ot_counts[i]) for i in worst])
    # HONEST cross-scene diversity: mean pairwise non-anchor passage overlap within each concept
    import itertools
    from collections import defaultdict
    byc = defaultdict(list)
    for o in out:
        byc[o["concept"]].append(set(r["cid"] for r in o["retrieved"] if not r["anchor"]))
    print("[sel] within-concept passage overlap (lower=more diverse; non-anchor of 5):")
    for c, sets in sorted(byc.items()):
        if len(sets) < 2: continue
        ovs = [len(a & b) for a, b in itertools.combinations(sets, 2)]
        print(f"     {c:24s} n={len(sets)} mean_overlap={sum(ovs)/len(ovs):.1f}/5")

if __name__ == "__main__":
    main()
