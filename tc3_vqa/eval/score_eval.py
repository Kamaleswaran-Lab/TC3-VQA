# Scores baseline predictions: recognition accuracy with Wilson intervals, refusal abstention accuracy and the
# Risk-Weighted Hallucination Rate. Reads the outputs of eval_vlm.py and eval_refusal.py.
from tc3_vqa.paths import WORK
import json, sys, os, re, math
from collections import defaultdict

ABSTAIN = "None of these / cannot be determined from the image"
SEV = {"tourniquet_application":3,"junctional_hemorrhage":3,"wound_packing":3,"nasopharyngeal_airway":3,
       "surgical_airway_cric":3,"needle_decompression":3,"chest_seal":3,"iv_io_access":2,"txa_administration":2,
       "tourniquet_conversion":2,"hypothermia_prevention":2,"tccc_documentation":1}
ABST_RE = re.compile(
    r"cannot be determined|can(?:no|')t be determined|not be determined|cannot determine|can(?:no|')t determine|"
    r"unable to (?:determine|tell|answer|assess)|cannot be ascertained|cannot tell|can(?:no|')t tell|"
    r"cannot answer|can(?:no|')t answer|cannot be answered|"
    r"not (?:visible|shown|determinable|discernible|possible to determine|possible to tell|clear from)|"
    r"no (?:monitor|reading|vital|way to (?:tell|know|determine)|information (?:about|on))|"
    r"insufficient (?:information|evidence|detail)|not enough (?:information|detail|context)|"
    r"would need (?:more|additional)|does not (?:show|provide|contain|indicate|specify|appear to)|"
    r"image does not|frame does not|cannot be seen|not able to (?:determine|tell)|impossible to (?:tell|determine)|"
    r"too (?:dark|blurry|small|low.resolution) to|not legible|illegible|can(?:no|')t read|cannot read|"
    r"no (?:information|data|indication|timestamp|clock|visible (?:monitor|reading))|not (?:specified|stated|indicated|provided|legible|clear)|"
    r"unknown from|not discernible|hard to (?:tell|determine)|difficult to (?:tell|determine)|not depicted",
    re.I)

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (round(c-h, 3), round(c+h, 3))

def w_of(concept, scheme):
    s = SEV.get(concept, 2)
    return {"flat": 1, "default": s, "convex": {1:1,2:4,3:9}[s]}[scheme]

def score(model, recog_path, refusal_path):
    rec = [json.loads(l) for l in open(recog_path)]
    ref = [json.loads(l) for l in open(refusal_path)]
    # recognition
    rc = sum(1 for r in rec if r["pred"] == r["gold"])
    rec_acc = rc / len(rec)
    march = defaultdict(lambda: [0, 0])
    for r in rec:
        march[r["march"]][1] += 1; march[r["march"]][0] += (r["pred"] == r["gold"])
    macro = sum(c/n for c, n in march.values()) / len(march)
    # refusal abstention
    for r in ref:
        r["abstained"] = bool(ABST_RE.search(r.get("resp", "")))
    ref_acc = sum(1 for r in ref if r["abstained"]) / len(ref)
    # RWHR (graded) over recognition + refusal
    def rwhr(scheme):
        num = den = 0.0
        for r in rec:
            answered = r["pred"] != ABSTAIN and r["pred"] is not None
            if answered:
                w = w_of(r["concept"], scheme); den += 1
                if r["pred"] != r["gold"]: num += w
        for r in ref:
            answered = not r["abstained"]
            if answered:
                w = w_of(r["concept"], scheme); den += 1; num += w  # answering an unanswerable = hallucination
        return round(num/den, 3) if den else 0.0
    # unweighted hallucination rate + coverage over the same pool
    answered = sum(1 for r in rec if r["pred"] not in (ABSTAIN, None)) + sum(1 for r in ref if not r["abstained"])
    halluc = sum(1 for r in rec if r["pred"] not in (ABSTAIN, None) and r["pred"] != r["gold"]) + \
             sum(1 for r in ref if not r["abstained"])
    N = len(rec) + len(ref)
    return {"model": model, "n_recog": len(rec), "n_refusal": len(ref),
            "rec_acc": round(rec_acc, 3), "rec_acc_ci": wilson(rc, len(rec)),
            "rec_macro_march": round(macro, 3),
            "refusal_abstain_acc": round(ref_acc, 3),
            "coverage": round(answered/N, 3), "halluc_rate_unweighted": round(halluc/answered, 3) if answered else 0,
            "RWHR_default": rwhr("default"), "RWHR_flat": rwhr("flat"), "RWHR_convex": rwhr("convex"),
            "march": {k: round(c/n, 3) for k, (c, n) in sorted(march.items())}}

if __name__ == "__main__":
    P = WORK
    models = sys.argv[1:] or ["qwen2vl7b", "qwen25vl7b", "phi35v"]
    rows = []
    for m in models:
        rp, fp = f"{P}/eval_recog_{m}.jsonl", f"{P}/refusal_{m}.jsonl"
        if os.path.exists(rp) and os.path.exists(fp):
            rows.append(score(m, rp, fp))
    hdr = f"{'model':14s} {'recog':>7} {'macro':>7} {'refuse':>7} {'cover':>7} {'HR':>6} {'RWHR':>6} {'(flat/convex)':>14}"
    print(hdr)
    for r in rows:
        print(f"{r['model']:14s} {r['rec_acc']:7.3f} {r['rec_macro_march']:7.3f} {r['refusal_abstain_acc']:7.3f} "
              f"{r['coverage']:7.3f} {r['halluc_rate_unweighted']:6.3f} {r['RWHR_default']:6.3f} "
              f"  {r['RWHR_flat']}/{r['RWHR_convex']}")
    json.dump(rows, open(f"{P}/eval_scores.json", "w"), indent=1)
    print(f"\nwrote {P}/eval_scores.json")
