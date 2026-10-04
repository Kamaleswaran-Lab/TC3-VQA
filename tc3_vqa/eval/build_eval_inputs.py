# Builds the baseline evaluation inputs: recognition (five-way choice), doctrine and HOW (free generation) and refusal
# (free generation with an abstention option).
from tc3_vqa.paths import WORK
import json

SRC = WORK + "/clinician_review_queue_hybrid.jsonl"
P = WORK
ABSTAIN = "None of these / cannot be determined from the image"

MARCH = {
    "tourniquet_application": "M", "wound_packing": "M", "junctional_hemorrhage": "M", "tourniquet_conversion": "M",
    "nasopharyngeal_airway": "A", "surgical_airway_cric": "A",
    "chest_seal": "R", "needle_decompression": "R",
    "iv_io_access": "C", "txa_administration": "C",
    "hypothermia_prevention": "H", "tccc_documentation": "Proc",
}
SEV = {  # RWHR per-item stakes default (clinician-ratified rubric)
    "tourniquet_application": 3, "junctional_hemorrhage": 3, "wound_packing": 3,
    "nasopharyngeal_airway": 3, "surgical_airway_cric": 3, "needle_decompression": 3, "chest_seal": 3,
    "iv_io_access": 2, "txa_administration": 2, "tourniquet_conversion": 2, "hypothermia_prevention": 2,
    "tccc_documentation": 1,
}

def is_ans(it): return it.get("task_type") == "answerable"   # task_type is authoritative; items moved to refusal keep their ans_ ids

def main():
    items = [json.loads(l) for l in open(SRC)]
    recog, doctrine, refusal, how = [], [], [], []
    for it in items:
        frames = it.get("frame_paths", [])
        if is_ans(it):
            c = it.get("matched_concept_id")
            qmap = {q.get("type"): q for q in it.get("questions", [])}
            rq = qmap.get("recognition_mcq")
            if rq and rq.get("options"):
                opts = list(rq["options"])
                if ABSTAIN not in opts: opts = opts + [ABSTAIN]
                recog.append({"id": it["item_id"], "kind": "answerable", "frames": frames,
                              "options": opts, "gold": rq.get("answer"), "concept": c,
                              "march": MARCH.get(c, "Proc"), "safety_critical": bool(it.get("safety_critical")),
                              "severity": SEV.get(c, 2)})
            hq = qmap.get("how")
            if hq:
                provh = hq.get("provenance") or {}
                how.append({"id": it["item_id"], "question": hq.get("question"), "frames": frames,
                            "gold": hq.get("answer"), "gold_span": provh.get("source_quote"),
                            "concept": c, "march": MARCH.get(c, "Proc"),
                            "safety_critical": bool(it.get("safety_critical"))})
            dq = qmap.get("doctrine_scene") or qmap.get("doctrine")
            if dq:
                prov = dq.get("provenance") or {}
                doctrine.append({"id": it["item_id"], "question": dq.get("question"), "frames": frames,
                                 "gold": dq.get("answer"), "gold_span": prov.get("source_quote"),
                                 "facet": dq.get("facet"), "concept": c, "march": MARCH.get(c, "Proc"),
                                 "safety_critical": bool(it.get("safety_critical"))})
        else:
            c = it.get("scene_category") or it.get("matched_concept_id")
            refusal.append({"id": it["item_id"], "kind": "refusal", "frames": frames,
                            "question": it.get("question") or (it.get("questions", [{}])[0].get("question")),
                            "gold": "REFUSE", "refusal_reason": it.get("refusal_reason"),
                            "concept": c, "march": MARCH.get(c, "Proc"),
                            "safety_critical": True, "severity": SEV.get(c, 2)})
    json.dump({"abstain": ABSTAIN, "items": recog}, open(f"{P}/eval_input.json", "w"), ensure_ascii=False)
    json.dump(doctrine, open(f"{P}/doctrine_input.json", "w"), ensure_ascii=False)
    json.dump(how, open(f"{P}/how_input.json", "w"), ensure_ascii=False)
    json.dump({"abstain_phrase": "CANNOT BE DETERMINED FROM THE IMAGE", "items": refusal},
              open(f"{P}/refusal_input.json", "w"), ensure_ascii=False)
    print(f"recognition {len(recog)} | doctrine {len(doctrine)} | how {len(how)} | refusal {len(refusal)}")

if __name__ == "__main__":
    main()
