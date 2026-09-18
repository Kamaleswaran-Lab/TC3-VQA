# Scores free-form doctrine answers with NLI: whole-span entailment and sentence-level partial credit.
from tccc_vqa.paths import WORK
import json, glob, sys, os, re
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

def split_gold(text):
    # split on bullets, line breaks, and sentence punctuation; drop very short fragments
    parts = re.split(r"[•\n]|(?<=[.;])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 15]

def main():
    paths = sys.argv[1:] or sorted(glob.glob(WORK + "/doc_*.jsonl"))
    tok = AutoTokenizer.from_pretrained(NLI)
    model = AutoModelForSequenceClassification.from_pretrained(NLI).cuda().eval()
    id2label = model.config.id2label
    ent_idx = [i for i, l in id2label.items() if l.lower().startswith("entail")][0]
    con_idx = [i for i, l in id2label.items() if l.lower().startswith("contra")][0]

    @torch.no_grad()
    def entail(premise, hyp):
        x = tok(premise, hyp, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        p = model(**x).logits.softmax(-1)[0]
        return float(p[ent_idx]), float(p[con_idx])

    rows = []
    for path in paths:
        recs = [json.loads(l) for l in open(path)]
        strict = 0
        partials = []
        for r in recs:
            e, c = entail(r["answer"], r["gold"])           # whole-span: answer entails full gold?
            strict += (e > 0.5) and (e > c)
            sents = split_gold(r["gold"]) or [r["gold"]]
            hit = sum(1 for s in sents if (lambda ec: ec[0] > 0.5 and ec[0] > ec[1])(entail(r["answer"], s)))
            partials.append(hit / len(sents))
        name = os.path.basename(path).replace("doc_", "").replace(".jsonl", "")
        rows.append({"model": name, "n": len(recs),
                     "doctrine_strict": strict / len(recs),
                     "doctrine_partial": sum(partials) / len(partials)})
        print(f"{name:16s} strict {strict/len(recs):.3f}  partial {sum(partials)/len(partials):.3f}  (n={len(recs)})", flush=True)
    json.dump(rows, open(WORK + "/doctrine_scores.json", "w"), indent=1)

if __name__ == "__main__":
    main()
