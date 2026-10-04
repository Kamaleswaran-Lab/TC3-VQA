# Scores free-form doctrine answers with NLI: whole-span entailment (strict), sentence-level partial credit (share of
# gold sentences entailed by the answer) and the unsupported-claim rate (share of answer sentences entailed neither by
# the gold answer nor by any window of the cited doctrine chunk).
from tc3_vqa.paths import RELEASE, WORK
import json, glob, sys, os, re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DATA = os.environ.get("TC3_VQA_DATA", RELEASE + "/data")   # items.jsonl and doctrine_chunks.jsonl
WINDOW_WORDS = 100                                    # chunk sentences are grouped into windows of about this length


def split_gold(text):
    # split on bullets, line breaks and sentence punctuation; drop fragments too short to carry a claim
    parts = re.split(r"[•\n]|(?<=[.;])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 15]


def windows(text, max_words=WINDOW_WORDS):
    """consecutive sentence groups of a chunk, each short enough for the NLI context"""
    sents = split_gold(text) or [text]
    out, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            out.append(" ".join(cur)); cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out


def cited_chunks():
    """doctrine question id -> text of the chunk its gold answer is quoted from"""
    try:
        chunks = {c["citation_id"]: c["text"] for c in map(json.loads, open(f"{DATA}/doctrine_chunks.jsonl"))}
        out = {}
        for it in map(json.loads, open(f"{DATA}/items.jsonl")):
            for q in it["questions"]:
                if q["type"] == "doctrine_scene":
                    out[it["item_id"]] = chunks.get(q["provenance"]["citation_id"], "")
        return out
    except FileNotFoundError:
        print("doctrine chunks not found; unsupported-claim rate skipped", file=sys.stderr)
        return {}


def load_nli():
    tok = AutoTokenizer.from_pretrained(NLI)
    model = AutoModelForSequenceClassification.from_pretrained(NLI).cuda().eval()
    id2label = model.config.id2label
    ent_idx = [i for i, l in id2label.items() if l.lower().startswith("entail")][0]
    con_idx = [i for i, l in id2label.items() if l.lower().startswith("contra")][0]

    @torch.no_grad()
    def entail(premise, hyp):
        x = tok(premise, hyp, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        p = model(**x).logits.softmax(-1)[0]
        e, c = float(p[ent_idx]), float(p[con_idx])
        return e > 0.5 and e > c
    return entail


def score_record(entail, r, chunk=None):
    """strict, partial and (when the cited chunk is known) the unsupported share of the answer's sentences"""
    out = {"strict": bool(entail(r["answer"], r["gold"]))}
    sents = split_gold(r["gold"]) or [r["gold"]]
    out["partial"] = sum(1 for s in sents if entail(r["answer"], s)) / len(sents)
    if chunk is not None:
        premises = [r["gold"]] + windows(chunk)
        claims = split_gold(r["answer"]) or [r["answer"]]
        out["unsupported"] = sum(1 for s in claims if not any(entail(p, s) for p in premises)) / len(claims)
    return out


def main():
    paths = sys.argv[1:] or sorted(glob.glob(WORK + "/doc_*.jsonl"))
    entail = load_nli()
    chunks = cited_chunks()
    rows = []
    for path in paths:
        recs = [json.loads(l) for l in open(path)]
        scores = [score_record(entail, r, chunks.get(r["id"]) if chunks else None) for r in recs]
        name = os.path.basename(path).replace("doc_", "").replace(".jsonl", "")
        row = {"model": name, "n": len(recs),
               "doctrine_strict": sum(s["strict"] for s in scores) / len(scores),
               "doctrine_partial": sum(s["partial"] for s in scores) / len(scores)}
        with_chunk = [s for s in scores if "unsupported" in s]
        if with_chunk:
            row["doctrine_unsupported"] = sum(s["unsupported"] for s in with_chunk) / len(with_chunk)
        rows.append(row)
        print(f"{name:16s} " + "  ".join(f"{k.replace('doctrine_', '')} {v:.3f}" for k, v in row.items() if k.startswith("doctrine_")) + f"  (n={len(recs)})", flush=True)
    json.dump(rows, open(WORK + "/doctrine_scores.json", "w"), indent=1)


if __name__ == "__main__":
    main()
