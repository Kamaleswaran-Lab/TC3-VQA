# Claim-level entailment gate: checks with an NLI model that every sentence of an answer is entailed by the corpus
# chunks of the item's concept.
import json, re, sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
INP=sys.argv[1]; OUT=sys.argv[2]; TH=float(sys.argv[3]) if len(sys.argv)>3 else 0.5
d=json.load(open(INP)); PB=d["premises_by_concept"]; answers=d["answers"]
M="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
dev="cuda" if torch.cuda.is_available() else "cpu"
tok=AutoTokenizer.from_pretrained(M); model=AutoModelForSequenceClassification.from_pretrained(M).to(dev).eval()
@torch.no_grad()
def max_ent(sent, prem):
    best=0.0
    for i in range(0,len(prem),48):
        ch=prem[i:i+48]
        x=tok(ch,[sent]*len(ch),truncation=True,padding=True,return_tensors="pt",max_length=320).to(dev)
        best=max(best, torch.softmax(model(**x).logits,-1)[:,0].max().item())
    return best
res=[]
for a in answers:
    prem=PB.get(a["concept"],[])
    sents=[s.strip() for s in re.split(r'(?<=[.!?])\s+',a["text"]) if len(s.strip())>15] or [a["text"]]
    mins=min((max_ent(s,prem) for s in sents), default=0.0)
    res.append({"id":a["id"],"faithful":mins>=TH,"min_claim":round(mins,3)})
ok=sum(r["faithful"] for r in res)
json.dump(res, open(OUT,"w"), ensure_ascii=False)
print(f"claim-level faithful: {ok}/{len(res)} = {100*ok//len(res)}%")
