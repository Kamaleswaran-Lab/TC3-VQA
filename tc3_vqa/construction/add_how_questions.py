# Adds the procedural HOW question from the batch outputs to each answerable item, snapping the answer to the exact
# corpus span and recording its offsets, and attaches the observed body region.
from tc3_vqa.paths import CORPUS, WORK
import json, glob, re
P=WORK
CH={c["citation_id"]:c["text"] for c in (json.loads(l) for l in open(CORPUS + "/index/chunks.jsonl"))}
def snap(a,cid):
    t=CH.get(cid,""); i=t.find(a)
    if i>=0: return a,i,i+len(a)
    m=re.compile(r"\s+".join(re.escape(w) for w in a.split())).search(t)
    return (t[m.start():m.end()],m.start(),m.end()) if m else None
how={}
for fp in glob.glob(f"{P}/how_out/batch_*.json"):
    for it in json.load(open(fp)).get("items",[]): how[it["item_id"]]=it
anat={r["item_id"]:r for r in (json.loads(l) for l in open(f"{P}/metadata.jsonl"))}
out=[]; nhow=0
for x in (json.loads(l) for l in open(f"{P}/clinician_review_queue_hybrid.jsonl")):
    iid=x.get("item_id")
    if x.get("task_type")!="refusal" and iid in how:
        h=how[iid]["how"]; sp=snap(h["answer"],h["cid"])
        if sp:
            ex,cs,ce=sp
            x=dict(x); x["questions"]=list(x["questions"])+[{"qid":f"{iid}#3","type":"how",
                "question":h["question"],"answer":ex,"source":h["cid"],"source_id":h.get("source_id"),
                "provenance":{"citation_id":h["cid"],"char_start":cs,"char_end":ce,"source_quote":ex},"faithful":1.0}]
            nhow+=1
    if iid in anat:
        x=dict(x); x["anatomy"]=anat[iid]["anatomy_region"]; x["anatomy_confidence"]=anat[iid].get("anatomy_confidence")
    out.append(x)
open(f"{P}/clinician_review_queue_hybrid.jsonl","w").write("\n".join(json.dumps(x,ensure_ascii=False) for x in out))
from collections import Counter
ans=[x for x in out if x.get("task_type")!="refusal" and x["item_id"].startswith("ans")]
print(f"total={len(out)} answerable={len(ans)} | HOW added={nhow}")
print("questions/item:",dict(Counter(len(x['questions']) for x in ans)))
