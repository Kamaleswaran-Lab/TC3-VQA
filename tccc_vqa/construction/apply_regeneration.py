# Replaces flagged doctrine and reasoning questions with their regenerated versions, snapping each answer to the exact
# corpus span and restoring character offsets.
from tccc_vqa.paths import CORPUS, WORK
import json, glob, re
P=WORK
CH={c["citation_id"]:c["text"] for c in (json.loads(l) for l in open(CORPUS + "/index/chunks.jsonl"))}
def snap(a,cid):
    t=CH.get(cid,""); i=t.find(a)
    if i>=0: return a,i,i+len(a)
    m=re.compile(r"\s+".join(re.escape(w) for w in a.split())).search(t)
    return (t[m.start():m.end()],m.start(),m.end()) if m else None
new={}
for fp in glob.glob(f"{P}/d_regen_out/batch_*.json"):
    for it in json.load(open(fp)).get("items",[]): new[it["item_id"]]=it
items=[json.loads(l) for l in open(f"{P}/clinician_review_queue_hybrid.jsonl")]
nrep=0; fail=0
for x in items:
    iid=x.get("item_id")
    if iid not in new: continue
    g=new[iid]
    for t,key in (("doctrine_scene","doctrine_scene"),("reasoning","reasoning")):
        if key not in g: continue
        q=g[key]; sp=snap(q.get("answer",""),q.get("cid",""))
        if not sp: fail+=1; continue
        ex,cs,ce=sp
        nq={"qid":next((qq["qid"] for qq in x["questions"] if qq["type"]==t),f"{iid}#x"),"type":t,
            "question":q["question"],"answer":ex,"source":q["cid"],"source_id":q.get("source_id"),
            "provenance":{"citation_id":q["cid"],"char_start":cs,"char_end":ce,"source_quote":ex},
            "facet":q.get("facet"),"reason_type":q.get("reason_type"),"dregen":True}
        x["questions"]=[nq if qq["type"]==t else qq for qq in x["questions"]]
        nrep+=1
open(f"{P}/clinician_review_queue_hybrid.jsonl","w").write("\n".join(json.dumps(x,ensure_ascii=False) for x in items))
print(f"D applied: replaced {nrep} questions | snap-fail {fail}")
