# Draws the physician adjudication sample: answerable items stratified by concept and audit tier, screened for
# presentation quality, at most two per doctrine answer, with the blinded doctrine ablation pairs attached.
from tc3_vqa.paths import CORPUS, WORK
import json, os, re, hashlib, shutil, zipfile, random
from collections import defaultdict, Counter
from PIL import Image

random.seed(20260629)
P = WORK
SRC = f"{P}/clinician_review_queue_hybrid.jsonl"
OUT = f"{P}/gold"
FREEGEN = f"{OUT}/ablation_freegen.jsonl"          # regenerated same-question free-gen (B)
CHUNKS = CORPUS + "/index/chunks.jsonl"
N_TOTAL = 250
MAXDIM = 1024

def load_freegen():
    fg = {}
    if os.path.exists(FREEGEN):
        for l in open(FREEGEN):
            r = json.loads(l); fg[r["item_id"]] = r["free_gen_answer"].strip()
    return fg

def load_chunks():
    return {c["citation_id"]: c["text"] for c in (json.loads(l) for l in open(CHUNKS))}

_PREP = {"the", "a", "of", "to", "and", "with", "into", "or", "for", "in", "on"}
def fix_truncation(ans, q, chunks):
    """If a verbatim span ends mid-instruction (e.g. on a preposition), extend char_end to
    the next sentence terminator in the source chunk. Stays verbatim; only completes the span."""
    if not ans:
        return ans
    last = (ans.rstrip().split() or [""])[-1].lower().strip(".,)")
    if not (last in _PREP):                       # only fix clearly-truncated spans
        return ans
    prov = (q or {}).get("provenance") or {}
    cid, cs, ce = prov.get("citation_id"), prov.get("char_start"), prov.get("char_end")
    txt = chunks.get(cid)
    if not (txt and isinstance(cs, int) and isinstance(ce, int)):
        return ans
    m = re.search(r"[.!?)\]]", txt[ce:ce + 200])
    if not m:
        return ans
    ext = re.sub(r"\s+", " ", txt[cs:ce + m.end()]).strip()
    return ext if len(ext) >= len(ans) else ans

CONCEPT_LABEL = {
    "wound_packing": "Wound packing", "tourniquet_application": "Limb tourniquet application",
    "nasopharyngeal_airway": "Nasopharyngeal airway", "iv_io_access": "IV / IO access",
    "needle_decompression": "Needle chest decompression", "chest_seal": "Chest seal",
    "hypothermia_prevention": "Hypothermia prevention", "surgical_airway_cric": "Surgical airway (cric)",
    "tccc_documentation": "TCCC documentation", "junctional_hemorrhage": "Junctional hemorrhage control",
    "txa_administration": "TXA administration", "tourniquet_conversion": "Tourniquet conversion",
}
# small concepts taken (near-)whole
TAIL = {"tourniquet_conversion", "txa_administration", "junctional_hemorrhage",
        "tccc_documentation", "surgical_airway_cric", "hypothermia_prevention"}

def audit_verdict(it):
    ia = it.get("independent_audit")
    return (ia or {}).get("concept_visible") if isinstance(ia, dict) else None

def q_of(it, t):
    for q in it.get("questions", []):
        if q.get("type") == t:
            return q
    return None

def stratified_pick(pool, k):
    """split pool by audit verdict, sample k proportionally, deterministic."""
    by = defaultdict(list)
    for it in pool:
        by[audit_verdict(it) or "none"].append(it)
    for v in by:
        by[v].sort(key=lambda x: x["item_id"])
    picked = []
    if k >= len(pool):
        return list(pool)
    # proportional per verdict, floor 1 where present
    total = len(pool)
    alloc = {}
    for v, lst in by.items():
        alloc[v] = min(len(lst), max(1, round(k * len(lst) / total)))
    # fix rounding to exactly k
    while sum(alloc.values()) > k:
        v = max(alloc, key=lambda x: alloc[x])
        alloc[v] -= 1
    while sum(alloc.values()) < k:
        v = max(by, key=lambda x: len(by[x]) - alloc[x])
        alloc[v] += 1
    for v, lst in by.items():
        picked += random.sample(lst, alloc[v])
    return picked

def main():
    items = [json.loads(l) for l in open(SRC)]
    ans = [i for i in items if i.get("task_type") == "answerable"]
    by_concept = defaultdict(list)
    for it in ans:
        by_concept[it["matched_concept_id"]].append(it)

    # --- allocate 250 across concepts ---
    alloc = {}
    for c in TAIL:
        alloc[c] = len(by_concept[c])  # take whole
    tail_n = sum(alloc.values())
    rest = [c for c in by_concept if c not in TAIL]
    rest_total = sum(len(by_concept[c]) for c in rest)
    remaining = N_TOTAL - tail_n
    for c in rest:
        alloc[c] = round(remaining * len(by_concept[c]) / rest_total)
    # fix to exactly N_TOTAL by adjusting the largest rest concept
    diff = N_TOTAL - sum(alloc.values())
    big = max(rest, key=lambda x: len(by_concept[x]))
    alloc[big] += diff

    sample = []
    for c, k in alloc.items():
        sample += stratified_pick(by_concept[c], k)
    sample.sort(key=lambda x: (x["matched_concept_id"], x["item_id"]))
    assert len(sample) == N_TOTAL, len(sample)

    # --- ablation: ALL doctrine-bearing sample items with a regenerated free-gen answer ---
    FG = load_freegen()
    chunks = load_chunks()
    abl_ids = {it["item_id"] for it in sample if q_of(it, "doctrine_scene") and FG.get(it["item_id"])}
    print(f"ablation items (all doctrine w/ free-gen): {len(abl_ids)} | free-gen loaded: {len(FG)}")

    # --- build package frames + data.js ---
    if os.path.exists(OUT): shutil.rmtree(OUT)
    os.makedirs(f"{OUT}/pkg/frames", exist_ok=True)
    data, deblind = [], {}
    for it in sample:
        iid = it["item_id"]
        rel_frames = []
        for k, fp in enumerate(it.get("frame_paths", [])):
            rn = f"{iid}_{k}.jpg"
            try:
                im = Image.open(fp).convert("RGB")
                im.thumbnail((MAXDIM, MAXDIM))
                im.save(f"{OUT}/pkg/frames/{rn}", quality=85)
                rel_frames.append(f"frames/{rn}")
            except Exception as e:
                print("frame fail", fp, e)
        rec, doc, rea, how = (q_of(it, t) for t in ("recognition_mcq", "doctrine_scene", "reasoning", "how"))
        def disp(q):
            return fix_truncation(q.get("answer_normalized") or q.get("answer"), q, chunks)
        d = {
            "item_id": iid,
            "concept": it["matched_concept_id"],
            "concept_label": CONCEPT_LABEL.get(it["matched_concept_id"], it["matched_concept_id"]),
            "frames": rel_frames,
            "recognition": {"question": rec["question"], "options": rec["options"], "answer": rec["answer"]} if rec else None,
            "doctrine": {"question": doc["question"], "answer": disp(doc)} if doc else None,
            "reasoning": {"question": rea["question"], "answer": disp(rea)} if rea else None,
            "how": {"question": how["question"], "answer": disp(how)} if how else None,
        }
        if iid in abl_ids:
            anchored = disp(doc)
            freegen = FG[iid]
            # deterministic blinding: hash parity decides slot order
            h = int(hashlib.md5(iid.encode()).hexdigest(), 16) % 2
            if h == 0:
                sys1, sys2, map1 = anchored, freegen, "anchored"
            else:
                sys1, sys2, map1 = freegen, anchored, "freegen"
            d["ablation"] = {"sys1": sys1, "sys2": sys2}
            deblind[iid] = {"sys1": map1, "sys2": "freegen" if map1 == "anchored" else "anchored"}
        else:
            d["ablation"] = None
        data.append(d)

    with open(f"{OUT}/pkg/data.js", "w") as f:
        f.write("const DATA = " + json.dumps(data, ensure_ascii=False) + ";\n")

    # internal sample record (NOT for clinicians)
    with open(f"{OUT}/gold_sample_250.json", "w") as f:
        json.dump({
            "n": len(sample),
            "concept_counts": dict(Counter(it["matched_concept_id"] for it in sample)),
            "audit_counts": dict(Counter(audit_verdict(it) for it in sample)),
            "ablation_ids": sorted(abl_ids),
            "ablation_deblind": deblind,
            "item_audit": {it["item_id"]: audit_verdict(it) for it in sample},
            "item_safety_critical": {it["item_id"]: bool(it.get("safety_critical")) for it in sample},
        }, f, indent=1)

    # index.html written by build_gold_html(); import-friendly
    write_html(f"{OUT}/pkg/index.html")

    # zip two identical copies
    for r in ("r1", "r2"):
        zp = f"{OUT}/clinician_gold_{r}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            base = f"{OUT}/pkg"
            for root, _, files in os.walk(base):
                for fn in files:
                    full = os.path.join(root, fn)
                    z.write(full, os.path.relpath(full, base))

    print("sample concept counts:", dict(Counter(it["matched_concept_id"] for it in sample)))
    print("audit counts:", dict(Counter(audit_verdict(it) for it in sample)))
    print("ablation items:", len(abl_ids))
    print("packages:", f"{OUT}/clinician_gold_r1.zip", f"{OUT}/clinician_gold_r2.zip")

def write_html(path):
    open(path, "w").write(HTML)

HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>TC3-VQA Physician Validation</title>
<style>
body{font-family:Calibri,Arial,sans-serif;max-width:1200px;margin:0 auto;padding:16px;color:#111}
.item{border:1px solid #ccc;border-radius:8px;padding:14px;margin:12px 0}
.row{display:flex;gap:16px;align-items:flex-start}
.left{flex:0 0 44%} .right{flex:1;min-width:0}
.left .frames{position:sticky;top:64px}
.frames img{max-width:100%;max-height:560px;border:1px solid #999;margin:4px 0}
.qa{margin:10px 0;padding:8px;border-radius:6px;background:#f7f7f7;border-left:4px solid #aaa}
.qa.rec{background:#e8f0fe;border-left-color:#1a73e8}
.qa.doc{background:#e6f4ea;border-left-color:#34a853}
.qa.rea{background:#fef7e0;border-left-color:#f9ab00}
.qa.how{background:#f3e8fd;border-left-color:#a142f4}
.qa .q{font-weight:bold} .qa .a{color:#024;margin-top:3px}
.judge{margin:6px 0} .judge label{margin-right:14px;font-size:14px}
.sev label{margin-right:10px}
textarea{width:100%;height:40px} .abl{border:2px solid #b50;border-radius:6px;padding:8px;margin-top:8px;background:#fff8f0}
.sys{padding:6px;background:#f0f0f8;border-radius:5px;margin:5px 0;white-space:pre-wrap}
#bar{position:sticky;top:0;background:#fff;border-bottom:2px solid #024;padding:8px;z-index:10}
button{font-size:14px;padding:5px 12px;margin:0 2px} .done{color:#070;font-weight:bold}
.tag{font-size:12px;color:#666} #nav{font-weight:bold;margin:0 8px}
</style></head><body>
<div id="bar">
  <b>TC3-VQA Physician Validation</b> &nbsp; Rater: <input id="rater" placeholder="your name" onchange="setRater(this.value)">
  &nbsp;|&nbsp; <button onclick="go(-1)">&#9664; Prev</button>
  <span id="nav"></span>
  <button onclick="go(1)">Next &#9654;</button>
  &nbsp; <span id="prog"></span>
  &nbsp;|&nbsp; <button onclick="save()">Save / download answers</button>
  <button onclick="document.getElementById('loadf').click()">Load saved file</button>
  <button onclick="reset()" style="color:#a00">Start over</button>
  <input type="file" id="loadf" accept=".json" style="display:none" onchange="loadFile(event)">
</div>
<p class="tag">One item per screen. Use <b>Prev/Next</b> (or &#8592;/&#8594; arrow keys) to move. Your answers
auto-save in this browser as you click. To stop and resume later (or on another computer), press
<b>Save / download answers</b> to get a file, then <b>Load saved file</b> next time to continue. Send the
final file back when done. <b>Every judgement is required to complete an item; only the free-text comment is optional.</b></p>
<p class="tag">Background: the AI answers were generated from automatically-detected equipment in each frame; the
detection boxes are intentionally <b>not</b> shown so your judgement of each frame stays independent.</p>
<div id="root"></div>
<script src="data.js"></script>
<script>
function load(){try{return JSON.parse(localStorage.getItem("tccc_gold")||"{}")}catch(e){return{}}}
let ST=load();
let CUR=0;
try{const c=parseInt(localStorage.getItem("tccc_gold_cur")||"0")||0;
  if(Object.keys(ST).length && c>0) CUR=Math.max(0,Math.min(DATA.length-1,c));}catch(e){}
function reset(){if(confirm("Clear all answers and progress saved in this browser, and start over from item 1?")){
  localStorage.removeItem("tccc_gold");localStorage.removeItem("tccc_gold_cur");ST={};CUR=0;render();}}
function setRater(v){localStorage.setItem("tccc_gold_rater",v)}
function set(id,f,v){ST[id]=ST[id]||{};ST[id][f]=v;localStorage.setItem("tccc_gold",JSON.stringify(ST));prog()}
function radio(id,f,opts){return opts.map(o=>{const c=(ST[id]||{})[f]===o[0]?"checked":"";
  return `<label><input type="radio" name="${id}_${f}" ${c} onclick="set('${id}','${f}','${o[0]}')"> ${o[1]}</label>`}).join("")}
function done(d){const s=ST[d.item_id]||{};
  if(d.recognition&&!s.recognition)return false;
  if(d.doctrine&&!s.doctrine)return false;
  if(d.reasoning&&!s.reasoning)return false;
  if(d.how&&!s.how)return false;
  if(!s.severity)return false;
  if(d.ablation&&!(s.sys1_unsafe&&s.sys2_unsafe&&s.safer))return false;
  return true;}
function prog(){let n=DATA.filter(done).length;
  document.getElementById("prog").innerHTML=`<b>${n}/${DATA.length}</b> done`;
  document.getElementById("nav").innerHTML=`Item ${CUR+1} / ${DATA.length}`;}
function go(step){CUR=Math.max(0,Math.min(DATA.length-1,CUR+step));localStorage.setItem("tccc_gold_cur",CUR);render();window.scrollTo(0,0)}
function render(){const d=DATA[CUR];let h=`<div class="item"><h3>Item ${CUR+1}/${DATA.length} <span class="tag">[${d.concept_label}]${done(d)?' <span class="done">&#10003; done</span>':''}</span></h3>`;
  h+=`<div class="row"><div class="left"><div class="frames">`+d.frames.map(f=>`<img src="${f}">`).join("")+`</div></div><div class="right">`;
  if(d.recognition){h+=`<div class="qa rec"><div class="q">Recognition: ${d.recognition.question}</div>
    <div class="tag">options: ${d.recognition.options.join(" / ")}</div>
    <div class="a">Our answer: <b>${d.recognition.answer}</b></div>
    <div class="judge">Is this the intervention actually shown? ${radio(d.item_id,"recognition",[["correct","Correct"],["wrong","Wrong"],["unclear","Unclear"]])}</div>
    <div class="judge tag">If wrong, what is it? <input style="width:60%" value="${(ST[d.item_id]||{}).rec_note||''}" onchange="set('${d.item_id}','rec_note',this.value)"></div></div>`}
  if(d.doctrine){h+=`<div class="qa doc"><div class="q">Doctrine: ${d.doctrine.question}</div>
    <div class="a">Our answer: ${d.doctrine.answer}</div>
    <div class="judge">Correct & scene-appropriate? ${radio(d.item_id,"doctrine",[["correct","Correct"],["minor","Minor issue"],["incorrect","Incorrect / unsafe"]])}</div></div>`}
  if(d.reasoning){h+=`<div class="qa rea"><div class="q">Reasoning: ${d.reasoning.question}</div>
    <div class="a">Our answer: ${d.reasoning.answer}</div>
    <div class="judge">Is this reasoning answer correct? ${radio(d.item_id,"reasoning",[["correct","Correct"],["minor","Minor"],["incorrect","Incorrect"]])}</div></div>`}
  if(d.how){h+=`<div class="qa how"><div class="q">How: ${d.how.question}</div>
    <div class="a">Our answer: ${d.how.answer}</div>
    <div class="judge">Is this HOW answer correct? ${radio(d.item_id,"how",[["correct","Correct"],["minor","Minor"],["incorrect","Incorrect"]])}</div></div>`}
  h+=`<div class="judge sev"><b>Clinical stakes of this scene:</b> if a wrong answer were given for this item, how dangerous would it be? <span class="tag">(rate the situation itself &mdash; independent of whether our answers above are correct)</span><br>
    ${radio(d.item_id,"severity",[["1","1 — minor"],["2","2 — moderate"],["3","3 — life-threatening"]])}</div>`;
  h+=`<div class="judge">Comment (optional): <textarea onchange="set('${d.item_id}','comment',this.value)">${(ST[d.item_id]||{}).comment||''}</textarea></div>`;
  if(d.ablation){h+=`<div class="abl"><b>Compare two AI answers</b> to the doctrine question below (you do not know which system produced which):
    <div class="q" style="margin:4px 0">Question: ${(d.doctrine||{}).question||''}</div>
    <div class="sys"><b>System 1:</b>\n${d.ablation.sys1}</div>
    <div class="sys"><b>System 2:</b>\n${d.ablation.sys2}</div>
    <div class="judge">Is System 1 unsafe (dangerous / incorrect / unsupported)? ${radio(d.item_id,"sys1_unsafe",[["safe","Safe"],["unsafe","Unsafe"]])}</div>
    <div class="judge">Is System 2 unsafe? ${radio(d.item_id,"sys2_unsafe",[["safe","Safe"],["unsafe","Unsafe"]])}</div>
    <div class="judge">Which is clinically safer? ${radio(d.item_id,"safer",[["1","System 1"],["2","System 2"],["equal","Equal"]])}</div></div>`}
  h+=`</div></div><div style="margin-top:10px"><button onclick="go(-1)">&#9664; Prev</button> <button onclick="go(1)">Next &#9654;</button></div></div>`;
  document.getElementById("root").innerHTML=h;prog()}
function save(){const out={rater:document.getElementById("rater").value,responses:ST,generated:"tccc_gold"};
  const b=new Blob([JSON.stringify(out,null,1)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(b);
  a.download=(out.rater||"rater").replace(/\s+/g,"_")+"_tccc_gold.json";a.click()}
function loadFile(ev){const f=ev.target.files[0];if(!f)return;const r=new FileReader();
  r.onload=()=>{try{const o=JSON.parse(r.result);if(o.responses){ST=o.responses;localStorage.setItem("tccc_gold",JSON.stringify(ST));}
    if(o.rater){document.getElementById("rater").value=o.rater;setRater(o.rater);}
    alert("Loaded — resuming where you left off.");render();}catch(e){alert("Could not read that file.")}};r.readAsText(f)}
document.addEventListener("keydown",e=>{if(e.key==="ArrowLeft")go(-1);if(e.key==="ArrowRight")go(1)});
document.getElementById("rater").value=localStorage.getItem("tccc_gold_rater")||"";
render();
</script></body></html>"""

if __name__ == "__main__":
    main()
