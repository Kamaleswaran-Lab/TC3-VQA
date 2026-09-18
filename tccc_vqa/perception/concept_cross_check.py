# Two passes of the perception model in one vLLM load. Pass A re-checks under a strict prompt that the assigned
# concept is visible. Pass B writes an ungated doctrine answer for the same question, kept only for the ablation.
from tccc_vqa.paths import WORK
import argparse, json, re, time
from pathlib import Path
from PIL import Image

IMG_MAX_SIDE = 896

CROSS_SYS = """You verify a single labeling decision for a TCCC dataset. You are shown 1-4 frames and
ONE candidate concept. Answer ONLY whether that concept's procedure is CLEARLY and UNAMBIGUOUSLY
visible in the frames. Output JSON: {"visible":"yes|partial|no","why":"<=12 words"}.
Be strict: 'no' if the frames are ambiguous or show a different/again-unclear procedure."""

FREEGEN_SYS = """You are a TCCC medic assistant. Looking at the casualty frame(s), answer the question
with concrete treatment guidance. Output JSON:
{"assessment":"...","action_steps":["imperative step", ...],"source":"TCCC document + section"}.
Give the actual steps a medic should perform."""


def parse_json(t):
    if not t: return None
    m=re.search(r"\{.*\}", t, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def load_resized(p):
    im=Image.open(p).convert("RGB"); w,h=im.size; s=IMG_MAX_SIDE/max(w,h)
    if s<1.0: im=im.resize((max(1,int(w*s)),max(1,int(h*s))),Image.LANCZOS)
    return im

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--capped", default=WORK + "/candidate_capped.json")
    ap.add_argument("--inventory", default=WORK + "/concept_inventory.json")
    ap.add_argument("--out", default=WORK + "/candidate_verified.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=16384)
    args=ap.parse_args()

    items=json.load(open(args.capped))
    inv={c["concept_id"]:c for c in json.load(open(args.inventory))}
    print(f"[v&a] {len(items)} items", flush=True)

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    proc=AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    t0=time.time()
    llm=LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len, gpu_memory_utilization=0.88,
            trust_remote_code=True, dtype="bfloat16", limit_mm_per_prompt={"image":4})
    print(f"[v&a] vLLM loaded {time.time()-t0:.0f}s", flush=True)
    sp_a=SamplingParams(temperature=0.0, max_tokens=120)
    sp_b=SamplingParams(temperature=0.0, max_tokens=400)

    # cache images per item
    imgs_cache=[]
    for it in items:
        imgs=[load_resized(p) for p in it["frame_paths"][:4] if Path(p).exists()]
        imgs_cache.append(imgs)

    def build(sys, user_text, imgs):
        content=[{"type":"image"} for _ in imgs]+[{"type":"text","text":user_text}]
        msgs=[{"role":"system","content":sys},{"role":"user","content":content}]
        return {"prompt":proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True),
                "multi_modal_data":{"image":imgs}}

    # Pass A: cross-check
    print("[v&a] Pass A cross-check ...", flush=True)
    inA=[]
    for it,imgs in zip(items,imgs_cache):
        c=inv[it["matched_concept_id"]]
        trig="; ".join(c["visual_triggers"][:4])
        inA.append(build(CROSS_SYS, f"Candidate concept: {it['matched_concept_id']} (signature: {trig}). "
                                    f"Is this clearly visible in the frame(s)?", imgs))
    outA=llm.generate(inA, sp_a)

    # Pass B: free-gen ablation
    print("[v&a] Pass B free-gen ablation ...", flush=True)
    inB=[build(FREEGEN_SYS, it["question"], imgs) for it,imgs in zip(items,imgs_cache)]
    outB=llm.generate(inB, sp_b)

    agree=disagree=0
    for it,oa,ob in zip(items,outA,outB):
        pa=parse_json(oa.outputs[0].text) or {}
        vis=pa.get("visible","").lower()
        it["cross_check"]={"visible":vis,"why":pa.get("why")}
        it["cross_check_agree"]= vis in ("yes","partial")   # 'no' = disagreement
        agree+= it["cross_check_agree"]; disagree+= (not it["cross_check_agree"])
        it["free_gen_doctrine"]=parse_json(ob.outputs[0].text) or {"raw":ob.outputs[0].text[:500]}

    json.dump(items, open(args.out,"w"), ensure_ascii=False, indent=2)
    print(f"[v&a] cross-check agree {agree}/{len(items)} ({100*agree/len(items):.0f}%) | disagree {disagree}", flush=True)
    print(f"[v&a] wrote {args.out}", flush=True)

if __name__=="__main__":
    main()
