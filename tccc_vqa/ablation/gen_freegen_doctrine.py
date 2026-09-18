# Ungated answers for the doctrine ablation: the perception model answers each doctrine question of the adjudication
# sample from the frames alone, without corpus anchor or entailment gate. Writes ablation_freegen.jsonl.
from tccc_vqa.paths import WORK
import argparse, base64, io, json, time
from pathlib import Path
from collections import defaultdict
from PIL import Image

IMG_MAX_SIDE = 896
MAX_IMG = 4
P = WORK

def data_url(path):
    im = Image.open(path).convert("RGB"); w, h = im.size
    s = IMG_MAX_SIDE / max(w, h)
    if s < 1.0:
        im = im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def prompt(q):
    return ("Answer the following Tactical Combat Casualty Care (TCCC) question about the scene shown "
            "in the image(s). Give the correct TCCC doctrine in 1-3 sentences. "
            "Respond with only the doctrinal answer, no preamble.\n\n"
            f"Question: {q}")

def doctrine_q(it):
    for qq in it.get("questions", []):
        if qq.get("type") == "doctrine_scene":
            return qq.get("question")
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--sample", default=f"{P}/gold/gold_sample_250.json")
    ap.add_argument("--src", default=f"{P}/clinician_review_queue_hybrid.jsonl")
    ap.add_argument("--out", default=f"{P}/gold/ablation_freegen.jsonl")
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    sample_ids = set(json.load(open(args.sample))["item_audit"].keys())
    items = {i["item_id"]: i for i in (json.loads(l) for l in open(args.src))
             if i.get("item_id") in sample_ids}
    targets = [it for it in items.values() if doctrine_q(it)]
    print(f"[freegen] {len(targets)} doctrine-bearing of {len(sample_ids)} sample items", flush=True)

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.9,
              trust_remote_code=True, dtype="bfloat16", limit_mm_per_prompt={"image": MAX_IMG})
    print(f"[freegen] loaded in {time.time()-t0:.0f}s", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=220)

    convs, meta = [], []
    for it in targets:
        frames = [f for f in it.get("frame_paths", []) if Path(f).exists()][:MAX_IMG]
        if not frames:
            continue
        content = [{"type": "text", "text": prompt(doctrine_q(it))}]
        for f in frames:
            content.append({"type": "image_url", "image_url": {"url": data_url(f)}})
        convs.append([{"role": "user", "content": content}])
        meta.append(it["item_id"])

    print(f"[freegen] generating {len(convs)} ...", flush=True)
    outs = llm.chat(convs, sampling_params=sp)
    with open(args.out, "w") as f:
        for iid, o in zip(meta, outs):
            f.write(json.dumps({"item_id": iid,
                                "free_gen_answer": o.outputs[0].text.strip()}) + "\n")
    print(f"[freegen] wrote {len(meta)} -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
