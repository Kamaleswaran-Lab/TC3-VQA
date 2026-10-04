# Runs a baseline VLM on the refusal questions with free-form generation. The prompt allows an explicit abstention;
# score_eval.py decides whether the model abstained.
from tc3_vqa.paths import WORK
import argparse, base64, io, json, time
from pathlib import Path
from PIL import Image

IMG_MAX_SIDE = 896
SYS = ("You are a TCCC (Tactical Combat Casualty Care) assistant. Look at the frame(s) and answer the "
       "question concisely and directly, based only on what is visible. If the image does not contain "
       "enough information to answer, say so rather than guessing.")

def data_url(path):
    im = Image.open(path).convert("RGB"); w, h = im.size
    s = IMG_MAX_SIDE / max(w, h)
    if s < 1.0: im = im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--refusal-input", default=WORK + "/refusal_input.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()
    items = json.load(open(args.refusal_input))["items"]
    print(f"[refusal] {args.model} on {len(items)} items", flush=True)
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=(32768 if any(k in args.model.lower() for k in ("internvl","minicpm","molmo")) else args.max_model_len), gpu_memory_utilization=0.9,
              trust_remote_code=True, dtype="bfloat16", limit_mm_per_prompt={"image": 4},
              mm_processor_kwargs=({"max_dynamic_patch": 8} if "internvl" in args.model.lower() else {}))
    print(f"[refusal] loaded {time.time()-t0:.0f}s", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=120)
    convs = []
    for it in items:
        content = [{"type": "text", "text": it["question"]}]
        for p in it["frames"][:4]:
            if Path(p).exists():
                content.append({"type": "image_url", "image_url": {"url": data_url(p)}})
        convs.append([{"role": "system", "content": SYS}, {"role": "user", "content": content}])
    cfmt = "string" if any(k in args.model.lower() for k in ("internvl","minicpm","molmo")) else "auto"
    outs = llm.chat(convs, sampling_params=sp, chat_template_content_format=cfmt)
    with open(args.out, "w") as f:
        for it, o in zip(items, outs):
            f.write(json.dumps({"id": it["id"], "kind": "refusal", "concept": it["concept"],
                                "march": it["march"], "safety_critical": it["safety_critical"],
                                "severity": it["severity"], "refusal_reason": it.get("refusal_reason"),
                                "resp": o.outputs[0].text.strip()}, ensure_ascii=False) + "\n")
    print(f"[refusal] wrote {args.out}", flush=True)

if __name__ == "__main__":
    main()
