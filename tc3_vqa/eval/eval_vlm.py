# Runs a baseline VLM on the recognition questions (four concepts plus an abstention option) and records the predictions.
from tc3_vqa.paths import WORK
import argparse, base64, io, json, re, time
from pathlib import Path
from PIL import Image

IMG_MAX_SIDE = 896
LETTERS = ["A", "B", "C", "D", "E"]

SYS = ("You are answering a multiple-choice question about a single frame from a Tactical Combat Casualty "
       "Care (TCCC) video. Choose the SINGLE best option for which intervention is being performed in the "
       "frame. If none of the listed interventions is clearly and unambiguously shown, choose the "
       "'None / cannot be determined' option. Reply with ONLY the option letter.")

def data_url(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = IMG_MAX_SIDE / max(w, h)
    if s < 1.0:
        im = im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def blank_url():
    im = Image.new("RGB", (448, 448), (128, 128, 128))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def parse_letter(text, n):
    if not text: return None
    m = re.search(r"\b([A-E])\b", text.strip().upper())
    if m and LETTERS.index(m.group(1)) < n: return m.group(1)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval-input", default=WORK + "/eval_input.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--no-visual", action="store_true",
                    help="contamination probe (LongVT No-Visual): omit frames; clean benchmark -> ~chance accuracy")
    args = ap.parse_args()

    data = json.load(open(args.eval_input))
    items = data["items"]
    print(f"[eval] {args.model} on {len(items)} items", flush=True)

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=(32768 if any(k in args.model.lower() for k in ("internvl","minicpm","molmo")) else args.max_model_len), gpu_memory_utilization=0.9,
              trust_remote_code=True, dtype="bfloat16", limit_mm_per_prompt={"image": 4},
              mm_processor_kwargs=({"max_dynamic_patch": 8} if "internvl" in args.model.lower() else {}))
    print(f"[eval] loaded in {time.time()-t0:.0f}s", flush=True)
    BLANK = blank_url()
    sp = SamplingParams(temperature=0.0, max_tokens=8)

    convs = []
    for it in items:
        opts = it["options"]
        optblock = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
        qtext = it.get("question") or "Which TCCC intervention is being performed in this frame?"
        content = [{"type": "text",
                    "text": f"{qtext}\n{optblock}\n\nAnswer with only the letter."}]
        if args.no_visual:
            content.append({"type": "image_url", "image_url": {"url": BLANK}})  # blank gray: no visual signal
        else:
            for p in it["frames"][:4]:
                if Path(p).exists():
                    content.append({"type": "image_url", "image_url": {"url": data_url(p)}})
        convs.append([{"role": "system", "content": SYS}, {"role": "user", "content": content}])

    print("[eval] generating ...", flush=True)
    # InternVL (and some other) HF chat templates expect string content, not the OpenAI
    # multi-part list; "string" makes vLLM flatten to text + inject the model's <image> placeholder.
    cfmt = "string" if any(k in args.model.lower() for k in ("internvl","minicpm","molmo")) else "auto"
    outs = llm.chat(convs, sampling_params=sp, chat_template_content_format=cfmt)

    recs = []
    for it, o in zip(items, outs):
        txt = o.outputs[0].text
        letter = parse_letter(txt, len(it["options"]))
        pred = it["options"][LETTERS.index(letter)] if letter else None
        recs.append({"id": it["id"], "kind": it["kind"], "concept": it["concept"],
                     "march": it["march"], "safety_critical": it["safety_critical"],
                     "gold": it["gold"], "pred": pred, "raw": txt[:20]})
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    parsed = sum(1 for r in recs if r["pred"] is not None)
    print(f"[eval] wrote {args.out} | parsed {parsed}/{len(recs)}", flush=True)

if __name__ == "__main__":
    main()
