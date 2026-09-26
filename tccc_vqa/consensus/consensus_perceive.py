# One recognition voter: a VLM served by vLLM labels the frames of each answerable item with at most one of the twelve
# concepts, using the perception prompt and no equipment boxes. Writes <tag>.jsonl, one record per item.
from tccc_vqa.paths import EXPERIMENTS, FRAMES, RELEASE, WORK
import argparse, base64, io, json, re
from pathlib import Path
from PIL import Image

ITEMS = RELEASE + '/data/items.jsonl'
FRAMES = FRAMES
INVENTORY = WORK + '/concept_inventory.json'
OUT = Path(EXPERIMENTS + '/consensus_perception')
IMG_MAX_SIDE, MAX_IMG = 896, 4

# perceive.py SYSTEM_PROMPT; the detection paragraph is included only with --with-detection
SYSTEM_HEAD = """You label frames for a Tactical Combat Casualty Care (TCCC) visual-QA dataset.
You are shown 1-4 frames from ONE moment of a video (time order){det_intro}, and a list of TCCC concepts with their
visual signatures.
Your ONLY job is to report what is VISIBLE and which ONE concept (if any) the frames clearly depict.

You DO NOT give medical advice, treatment steps, doses, or doctrine. You only describe what you see.
{det_rule}
Output ONE JSON object:
{{
  "visual_observation": "1-2 sentence factual description of what is visible",
  "visible_evidence": "the single most specific medical element you can point to IN THE FRAME, or 'none'",
  "matched_concept_id": "<exactly one concept_id from the provided list, or 'none'>",
  "match_confidence": "high|medium|low",
  "why": "<=15 words naming the visual signature you saw"
}}

Rules:
- Choose a concept ONLY if its visual signature is clearly present. If ambiguous or matching no concept, answer 'none'.
- 'none' is the CORRECT answer for ambiguous or non-procedural frames. Do not guess.
- Describe only what is visible. Never state what should be done."""
DET_INTRO = ', an independently-detected list of equipment verified to be present in these frames'
DET_RULE = ('\nUse the detected-equipment list as supporting visual evidence, but rely on what you actually see; if the\n'
            'detected equipment and the scene disagree, trust the frame.\n')


def data_url(path):
    im = Image.open(path).convert('RGB'); w, h = im.size
    s = IMG_MAX_SIDE / max(w, h)
    if s < 1.0:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=88)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def parse_json(t):
    t = (t or '').split('</think>')[-1]
    m = re.search(r'\{.*\}', t, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--tensor-parallel-size', type=int, default=1)
    ap.add_argument('--max-model-len', type=int, default=16384)
    ap.add_argument('--with-detection', action='store_true', help='inject released detections as in the construction-time perception')
    ap.add_argument('--mistral', action='store_true')
    ap.add_argument('--hf-overrides', default=None)
    ap.add_argument('--mm-processor-kwargs', default=None)
    ap.add_argument('--chat-content-format', default='auto', choices=['auto', 'string', 'openai'])
    ap.add_argument('--n-items', type=int, default=0, help='0 = all answerable items')
    ap.add_argument('--items', default=ITEMS)
    args = ap.parse_args()

    inv = json.load(open(INVENTORY))
    cblock = '\n'.join(['TCCC concepts (pick at most one that is clearly visible):'] +
                       [f"- {c['concept_id']}: {'; '.join(c['visual_triggers'][:4])}" for c in inv])
    valid = {c['concept_id'] for c in inv}
    system = SYSTEM_HEAD.format(det_intro=DET_INTRO if args.with_detection else '',
                                det_rule=DET_RULE if args.with_detection else '')
    items = [x for x in map(json.loads, open(args.items)) if x['task_type'] == 'answerable']
    if args.n_items:
        items = items[:args.n_items]

    from vllm import LLM, SamplingParams
    kw = dict(model=args.model, tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.9, trust_remote_code=True, limit_mm_per_prompt={'image': MAX_IMG}, enforce_eager=True)
    kw.update(dict(tokenizer_mode='mistral', config_format='mistral', load_format='mistral') if args.mistral else dict(dtype='bfloat16'))
    if args.hf_overrides:
        kw['hf_overrides'] = json.loads(args.hf_overrides)
    if args.mm_processor_kwargs:
        kw['mm_processor_kwargs'] = json.loads(args.mm_processor_kwargs)
    llm = LLM(**kw)

    convs = []
    for it in items:
        frames = [f"{FRAMES}/{r['frame_id']}" for r in it['frame_refs']][:MAX_IMG]
        text = cblock + '\n\n'
        if args.with_detection:
            det = sorted({d['label'] for d in it.get('detection', []) if d['label'] not in ('hand', 'gloved_hands')}) or ['(none detected)']
            text += f"Independently detected equipment verified present in these frame(s): [{', '.join(det)}].\n\n"
        text += f'Here are {len(frames)} frame(s) from one window, in time order. Report visible content and pick at most one concept.'
        content = [{'type': 'image_url', 'image_url': {'url': data_url(f)}} for f in frames] + [{'type': 'text', 'text': text}]
        convs.append([{'role': 'system', 'content': system}, {'role': 'user', 'content': content}])
    outs = llm.chat(convs, sampling_params=SamplingParams(temperature=0.0, max_tokens=600),
                    chat_template_content_format=args.chat_content_format)

    OUT.mkdir(parents=True, exist_ok=True)
    n_ok = n_none = 0
    with open(OUT / f'{args.tag}.jsonl', 'w') as f:
        for it, o in zip(items, outs):
            t = o.outputs[0].text; p = parse_json(t) or {}
            cid = str(p.get('matched_concept_id', '')).strip(); conf = str(p.get('match_confidence', 'low')).strip().lower()
            parsed = cid in valid or cid == 'none'
            vote = cid if (cid in valid and conf in ('high', 'medium')) else 'none'   # construction-time rule: low confidence counts as none
            n_ok += parsed; n_none += vote == 'none'
            f.write(json.dumps({'item_id': it['item_id'], 'released_concept': it['concept_id'], 'model': args.model,
                                'with_detection': args.with_detection, 'parsed': parsed, 'concept': cid if parsed else None,
                                'confidence': conf, 'vote': vote, 'observation': p.get('visual_observation'),
                                'raw': t[:600]}) + '\n')
    agree = sum(json.loads(l)['vote'] == json.loads(l)['released_concept'] for l in open(OUT / f'{args.tag}.jsonl'))
    print(f'[consensus] {args.tag}: {len(items)} items, parsed {n_ok}, vote none {n_none}, agrees with released {agree}', flush=True)


if __name__ == '__main__':
    main()
