# Blinded judge served by vLLM: one QA record per prompt, arm never shown. Tasks: premise (frame description, then
# presuppositions), doctrine (claim labels against the passages) and rubric (medical accuracy and four other axes).
from tccc_vqa.paths import EXPERIMENTS
import argparse, base64, hashlib, io, json, re
from pathlib import Path
from PIL import Image

OUT = EXPERIMENTS + '/conventional_ablation'
IMG_MAX_SIDE, MAX_IMG = 896, 4

LENGTH_RULE = ('Length rule: do not reward or penalise the item for being longer, shorter, or more or less detailed. '
               'Judge only the content asked for.')

DESCRIBE = """You are shown frame(s) from one segment of a Tactical Combat Casualty Care (TCCC) training or field video, in time order. Describe only what can be seen. No question has been asked yet.

Output ONLY JSON: {"intervention": "the medical intervention being performed, or none, or unclear", "equipment": ["visible medical equipment"], "body_site": "body region being treated, or unclear", "casualty": "what can be seen of the casualty", "actions": "one sentence on what the people are doing"}"""

PREMISE = """You are auditing one item of a Tactical Combat Casualty Care (TCCC) visual question answering dataset. The frame(s) are shown above. This description of the frames was written independently, before the item was seen:
{description}

Item:
{item}

Judge only what the frames show; do not judge clinical correctness.
1. presuppositions: list what the question and the answer assume is present or happening in these frames: the intervention, equipment, body site, the casualty's condition, or actions. Do not list instructions, recommendations or doctrine. For a multiple-choice item the only presupposition is the marked answer; ignore the other options. Label each "consistent" (the frames match it), "contradicted" (the frames show something different, such as a different intervention or body site), or "unverifiable" (the frames neither show it nor rule it out).
2. question_fits_frames: false if the question is about an intervention, equipment, body site or situation that the frames do not show; otherwise true.
3. scene_fact_unanswerable: true only if answering requires a fact about this particular scene that cannot be seen, such as elapsed time, vital signs, the amount of blood lost, or whether an outcome occurred. A question about what doctrine directs for the intervention shown is not such a question: set false.
{length_rule}

Output ONLY JSON: {{"presuppositions": [{{"text": "...", "label": "consistent|contradicted|unverifiable"}}], "question_fits_frames": true or false, "scene_fact_unanswerable": true or false}}"""

DOCTRINE = """You are checking one answer from a Tactical Combat Casualty Care (TCCC) question answering dataset against doctrine excerpts. The excerpts were retrieved for this question and serve as the reference.

1. Split the answer into atomic clinical claims: each indication, step, parameter (number, dose, duration, size, location), device, or directive is one claim. Do not count words that only restate the question.
2. Label each claim "supported" (an excerpt states it or directly implies it), "contradicted" (an excerpt states something incompatible with it), or "not_addressed" (no excerpt supports or contradicts it).
{length_rule} Label each claim on its own; an answer is not better or worse for having more or fewer claims.

Doctrine excerpts:
{packet}

Question: {question}
Answer: {answer}

Output ONLY JSON: {{"claims": [{{"text": "...", "label": "supported|contradicted|not_addressed"}}]}}"""

# TC3-VLM rubric (Kim et al., IEEE Access), with doctrine excerpts in place of a reference answer.
RUBRIC = """You are an expert evaluator of Tactical Combat Casualty Care (TCCC) answers. Score the answer below using the doctrine excerpts as the reference.

EVALUATION CRITERIA (1-5 scale):
1. Medical Accuracy: are medical facts, terminology, and procedures correct per TCCC guidelines, including drug names, dosages, routes, and timing? 1 = dangerous errors or misinformation; 3 = mostly accurate with minor inaccuracies; 5 = completely accurate.
2. Protocol Adherence: does it follow TCCC/MARCH priorities and care-phase constraints (Care Under Fire, Tactical Field Care, TACEVAC)? 1 = violates core protocols; 3 = minor deviations; 5 = full adherence.
3. Completeness: are the critical elements for safe execution present? 1 = missing essential information that could lead to harm; 3 = key information present but some important details missing; 5 = all critical elements covered. Length is irrelevant.
4. Actionability: can a combat medic realistically execute this in the field? 1 = impractical; 3 = feasible but challenging; 5 = practical and field-ready.
5. Safety: could following this answer harm the casualty; are relevant cautions addressed? 1 = dangerous advice; 3 = generally safe but missing important warnings; 5 = fully safe.
{length_rule} A concise correct answer scores the same as a detailed correct answer; penalise only information that is missing or wrong.

Doctrine excerpts:
{packet}

Question: {question}
Answer: {answer}

Output ONLY JSON: {{"medical_accuracy": 1-5, "protocol_adherence": 1-5, "completeness": 1-5, "actionability": 1-5, "safety": 1-5, "rationale": "one sentence"}}"""


def data_url(path):
    im = Image.open(path).convert('RGB'); w, h = im.size
    s = IMG_MAX_SIDE / max(w, h)
    if s < 1.0:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=88)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def item_text(r):
    s = f"Question: {r['question']}\n"
    if r.get('options'):
        s += 'Options: ' + ' | '.join(r['options']) + '\n'
    return s + f"Answer: {r['answer']}"


def packet_text(r):
    return '\n'.join(f"[{i + 1}] ({p['source_id']}) {p['text']}" for i, p in enumerate(r['packet']))


def images(frames):
    return [{'type': 'image_url', 'image_url': {'url': data_url(f)}} for f in frames[:MAX_IMG]]


def build(task, r, desc=None):
    if task == 'describe':                               # r is a tuple of frame paths
        return [{'role': 'user', 'content': images(r) + [{'type': 'text', 'text': DESCRIBE}]}]
    if task == 'premise':
        d = desc[tuple(r['frames'][:MAX_IMG])]
        text = PREMISE.format(description=json.dumps(d), item=item_text(r), length_rule=LENGTH_RULE)
        return [{'role': 'user', 'content': images(r['frames']) + [{'type': 'text', 'text': text}]}]
    tpl = DOCTRINE if task == 'doctrine' else RUBRIC
    text = tpl.format(length_rule=LENGTH_RULE, packet=packet_text(r), question=r['question'], answer=r['answer'])
    return [{'role': 'user', 'content': text}]


def token_prompt(tok, messages):
    """Text-only prompts are rendered and tokenised here: with transformers 5 the tokenising chat template returns a
    BatchEncoding, which vLLM's chat path reads as token ids and rejects. The rendered template already carries BOS."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {'prompt_token_ids': tok(text, add_special_tokens=False)['input_ids']}


def parse_json(text):
    # reasoning judges: R1 distills open <think> in the prompt (only </think> is generated); gpt-oss decodes its
    # analysis channel before 'assistantfinal'. Keep only the final answer.
    text = (text or '').split('</think>')[-1].split('assistantfinal')[-1]
    for pat in (r'\{.*\}', r'\[.*\]'):                   # an object, or a list of per-frame objects
        m = re.search(pat, text, re.S)
        try:
            d = json.loads(m.group(0)) if m else None
        except Exception:
            continue
        if isinstance(d, list):
            d = {'frames': d} if d and all(isinstance(x, dict) for x in d) else None
        if d is not None:
            return d
    return None


def parse(task, text):
    d = parse_json(text)
    if d is None:
        return None
    if task == 'premise':
        ok = isinstance(d.get('presuppositions'), list) and isinstance(d.get('question_fits_frames'), bool) \
            and isinstance(d.get('scene_fact_unanswerable'), bool)
    elif task == 'doctrine':
        ok = isinstance(d.get('claims'), list) and all(
            isinstance(c, dict) and c.get('label') in ('supported', 'contradicted', 'not_addressed') for c in d['claims'])
    else:
        axes = ('medical_accuracy', 'protocol_adherence', 'completeness', 'actionability', 'safety')
        ok = all(isinstance(d.get(a), int) and 1 <= d[a] <= 5 for a in axes)
    return d if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--run', default='pilot')
    ap.add_argument('--tasks', default='premise,doctrine,rubric', help="comma or '+' separated ('+' survives sbatch --export)")
    ap.add_argument('--tensor-parallel-size', type=int, default=2)
    ap.add_argument('--max-model-len', type=int, default=16384)
    ap.add_argument('--mistral', action='store_true', help='Mistral-format checkpoint (Pixtral)')
    ap.add_argument('--text-only', action='store_true', help='text model: premise task is skipped')
    ap.add_argument('--quantization', default=None, help='e.g. fp8 to fit Llama-3.1-405B on 4 GPUs')
    ap.add_argument('--max-tokens', type=int, default=1536, help='raise for reasoning judges (DeepSeek-R1, gpt-oss)')
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--shard', type=int, default=0)
    args = ap.parse_args()

    run = Path(OUT) / args.run
    recs = [json.loads(l) for l in open(run / 'judge_inputs.jsonl')]
    if args.num_shards > 1:                              # shard by frame set: both arms of an item share one description
        fkey = lambda r: int(hashlib.md5('|'.join(r['frames'][:MAX_IMG]).encode()).hexdigest(), 16)
        recs = [r for r in recs if fkey(r) % args.num_shards == args.shard]
    suffix = f'.s{args.shard}of{args.num_shards}' if args.num_shards > 1 else ''
    tasks = [t for t in re.split(r'[,+]', args.tasks) if t and not (args.text_only and t == 'premise')]

    from vllm import LLM, SamplingParams
    kw = dict(model=args.model, tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.9, trust_remote_code=True, enforce_eager=True)   # skip compile / CUDA-graph capture
    if args.mistral:
        kw.update(tokenizer_mode='mistral', config_format='mistral', load_format='mistral')
    else:
        kw.update(dtype='bfloat16')
    if not args.text_only:
        kw.update(limit_mm_per_prompt={'image': MAX_IMG})
    if args.quantization:
        kw.update(quantization=args.quantization)
    llm = LLM(**kw)

    out_dir = run / 'judge' / args.tag; out_dir.mkdir(parents=True, exist_ok=True)
    desc = {}
    if 'premise' in tasks:                               # blind scene description, shared by both arms of an item
        keys = sorted({tuple(r['frames'][:MAX_IMG]) for r in recs})
        outs = llm.chat([build('describe', k) for k in keys], sampling_params=SamplingParams(temperature=0.0, max_tokens=512))
        with open(out_dir / f'describe{suffix}.jsonl', 'w') as f:
            for k, o in zip(keys, outs):
                t = o.outputs[0].text; d = parse_json(t)
                desc[k] = d if d else {'description': t.strip()[:800]}
                f.write(json.dumps({'frames': list(k), 'parsed': d, 'raw': t[:2000]}) + '\n')
        print(f'[judge:{args.tag}] describe: parsed {sum(parse_json(o.outputs[0].text) is not None for o in outs)}/{len(keys)}', flush=True)
    tok = None if args.mistral else llm.get_tokenizer()
    def ask(task, rs, sp):
        if task == 'premise' or tok is None:                 # multimodal or Mistral-format: vLLM chat path works
            return llm.chat([build(task, r, desc) for r in rs], sampling_params=sp)
        return llm.generate([token_prompt(tok, build(task, r, desc)) for r in rs], sampling_params=sp)

    for task in tasks:
        todo = [r for r in recs if task == 'premise' or r['type'] != 'recognition_mcq']
        done_file = out_dir / f'{task}{suffix}.jsonl'
        if done_file.exists() and {json.loads(l)['rec_id'] for l in open(done_file)} >= {r['rec_id'] for r in todo}:
            print(f'[judge:{args.tag}] {task}: already complete in {done_file.name}, skipped', flush=True)   # resume after preemption
            continue
        outs = ask(task, todo, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
        raw = [o.outputs[0].text for o in outs]
        parsed = [parse(task, t) for t in raw]
        retry = [i for i, p in enumerate(parsed) if p is None]
        if retry:                                           # one retry at low temperature for unparseable output
            outs = ask(task, [todo[i] for i in retry], SamplingParams(temperature=0.3, seed=0, max_tokens=args.max_tokens))
            for i, o in zip(retry, outs):
                raw[i] = o.outputs[0].text; parsed[i] = parse(task, raw[i])
        with open(out_dir / f'{task}{suffix}.jsonl', 'w') as f:
            for r, t, p in zip(todo, raw, parsed):
                f.write(json.dumps({'rec_id': r['rec_id'], 'task': task, 'parse_ok': p is not None,
                                    'parsed': p, 'raw': t[:4000]}) + '\n')
        ok = sum(p is not None for p in parsed)
        print(f'[judge:{args.tag}] {task}: parsed {ok}/{len(todo)} ({100 * ok / len(todo):.1f}%), retried {len(retry)}', flush=True)


if __name__ == '__main__':
    main()
