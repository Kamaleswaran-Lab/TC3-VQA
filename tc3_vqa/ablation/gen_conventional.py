# Model-authored arm: a VLM served by vLLM sees the frames and writes each question and answer itself, one per released
# question type, with the answer length matched to the reference. --concept-given tells it the audited intervention;
# --concept-from <tag> tells it the intervention the same model chose from the 12 concepts in the consensus run <tag>.
from tc3_vqa.paths import EXPERIMENTS, FRAMES, RELEASE
import argparse, base64, io, json, random, re, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ITEMS = RELEASE + '/data/items.jsonl'
FRAMES = FRAMES
OUT = EXPERIMENTS + '/conventional_ablation'
IMG_MAX_SIDE, MAX_IMG = 896, 4
MCQ_Q = 'Which TCCC intervention is being performed in this frame?'

SYSTEM = ('You are building a visual question answering dataset for Tactical Combat Casualty Care (TCCC) training. '
          'You are shown frame(s) from one segment of a video, in time order. Write one item of the requested type '
          'about this scene. Answer according to TCCC doctrine.')

TYPE_SPEC = {
    'doctrine_scene': 'a question about what TCCC doctrine directs for the intervention shown in the scene (for example '
                      'its indication, technique, placement or site, sequence or next step, how to verify that it worked, '
                      'or a caution or contraindication), and its answer',
    'reasoning': 'a scene-specific question that requires reasoning from what is visible to a doctrinal decision (for '
                 'example the next step, the rationale for an action, a consequence, or what to do if the intervention '
                 'is not working), and its answer',
    'how': "a question of the form 'Given the <equipment or situation visible in the scene>, how should the responder "
           "<carry out the procedure>?', and its answer describing how to do it",
}


def words(t):
    return len(re.findall(r'\S+', t or ''))


def band(n):
    return max(1, round(min(0.8 * n, n - 5))), round(max(1.25 * n, n + 5))


def data_url(path):
    im = Image.open(path).convert('RGB'); w, h = im.size
    s = IMG_MAX_SIDE / max(w, h)
    if s < 1.0:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=88)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def prompt(job):
    if job['type'] == 'recognition_mcq':
        return (f"Write a multiple-choice recognition item. The question is fixed: '{MCQ_Q}' "
                'Give 4 distinct options naming TCCC interventions, exactly one of which is correct for this scene, '
                'and mark the correct one.\n'
                'Output ONLY JSON: {"options": [4 strings], "answer": "<one of the options>"}')
    lo, hi = job['band']
    given = (f"The intervention shown in these frames is: {job['concept_text']}.\n" if job.get('concept_text') else '')
    return (given + f"Write {TYPE_SPEC[job['type']]}.\n"
            f"The question should be about {job['target_q_words']} words. The answer must contain between {lo} and "
            f"{hi} words, aiming for {job['target_words']} words; an answer shorter than {lo} words is rejected.\n"
            'Write plain text only: no bullet points, no numbering, no preamble.\n'
            'Output ONLY JSON: {"question": "...", "answer": "..."}')


def parse(job, text):
    m = re.search(r'\{.*\}', text or '', re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if job['type'] == 'recognition_mcq':
        opts = d.get('options')
        if not (isinstance(opts, list) and len(opts) == 4 and d.get('answer') in opts):
            return None
        return {'question': MCQ_Q, 'options': opts, 'answer': d['answer']}
    if not (isinstance(d.get('question'), str) and isinstance(d.get('answer'), str) and d['answer'].strip()):
        return None
    return {'question': d['question'].strip(), 'answer': d['answer'].strip()}


def in_band(job, r):
    if r is None:
        return False
    if job['type'] == 'recognition_mcq':
        return True
    lo, hi = job['band']
    return lo <= words(r['answer']) <= hi


def select_items(items, n, seed):
    if not n:
        return items
    by = defaultdict(list)
    for it in sorted(items, key=lambda x: x['item_id']):
        by[it['concept_id']].append(it)
    rnd = random.Random(seed)
    for v in by.values():
        rnd.shuffle(v)
    picked = []
    while len(picked) < n and any(by.values()):          # round-robin over concepts, largest first
        for c in sorted(by, key=lambda c: -len(by[c])):
            if by[c] and len(picked) < n:
                picked.append(by[c].pop())
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2.5-VL-72B-Instruct')
    ap.add_argument('--run', default='pilot')
    ap.add_argument('--n-items', type=int, default=60, help='0 = all answerable items')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tensor-parallel-size', type=int, default=2)
    ap.add_argument('--max-model-len', type=int, default=16384)
    ap.add_argument('--mistral', action='store_true', help='Mistral-format checkpoint (Pixtral, Mistral-Small)')
    ap.add_argument('--hf-overrides', default=None, help='JSON config overrides passed to vLLM (e.g. Qwen3-VL text_config fix)')
    ap.add_argument('--mm-processor-kwargs', default=None, help='JSON processor kwargs passed to vLLM (e.g. Qwen3-VL image pixel cap)')
    ap.add_argument('--chat-content-format', default='auto', choices=['auto', 'string', 'openai'], help="'string' for templates that concatenate message content (InternVL3)")
    ap.add_argument('--concept-given', action='store_true',
                    help='concept-given arm: the released (audited) intervention is stated in the prompt; open types only')
    ap.add_argument('--concept-from', default=None, metavar='TAG',
                    help='self-pick arm: the intervention this model chose in the consensus run <TAG> is stated in the prompt; open types only')
    ap.add_argument('--items', default=ITEMS)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.items)]
    names = {x['concept_id']: q['answer'] for x in items for q in x['questions'] if q['type'] == 'recognition_mcq'}
    picks = {}
    if args.concept_from:
        picks = {r['item_id']: r['vote'] for r in map(json.loads, open(f'{EXPERIMENTS}/consensus_perception/{args.concept_from}.jsonl'))}
    given = args.concept_given or bool(args.concept_from)
    items = select_items([x for x in items if x['task_type'] == 'answerable'], args.n_items, args.seed)
    jobs = []
    for it in items:
        frames = [f"{FRAMES}/{r['frame_id']}" for r in it['frame_refs']][:MAX_IMG]
        assert frames and all(Path(f).exists() for f in frames), it['item_id']
        concept_text = next((q['answer'] for q in it['questions'] if q['type'] == 'recognition_mcq'), it['concept_id'])
        if args.concept_from:
            if not picks.get(it['item_id']):
                continue                                   # the model named none of the 12: no self-pick arm for this item
            concept_text = names.get(picks[it['item_id']], picks[it['item_id']])
        for q in it['questions']:
            if given and q['type'] == 'recognition_mcq':
                continue
            n = words(q.get('answer_normalized') or q['answer'])
            jobs.append({'item_id': it['item_id'], 'concept_id': it['concept_id'], 'video_id': it['video_id'],
                         'frames': frames, 'type': q['type'], 'target_words': n,
                         'target_q_words': words(q['question']), 'band': band(n),
                         'concept_text': concept_text if given else None})
    print(f'[gen] {len(items)} items, {len(jobs)} questions: {dict(Counter(j["type"] for j in jobs))}', flush=True)

    from vllm import LLM, SamplingParams
    kw = dict(model=args.model, tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.9, trust_remote_code=True, limit_mm_per_prompt={'image': MAX_IMG},
              enforce_eager=True)                                  # skip compile / CUDA-graph capture
    kw.update(dict(tokenizer_mode='mistral', config_format='mistral', load_format='mistral') if args.mistral else dict(dtype='bfloat16'))
    if args.hf_overrides:   # vLLM 0.15 + transformers 5: Qwen3VLTextConfig lacks tie_word_embeddings
        kw['hf_overrides'] = json.loads(args.hf_overrides)
    if args.mm_processor_kwargs:   # frames are already <= IMG_MAX_SIDE; a cap above that only bounds the profiling pass
        kw['mm_processor_kwargs'] = json.loads(args.mm_processor_kwargs)
    llm = LLM(**kw)

    urls = {}
    def conv(job):
        for f in job['frames']:
            if f not in urls:
                urls[f] = data_url(f)
        content = [{'type': 'image_url', 'image_url': {'url': urls[f]}} for f in job['frames']]
        content.append({'type': 'text', 'text': prompt(job)})
        return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]

    convs = [conv(j) for j in jobs]
    outs = llm.chat(convs, sampling_params=SamplingParams(temperature=0.0, max_tokens=512), chat_template_content_format=args.chat_content_format)
    for j, o in zip(jobs, outs):
        j['result'] = parse(j, o.outputs[0].text); j['pass'] = 1

    redo = [i for i, j in enumerate(jobs) if not in_band(j, j['result'])]
    print(f'[gen] pass 1: {len(jobs) - len(redo)}/{len(jobs)} parsed and in band; resampling {len(redo)}', flush=True)
    if redo:
        sp = SamplingParams(n=16, temperature=0.7, top_p=0.95, seed=args.seed, max_tokens=512)
        outs = llm.chat([convs[i] for i in redo], sampling_params=sp, chat_template_content_format=args.chat_content_format)
        for i, o in zip(redo, outs):
            j = jobs[i]
            cands = [r for r in (parse(j, c.text) for c in o.outputs) if r is not None]
            if j['result'] is not None:
                cands.insert(0, j['result'])
            if not cands:
                continue
            key = (lambda r: 0) if j['type'] == 'recognition_mcq' else (lambda r: abs(words(r['answer']) - j['target_words']))
            ok = [r for r in cands if in_band(j, r)]
            j['result'] = min(ok or cands, key=key); j['pass'] = 2

    out_dir = Path(OUT) / args.run; out_dir.mkdir(parents=True, exist_ok=True)
    by_item = defaultdict(list)
    for j in jobs:
        r = j['result'] or {}
        by_item[j['item_id']].append({
            'type': j['type'], 'question': r.get('question'), 'options': r.get('options'), 'answer': r.get('answer'),
            'target_words': j['target_words'], 'words': words(r.get('answer')) if r else None, 'band': j['band'],
            'parsed': bool(r), 'in_band': in_band(j, j['result']), 'pass': j['pass']})
    meta = {j['item_id']: j for j in jobs}
    with open(out_dir / 'arm_c.jsonl', 'w') as f:
        for iid, qs in by_item.items():
            m = meta[iid]
            f.write(json.dumps({'item_id': iid, 'concept_id': m['concept_id'], 'video_id': m['video_id'],
                                'frames': m['frames'], 'generator': args.model, 'questions': qs}) + '\n')

    report = {}
    for t in sorted({j['type'] for j in jobs}):
        js = [j for j in jobs if j['type'] == t]
        rec = {'n': len(js), 'parsed': sum(j['result'] is not None for j in js),
               'in_band': sum(in_band(j, j['result']) for j in js), 'resampled': sum(j['pass'] == 2 for j in js)}
        if t != 'recognition_mcq':
            d = [words(j['result']['answer']) - j['target_words'] for j in js if j['result']]
            rec.update({'target_words_median': st.median(j['target_words'] for j in js),
                        'words_median': st.median(words(j['result']['answer']) for j in js if j['result']),
                        'diff_median': st.median(d), 'diff_iqr': [round(x, 1) for x in st.quantiles(d, n=4)[::2]]})
        report[t] = rec
    json.dump(report, open(out_dir / 'length_report.json', 'w'), indent=1)
    print(json.dumps(report, indent=1))
    print(f'[gen] wrote {out_dir}/arm_c.jsonl', flush=True)


if __name__ == '__main__':
    main()
