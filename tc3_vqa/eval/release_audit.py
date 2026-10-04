# Release-wide audit by one VLM served by vLLM. Each judgement covers one question and its frames, blind to item type
# and concept, and is paired with a control the model should reject. Writes <tag>.jsonl, one record per judgement.
from tc3_vqa.paths import EXPERIMENTS
import argparse, json
from collections import defaultdict
from pathlib import Path

from tc3_vqa.consensus.consensus_perceive import FRAMES, ITEMS, MAX_IMG, data_url, parse_json

OUT = Path(EXPERIMENTS + '/release_audit')

HEAD = """You review questions from a visual question-answering dataset about Tactical Combat Casualty Care (TCCC).
You are shown 1-4 frames from one moment of a video, in time order, and {what}."""

PROMPTS = {
    'answerability': (HEAD.format(what='one question about them') + """
Decide whether the question can be answered from these frames together with standard TCCC knowledge.
- "answerable": a careful clinician could give a specific, justified answer using what the frames show and standard TCCC knowledge.
- "unanswerable": the question asks for information that neither the frames nor standard knowledge can supply, so the correct response is to decline.

Output one JSON object: {"reason": "<=25 words", "verdict": "answerable|unanswerable"}""", ('answerable', 'unanswerable')),

    'anchoring': (HEAD.format(what='one question about them') + """
Decide how the question relates to what these frames show. Judge only what the question states or assumes about the scene,
not whether its answer can be seen.
- "specific": the question points to something visible in these frames, such as the equipment, step, body region, injury or
  situation shown.
- "generic": the question asks about care of this kind without pointing to anything in these particular frames; a question
  about indications, criteria or general doctrine that could be asked about any image of this kind is generic.
- "mismatch": the question states or assumes something about the scene (equipment, step, body region or injury) that these
  frames do not show or that they contradict.

Output one JSON object: {"reason": "<=25 words", "verdict": "specific|generic|mismatch"}""", ('specific', 'generic', 'mismatch')),

    'reasoning': (HEAD.format(what="one clinical-reasoning question about the scene with the dataset's answer") + """
Judge whether the answer is correct for this question and this scene.
- "correct": the answer is clinically correct and appropriate for the question and the scene.
- "minor": the answer is broadly correct but has a minor inaccuracy, omission or imprecision.
- "incorrect": the answer is wrong, unsafe, or does not answer the question.

Output one JSON object: {"reason": "<=25 words", "verdict": "correct|minor|incorrect"}""", ('correct', 'minor', 'incorrect')),
}


def build_tasks(items):
    """one entry per judgement: (task, condition, item_id, frame_ids, user_text)"""
    by_id = {x['item_id']: x for x in items}
    qof = lambda it, t: next((q for q in it['questions'] if q['type'] == t), None)
    frames = lambda it: [r['frame_id'] for r in it['frame_refs']]
    ans = sorted((x for x in items if x['task_type'] == 'answerable'), key=lambda x: x['item_id'])
    # substitution partner: the first released answerable item of a different concept, starting half the list away
    swap = {}
    for k, it in enumerate(ans):
        order = ans[k + len(ans) // 2:] + ans[:k + len(ans) // 2]
        swap[it['item_id']] = frames(next(o for o in order if o['concept_id'] != it['concept_id']))
    tasks = []
    # (1) answerability: every refusal question, and the HOW question of the item whose frames it reuses
    for it in items:
        if it['task_type'] == 'answerable':
            continue
        tasks.append(('answerability', 'refusal', it['item_id'], frames(it), f"Question: {qof(it, 'refusal')['question']}"))
        src = by_id.get(it.get('derived_from'))
        if src and qof(src, 'how'):
            tasks.append(('answerability', 'how_same_frames', it['item_id'], frames(it), f"Question: {qof(src, 'how')['question']}"))
    # (2) anchoring: doctrine and HOW questions on their own frames and on the substitution partner's frames
    for it in ans:
        for t in ('doctrine_scene', 'how'):
            q = qof(it, t)
            if q:
                tasks.append(('anchoring', f'{t}_own', it['item_id'], frames(it), f"Question: {q['question']}"))
                tasks.append(('anchoring', f'{t}_swapped', it['item_id'], swap[it['item_id']], f"Question: {q['question']}"))
    # (3) reasoning: own answer, and the reasoning answer of the next item (by id) of the same concept with a different answer
    by_concept = defaultdict(list)
    for it in sorted(ans, key=lambda x: x['item_id']):
        if qof(it, 'reasoning'):
            by_concept[it['concept_id']].append(it)
    for group in by_concept.values():
        for k, it in enumerate(group):
            q = qof(it, 'reasoning')
            tasks.append(('reasoning', 'own_answer', it['item_id'], frames(it), f"Question: {q['question']}\nAnswer: {q['answer']}"))
            other = next((o for o in group[k + 1:] + group[:k] if qof(o, 'reasoning')['answer'] != q['answer']), None)
            if other:
                tasks.append(('reasoning', 'swapped_answer', it['item_id'], frames(it),
                              f"Question: {q['question']}\nAnswer: {qof(other, 'reasoning')['answer']}"))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--tensor-parallel-size', type=int, default=1)
    ap.add_argument('--max-model-len', type=int, default=16384)
    ap.add_argument('--mistral', action='store_true')
    ap.add_argument('--hf-overrides', default=None)
    ap.add_argument('--mm-processor-kwargs', default=None)
    ap.add_argument('--chat-content-format', default='auto', choices=['auto', 'string', 'openai'])
    ap.add_argument('--tasks', default='', help='comma-separated subset of tasks; default all')
    ap.add_argument('--n-per-condition', type=int, default=0, help='smoke test: first N judgements of each condition; 0 = all')
    args = ap.parse_args()

    tasks = build_tasks([json.loads(l) for l in open(ITEMS)])
    if args.tasks:
        tasks = [t for t in tasks if t[0] in args.tasks.split(',')]
    if args.n_per_condition:
        seen = defaultdict(int); keep = []
        for t in tasks:
            if seen[(t[0], t[1])] < args.n_per_condition:
                keep.append(t); seen[(t[0], t[1])] += 1
        tasks = keep

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
    for task, cond, iid, fids, text in tasks:
        content = [{'type': 'image_url', 'image_url': {'url': data_url(f'{FRAMES}/{f}')}} for f in fids[:MAX_IMG]]
        content.append({'type': 'text', 'text': text + '\n\nOutput the JSON object.'})
        convs.append([{'role': 'system', 'content': PROMPTS[task][0]}, {'role': 'user', 'content': content}])
    outs = llm.chat(convs, sampling_params=SamplingParams(temperature=0.0, max_tokens=400),
                    chat_template_content_format=args.chat_content_format)

    OUT.mkdir(parents=True, exist_ok=True)
    n_parsed = 0
    with open(OUT / f'{args.tag}.jsonl', 'w') as f:
        for (task, cond, iid, fids, text), o in zip(tasks, outs):
            t = o.outputs[0].text; p = parse_json(t) or {}
            v = str(p.get('verdict', '')).strip().lower()
            v = v if v in PROMPTS[task][1] else None
            n_parsed += v is not None
            f.write(json.dumps({'task': task, 'condition': cond, 'item_id': iid, 'model': args.model, 'verdict': v,
                                'reason': p.get('reason'), 'raw': t[:800]}) + '\n')
    print(f'[release_audit] {args.tag}: {len(tasks)} judgements, parsed {n_parsed}', flush=True)


if __name__ == '__main__':
    main()
