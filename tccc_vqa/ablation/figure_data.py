# Computes the numbers behind the model-authored QA figures on audit-confirmed items: wrong-intervention rates per
# generator, and clean-QA and rubric rates per judge. Writes figure_data.json.
from tccc_vqa.paths import CANDIDATES
import json, math, random, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

from tccc_vqa.ablation import analyze as A

EXP = Path(A.OUT)
# The comparison is reported on the candidate set before the recognition consensus: the consensus voters include
# most panel-d generators, so restricting to gated items would select scenes those models already recognise.
ITEMS = CANDIDATES + '/data/items.jsonl'
OUT = EXP / 'figure_data.json'
# judged comparison arms: (name, leave-source-out run, side in that run)
ARMS = [('released', 'full_lso', 'A'), ('qwen_concept', 'full_concept_lso', 'C'),
        ('claude_alone', 'claude_full_lso', 'C'), ('qwen_alone', 'full_lso', 'C')]
# generator runs for panel d (name, generation run, label)
GENERATORS = [('qwen_concept', 'full_concept', 'Qwen2.5-VL + verified concept'), ('claude_alone', 'claude_full', 'Claude Opus 5'),
              ('qwen_alone', 'full', 'Qwen2.5-VL-72B'), ('qwen3vl', 'gen_qwen3vl', 'Qwen3-VL-32B'),
              ('medgemma', 'gen_medgemma', 'MedGemma-27B'), ('mistral_small', 'gen_mistralsmall', 'Mistral-Small-3.2-24B'),
              ('pixtral', 'gen_pixtral', 'Pixtral-Large'), ('internvl3', 'gen_internvl3', 'InternVL3-78B')]
JUDGE_ORDER = [('llama70b', 'Llama-3.3-70B'), ('medgemma27b', 'MedGemma-27B'), ('pixtral_large', 'Pixtral-Large'),
               ('deepseek70b', 'DeepSeek-R1-Distill-70B'), ('gptoss120b', 'gpt-oss-120b'), ('phi4', 'Phi-4')]
REPS = 2000
OVERLAP_MAX = 0.5


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {'k': k, 'n': n, 'rate': p, 'lo': (c - h) / d, 'hi': (c + h) / d}


def boot(rows_by_video, fn, reps=REPS, seed=0):
    vids = [v for v, rs in rows_by_video.items() if rs]
    rnd = random.Random(seed); est = []
    for _ in range(reps):
        rows = [r for v in (rnd.choice(vids) for _ in vids) for r in rows_by_video[v]]
        est.append(fn(rows))
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def wrong_rates(questions_by_item, keep, items):
    o_k = o_n = m_k = m_n = m_other = 0
    for iid, (concept, qs) in questions_by_item.items():
        if iid not in keep:
            continue
        for q in qs:
            if q['type'] == 'recognition_mcq':
                n = A.concepts_named(q['answer']); m_n += 1; m_k += (not n or n[0] != concept)
                m_other += bool(n) and n[0] != concept
            else:
                n = A.concepts_named(q['question']); o_n += 1; o_k += bool(n) and concept not in n
    mcq = wilson(m_k, m_n)
    if mcq:   # a miss either names another listed intervention or none of the 12 (e.g. 'Direct pressure')
        mcq['k_other_listed'] = m_other; mcq['rate_other_listed'] = m_other / m_n
    return wilson(o_k, o_n), mcq


def tok(t):
    return re.findall(r'[a-z0-9]+', (t or '').lower())


def grams(ws, n=4):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def packet_overlap(answer, packet):
    g = grams(tok(answer))
    pg = set().union(*(grams(tok(p['text'])) for p in packet)) if packet else set()
    return len(g & pg) / len(g) if g else 0.0


def judge_complete(run, judge, n_open):
    d = EXP / run / 'judge' / judge
    if not d.exists():
        return False
    for task in ('doctrine', 'rubric'):
        ids = set()
        for f in d.glob(f'{task}*.jsonl'):
            ids |= {json.loads(l)['rec_id'] for l in open(f)}
        if len(ids) < n_open:
            return False
    return True


def main():
    items = {json.loads(l)['item_id']: json.loads(l) for l in open(ITEMS)}
    keep = {i for i, x in items.items() if x['task_type'] == 'answerable' and x['audit']['concept_visible'] == 'yes'}
    data = {'subset': 'audit_yes', 'n_items': len(keep), 'generators': {}, 'arms': {}, 'judges': []}

    # (d)
    released = {i: (x['concept_id'], x['questions']) for i, x in items.items() if x['task_type'] == 'answerable'}
    wo, wm = wrong_rates(released, keep, items)
    data['generators']['released'] = {'label': 'Released', 'wrong_open': wo, 'wrong_mcq': wm}
    for name, run, label in GENERATORS:
        f = EXP / run / 'arm_c.jsonl'
        if not f.exists():
            print(f'[skip d] {label}: {f} not found'); continue
        gen = {}
        for l in open(f):
            x = json.loads(l)
            gen[x['item_id']] = (x['concept_id'], [q for q in x['questions'] if q['parsed']])
        wo, wm = wrong_rates(gen, keep, items)
        data['generators'][name] = {'label': label, 'wrong_open': wo, 'wrong_mcq': wm}
        print(f"[d] {label}: open {wo['rate']:.3f} ({wo['k']}/{wo['n']})" + (f", mcq {wm['rate']:.3f}" if wm else ''), flush=True)

    # (e)/(f): judges complete in every leave-source-out run
    runs = sorted({run for _, run, _ in ARMS})
    for judge, jlabel in JUDGE_ORDER:
        ok = True
        for run in runs:
            key = json.load(open(EXP / run / 'arm_key.json'))
            n_open = sum(1 for k in key.values() if k['type'] in A.OPEN)
            ok &= judge_complete(run, judge, n_open)
        if ok:
            data['judges'].append({'key': judge, 'label': jlabel})
        else:
            print(f'[skip e/f] {jlabel}: incomplete')
    judges = [j['key'] for j in data['judges']]
    # low-overlap pairs, defined once from the released records (identical question and packet in every run)
    key0 = json.load(open(EXP / ARMS[0][1] / 'arm_key.json'))
    recs0 = {json.loads(l)['rec_id']: json.loads(l) for l in open(EXP / ARMS[0][1] / 'judge_inputs.jsonl')}
    low = {(k['item_id'], k['type']) for rid, k in key0.items() if k['arm'] == 'A' and k['type'] in A.OPEN
           and k['item_id'] in keep and packet_overlap(recs0[rid]['answer'], recs0[rid]['packet']) < OVERLAP_MAX}
    data['low_overlap_pairs'] = len(low); data['overlap_max'] = OVERLAP_MAX
    print(f'[e/f] low-overlap (item, type) pairs: {len(low)}', flush=True)

    for arm, run, side in ARMS:
        key, J = A.load(EXP / run)
        M = A.record_metrics(key, J)
        recs = {json.loads(l)['rec_id']: json.loads(l) for l in open(EXP / run / 'judge_inputs.jsonl')}
        data['arms'][arm] = {}
        for judge in judges:
            by_video = defaultdict(list)
            for rid, k in key.items():
                if k['arm'] != side or k['type'] not in A.OPEN or k['item_id'] not in keep:
                    continue
                m = M.get(judge, {}).get(rid)
                if not m or 'claims' not in m:
                    continue
                named = A.concepts_named(recs[rid]['question'])
                by_video[k['video_id']].append(dict(m, wrong_concept=bool(named) and k['concept_id'] not in named,
                                                    low_overlap=(k['item_id'], k['type']) in low))
            rows = [r for v in by_video.values() for r in v]
            clean_fn = lambda rs: float(np.mean([A.clean(r) for r in rs]))
            lo, hi = boot(by_video, clean_fn)
            rows_low = [r for r in rows if r['low_overlap']]
            entry = {'n': len(rows), 'n_low_overlap': len(rows_low), 'clean': {'rate': clean_fn(rows), 'lo': lo, 'hi': hi},
                     'unsupported_claim_rate': A.rate(rows_low, 'unsupported', 'claims'),
                     'unsupported_claim_rate_all': A.rate(rows, 'unsupported', 'claims'),
                     'contradicted_claim_rate': A.rate(rows, 'contradicted', 'claims'), 'rubric': {}, 'rubric_all': {}}
            for field, subset in (('rubric', True), ('rubric_all', False)):
                bv = {v: [r for r in rs if (r['low_overlap'] or not subset)] for v, rs in by_video.items()}
                for ax in A.AXES:
                    vals = [r for rs in bv.values() for r in rs if ax in r]
                    if not vals:
                        continue
                    fn = lambda rs, ax=ax: float(np.mean([r[ax] for r in rs if ax in r]))
                    blo, bhi = boot({v: [r for r in rs if ax in r] for v, rs in bv.items()}, fn, reps=1000)
                    entry[field][ax] = {'mean': fn(vals), 'lo': blo, 'hi': bhi}
            data['arms'][arm][judge] = entry
        print(f'[e/f] {arm}: ' + '; '.join(f"{j} clean {data['arms'][arm][j]['clean']['rate']:.3f}" for j in judges), flush=True)
    json.dump(data, open(OUT, 'w'), indent=1)
    print('judges:', judges, '| wrote', OUT)


if __name__ == '__main__':
    main()
