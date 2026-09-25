# Recomputes every number in the manuscript tables from the artifacts and prints one pass or fail line per cell.
from tccc_vqa.paths import EXPERIMENTS, FRAMES, RELEASE, WORK
import json, math, os
from collections import Counter

DEP = RELEASE
POOL = WORK
ok = fail = 0


def chk(label, got, want, tol=0.006):
    global ok, fail
    if got is None:
        print(f'  ??   {label:56s} paper {want}  (no artifact)'); return
    good = abs(got - want) <= tol if isinstance(want, float) else got == want
    ok, fail = ok + good, fail + (not good)
    print(f'  {"OK  " if good else "FAIL"} {label:56s} paper {want}   computed {round(got,4) if isinstance(got,float) else got}')


def load_items():
    return [json.loads(l) for l in open(f'{DEP}/data/items.jsonl')]


# ---------------------------------------------------------------- release-wide VLM audit (tab:release_audit)
def table_release_audit():
    print('\nTable  release-wide audit (Qwen3-VL-32B, InternVL3-78B, MedGemma-27B)')
    D = EXPERIMENTS + '/release_audit'
    tags = ['qwen3vl32b', 'internvl3_78b', 'medgemma27b']
    V = {t: {(r['condition'], r['item_id']): r['verdict'] for r in map(json.loads, open(f'{D}/{t}.jsonl'))} for t in tags}
    want = {('refusal', 'unanswerable'): ([98.7, 99.3, 88.0], 98.7), ('how_same_frames', 'unanswerable'): ([12.2, 34.5, 3.6], 10.8),
            ('doctrine_scene_own', 'specific'): ([53.9, 50.4, 63.5], 58.8), ('doctrine_scene_swapped', 'specific'): ([4.7, 6.6, 52.7], 7.7),
            ('how_own', 'specific'): ([87.8, 93.4, 94.1], 94.6), ('how_swapped', 'specific'): ([4.9, 9.1, 89.2], 11.0),
            ('own_answer', 'correct'): ([86.4, 89.4, 90.8], 91.5), ('swapped_answer', 'correct'): ([30.8, 33.2, 38.0], 33.0)}
    for (cond, ok), (per, maj) in want.items():
        ids = sorted({i for (c, i) in V[tags[0]] if c == cond})
        for t, w in zip(tags, per):
            rated = [V[t][(cond, i)] for i in ids if V[t].get((cond, i)) is not None]
            chk(f'{cond} {t}', round(100 * sum(v == ok for v in rated) / len(rated), 1), w)
        full = [sum(V[t][(cond, i)] == ok for t in tags) for i in ids if all(V[t].get((cond, i)) is not None for t in tags)]
        chk(f'{cond} majority', round(100 * sum(k >= 2 for k in full) / len(full), 1), maj)


# ---------------------------------------------------------------- paired rubric table (tab:conventional_rubric)
def table_conventional_rubric():
    print('\nTable  paired medical accuracy, reference vs Claude Opus 5 on the same questions')
    d = json.load(open(EXPERIMENTS + '/conventional_ablation/claude_answer_only_lso/paired_rubric.json'))
    chk('pairs', d['pairs'], 849); chk('same question', d['same_question'], 849); chk('same packet', d['same_packet'], 849)
    want = {'llama70b': (4.62, 4.61, -0.01, -0.10, 0.09, 108, 623, 118), 'medgemma27b': (4.79, 4.82, 0.03, -0.03, 0.09, 53, 757, 39),
            'phi4': (4.45, 4.52, 0.07, -0.01, 0.15, 150, 586, 113), 'deepseek70b': (4.31, 4.37, 0.07, -0.05, 0.18, 227, 414, 208),
            'gptoss120b': (4.08, 4.20, 0.12, 0.00, 0.24, 266, 384, 199)}
    for j, w in want.items():
        v = d['judges'][j]
        got = (round(v['released'], 2), round(v['opus'], 2), round(v['diff_opus_minus_released'], 2), round(v['ci'][0], 2),
               round(v['ci'][1], 2), v['opus_higher'], v['tie'], v['released_higher'])
        for name, g, x in zip(('reference', 'opus', 'diff', 'ci_lo', 'ci_hi', 'higher', 'equal', 'lower'), got, w):
            chk(f'{j} {name}', g, x)


# ---------------------------------------------------------------- Table 2: released-set summary
def table_stats():
    print('\nTable 2  released-set summary')
    it = load_items()
    ans = [x for x in it if x['task_type'] == 'answerable']
    ref = [x for x in it if x['task_type'] == 'refusal']
    chk('items', len(it), 581); chk('answerable', len(ans), 431); chk('refusal', len(ref), 150)
    chk('concepts', len({x['concept_id'] for x in ans}), 11)
    chk('source videos', len({x['video_id'] for x in it}), 190)
    q = Counter(qq['type'] for x in it for qq in x['questions'])
    chk('questions total', sum(q.values()), 1860)
    for t, want in [('recognition_mcq', 431), ('doctrine_scene', 427), ('reasoning', 425), ('how', 427), ('refusal', 150)]:
        chk(f'  {t}', q[t], want)
    nf = sorted(len(x['frame_refs']) for x in it)
    chk('median frames per item', nf[len(nf) // 2], 1)
    chk('multi-frame items', sum(1 for x in it if len(x['frame_refs']) > 1), 272)
    coco = json.load(open(f'{DEP}/data/detection_coco.json'))
    chk('equipment boxes', len(coco['annotations']), 1565)
    chk('equipment classes', len(coco['categories']), 17)
    chk('frames with boxes', len({i['file_name'] for i in coco['images']}), 743)
    from PIL import Image
    fr = FRAMES
    kept = {f['frame_id'] for x in it for f in x['frame_refs']}
    sizes = Counter(Image.open(os.path.join(fr, f)).size for f in os.listdir(fr) if f in kept)
    lo = min(sizes, key=lambda s: s[0] * s[1]); hi = max(sizes, key=lambda s: s[0] * s[1])
    chk('smallest resolution', lo, (320, 240)); chk('largest resolution', hi, (3840, 2160))
    chk('modal resolution', sizes.most_common(1)[0][0], (1920, 1080))
    sc = sum(1 for x in ans if x['safety_critical'])
    chk('safety-critical answerable', sc, 346)
    chk('safety-critical %', round(100 * sc / len(ans)), 80)


# ---------------------------------------------------------------- Table 4: agreement contingency
def table_agreement():
    print('\nTable 4  cross-check x audit contingency (full 1,515 pool)')
    rows = []
    for path in (f'{POOL}/_archive/_archive_pre0624/clinician_review_queue.jsonl',
                 f'{POOL}/_archive/loose_pre/audit_rejected.jsonl'):
        for line in open(path):
            r = json.loads(line)
            if r.get('task_type') == 'refusal': continue
            cc = (r.get('checks') or {}).get('cross_check_visible')
            au = (r.get('independent_audit') or {}).get('concept_visible')
            if cc and au: rows.append((cc, au))
    c = lambda a, b: sum(1 for x, y in rows if x == a and y == b)
    want = {('yes', 'yes'): 280, ('yes', 'partial'): 289, ('yes', 'no'): 234,
            ('partial', 'yes'): 3, ('partial', 'partial'): 12, ('partial', 'no'): 6,
            ('no', 'yes'): 53, ('no', 'partial'): 169, ('no', 'no'): 469}
    for k, v in want.items(): chk(f'cell {k[0]}/{k[1]}', c(*k), v)
    chk('total', len(rows), 1515)
    for col, v in [('yes', 336), ('partial', 470), ('no', 709)]:
        chk(f'audit total {col}', sum(1 for _, y in rows if y == col), v)
    for row, v in [('yes', 803), ('partial', 21), ('no', 691)]:
        chk(f'cross-check total {row}', sum(1 for x, _ in rows if x == row), v)


# ---------------------------------------------------------------- Table 5: four-rater adjudication
def table_gradient():
    print('\nTable 5  four-rater adjudication (88-item sample)')
    g = json.load(open(f'{DEP}/gold/gold_adjudication.json'))
    items = g['items']
    chk('n items', len(items), 88)
    roles = ['physician_1', 'physician_2', 'student_1', 'student_2']
    want = {'recognition': [69, 69, 93, 94], 'doctrine': [74, 91, 95, 76],
            'reasoning': [59, 92, 92, 60], 'how': [68, 93, 93, 41]}
    ac1_want = {'recognition': .75, 'doctrine': .68, 'reasoning': .57, 'how': .47}
    # 5.4: mean physician severity by the concept-level RWHR weight (score_eval.py SEV table)
    import re as _re
    sev = eval(_re.search(r'SEV = (\{.*?\})', open(f'{DEP}/eval/scripts/score_eval.py').read(), _re.S).group(1))
    concept = {json.loads(l)['item_id']: json.loads(l)['concept_id'] for l in open(f'{DEP}/data/items.jsonl')}
    byw = {1: [], 2: [], 3: []}
    for k in items:
        ps = [int(items[k]['labels'][r]['severity']) for r in roles[:2] if items[k]['labels'][r].get('severity') in ('1', '2', '3')]
        if ps:
            byw[sev.get(concept[k], 2)].append(sum(ps) / len(ps))
    for w, want_mean, want_n in ((1, 1.0, 7), (2, 2.05, 10), (3, 2.4, 71)):
        chk(f'severity mean, weight {w} (n={want_n})', (round(sum(byw[w]) / len(byw[w]), 2), len(byw[w])), (want_mean, want_n))
    pair_want = {('physician_1', 'physician_2'): {'recognition': .80, 'doctrine': .66, 'reasoning': .48, 'how': .61, 'severity': .27},
                 ('student_1', 'student_2'): {'recognition': .95, 'doctrine': .71, 'reasoning': .52, 'how': .26}}
    for axis, vals in list(want.items()) + [('severity', None)]:   # severity: text only (physician-pair AC1)
        for role, w in zip(roles, vals or []):
            lab = [items[k]['labels'][role].get(axis) for k in items]
            lab = [x for x in lab if x is not None]
            got = round(100 * sum(1 for x in lab if x == 'correct') / len(lab))
            chk(f'{axis} {role}', got, w)
        # Gwet AC1 over the four raters, complete cases
        cases = [[items[k]['labels'][r].get(axis) for r in roles] for k in items]
        cases = [c for c in cases if all(x is not None for x in c)]
        cats = sorted({x for c in cases for x in c})
        n, m = len(cases), len(roles)
        pa = sum(sum(c.count(x) * (c.count(x) - 1) for x in set(c)) / (m * (m - 1)) for c in cases) / n
        pi = {x: sum(c.count(x) for c in cases) / (n * m) for x in cats}
        K = len(cats)
        pe = sum(pi[x] * (1 - pi[x]) for x in cats) / (K - 1)
        if axis in ac1_want:
            chk(f'{axis} AC1', (pa - pe) / (1 - pe), ac1_want[axis], tol=0.011)
        for pair, want_pair in pair_want.items():
            if axis not in want_pair:
                continue
            cases = [[items[k]['labels'][r].get(axis) for r in pair] for k in items]
            cases = [c for c in cases if all(x is not None for x in c)]
            cats = sorted({x for c in cases for x in c}); n = len(cases)
            pa = sum(sum(c.count(x) * (c.count(x) - 1) for x in set(c)) / 2 for c in cases) / n
            pi = {x: sum(c.count(x) for c in cases) / (n * 2) for x in cats}
            pe = sum(pi[x] * (1 - pi[x]) for x in cats) / (len(cats) - 1)
            chk(f'{axis} AC1 {pair[0][:-2]} pair', (pa - pe) / (1 - pe), want_pair[axis], tol=0.011)


# ---------------------------------------------------------------- Table 6: baseline leaderboard
def table_results():
    print('\nTable 6  baseline leaderboard')
    lb = {r['model']: r for r in json.load(open(f'{POOL}/leaderboard.json'))}
    doc = json.load(open(EXPERIMENTS + '/consensus_perception/doctrine_scores.json'))
    dd = {r['model']: r['3'] for r in doc}
    want = {'qwen2vl7b': (.854, .951, .864, .267, .260, .921),
            'qwen25vl7b': (.805, .963, .874, .285, .833, .546),
            'phi35v': (.666, .867, .703, .225, .933, .888),
            'internvl3_8b': (.840, .948, .845, .243, .860, .479),
            'internvl3_38b': (.740, .986, .913, .264, .933, .420)}
    mcq = {}
    for kind in ('easy', 'hard'):
        for m in want:
            f = f'{POOL}/doctrine_mcq_{kind}_{m}.jsonl'
            if os.path.exists(f):
                rs = [json.loads(l) for l in open(f)]
                rel = {json.loads(l)['item_id'] for l in open(f'{DEP}/data/items.jsonl')}
                rs = [r for r in rs if r.get('id') in rel]
                mcq[(kind, m)] = sum(1 for r in rs if r.get('pred') == r.get('gold')) / len(rs)
    unsup = {'qwen2vl7b': .576, 'qwen25vl7b': .574, 'phi35v': .588, 'internvl3_8b': .579, 'internvl3_38b': .523}
    for m, (rec, e, h, dop, ref, rw) in want.items():
        chk(f'{m} recognition', lb[m]['rec_acc'], rec)
        chk(f'{m} MCQ easy', mcq.get(('easy', m)), e)
        chk(f'{m} MCQ hard', mcq.get(('hard', m)), h)
        chk(f'{m} doctrine-open (partial-credit NLI)', dd.get(m, {}).get('doctrine_partial'), dop)
        chk(f'{m} doctrine-open n', dd.get(m, {}).get('n'), 427)
        chk(f'{m} doctrine-open unsupported-claim rate', dd.get(m, {}).get('doctrine_unsupported'), unsup[m])
        chk(f'{m} refusal', lb[m]['refusal_abstain_acc'], ref)
        chk(f'{m} RWHR', lb[m]['RWHR_default'], rw)


# ---------------------------------------------------------------- Table 8: difficulty strata
def table_difficulty():
    print('\nTable 8  difficulty-stratified recognition (adjudicated sample)')
    g = json.load(open(f'{DEP}/gold/gold_adjudication.json'))['items']
    easy = {k for k, v in g.items() if v['difficulty_flag'] == 0}
    hard = {k for k, v in g.items() if v['difficulty_flag'] == 2}
    chk('n easy', len(easy), 58); chk('n hard', len(hard), 24)
    want = {'qwen2vl7b': (.845, .750), 'qwen25vl7b': (.724, .583), 'phi35v': (.586, .625),
            'internvl3_8b': (.879, .708), 'internvl3_38b': (.879, .583)}
    me, mh = [], []
    for m, (we, wh) in want.items():
        rs = {r['id']: r for r in (json.loads(l) for l in open(f'{POOL}/eval_recog_{m}.jsonl'))}
        acc = lambda ids: sum(1 for i in ids if i in rs and rs[i]['pred'] == rs[i]['gold']) / len([i for i in ids if i in rs])
        ae, ah = acc(easy), acc(hard)
        me.append(ae); mh.append(ah)
        chk(f'{m} easy', ae, we); chk(f'{m} hard', ah, wh)
    chk('panel mean easy', sum(me) / len(me), .783)
    chk('panel mean hard', sum(mh) / len(mh), .650)
    chk('gap', sum(me) / len(me) - sum(mh) / len(mh), .133)


# ---------------------------------------------------------------- concept table
def table_concepts():
    print('\nTable 1  closed concept set')
    it = load_items()
    ans = [x for x in it if x['task_type'] == 'answerable']
    ref = [x for x in it if x['task_type'] == 'refusal']
    want = {'wound_packing': (129, 28, 129, 'M'), 'tourniquet_application': (137, 55, 137, 'M'),
            'junctional_hemorrhage': (4, 2, 4, 'M'), 'tourniquet_conversion': (0, 0, 0, None),
            'nasopharyngeal_airway': (41, 17, 2, 'A'), 'surgical_airway_cric': (11, 5, 11, 'A'),
            'chest_seal': (29, 11, 29, 'R'), 'needle_decompression': (27, 8, 27, 'R'),
            'iv_io_access': (17, 9, 0, 'C'), 'txa_administration': (9, 3, 7, 'C'),
            'hypothermia_prevention': (13, 6, 0, 'H'), 'tccc_documentation': (14, 6, 0, 'process')}
    for c, (a, f, sc, m) in want.items():
        chk(f'{c} answerable', sum(1 for x in ans if x['concept_id'] == c), a)
        chk(f'{c} refusal', sum(1 for x in ref if x['concept_id'] == c), f)
        chk(f'{c} safety-critical', sum(1 for x in ans if x['concept_id'] == c and x['safety_critical']), sc)
        got = {x['march_category'] for x in ans if x['concept_id'] == c}
        if m is not None: chk(f'{c} MARCH group', got.pop() if len(got) == 1 else str(got), m)
    chk('concepts listed', len(want), 12)
    chk('column total answerable', sum(v[0] for v in want.values()), 431)
    chk('column total refusal', sum(v[1] for v in want.values()), 150)
    chk('column total safety-critical', sum(v[2] for v in want.values()), 346)


if __name__ == '__main__':
    for f in (table_concepts, table_release_audit, table_conventional_rubric, table_stats, table_agreement, table_gradient, table_results, table_difficulty):
        try: f()
        except Exception as e: print(f'\n{f.__name__}: ERROR {type(e).__name__}: {e}')
    print(f'\n==== {ok} checks passed, {fail} failed ====')
