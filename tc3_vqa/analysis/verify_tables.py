# Checks the manuscript's tables and the counts quoted in its text against the released package and prints a
# pass/fail line per value. Needs only the package: set TC3_VQA_RELEASE to its root.
from tc3_vqa.paths import RELEASE
import json, re, statistics as st
from collections import Counter

DEP = RELEASE
ok = fail = 0


def chk(label, got, want, tol=0.006):
    global ok, fail
    good = abs(got - want) <= tol if isinstance(want, float) else got == want
    ok, fail = ok + good, fail + (not good)
    print(f'  {"OK  " if good else "FAIL"} {label:56s} paper {want}   computed {round(got, 4) if isinstance(got, float) else got}')


def load(path):
    return json.load(open(f'{DEP}/{path}'))


def load_items():
    return [json.loads(l) for l in open(f'{DEP}/data/items.jsonl')]


def table_concepts():
    print('\nTable 1  concepts')
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
        if m is not None:
            chk(f'{c} MARCH group', got.pop() if len(got) == 1 else str(got), m)
    chk('column total answerable', sum(v[0] for v in want.values()), 431)
    chk('column total refusal', sum(v[1] for v in want.values()), 150)
    chk('column total safety-critical', sum(v[2] for v in want.values()), 346)


def data_overview():
    print('\nData Overview and Usage Notes')
    it = load_items()
    ans = [x for x in it if x['task_type'] == 'answerable']
    chk('items', len(it), 581); chk('answerable', len(ans), 431); chk('refusal', len(it) - len(ans), 150)
    chk('concepts', len({x['concept_id'] for x in ans}), 11)
    chk('source videos', len({x['video_id'] for x in it}), 190)
    q = Counter(qq['type'] for x in it for qq in x['questions'])
    chk('questions total', sum(q.values()), 1860)
    for t, want in [('recognition_mcq', 431), ('doctrine_scene', 427), ('reasoning', 425), ('how', 427), ('refusal', 150)]:
        chk(f'  {t}', q[t], want)
    four = {'recognition_mcq', 'doctrine_scene', 'reasoning', 'how'}
    chk('answerable items with all four types', sum(1 for x in ans if four <= {qq['type'] for qq in x['questions']}), 418)
    seg = sorted(x['segment']['t_end_s'] - x['segment']['t_start_s'] for x in it if x.get('segment'))
    chk('segment seconds, median', round(st.median(seg)), 10)
    chk('segment seconds, quartiles', (round(seg[len(seg) // 4]), round(seg[3 * len(seg) // 4])), (4, 30))
    nf = Counter(len(x['frame_refs']) for x in it)
    chk('single-frame items', nf[1], 309)
    chk('multi-frame items', sum(v for k, v in nf.items() if k > 1), 272)
    per = Counter(x['video_id'] for x in it)
    chk('items per video, median / max', (st.median(per.values()), max(per.values())), (3.0, 8))
    coco = load('data/detection_coco.json')
    chk('equipment boxes', len(coco['annotations']), 1565)
    chk('frames with boxes', len({i['file_name'] for i in coco['images']}), 743)
    chk('classes with boxes', len({a['category_id'] for a in coco['annotations']}), 16)
    chk('ontology classes', len(coco['categories']), 17)
    chk('frames referenced', len({f['frame_id'] for x in it for f in x['frame_refs']}), 903)
    spans, chunks, srcs = Counter(), Counter(), Counter()
    for x in it:
        for qq in x['questions']:
            p = qq.get('provenance')
            if p:
                spans[(p['citation_id'], p['char_start'], p['char_end'])] += 1
                chunks[p['citation_id']] += 1
                srcs[p['source_id']] += 1
    chk('quoted answers', sum(spans.values()), 1279)
    chk('distinct spans / chunks / documents', (len(spans), len(chunks), len(srcs)), (313, 49, 6))


def table_release_audit():
    print('\nTable 2  release-wide audit')
    r = load('results/release_audit.json')
    models = ['qwen3vl32b', 'internvl3_78b', 'medgemma27b']
    want = {'answerability': ('refusal', 'how_same_frames', [98.7, 99.3, 88.0], [12.2, 34.5, 3.6], 98.7, 10.8, 0.91),
            'anchoring_doctrine': ('doctrine_scene_own', 'doctrine_scene_swapped', [53.9, 50.4, 63.5], [4.7, 6.6, 52.7], 58.8, 7.7, 0.46),
            'anchoring_how': ('how_own', 'how_swapped', [87.8, 93.4, 94.1], [4.9, 9.1, 89.2], 94.6, 11.0, 0.86),
            'reasoning': ('own_answer', 'swapped_answer', [86.4, 89.4, 90.8], [30.8, 33.2, 38.0], 91.5, 33.0, 0.84)}
    for task, (own, ctl, w_own, w_ctl, maj, maj_ctl, ac1) in want.items():
        t = r['tasks'][task]; v = t['pass_verdict']
        rate = lambda d: 100 * d[v] / (100 - d.get('null', 0))   # the paper excludes unparsed verdicts
        for m, a, b in zip(models, w_own, w_ctl):
            chk(f'{task} {m}', rate(t['per_model'][m][own]), a, tol=0.06)
            chk(f'{task} {m} control', rate(t['per_model'][m][ctl]), b, tol=0.06)
        chk(f'{task} majority', t['majority_pass_pct'], maj, tol=0.06)
        chk(f'{task} majority control', t['control_majority_pass_pct'], maj_ctl, tol=0.06)
        chk(f'{task} AC1', t['ac1'], ac1, tol=0.006)
    rv = r['reasoning_vs_physicians']['majority']
    chk('reasoning audit, both physicians accept', tuple(rv['model_correct_when_both_physicians_correct']), (48, 49))
    chk('reasoning audit, a physician flags', tuple(rv['model_correct_when_a_physician_flags']), (35, 38))


def ac1(cases, m):
    cats = sorted({x for c in cases for x in c}); n = len(cases)
    pa = sum(sum(c.count(x) * (c.count(x) - 1) for x in set(c)) / (m * (m - 1)) for c in cases) / n
    pi = {x: sum(c.count(x) for c in cases) / (n * m) for x in cats}
    pe = sum(pi[x] * (1 - pi[x]) for x in cats) / (len(cats) - 1)
    return (pa - pe) / (1 - pe)


def table_ratings():
    print('\nTable 3  four-rater ratings (88 items)')
    items = load('gold/gold_adjudication.json')['items']
    chk('n items', len(items), 88)
    roles = ['physician_1', 'physician_2', 'student_1', 'student_2']
    want = {'recognition': [69, 69, 93, 94], 'doctrine': [74, 91, 95, 76], 'reasoning': [59, 92, 92, 60], 'how': [68, 93, 93, 41]}
    all_four = {'recognition': .75, 'doctrine': .68, 'reasoning': .57, 'how': .47}
    pairs = {('physician_1', 'physician_2'): {'recognition': .80, 'doctrine': .66, 'reasoning': .48, 'how': .61, 'severity': .27},
             ('student_1', 'student_2'): {'recognition': .95, 'doctrine': .71, 'reasoning': .52, 'how': .26}}
    for axis in list(want) + ['severity']:
        for role, w in zip(roles, want.get(axis, [])):
            lab = [items[k]['labels'][role].get(axis) for k in items]
            lab = [x for x in lab if x is not None]
            chk(f'{axis} {role} % correct', round(100 * lab.count('correct') / len(lab)), w)
        if axis in all_four:
            cases = [[items[k]['labels'][r].get(axis) for r in roles] for k in items]
            chk(f'{axis} AC1, all four', ac1([c for c in cases if None not in c], 4), all_four[axis], tol=0.011)
        for pair, w in pairs.items():
            if axis in w:
                cases = [[items[k]['labels'][r].get(axis) for r in pair] for k in items]
                chk(f'{axis} AC1, {pair[0][:-2]} pair', ac1([c for c in cases if None not in c], 2), w[axis], tol=0.011)
    sev = eval(re.search(r'SEV = (\{.*?\})', open(f'{DEP}/eval/scripts/score_eval.py').read(), re.S).group(1))
    concept = {x['item_id']: x['concept_id'] for x in load_items()}
    byw = {1: [], 2: [], 3: []}
    for k in items:
        ps = [int(items[k]['labels'][r]['severity']) for r in roles[:2] if items[k]['labels'][r].get('severity') in ('1', '2', '3')]
        if ps:
            byw[sev.get(concept[k], 2)].append(sum(ps) / len(ps))
    for w, mean, n in ((1, 1.0, 7), (2, 2.05, 10), (3, 2.4, 71)):
        chk(f'physician severity, weight {w}', (round(sum(byw[w]) / len(byw[w]), 2), len(byw[w])), (mean, n))


def table_results():
    print('\nTable 4  baselines')
    b = load('results/baselines.json')
    doc = {r['model']: r for r in load('results/doctrine_open.json')}
    want = {'qwen2vl7b': (.854, .951, .864, .267, .576, .260, .921),
            'qwen25vl7b': (.805, .963, .874, .285, .574, .833, .546),
            'phi35v': (.666, .867, .703, .225, .588, .933, .888),
            'internvl3_8b': (.840, .948, .845, .243, .579, .860, .479),
            'internvl3_38b': (.740, .986, .913, .264, .523, .933, .420)}
    for m, (rec, easy, hard, partial, unsup, refusal, rwhr) in want.items():
        chk(f'{m} recognition', b[m]['rec_acc'], rec)
        chk(f'{m} doctrine-MCQ easy', b[m]['mcq']['easy'][0], easy)
        chk(f'{m} doctrine-MCQ hard', b[m]['mcq']['hard'][0], hard)
        chk(f'{m} doctrine-open', doc[m]['doctrine_partial'], partial)
        chk(f'{m} doctrine-open unsupported', doc[m]['doctrine_unsupported'], unsup)
        chk(f'{m} refusal', b[m]['refusal_abstain_acc'], refusal)
        chk(f'{m} RWHR', b[m]['RWHR_default'], rwhr)


def table_controls():
    print('\nTable 5  frame substitution and first-frame truncation')
    c = load('results/controls.json')
    want = {'qwen2vl7b': (.854, .139, -.715, 2.3, .853, .828, -.025, .42),
            'qwen25vl7b': (.807, .081, -.726, 18.8, .814, .686, -.127, None),
            'phi35v': (.666, .332, -.334, 10.2, .716, .637, -.078, .014),
            'internvl3_8b': (.840, .123, -.717, 17.9, .873, .745, -.127, None),
            'internvl3_38b': (.740, .044, -.696, 44.8, .819, .725, -.093, .002)}
    for m, (own, sw, d1, ab, win, one, d2, p) in want.items():
        r = c[m]
        chk(f'{m} own', r['base'], own); chk(f'{m} swapped', r['swap'], sw); chk(f'{m} swap delta', r['swap'] - r['base'], d1)
        chk(f'{m} swap abstain %', round(100 * r['swap_abstain'], 1), ab, tol=0.06)
        chk(f'{m} window', r['window'], win); chk(f'{m} first frame', r['one'], one); chk(f'{m} frame delta', r['one'] - r['window'], d2)
        if p is None:
            chk(f'{m} p < 0.001', r['p'] < 0.001, True)
        else:
            chk(f'{m} p', r['p'], p)
        chk(f'{m} n', (r['n_swap'], r['n_multi']), (431, 204))


def direct_generation():
    print('\nComparison with direct generation')
    m = load('results/model_authored_qa.json')
    g = m['generators']
    opn = ['qwen_alone', 'qwen3vl', 'medgemma', 'mistral_small', 'pixtral', 'internvl3']
    r = [g[k]['wrong_open']['rate'] for k in opn]
    chk('open VLMs, other-concept questions %', (round(100 * min(r)), round(100 * max(r))), (16, 31))
    chk('pipeline, other-concept questions %', round(100 * g['released']['wrong_open']['rate'], 1), 0.4)
    r = [g[k]['wrong_open_selfpick']['rate'] for k in opn]
    chk('self-selected concept, inconsistent %', (round(100 * min(r)), round(100 * max(r))), (11, 42))
    chk('Qwen2.5-VL-72B given the concept, inconsistent', g['qwen_concept']['wrong_open']['rate'], 0.0)
    for arm, want in (('released', (95, 99)), ('qwen_alone', (72, 78))):
        c = [v['clean']['rate'] for v in m['arms'][arm].values()]
        chk(f'{arm} clean QA %, range over judges', (round(100 * min(c)), round(100 * max(c))), want)


if __name__ == '__main__':
    for f in (table_concepts, data_overview, table_release_audit, table_ratings, table_results, table_controls, direct_generation):
        try:
            f()
        except Exception as e:
            print(f'\n{f.__name__}: ERROR {type(e).__name__}: {e}')
    print(f'\n==== {ok} checks passed, {fail} failed ====')
