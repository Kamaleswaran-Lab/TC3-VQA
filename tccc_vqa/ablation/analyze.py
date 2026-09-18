# Aggregates the blinded judge outputs of the reference-versus-model-authored comparison: clean-QA rate, claim
# labels, rubric means, video-clustered bootstrap intervals and inter-judge agreement, per arm and question type.
from tccc_vqa.paths import EXPERIMENTS, RELEASE
import argparse, json, random, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

OUT = EXPERIMENTS + '/conventional_ablation'
ITEMS = RELEASE + '/data/items.jsonl'
AXES = ('medical_accuracy', 'protocol_adherence', 'completeness', 'actionability', 'safety')
OPEN = ('doctrine_scene', 'reasoning', 'how')


def load(run):
    key = json.load(open(run / 'arm_key.json'))
    J = defaultdict(dict)                                     # J[tag][task][rec_id] = parsed or None
    for d in sorted((run / 'judge').glob('*')):
        for f in sorted(d.glob('*.jsonl')):                   # task.jsonl or task.s<k>of<n>.jsonl shards
            task = f.name.split('.')[0]
            if task == 'describe':
                continue
            J[d.name].setdefault(task, {}).update({r['rec_id']: r['parsed'] for r in map(json.loads, open(f))})
    return key, J


def record_metrics(key, J):
    """per judge, per record: dict of metric -> value (None when the judge has no parsed verdict)."""
    M = defaultdict(dict)
    for tag, tasks in J.items():
        for rid, k in key.items():
            m = {}
            p = tasks.get('premise', {}).get(rid)
            if p and 'presuppositions' in p:
                contra = any(a.get('label') == 'contradicted' for a in p['presuppositions'] if isinstance(a, dict))
                m['premise_error'] = (not p['question_fits_frames']) or contra
                m['unanswerable'] = p['scene_fact_unanswerable']
            d = tasks.get('doctrine', {}).get(rid)
            if d is not None and k['type'] in OPEN:
                labels = [c['label'] for c in d['claims']]
                m['claims'] = len(labels)
                m['unsupported'] = sum(l != 'supported' for l in labels)
                m['contradicted'] = sum(l == 'contradicted' for l in labels)
            r = tasks.get('rubric', {}).get(rid)
            if r is not None and k['type'] in OPEN:
                for a in AXES:
                    m[a] = r[a]
            if m:
                M[tag][rid] = m
    return M


def rate(rows, num, den=None):
    rows = [r for r in rows if num in r and (den is None or den in r)]
    if not rows:
        return None
    if den:
        d = sum(r[den] for r in rows)
        return sum(r[num] for r in rows) / d if d else None
    return float(np.mean([float(r[num]) for r in rows]))


def summarise(key, M, rids):
    out = {}
    for tag, recs in M.items():
        rows = [recs[i] for i in rids if i in recs]
        s = {'n': len(rows),
             'premise_error': rate(rows, 'premise_error'), 'unanswerable': rate(rows, 'unanswerable'),
             'unsupported_claim_rate': rate(rows, 'unsupported', 'claims'),
             'contradicted_claim_rate': rate(rows, 'contradicted', 'claims'),
             'any_contradicted': rate([{**r, 'ac': r['contradicted'] > 0} for r in rows if 'contradicted' in r], 'ac')}
        for a in AXES:
            v = [r[a] for r in rows if a in r]
            s[a] = float(np.mean(v)) if v else None
        out[tag] = s
    return out


def clean(m):
    """clean QA = the question does not name only another concept (judge-free) and no claim is contradicted.
    VLM premise verdicts are not used: both open VLM judges proved unreliable in the pilot."""
    return not m.get('wrong_concept', False) and m.get('contradicted', 0) == 0


def video_bootstrap(key, M, tag, arm, metric_fn, reps, seed=0):
    by_vid = defaultdict(list)
    for rid, k in key.items():
        if k['arm'] == arm and k['type'] in OPEN and rid in M[tag] and 'claims' in M[tag][rid]:
            by_vid[k['video_id']].append(M[tag][rid])
    vids = list(by_vid); rnd = random.Random(seed); est = []
    for _ in range(reps):
        rows = [m for v in (rnd.choice(vids) for _ in vids) for m in by_vid[v]]
        est.append(metric_fn(rows))
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def ac1(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    pa = np.mean(x == y); pi = (x.mean() + y.mean()) / 2; pe = 2 * pi * (1 - pi)
    return float((pa - pe) / (1 - pe)) if pe < 1 else float('nan')


# judge-free concept check: which of the 12 concepts a text names, by keyword (junctional before limb tourniquet)
CONCEPT_KEYS = [
    ('junctional_hemorrhage', r'junctional'), ('tourniquet_conversion', r'convert|conversion'),
    ('tourniquet_application', r'tourniquet'), ('needle_decompression', r'decompression'),
    ('chest_seal', r'chest seal|occlusive|sucking chest|seal (a|the) chest'), ('surgical_airway_cric', r'cricothyr|surgical airway'),
    ('nasopharyngeal_airway', r'nasopharyngeal|\bnpa\b|nasal airway'), ('txa_administration', r'tranexamic|\btxa\b'),
    ('iv_io_access', r'\biv\b|\bio\b|intravenous|intraosseous|vascular access'),
    ('hypothermia_prevention', r'hypotherm|thermal|blanket|warming'),
    ('tccc_documentation', r'1380|casualty card|document'), ('wound_packing', r'pack|hemostatic|gauze')]


def concepts_named(text):
    t = (text or '').lower()
    return [c for c, pat in CONCEPT_KEYS if re.search(pat, t)]


def concept_check(run, key):
    """strict = the released concept; lenient = the same MARCH category, so adjacent concepts (tourniquet and
    tourniquet conversion, junctional haemorrhage and wound packing) are not counted as mismatches."""
    recs = {json.loads(l)['rec_id']: json.loads(l) for l in open(run / 'judge_inputs.jsonl')}
    march = {x['concept_id']: x['march_category'] for x in map(json.loads, open(ITEMS)) if x['task_type'] == 'answerable'}
    out = {}
    for arm in ('A', 'C'):
        mcq = Counter(); q = Counter()
        for rid, k in key.items():
            if k['arm'] != arm:
                continue
            r = recs[rid]
            named = concepts_named(r['answer'] if k['type'] == 'recognition_mcq' else r['question'])
            if k['type'] == 'recognition_mcq':
                if not named:
                    mcq['unmapped'] += 1
                else:
                    mcq['strict_correct'] += named[0] == k['concept_id']
                    mcq['march_correct'] += march.get(named[0]) == march[k['concept_id']]
                    mcq['mapped'] += 1
            elif named:
                q['naming_a_concept'] += 1
                q['strict_mismatch'] += k['concept_id'] not in named
                q['march_mismatch'] += march[k['concept_id']] not in {march.get(c) for c in named}
        out[arm] = {'mcq': dict(mcq), 'open_questions': dict(q)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='pilot')
    ap.add_argument('--reps', type=int, default=1000)
    args = ap.parse_args()
    run = Path(OUT) / args.run
    key, J = load(run)
    M = record_metrics(key, J)
    recs = {json.loads(l)['rec_id']: json.loads(l) for l in open(run / 'judge_inputs.jsonl')}
    for rid, k in key.items():                              # attach the judge-free wrong-concept flag to every judge's row
        if k['type'] in OPEN:
            named = concepts_named(recs[rid]['question'])
            wc = bool(named) and k['concept_id'] not in named
            for tag in M:
                if rid in M[tag]:
                    M[tag][rid]['wrong_concept'] = wc
    S = {'parse_rate': {t: {task: sum(v is not None for v in d.values()) / len(d) for task, d in tasks.items()}
                        for t, tasks in J.items()}}

    groups = {}
    for arm in ('A', 'C'):
        for typ in OPEN + ('recognition_mcq', 'open_pooled'):
            rids = [r for r, k in key.items() if k['arm'] == arm and (k['type'] == typ or (typ == 'open_pooled' and k['type'] in OPEN))]
            groups[f'{arm}:{typ}'] = summarise(key, M, rids)
    S['by_arm_type'] = groups

    # clean-QA rate with video-bootstrap CI, and item-level McNemar (item clean = all its open QA clean)
    S['clean'] = {}
    for tag in M:
        res = {}
        for arm in ('A', 'C'):
            fn = lambda rows: float(np.mean([clean(m) for m in rows])) if rows else float('nan')
            rows = [M[tag][r] for r, k in key.items() if k['arm'] == arm and k['type'] in OPEN and r in M[tag]
                    and 'claims' in M[tag][r]]                  # a clean verdict needs this judge's doctrine labels
            res[arm] = {'rate': fn(rows), 'ci95': video_bootstrap(key, M, tag, arm, fn, args.reps)}
        item = defaultdict(dict)
        for r, k in key.items():
            if r in M[tag] and k['type'] in OPEN and 'claims' in M[tag][r]:
                item[k['item_id']].setdefault(k['arm'], []).append(clean(M[tag][r]))
        pairs = [(all(v['A']), all(v['C'])) for v in item.values() if 'A' in v and 'C' in v]
        b = sum(a and not c for a, c in pairs); c_ = sum(c and not a for a, c in pairs)
        res['item_mcnemar'] = {'items': len(pairs), 'A_clean_C_not': b, 'C_clean_A_not': c_,
                               'p_exact': stats.binomtest(b, b + c_, 0.5).pvalue if b + c_ else None}
        S['clean'][tag] = res

    # length diagnostics: answer words by arm and type; within-arm Spearman of rubric mean vs words
    S['length'] = {}
    for typ in OPEN:
        row = {}
        for arm in ('A', 'C'):
            w = [k['answer_words'] for k in key.values() if k['arm'] == arm and k['type'] == typ]
            row[arm] = {'median_words': float(np.median(w)) if w else None}
            if arm == 'C':
                ib = [k['in_band'] for k in key.values() if k['arm'] == 'C' and k['type'] == typ]
                row[arm]['in_band'] = float(np.mean(ib)) if ib else None
            for tag in M:
                pts = [(k['answer_words'], np.mean([M[tag][r][a] for a in AXES])) for r, k in key.items()
                       if k['arm'] == arm and k['type'] == typ and r in M[tag] and AXES[0] in M[tag][r]]
                if len(pts) > 5:
                    row[arm][f'spearman_rubric_words:{tag}'] = float(stats.spearmanr(*zip(*pts)).statistic)
        S['length'][typ] = row

    S['concept_check'] = concept_check(run, key)

    # fairness and length diagnostics for the doctrine endpoint, per judge
    #  - arm A split by whether its cited passage is inside the question-retrieved packet (the 'not in packet' rows are
    #    the fairer comparison with arm C, which has no source of its own in the packet)
    #  - arm C restricted to answers inside the length band
    #  - unsupported-claim rate within answer-length tertiles, arm A vs arm C
    S['doctrine_diagnostics'] = {}
    open_rows = [(r, k) for r, k in key.items() if k['type'] in OPEN and k['arm'] in ('A', 'C')]
    cuts = np.quantile([k['answer_words'] for _, k in open_rows], [1 / 3, 2 / 3])
    def tertile(w):
        return 0 if w <= cuts[0] else (1 if w <= cuts[1] else 2)
    for tag in M:
        def urate(pred):
            rows = [M[tag][r] for r, k in open_rows if r in M[tag] and 'claims' in M[tag][r] and pred(k)]
            return {'n': len(rows), 'unsupported_claim_rate': rate(rows, 'unsupported', 'claims'),
                    'contradicted_claim_rate': rate(rows, 'contradicted', 'claims')}
        S['doctrine_diagnostics'][tag] = {
            'A_cited_in_packet': urate(lambda k: k['arm'] == 'A' and k['cited_in_packet']),
            'A_cited_not_in_packet': urate(lambda k: k['arm'] == 'A' and not k['cited_in_packet']),
            'C_all': urate(lambda k: k['arm'] == 'C'),
            'C_in_band': urate(lambda k: k['arm'] == 'C' and k['in_band']),
            'length_tertiles_words': [float(c) for c in cuts],
            'by_length_tertile': {f't{t}': {arm: urate(lambda k, t=t, arm=arm: k['arm'] == arm and tertile(k['answer_words']) == t)
                                            for arm in ('A', 'C')} for t in range(3)}}

    # length probe: arm A answer vs the same answer plus the next source sentence (same packet)
    S['length_probe'] = {}
    base = {(k['item_id'], k['type']): r for r, k in key.items() if k['arm'] == 'A'}
    pairs = [(base[(k['item_id'], k['type'])], r) for r, k in key.items()
             if k['arm'] == 'A_probe' and (k['item_id'], k['type']) in base]
    for tag in M:
        pp = [(M[tag][a], M[tag][p]) for a, p in pairs if a in M[tag] and p in M[tag]]
        uns = [(a['unsupported'] / a['claims'] if a['claims'] else 0, p['unsupported'] / p['claims'] if p['claims'] else 0)
               for a, p in pp if 'claims' in a and 'claims' in p]
        rub = [(np.mean([a[x] for x in AXES]), np.mean([p[x] for x in AXES])) for a, p in pp if AXES[0] in a and AXES[0] in p]
        S['length_probe'][tag] = {
            'pairs': len(pp),
            'unsupported_rate_diff_mean': float(np.mean([q - o for o, q in uns])) if uns else None,
            'rubric_mean_diff': float(np.mean([q - o for o, q in rub])) if rub else None,
            'rubric_changed_share': float(np.mean([abs(q - o) > 0 for o, q in rub])) if rub else None}

    # inter-judge agreement
    S['agreement'] = {}
    tags = sorted(M)
    for i, t1 in enumerate(tags):
        for t2 in tags[i + 1:]:
            both = [r for r in M[t1] if r in M[t2]]
            ag = {}
            for f in ('premise_error', 'unanswerable'):
                xs = [(M[t1][r][f], M[t2][r][f]) for r in both if f in M[t1][r] and f in M[t2][r]]
                if xs:
                    ag[f'ac1_{f}'] = ac1(*zip(*xs))
            xs = [(M[t1][r]['contradicted'] > 0, M[t2][r]['contradicted'] > 0) for r in both
                  if 'contradicted' in M[t1][r] and 'contradicted' in M[t2][r]]
            if xs:
                ag['ac1_any_contradicted'] = ac1(*zip(*xs))
            for a in AXES:
                xs = [(M[t1][r][a], M[t2][r][a]) for r in both if a in M[t1][r] and a in M[t2][r]]
                if len(xs) > 5:
                    x, y = map(np.array, zip(*xs))
                    ag[a] = {'pearson': float(stats.pearsonr(x, y).statistic), 'spearman': float(stats.spearmanr(x, y).statistic),
                             'agr_pm1': float(np.mean(abs(x - y) <= 1)), 'mae': float(np.mean(abs(x - y)))}
            S['agreement'][f'{t1}~{t2}'] = ag

    json.dump(S, open(run / 'summary.json', 'w'), indent=1)
    print(json.dumps({'parse_rate': S['parse_rate'], 'concept_check': S['concept_check'], 'clean': S['clean'],
                      'doctrine_diagnostics': S['doctrine_diagnostics'], 'length_probe': S['length_probe'],
                      'length': S['length']}, indent=1))
    for g, v in groups.items():
        for tag, s in v.items():
            print(g, tag, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in s.items()})
    print(f'wrote {run}/summary.json')


if __name__ == '__main__':
    main()
