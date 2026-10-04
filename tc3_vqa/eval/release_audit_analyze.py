# Summarises the release-wide audit: verdict distributions per model for each judgement and its control, majority
# verdicts, Gwet's AC1, and agreement of the reasoning verdicts with the physicians. Writes release_audit_summary.json.
from tc3_vqa.paths import EXPERIMENTS, RELEASE
import argparse, json
from collections import Counter
from pathlib import Path

ROOT = Path(EXPERIMENTS + '/release_audit')
GOLD = RELEASE + '/gold/gold_adjudication.json'
TAGS = ['qwen3vl32b', 'internvl3_78b', 'mistralsmall24b', 'medgemma27b']
PHYS = ('physician_1', 'physician_2')
# task -> (primary condition, control condition(s), verdict that counts as a pass on the primary condition)
TASKS = {'answerability': ('refusal', ['how_same_frames'], 'unanswerable'),
         'anchoring_doctrine': ('doctrine_scene_own', ['doctrine_scene_swapped'], 'specific'),
         'anchoring_how': ('how_own', ['how_swapped'], 'specific'),
         'reasoning': ('own_answer', ['swapped_answer'], 'correct')}


def ac1_binary(rows):
    """multi-rater Gwet AC1 for binary ratings; rows = per-item lists of booleans with the same number of raters"""
    if not rows or len(rows[0]) < 2:
        return None
    r = len(rows[0])
    pa = sum((k * (k - 1) + (r - k) * (r - k - 1)) / (r * (r - 1)) for k in (sum(x) for x in rows)) / len(rows)
    pi = sum(sum(x) / r for x in rows) / len(rows)
    pe = 2 * pi * (1 - pi)
    return round((pa - pe) / (1 - pe), 3) if pe < 1 else None


def dist(recs):
    c = Counter(r['verdict'] for r in recs); n = len(recs)
    return {k: round(100 * v / n, 1) for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))} | {'n': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='', help="e.g. 'smoke_'")
    ap.add_argument('--models', default=','.join(TAGS), help='comma-separated auditor tags to include')
    args = ap.parse_args()
    tags = [t for t in args.models.split(',') if (ROOT / f'{args.prefix}{t}.jsonl').exists()]
    R = {t: [json.loads(l) for l in open(ROOT / f'{args.prefix}{t}.jsonl')] for t in tags}
    out = {'models': tags, 'tasks': {}}
    print('models:', ', '.join(tags))
    for t in tags:
        print(f"  {t}: parsed {sum(r['verdict'] is not None for r in R[t])}/{len(R[t])}")

    for name, (prim, controls, ok) in TASKS.items():
        rec = {'pass_verdict': ok, 'per_model': {}}
        print(f'\n== {name} (pass on {prim} = {ok})')
        for t in tags:
            rec['per_model'][t] = {c: dist([r for r in R[t] if r['condition'] == c]) for c in [prim] + controls}
            print(f'  {t:16s} ' + ' | '.join(f'{c}: {rec["per_model"][t][c]}' for c in [prim] + controls))
        # majority and agreement on the primary condition, over judgements every model parsed
        key = lambda r: r['item_id']
        V = {t: {key(r): r['verdict'] for r in R[t] if r['condition'] == prim and r['verdict'] is not None} for t in tags}
        common = sorted(set.intersection(*(set(v) for v in V.values()))) if tags else []
        rows = [[V[t][i] == ok for t in tags] for i in common]
        need = len(tags) // 2 + 1
        rec['primary_all_parsed'] = len(common)
        rec['majority_pass_pct'] = round(100 * sum(sum(x) >= need for x in rows) / max(1, len(rows)), 1)
        rec['ac1'] = ac1_binary(rows)
        Vc = {t: {key(r): r['verdict'] for r in R[t] if r['condition'] == controls[0] and r['verdict'] is not None} for t in tags}
        cc = sorted(set.intersection(*(set(v) for v in Vc.values()))) if tags else []
        rec['control_majority_pass_pct'] = round(100 * sum(sum(Vc[t][i] == ok for t in tags) >= need for i in cc) / max(1, len(cc)), 1)
        print(f"  majority pass: primary {rec['majority_pass_pct']}% (n={len(common)}) vs control {rec['control_majority_pass_pct']}% (n={len(cc)}); AC1 {rec['ac1']}")
        out['tasks'][name] = rec

    # reasoning against the physicians on the adjudicated sample
    gold = json.load(open(GOLD))['items']
    cal = {}
    for t in tags:
        v = {r['item_id']: r['verdict'] for r in R[t] if r['condition'] == 'own_answer' and r['verdict'] is not None}
        pairs = [(v[i], [g['labels'][p].get('reasoning') for p in PHYS]) for i, g in gold.items() if i in v]
        pairs = [(m, p) for m, p in pairs if None not in p]
        ok_items = [m for m, p in pairs if all(x == 'correct' for x in p)]
        flagged = [m for m, p in pairs if any(x != 'correct' for x in p)]
        cal[t] = {'n': len(pairs),
                  'model_correct_when_both_physicians_correct': [sum(m == 'correct' for m in ok_items), len(ok_items)],
                  'model_correct_when_a_physician_flags': [sum(m == 'correct' for m in flagged), len(flagged)]}
        print(f'  physicians vs {t}: {cal[t]}')
    # the same comparison for the majority verdict over the auditors that parsed the item
    need = len(tags) // 2 + 1
    V = {t: {r['item_id']: r['verdict'] for r in R[t] if r['condition'] == 'own_answer' and r['verdict'] is not None} for t in tags}
    common = [i for i in gold if all(i in V[t] for t in tags)]
    pairs = [(sum(V[t][i] == 'correct' for t in tags) >= need, [gold[i]['labels'][p].get('reasoning') for p in PHYS]) for i in common]
    pairs = [(m, p) for m, p in pairs if None not in p]
    ok_items = [m for m, p in pairs if all(x == 'correct' for x in p)]
    flagged = [m for m, p in pairs if any(x != 'correct' for x in p)]
    cal['majority'] = {'n': len(pairs), 'model_correct_when_both_physicians_correct': [sum(ok_items), len(ok_items)],
                       'model_correct_when_a_physician_flags': [sum(flagged), len(flagged)]}
    print(f"  physicians vs majority: {cal['majority']}")
    out['reasoning_vs_physicians'] = cal
    dst = ROOT / f'{args.prefix}release_audit_summary.json'
    json.dump(out, open(dst, 'w'), indent=1)
    print(f'\nwrote {dst}')


if __name__ == '__main__':
    main()
