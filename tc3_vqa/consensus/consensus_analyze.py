# Analyses the recognition consensus: per-model agreement with the assigned concept, the vote per item, physician
# acceptance split by consensus, and the items a given rule keeps. Writes consensus_summary.json.
from tc3_vqa.paths import EXPERIMENTS, RELEASE
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(EXPERIMENTS + '/consensus_perception')
ITEMS = RELEASE + '/data/items.jsonl'
GOLD = RELEASE + '/gold/gold_adjudication.json'
PHYS = ('physician_1', 'physician_2')


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(p, 4), round((c - h) / d, 4), round((c + h) / d, 4)]


def consensus(votes, need):
    """most common non-none concept if at least `need` voters chose it; otherwise None (no consensus)"""
    c = Counter(v for v in votes if v != 'none')
    if not c:
        return None
    top, n = c.most_common(1)[0]
    return top if n >= need else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', default='', help='comma-separated voter tags (default: all jsonl files present)')
    args = ap.parse_args()
    tags = [t for t in args.tags.split(',') if t] or sorted(p.stem for p in ROOT.glob('*.jsonl'))
    V = {t: {r['item_id']: r for r in map(json.loads, open(ROOT / f'{t}.jsonl'))} for t in tags}
    items = {x['item_id']: x for x in map(json.loads, open(ITEMS)) if x['task_type'] == 'answerable'}
    gold = json.load(open(GOLD))['items']
    out = {'voters': tags, 'per_voter': {}, 'consensus': {}, 'physician': {}}

    print(f'{"voter":18s} {"n":>4s} {"parsed":>7s} {"none":>6s} {"agree":>7s}  agree on physician-confirmed / rejects when a physician says wrong')
    for t in tags:
        rs = [V[t][i] for i in items if i in V[t]]
        conf_items = [i for i in gold if all(gold[i]['labels'][p].get('recognition') == 'correct' for p in PHYS)]
        wrong_items = [i for i in gold if any(gold[i]['labels'][p].get('recognition') == 'wrong' for p in PHYS)]
        a_conf = sum(V[t][i]['vote'] == items[i]['concept_id'] for i in conf_items if i in V[t])
        rej_wrong = sum(V[t][i]['vote'] != items[i]['concept_id'] for i in wrong_items if i in V[t])
        rec = {'n': len(rs), 'parsed': sum(r['parsed'] for r in rs), 'none': sum(r['vote'] == 'none' for r in rs),
               'agree_released': sum(r['vote'] == r['released_concept'] for r in rs),
               'agree_on_physician_confirmed': [a_conf, len(conf_items)], 'reject_when_physician_wrong': [rej_wrong, len(wrong_items)]}
        out['per_voter'][t] = rec
        print(f"{t:18s} {rec['n']:4d} {rec['parsed']:7d} {rec['none']:6d} {rec['agree_released']:4d} ({100 * rec['agree_released'] / max(1, rec['n']):.1f}%)"
              f"  {a_conf}/{len(conf_items)} ; {rej_wrong}/{len(wrong_items)}")

    k = len(tags)
    for need in sorted({math.ceil((k + 1) / 2), math.ceil(k / 2)}):     # strict majority and at-least-half
        cons = {i: consensus([V[t][i]['vote'] for t in tags if i in V[t]], need) for i in items}
        agree = {i for i, c in cons.items() if c == items[i]['concept_id']}
        other = {i for i, c in cons.items() if c is not None and c != items[i]['concept_id']}
        nocon = {i for i, c in cons.items() if c is None}
        key = f'>={need}_of_{k}'
        rec = {'agree': len(agree), 'other_concept': len(other), 'no_consensus': len(nocon),
               'other_concept_pairs': Counter(f"{items[i]['concept_id']}->{cons[i]}" for i in other).most_common(10)}
        phys = {}
        for name, group in (('agree', agree), ('other_concept', other), ('no_consensus', nocon), ('all', set(items))):
            g = [i for i in gold if i in group]
            labels = [gold[i]['labels'][p].get('recognition') for i in g for p in PHYS]
            both = sum(all(gold[i]['labels'][p].get('recognition') == 'correct' for p in PHYS) for i in g)
            phys[name] = {'items': len(g), 'labels': dict(Counter(labels)),
                          'label_correct': wilson(labels.count('correct'), len(labels)),
                          'both_physicians_correct': wilson(both, len(g))}
        rec['physician'] = phys
        out['consensus'][key] = rec
        print(f'\n== consensus {key}: agree {len(agree)}, other concept {len(other)}, no consensus {len(nocon)} (of {len(items)})')
        for name in ('agree', 'other_concept', 'no_consensus', 'all'):
            p = phys[name]
            if p['items']:
                print(f"   physician items {name:13s} n={p['items']:3d}  label correct {100 * p['label_correct'][0]:5.1f}% "
                      f"[{100 * p['label_correct'][1]:.1f}, {100 * p['label_correct'][2]:.1f}]  both correct {100 * p['both_physicians_correct'][0]:5.1f}%  {p['labels']}")
        print('   top released->consensus changes:', rec['other_concept_pairs'][:6])
    json.dump(out, open(ROOT / 'consensus_summary.json', 'w'), indent=1)
    print(f'\nwrote {ROOT / "consensus_summary.json"}')


if __name__ == '__main__':
    main()
