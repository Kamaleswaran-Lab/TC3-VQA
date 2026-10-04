# Collects the batch outputs of the API-served recognition model into the per-item record format written by
# consensus_perceive.py.
from tc3_vqa.paths import EXPERIMENTS, RELEASE, WORK
import json
from pathlib import Path

ITEMS = RELEASE + '/data/items.jsonl'
INVENTORY = WORK + '/concept_inventory.json'
ROOT = Path(EXPERIMENTS + '/consensus_perception')


def main():
    valid = {c['concept_id'] for c in json.load(open(INVENTORY))}
    released = {x['item_id']: x['concept_id'] for x in map(json.loads, open(ITEMS)) if x['task_type'] == 'answerable'}
    got, bad = {}, []
    for f in sorted((ROOT / 'opus' / 'out').glob('batch_*.json')):
        try:
            for r in json.load(open(f))['items']:
                got[r['item_id']] = r
        except Exception as e:
            bad.append((f.name, str(e)[:80]))
    n_ok = n_none = agree = 0
    with open(ROOT / 'opus5.jsonl', 'w') as fo:
        for iid, rc in released.items():
            p = got.get(iid)
            if p is None:
                continue
            cid = str(p.get('matched_concept_id', '')).strip(); conf = str(p.get('match_confidence', 'low')).strip().lower()
            parsed = cid in valid or cid == 'none'
            vote = cid if (cid in valid and conf in ('high', 'medium')) else 'none'
            n_ok += parsed; n_none += vote == 'none'; agree += vote == rc
            fo.write(json.dumps({'item_id': iid, 'released_concept': rc, 'model': 'claude-opus-5',
                                 'with_detection': False, 'parsed': parsed, 'concept': cid if parsed else None,
                                 'confidence': conf, 'vote': vote, 'observation': p.get('visual_observation'),
                                 'raw': json.dumps(p)[:600]}) + '\n')
    print(f'[opus] items {len(released)}, collected {len(got)}, missing {len(set(released) - set(got))}, '
          f'parsed {n_ok}, vote none {n_none}, agrees with released {agree}, bad files {bad}')


if __name__ == '__main__':
    main()
