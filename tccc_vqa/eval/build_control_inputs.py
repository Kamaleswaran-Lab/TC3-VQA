# Builds the two recognition-control inputs from the recognition input: frames substituted from an item of another
# concept, and multi-frame items truncated to their first frame.
from tccc_vqa.paths import RELEASE, WORK
import json, random

Q = WORK
REL = RELEASE + '/data/items.jsonl'


def main():
    released = {json.loads(l)['item_id'] for l in open(REL)}
    data = json.load(open(f'{Q}/eval_input.json'))
    items = [it for it in data['items'] if it['id'] in released]
    print(f'released answerable items in eval input: {len(items)}')

    # ---- frame swap -----------------------------------------------------------------------------
    rng = random.Random(20260909)
    donors = list(items)
    ok = False
    for _ in range(200):
        rng.shuffle(donors)
        bad = [i for i, (it, d) in enumerate(zip(items, donors))
               if d['id'] == it['id'] or d['concept'] in {o.lower().replace(' ', '_') for o in it['options']}
               or d['concept'] == it['concept']]
        if not bad:
            ok = True
            break
    # concept names in options are prose, so match on the donor's concept id against the recipient's
    # concept id only; a residual overlap is possible and is repaired greedily below.
    used = set()
    swapped = []
    pool = sorted(items, key=lambda x: x['id'])
    for it in items:
        cand = [d for d in pool if d['concept'] != it['concept'] and d['id'] not in used and d['id'] != it['id']]
        d = rng.choice(cand) if cand else rng.choice([x for x in pool if x['concept'] != it['concept']])
        used.add(d['id'])
        swapped.append({**it, 'frames': d['frames'], 'swapped_from': d['id'], 'swapped_concept': d['concept']})
    same = sum(1 for it, s in zip(items, swapped) if s['swapped_concept'] == it['concept'])
    print(f'frameswap built: {len(swapped)} items | donor concept equals gold concept in {same} (must be 0)')
    json.dump({'abstain': data.get('abstain', False), 'items': swapped},
              open(f'{Q}/eval_input_frameswap.json', 'w'), ensure_ascii=False, indent=1)

    # ---- single frame ---------------------------------------------------------------------------
    one = [{**it, 'frames': it['frames'][:1]} for it in items]
    multi = [it['id'] for it in items if len(it['frames']) > 1]
    print(f'oneframe built: {len(one)} items | {len(multi)} of them were multi-frame')
    json.dump({'abstain': data.get('abstain', False), 'items': one},
              open(f'{Q}/eval_input_oneframe.json', 'w'), ensure_ascii=False, indent=1)

    # ---- matched full-frame baseline over the same 543 items ------------------------------------
    json.dump({'abstain': data.get('abstain', False), 'items': items},
              open(f'{Q}/eval_input_543.json', 'w'), ensure_ascii=False, indent=1)
    json.dump(multi, open(f'{Q}/eval_multiframe_ids.json', 'w'))
    print('wrote eval_input_frameswap.json, eval_input_oneframe.json, eval_input_543.json')


if __name__ == '__main__':
    main()
