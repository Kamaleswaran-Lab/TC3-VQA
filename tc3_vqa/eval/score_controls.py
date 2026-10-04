# Scores the two recognition controls against the matched full-frame run: frame substitution and first frame only,
# with an exact McNemar test on the discordant pairs.
from tc3_vqa.paths import WORK
import json, math
from collections import Counter

Q = WORK
MODELS = [('qwen2vl7b', 'Qwen2-VL-7B'), ('qwen25vl7b', 'Qwen2.5-VL-7B'), ('phi35v', 'Phi-3.5-Vision'),
          ('internvl3_8b', 'InternVL3-8B'), ('internvl3_38b', 'InternVL3-38B')]
ABSTAIN = 'None of these / cannot be determined from the image'


def load(path):
    return {r['id']: r for r in (json.loads(l) for l in open(path))}


def acc(rows, ids=None):
    rs = [r for r in rows.values() if ids is None or r['id'] in ids]
    return sum(1 for r in rs if r.get('pred') == r.get('gold')) / len(rs), len(rs)


def abstain_rate(rows, ids=None):
    rs = [r for r in rows.values() if ids is None or r['id'] in ids]
    return sum(1 for r in rs if r.get('pred') == ABSTAIN) / len(rs)


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def mcnemar(a, b, ids):
    """exact-ish McNemar on paired correctness (b01, b10)."""
    b01 = sum(1 for i in ids if a[i]['pred'] == a[i]['gold'] and b[i]['pred'] != b[i]['gold'])
    b10 = sum(1 for i in ids if a[i]['pred'] != a[i]['gold'] and b[i]['pred'] == b[i]['gold'])
    n = b01 + b10
    if n == 0: return b01, b10, 1.0
    # two-sided binomial test against p=0.5
    k = min(b01, b10)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b01, b10, min(1.0, p)


def main():
    multi = set(json.load(open(f'{Q}/eval_multiframe_ids.json')))
    print('=' * 96)
    print('CONTROL 1  frame swap: same question and options, frames from a different-concept item')
    print(f"{'model':16s}{'baseline':>10s}{'swapped':>10s}{'delta':>9s}{'abstain%':>10s}{'chance':>9s}")
    swap_rows = []
    for tag, name in MODELS:
        base, swap = load(f'{Q}/ctrl_base543_{tag}.jsonl'), load(f'{Q}/ctrl_frameswap_{tag}.jsonl')
        ab, _ = acc(base); asw, n = acc(swap)
        lo, hi = wilson(round(asw * n), n)
        swap_rows.append((name, ab, asw, lo, hi, abstain_rate(swap)))
        print(f'{name:16s}{ab:10.3f}{asw:10.3f}{asw - ab:+9.3f}{100 * abstain_rate(swap):9.1f}%{0.20:9.2f}')
    print('  chance = 0.20 (four concepts + an explicit "cannot be determined" option)')
    print('  Wilson 95% CI on the swapped accuracy:')
    for name, ab, asw, lo, hi, _ in swap_rows:
        print(f'    {name:16s}{asw:6.3f}  [{lo:.3f}, {hi:.3f}]')

    print()
    print('=' * 96)
    print(f'CONTROL 2  single frame vs the full window, on the {len(multi)} multi-frame items')
    print(f"{'model':16s}{'window':>9s}{'1 frame':>9s}{'delta':>8s}{'b01':>6s}{'b10':>6s}{'p':>8s}")
    for tag, name in MODELS:
        base, one = load(f'{Q}/ctrl_base543_{tag}.jsonl'), load(f'{Q}/ctrl_oneframe_{tag}.jsonl')
        ids = [i for i in multi if i in base and i in one]
        aw, _ = acc(base, set(ids)); ao, _ = acc(one, set(ids))
        b01, b10, p = mcnemar(base, one, ids)
        print(f'{name:16s}{aw:9.3f}{ao:9.3f}{ao - aw:+8.3f}{b01:6d}{b10:6d}{p:8.3f}')
    print('  b01 = correct only with the full window, b10 = correct only with one frame;')
    print('  p is a two-sided exact McNemar test on the discordant pairs.')

    print()
    print('sanity: the matched baseline reproduces the released leaderboard')
    lb = {r['model']: r['rec_acc'] for r in json.load(open(f'{Q}/leaderboard.json'))}
    for tag, name in MODELS:
        a, n = acc(load(f'{Q}/ctrl_base543_{tag}.jsonl'))
        print(f'  {name:16s} rerun {a:.3f}  leaderboard {lb[tag]:.3f}  diff {a - lb[tag]:+.3f}')


if __name__ == '__main__':
    main()
