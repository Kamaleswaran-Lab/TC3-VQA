# Paired rubric comparison on the answer-only run: reference and model answers to the same question, judged with the
# same passages. Reports per-judge means, the paired difference with a bootstrap interval, and win/tie/loss counts.
import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

_argv = sys.argv[1:]
from tccc_vqa.ablation import analyze as A

JUDGES = [('llama70b', 'Llama-3.3-70B'), ('medgemma27b', 'MedGemma-27B'), ('phi4', 'Phi-4'),
          ('deepseek70b', 'DeepSeek-R1-Distill-70B'), ('gptoss120b', 'gpt-oss-120b')]


def boot(pairs_by_video, reps=2000, seed=0):
    vids = [v for v, d in pairs_by_video.items() if d]
    rnd = random.Random(seed); est = []
    for _ in range(reps):
        d = [x for v in (rnd.choice(vids) for _ in vids) for x in pairs_by_video[v]]
        est.append(float(np.mean(d)))
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='claude_answer_only_lso')
    ap.add_argument('--axis', default='medical_accuracy')
    args = ap.parse_args(_argv)
    run = Path(A.OUT) / args.run
    key, J = A.load(run); M = A.record_metrics(key, J)
    recs = {json.loads(l)['rec_id']: json.loads(l) for l in open(run / 'judge_inputs.jsonl')}
    pair = defaultdict(dict)
    for rid, k in key.items():
        if k['type'] in A.OPEN:
            pair[(k['item_id'], k['type'])][k['arm']] = (rid, k['video_id'])
    pairs = {p: d for p, d in pair.items() if 'A' in d and 'C' in d}
    same_q = sum(recs[d['A'][0]]['question'] == recs[d['C'][0]]['question'] for d in pairs.values())
    same_p = sum(recs[d['A'][0]]['packet'] == recs[d['C'][0]]['packet'] for d in pairs.values())
    out = {'run': args.run, 'axis': args.axis, 'pairs': len(pairs), 'same_question': same_q, 'same_packet': same_p, 'judges': {}}
    print(f'pairs {len(pairs)} | same question {same_q} | same packet {same_p}')
    for j, label in JUDGES:
        if j not in M:
            print(f'  {label}: no outputs'); continue
        a, c, by_video = [], [], defaultdict(list)
        win = tie = loss = 0
        for (iid, t), d in pairs.items():
            sa, sc = M[j].get(d['A'][0], {}).get(args.axis), M[j].get(d['C'][0], {}).get(args.axis)
            if sa is None or sc is None:
                continue
            a.append(sa); c.append(sc); by_video[d['A'][1]].append(sc - sa)
            win += sc > sa; tie += sc == sa; loss += sc < sa
        lo, hi = boot(by_video)
        out['judges'][j] = {'label': label, 'n': len(a), 'released': float(np.mean(a)), 'opus': float(np.mean(c)),
                            'diff_opus_minus_released': float(np.mean(c) - np.mean(a)), 'ci': [lo, hi],
                            'opus_higher': win, 'tie': tie, 'released_higher': loss}
        print(f"  {label:26s} n={len(a)} released {np.mean(a):.2f} opus {np.mean(c):.2f} diff {np.mean(c) - np.mean(a):+.2f} "
              f"[{lo:+.2f}, {hi:+.2f}] opus>rel {win} tie {tie} rel>opus {loss}")
    json.dump(out, open(run / 'paired_rubric.json', 'w'), indent=1)
    print('wrote', run / 'paired_rubric.json')


if __name__ == '__main__':
    main()
