# Collects the batch outputs of the API-served model into the arm file format written by gen_conventional.py,
# so build_judge_inputs.py and analyze.py read both arms alike.
from tc3_vqa.paths import EXPERIMENTS
import argparse, json, re
from collections import Counter
from pathlib import Path

OUT = EXPERIMENTS + '/conventional_ablation'
MCQ_Q = 'Which TCCC intervention is being performed in this frame?'


def words(t):
    return len(re.findall(r'\S+', t or ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='claude_pilot')
    ap.add_argument('--answer-only', action='store_true', help='questions are fixed in jobs.json; workers wrote answers only')
    args = ap.parse_args()
    run = Path(OUT) / args.run
    jobs = {it['item_id']: it for it in json.load(open(run / 'jobs.json'))['items']}
    got = {}
    bad_files = []
    for f in sorted((run / 'out').glob('batch_*.json')):
        try:
            for it in json.load(open(f))['items']:
                got[it['item_id']] = it
        except Exception as e:
            bad_files.append((f.name, str(e)[:80]))
    stats = Counter(); per_type = {}
    with open(run / 'arm_c.jsonl', 'w') as fo:
        for iid, job in jobs.items():
            out = got.get(iid)
            if out is None:
                stats['missing_items'] += 1
                continue
            produced = list(out.get('questions') or [])
            qs = []
            for k, jq in enumerate(job['questions']):
                oq = next((q for q in produced if q.get('type') == jq['type']), None)
                if oq is not None:
                    produced.remove(oq)
                t = jq['type']; lo, hi = jq['band']
                if t == 'recognition_mcq':
                    opts = (oq or {}).get('options'); ans = (oq or {}).get('answer')
                    ok = isinstance(opts, list) and len(opts) == 4 and ans in opts
                    rec = {'type': t, 'question': MCQ_Q, 'options': opts if ok else None, 'answer': ans if ok else None,
                           'target_words': jq['target_words'], 'words': None, 'band': [lo, hi], 'parsed': ok, 'in_band': ok, 'pass': 1}
                else:
                    q = jq['question'] if args.answer_only else (oq or {}).get('question')
                    a = (oq or {}).get('answer')
                    ok = isinstance(q, str) and isinstance(a, str) and bool(a.strip())
                    w = words(a) if ok else None
                    rec = {'type': t, 'question': q.strip() if ok else None, 'options': None, 'answer': a.strip() if ok else None,
                           'target_words': jq['target_words'], 'words': w, 'band': [lo, hi], 'parsed': ok,
                           'in_band': bool(ok and lo <= w <= hi), 'pass': 1}
                qs.append(rec)
                d = per_type.setdefault(t, Counter()); d['n'] += 1; d['parsed'] += rec['parsed']; d['in_band'] += rec['in_band']
            fo.write(json.dumps({'item_id': iid, 'concept_id': job['concept_id'], 'video_id': job['video_id'],
                                 'frames': job['frames'], 'generator': 'claude-opus-4-8',
                                 'questions': qs}) + '\n')
            stats['items'] += 1
    report = {'items_expected': len(jobs), **stats, 'bad_files': bad_files, 'by_type': {t: dict(c) for t, c in per_type.items()}}
    json.dump(report, open(run / 'length_report.json', 'w'), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
