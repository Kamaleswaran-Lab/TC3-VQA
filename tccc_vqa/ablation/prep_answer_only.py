# Answer-only arm: the model receives each released doctrine, reasoning and HOW question with the frames and writes
# only the answer, length-matched to the reference. Writes jobs.json and the batch files for the prompt runner.
from tccc_vqa.paths import CANDIDATES
import argparse, json
from pathlib import Path

from tccc_vqa.ablation.gen_conventional import FRAMES, MAX_IMG, OUT, band, words

ITEMS = CANDIDATES + '/data/items.jsonl'
OPEN = ('doctrine_scene', 'reasoning', 'how')
SYSTEM = ('You are answering questions for a Tactical Combat Casualty Care (TCCC) visual question answering dataset. '
          'You are shown frame(s) from one segment of a video, in time order, and a question about this scene. '
          'Answer the question according to TCCC doctrine.')


def instruction(q, lo, hi, n):
    return (f"Question: {q}\n"
            f"Write only the answer. The answer must contain between {lo} and {hi} words, aiming for {n} words; "
            f"an answer shorter than {lo} words is rejected.\n"
            'Write plain text only: no bullet points, no numbering, no preamble, and do not restate the question.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='claude_answer_only')
    ap.add_argument('--batch-size', type=int, default=5)
    args = ap.parse_args()
    run = Path(OUT) / args.run
    (run / 'batches').mkdir(parents=True, exist_ok=True); (run / 'out').mkdir(exist_ok=True)

    items = [json.loads(l) for l in open(ITEMS)]
    keep = [x for x in items if x['task_type'] == 'answerable' and x['audit']['concept_visible'] == 'yes']
    jobs = []
    for it in keep:
        frames = [f"{FRAMES}/{r['frame_id']}" for r in it['frame_refs']][:MAX_IMG]
        assert frames and all(Path(f).exists() for f in frames), it['item_id']
        qs = []
        for q in it['questions']:
            if q['type'] not in OPEN:
                continue
            n = words(q.get('answer_normalized') or q['answer'])
            qs.append({'type': q['type'], 'question': q['question'], 'target_words': n, 'band': list(band(n))})
        jobs.append({'item_id': it['item_id'], 'concept_id': it['concept_id'], 'video_id': it['video_id'],
                     'frames': frames, 'questions': qs})
    json.dump({'items': jobs, 'items_file': ITEMS}, open(run / 'jobs.json', 'w'), indent=1)

    nb = 0
    for k in range(0, len(jobs), args.batch_size):
        batch = [{'item_id': j['item_id'], 'frames': j['frames'],
                  'questions': [{'type': q['type'], 'instruction': instruction(q['question'], *q['band'], q['target_words']),
                                 'target_words': q['target_words'], 'band': q['band']} for q in j['questions']]}
                 for j in jobs[k:k + args.batch_size]]
        json.dump({'system': SYSTEM, 'items': batch}, open(run / 'batches' / f'batch_{nb:03d}.json', 'w'), indent=1)
        nb += 1
    nq = sum(len(j['questions']) for j in jobs)
    print(f'[prep] {len(jobs)} items, {nq} questions, {nb} batches -> {run}')


if __name__ == '__main__':
    main()
