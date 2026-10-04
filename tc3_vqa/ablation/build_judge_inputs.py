# Builds blinded judge inputs: pools reference and model-authored QA of the same items, normalises the surface form,
# shuffles them under random ids and attaches doctrine passages retrieved from the question text.
from tc3_vqa.paths import CORPUS, EXPERIMENTS, RELEASE
import argparse, json, random, re
from pathlib import Path
import numpy as np

ITEMS = RELEASE + '/data/items.jsonl'
OUT = EXPERIMENTS + '/conventional_ablation'
IDX = CORPUS + '/index'
QPREFIX = 'Represent this sentence for searching relevant passages: '
TOP_K, SEARCH_K = 8, 200
TEXT_OVERLAP = 0.5   # --exclude-item-text: share of a released answer's 4-grams a passage may hold before it is dropped


def tok(t):
    return re.findall(r'[a-z0-9]+', (t or '').lower())


def grams(ws, n=4):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def norm(t, question=False):
    # a bullet after an unpunctuated line ends that sentence
    t = re.sub(r'([^\s.?!:;])\s*[•●▪◦]\s*(\w?)', lambda m: m.group(1) + '. ' + m.group(2).upper(), t or '')
    t = re.sub(r'(^|\s)[•●▪◦]\s*', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    if not t:
        return t
    t = t[0].upper() + t[1:]
    if t[-1] not in '.?!':
        t += '?' if question else '.'
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='pilot')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--probe', type=int, default=100, help='length-probe pairs to add')
    ap.add_argument('--src-run', default=None, help='read arm_c.jsonl from this run (default: --run)')
    ap.add_argument('--items', default=ITEMS, help='released items file the arm A records are taken from')
    ap.add_argument('--exclude-item-cited', action='store_true',
                    help='leave-source-out: drop every passage cited by the item\'s released answers from both arms\' packets')
    ap.add_argument('--exclude-item-text', action='store_true',
                    help='also drop any passage (from any source) that contains the text of one of the item\'s released '
                         'answers, so duplicated doctrine text in other documents cannot support the released answer')
    args = ap.parse_args()
    run = Path(OUT) / args.run; run.mkdir(parents=True, exist_ok=True)

    arm_c = [json.loads(l) for l in open(Path(OUT) / (args.src_run or args.run) / 'arm_c.jsonl')]
    ids = {r['item_id'] for r in arm_c}
    released = {x['item_id']: x for x in map(json.loads, open(args.items)) if x['item_id'] in ids}

    recs = []
    for c in arm_c:
        a = released[c['item_id']]
        base = {'item_id': c['item_id'], 'concept_id': c['concept_id'], 'video_id': c['video_id'], 'frames': c['frames']}
        for q in a['questions']:
            recs.append({**base, 'arm': 'A', 'type': q['type'], 'question': norm(q['question'], True),
                         'options': q.get('options'),
                         'answer': q['answer'] if q['type'] == 'recognition_mcq' else norm(q.get('answer_normalized') or q['answer']),
                         'cited': (q.get('provenance') or {}).get('citation_id')})
        for q in c['questions']:
            if not q['parsed']:
                continue
            recs.append({**base, 'arm': 'C', 'type': q['type'], 'question': norm(q['question'], True),
                         'options': q.get('options'),
                         'answer': q['answer'] if q['type'] == 'recognition_mcq' else norm(q['answer']),
                         'cited': None, 'in_band': q['in_band']})

    # doctrine packets: doctrinal and primary sources only, retrieved from the record's own question text
    meta = json.load(open(f'{IDX}/metadata.json'))
    allowed = set(meta['primary_sources']) | set(meta['doctrinal_sources'])
    chunks = [json.loads(l) for l in open(f'{IDX}/chunks.jsonl')]
    import faiss
    from sentence_transformers import SentenceTransformer
    index = faiss.read_index(f'{IDX}/faiss.index')
    model = SentenceTransformer(meta['model_name'])
    open_recs = [r for r in recs if r['type'] != 'recognition_mcq']
    item_cited = {iid: {(q.get('provenance') or {}).get('citation_id') for q in x['questions']} - {None}
                  for iid, x in released.items()}
    qe = model.encode([QPREFIX + r['question'] for r in open_recs], normalize_embeddings=True,
                      convert_to_numpy=True, batch_size=32).astype(np.float32)
    _, hits = index.search(qe, SEARCH_K)
    # text exclusion: the released answers' 4-grams per item; a passage holding >= half of one answer's 4-grams is dropped
    item_grams = {iid: [g for g in (grams(tok(q.get('answer_normalized') or q['answer'])) for q in x['questions']
                                    if q['type'] != 'recognition_mcq') if g] for iid, x in released.items()}
    chunk_grams = {}
    def holds_answer(i, iid):
        if i not in chunk_grams:
            chunk_grams[i] = grams(tok(chunks[i]['text']))
        return any(len(g & chunk_grams[i]) >= TEXT_OVERLAP * len(g) for g in item_grams[iid])
    for r, row in zip(open_recs, hits):
        banned = item_cited[r['item_id']] if args.exclude_item_cited else set()
        top = [chunks[i] for i in row if i >= 0 and chunks[i]['source_id'] in allowed
               and chunks[i]['citation_id'] not in banned
               and not (args.exclude_item_text and holds_answer(i, r['item_id']))][:TOP_K]
        r['packet'] = [{'citation_id': ch['citation_id'], 'source_id': ch['source_id'],
                        'text': re.sub(r'\s+', ' ', ch['text']).strip()} for ch in top]
        r['cited_in_packet'] = r['cited'] in {ch['citation_id'] for ch in top} if r['arm'] == 'A' else None
    if args.exclude_item_text:
        ov = []
        for r in open_recs:
            if r['arm'] == 'A':
                g = grams(tok(r['answer'])); pg = set().union(*(grams(tok(p['text'])) for p in r['packet'])) if r['packet'] else set()
                ov.append(len(g & pg) / len(g) if g else 0.0)
        ov = np.array(ov)
        print(f'[inputs] released answer 4-gram overlap with its packet: median {np.median(ov):.2f}, '
              f'>= 0.5 {100 * np.mean(ov >= 0.5):.1f}%, >= 0.9 {100 * np.mean(ov >= 0.9):.1f}%; '
              f'packets shorter than {TOP_K}: {sum(len(r["packet"]) < TOP_K for r in open_recs)}')

    # length probe: an arm A answer plus the next sentence of its own source passage (still doctrine, only longer),
    # judged with the same packet; verdicts should not move where the cited passage is inside the packet
    by_cid = {ch['citation_id']: ch['text'] for ch in chunks}
    src = {(x['item_id'], q['type']): q for x in released.values() for q in x['questions']}
    cands = [r for r in open_recs if r['arm'] == 'A' and r['cited_in_packet']]
    random.Random(args.seed).shuffle(cands)
    probes = []
    for r in cands:
        if len(probes) >= args.probe:
            break
        pv = src[(r['item_id'], r['type'])].get('provenance') or {}
        text = by_cid.get(pv.get('citation_id'), '')
        m = re.match(r'\s*[•●▪◦]?\s*([A-Z][^.?!]{15,}?[.?!])(\s|$)', text[pv.get('char_end', 0):])
        if not m or len(m.group(1).split()) > 40:
            continue
        probes.append({**r, 'arm': 'A_probe', 'answer': norm(r['answer'] + ' ' + m.group(1)),
                       'probe_added_words': len(m.group(1).split())})
    recs += probes

    rnd = random.Random(args.seed)
    rnd.shuffle(recs)
    key = {}
    with open(run / 'judge_inputs.jsonl', 'w') as f:
        for r in recs:
            rid = f'r{rnd.getrandbits(40):010x}'
            key[rid] = {k: r.get(k) for k in ('arm', 'item_id', 'concept_id', 'video_id', 'type', 'in_band', 'cited', 'cited_in_packet', 'probe_added_words')}
            key[rid]['answer_words'] = len(r['answer'].split())
            f.write(json.dumps({'rec_id': rid, 'type': r['type'], 'frames': r['frames'], 'question': r['question'],
                                'options': r['options'], 'answer': r['answer'], 'packet': r.get('packet')}) + '\n')
    json.dump(key, open(run / 'arm_key.json', 'w'), indent=0)
    n_arm = {a: sum(v['arm'] == a for v in key.values()) for a in ('A', 'C', 'A_probe')}
    cip = [v['cited_in_packet'] for v in key.values() if v['arm'] == 'A' and v['type'] != 'recognition_mcq']
    print(f'[inputs] {len(key)} records {n_arm}; '
          f'arm A cited passage inside packet {sum(cip)}/{len(cip)} -> {run}/judge_inputs.jsonl')


if __name__ == '__main__':
    main()
