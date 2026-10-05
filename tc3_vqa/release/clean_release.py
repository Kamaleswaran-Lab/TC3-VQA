# Writes the public package layout: public record schema, per-record detection from the COCO file, the cited doctrine
# chunks, reference results, the source manifest, the shipped scripts and the field dictionary.
from tc3_vqa.paths import CORPUS, EXPERIMENTS, WORK
import json, os, shutil, csv, sys

REL, DEP = sys.argv[1], sys.argv[2]
POOL = WORK          # source of the control inputs
CORPUS = CORPUS + '/index'
EXP = EXPERIMENTS
AVAILABILITY = f'{EXP}/source_availability.csv'
CHECK_DATE = '2026-09-17'
FIG = f'{EXP}/conventional_ablation/figure_data.json'
QA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eval')
VERSION = '1.1'

ANS_FIELDS = [  # (public name, description) in output order
    ('item_id', 'stable item identifier (`ans_*` answerable, `ref_*` refusal)'),
    ('task_type', '`answerable` or `refusal`'),
    ('release_version', 'release tag this record belongs to'),
    ('concept_id', 'the TCCC concept the frame depicts (12-concept closed set; for refusal items, the concept of the scene)'),
    ('march_category', 'MARCH phase of the concept: M / A / R / C / H / process'),
    ('safety_critical', 'whether the cited doctrine passage is flagged safety-critical by the stakes rubric (answerable only); RWHR weights are per concept, see eval/scripts/score_eval.py'),
    ('review_priority', 'clinician review tier from the full-pool audit: `primary` or `review` (answerable only)'),
    ('video_id', 'YouTube video identifier of the source'),
    ('source_url', 'public URL of the source video'),
    ('source_license', '`standard_youtube`, `age_restricted_standard`, or `creative_commons`'),
    ('source_channel', 'uploading channel (empty when metadata was unavailable)'),
    ('channel_category', '`us_government_public_domain`, `creative_commons`, `standard`, or `standard_unknown_channel`'),
    ('requires_login', 'true for age-restricted sources (fetch with cookies)'),
    ('source_status', 'reachability of the source video at the last check, recorded in `meta/source_videos.csv`'),
    ('pixels_released', 'true when the item\'s frames ship in `frames/`'),
    ('segment', 'exact source segment: `shot_idx`, `window_idx`, `t_start_s`, `t_end_s`, `source`'),
    ('frame_refs', 'frames of the item: `frame_id`, `video_id`, `timestamp_s`, shot/window/candidate index, `masked_in_release`, `mask_rects` (normalized x0,y0,x1,y1), `local_path` when shipped'),
    ('questions', 'question objects (see below)'),
    ('detection', 'reviewed equipment boxes on the item\'s frames, identical to `data/detection_coco.json`: `frame_id`, `label` (17 classes), `bbox_norm` (x0,y0,x1,y1), `source` (`reviewed` / `recovered`) (answerable only)'),
    ('answer_region', 'normalized box of the evidence region (answerable only)'),
    ('observed_anatomy', '`region` and `confidence` of the body region visible in the frames (answerable only)'),
    ('audit', 'independent visual audit of the frames, made without the perception model\'s reasoning: `concept_visible` (yes/partial/no), `answerable`, `what_you_see` (answerable only)'),
    ('auto_checks', 'automatic verification flags: `citation_offset_ok`, `visible_evidence_present`, `cross_check_visible`, `audit_concept_visible`, `audit_answerable` (answerable only)'),
    ('refusal_kind', 'refusal items: how the item was derived'),
    ('refusal_reason', 'refusal items: category of missing evidence (`vital_signs`, `elapsed_time`, `step_outcome_not_shown`, ...)'),
    ('derived_from', 'refusal items: the answerable item whose frames this refusal question reuses'),
]
Q_FIELDS = [
    ('qid', '`<item_id>#<k>`'),
    ('type', '`recognition_mcq`, `doctrine_scene`, `reasoning`, `how`, or `refusal`'),
    ('question', 'prompt text (scene-grounded for doctrine/reasoning/how)'),
    ('options', 'recognition_mcq only: four concept names, one correct'),
    ('answer', 'gold answer: the concept (MCQ); the verbatim corpus span (doctrine/reasoning/how); `REFUSE` (refusal)'),
    ('answer_normalized', 'doctrine/reasoning/how: readable rendering of `answer` with corpus formatting artifacts removed'),
    ('provenance', 'doctrine/reasoning/how: `citation_id`, `source_id`, `char_start`, `char_end`, `source_quote`; `answer` equals `text[char_start:char_end]` of the chunk with that `citation_id` in `data/doctrine_chunks.jsonl`'),
    ('facet', 'doctrine_scene: `indication`, `technique`, `placement_or_site`, `sequence_or_next_step`, `effectiveness_or_verification`, `caution_or_contraindication`, or `other`'),
    ('reason_type', 'reasoning: free-text tag of the reasoning asked for (e.g. `next_step`, `indication`, `consequence`, `verification`)'),
    ('faithful', 'how: passed the claim-level NLI faithfulness gate'),
    ('rationale', 'refusal: why the frames cannot support the question'),
]
HEADERS = {  # English first-line comments for shipped scripts (replace the working-language headers)
    'eval_vlm.py': '# Runs a baseline VLM on the recognition axis (4 concepts + "cannot be determined"; model-agnostic, vLLM chat API) and records predictions.',
    'eval_refusal.py': '# Runs a baseline VLM on the in-domain refusal axis: free-form answers with an explicit abstention phrase allowed; records whether the model abstained.',
    'eval_doctrine.py': '# Runs a baseline VLM on the doctrine axis (free-form answers); scoring is done by score_doctrine.py with NLI.',
    'score_eval.py': '# Scores recognition accuracy (+MARCH macro, Wilson CI), in-domain refusal abstention accuracy, and the graded Risk-Weighted Hallucination Rate (+weight sensitivity) from eval_vlm.py / eval_refusal.py outputs.',
}


def patch(path, pairs, header=None):
    s = open(path).read()
    if header:
        lines = s.split('\n')
        while lines and lines[0].startswith('#') and any(ord(c) > 127 for c in lines[0]): lines.pop(0)
        s = header + '\n' + '\n'.join(lines)
    for old, new, *rest in pairs:
        n = s.count(old)
        assert n >= 1 and (n == 1 or rest == ['all']), (path, n, old[:70])
        s = s.replace(old, new)
    open(path, 'w').write(s)


FACETS = ('indication', 'technique', 'placement_or_site', 'sequence_or_next_step', 'effectiveness_or_verification',
          'caution_or_contraindication')
FACET_WORDS = [  # free-text facet labels written during regeneration are folded into the six classes by keyword
    ('caution_or_contraindication', ('caution', 'precaution', 'contraindication', 'avoid')),
    ('effectiveness_or_verification', ('reassess', 'verification', 'effective', 'endpoint', 'hold time', 'duration', 'confirm')),
    ('sequence_or_next_step', ('timing', 'next_step', 'sequence', 'conversion')),
    ('placement_or_site', ('placement', 'site', 'location')),
    ('indication', ('indication', 'purpose', 'preferred', 'adjunct')),
    ('technique', ('technique', 'application', 'pressure', 'packing', 'dressing', 'closure', 'lubrication', 'sizing', 'measurement',
                   'locate', 'augment', 'airway', 'hemorrhage', 'extent', 'extends')),
]


def facet_class(v):
    v = (v or '').strip()
    if v in FACETS:
        return v
    low = v.lower()
    for cls, words in FACET_WORDS:
        if any(w in low for w in words):
            return cls
    return 'other'


def clean_question(q):
    out = {}
    for k in ('qid', 'type', 'question', 'options', 'answer', 'answer_normalized'):
        if k in q: out[k] = q[k]
    if q.get('provenance'):
        p = q['provenance']
        out['provenance'] = {'citation_id': p.get('citation_id'), 'source_id': p.get('source_id') or q.get('source_id'),
                             **{k: v for k, v in p.items() if k not in ('citation_id', 'source_id')}}
    for k in ('facet', 'reason_type', 'faithful', 'rationale'):
        if q.get(k) is not None: out[k] = bool(q[k]) if k == 'faithful' else q[k]
    if out.get('facet'): out['facet'] = facet_class(out['facet'])
    internal = {k: v for k, v in q.items() if k not in out and k not in ('source', 'source_id', 'provenance')}
    return out, internal


def clean_record(r, coco_boxes):
    ref = r['task_type'] == 'refusal'
    o = {'item_id': r['item_id'], 'task_type': r['task_type'], 'release_version': VERSION,
         'concept_id': r['scene_category'] if ref else r['matched_concept_id'], 'march_category': r['march_category']}
    if not ref: o['safety_critical'] = r['safety_critical']
    if not ref: o['review_priority'] = r['review_priority']
    for k in ('video_id', 'source_url', 'source_license', 'source_channel', 'channel_category', 'requires_login', 'pixels_released', 'segment', 'frame_refs'):
        o[k] = r[k]
    qs, q_internal = [], {}
    for q in r['questions']:
        cq, internal = clean_question(q); qs.append(cq)
        if internal: q_internal[q['qid']] = internal
    o['questions'] = qs
    if not ref:
        o['detection'] = [b for f in r['frame_refs'] for b in coco_boxes.get(f['frame_id'], [])]
        o['answer_region'] = r.get('answer_region')
        o['observed_anatomy'] = {'region': r.get('anatomy'), 'confidence': r.get('anatomy_confidence')}
        o['audit'] = r.get('independent_audit')
        o['auto_checks'] = r.get('checks')
    else:
        o['refusal_kind'] = r.get('refusal_kind'); o['refusal_reason'] = r.get('refusal_reason'); o['derived_from'] = r.get('src_item')
    consumed = {'item_id', 'task_type', 'release_version', 'matched_concept_id', 'scene_category', 'march_category', 'safety_critical', 'tier',
                'review_priority', 'video_id', 'source_url', 'source_license', 'source_channel', 'channel_category', 'requires_login',
                'pixels_released', 'segment', 'frame_refs', 'questions', 'answer_region', 'anatomy', 'anatomy_confidence', 'tier',
                'independent_audit', 'checks', 'refusal_kind', 'refusal_reason', 'src_item', 'gold_label'}
    if ref: consumed |= {'question', 'rationale'}          # pure duplicates of questions[0]
    hist = {('detection_raw' if k == 'detection' else k): v for k, v in r.items() if k not in consumed}
    if q_internal: hist['questions_internal'] = q_internal
    return o, hist


def main():
    if os.path.exists(DEP): shutil.rmtree(DEP)
    for d in ('data', 'meta', 'gold', 'frames', 'scripts', 'mcq', 'eval'): os.makedirs(f'{DEP}/{d}', exist_ok=True)
    # reviewed detection layer -> per-frame normalized boxes
    coco = json.load(open(f'{REL}/core/detection_coco.json'))
    cat = {c['id']: c['name'] for c in coco['categories']}; img = {i['id']: i for i in coco['images']}
    coco_boxes = {}
    for a in coco['annotations']:
        im = img[a['image_id']]; W, H = im['width'], im['height']; x, y, w, h = a['bbox']
        coco_boxes.setdefault(im['file_name'], []).append({'frame_id': im['file_name'], 'label': cat[a['category_id']],
                                                           'bbox_norm': [round(x / W, 4), round(y / H, 4), round((x + w) / W, 4), round((y + h) / H, 4)], 'source': a.get('source')})
    recs = [json.loads(l) for l in open(f'{REL}/core/tc3_vqa.jsonl')]
    n_hist = n_box = 0
    with open(f'{DEP}/data/items.jsonl', 'w') as fo:
        for r in recs:
            o, h = clean_record(r, coco_boxes)
            n_box += len(o.get('detection') or [])
            fo.write(json.dumps(o, ensure_ascii=False) + '\n')
            n_hist += bool(h)                                 # construction-time fields are not shipped
    shutil.copy2(f'{REL}/core/detection_coco.json', f'{DEP}/data/detection_coco.json')
    shutil.copy2(f'{REL}/segments/segments.jsonl', f'{DEP}/data/segments.jsonl')
    shutil.copy2(f'{REL}/meta/mask_rects.json', f'{DEP}/meta/mask_rects.json')
    shutil.copy2(f'{REL}/meta/dropped_items.json', f'{DEP}/meta/dropped_items.json')
    g = json.load(open(f'{REL}/gold/gold_adjudication.json')); g['release_version'] = VERSION
    json.dump(g, open(f'{DEP}/gold/gold_adjudication.json', 'w'), indent=1, ensure_ascii=False)
    shutil.copytree(f'{REL}/frames', f'{DEP}/frames', dirs_exist_ok=True)
    shutil.copytree(f'{REL}/mcq', f'{DEP}/mcq', dirs_exist_ok=True)
    shutil.copytree(f'{REL}/eval', f'{DEP}/eval', dirs_exist_ok=True)
    shutil.copy2(f'{REL}/scripts/fetch_frames.py', f'{DEP}/scripts/fetch_frames.py')
    # recognition controls: same items, substituted or truncated frames (Technical Validation)
    os.makedirs(f'{DEP}/eval/controls', exist_ok=True)
    kept_ids = {r['item_id'] for r in recs}                # controls are built on the full pool; ship only released items
    for src, dst in (('eval_input_frameswap.json', 'frameswap_input.json'),
                     ('eval_input_oneframe.json', 'oneframe_input.json')):
        d = json.load(open(f'{POOL}/{src}'))
        d['items'] = [x for x in d['items'] if x['id'] in kept_ids]
        for x in d['items']:
            x['frame_ids'] = [os.path.basename(z) for z in x.pop('frames')]
        json.dump(d, open(f'{DEP}/eval/controls/{dst}', 'w'), ensure_ascii=False, indent=1)
    mf = json.load(open(f'{POOL}/eval_multiframe_ids.json'))
    mf = [i for i in mf if i in kept_ids] if isinstance(mf, list) else {k: ([i for i in v if i in kept_ids] if isinstance(v, list) else v) for k, v in mf.items()}
    json.dump(mf, open(f'{DEP}/eval/controls/multiframe_item_ids.json', 'w'), ensure_ascii=False, indent=1)
    shutil.copy2(f'{QA}/score_controls.py', f'{DEP}/eval/scripts/score_controls.py')
    shutil.copy2(f'{QA}/score_doctrine.py', f'{DEP}/eval/scripts/score_doctrine.py')
    patch(f'{DEP}/eval/scripts/score_controls.py', [("from tc3_vqa.paths import WORK\n", ""), ("import json, math", "import json, math, os"),
          ("Q = WORK", "Q = os.environ.get('TC3_VQA_RUNS', '.')  # directory holding ctrl_base_<model>.jsonl, ctrl_frameswap_<model>.jsonl and ctrl_oneframe_<model>.jsonl from eval_vlm.py\nHERE = os.path.dirname(os.path.abspath(__file__))"),
          ("json.load(open(f'{Q}/eval_multiframe_ids.json'))", "json.load(open(f'{HERE}/../controls/multiframe_item_ids.json'))"),
          ("lb = {r['model']: r['rec_acc'] for r in json.load(open(f'{Q}/leaderboard.json'))}", "lb = {m: r['rec_acc'] for m, r in json.load(open(f'{HERE}/../../results/baselines.json')).items()}"),
          ("ctrl_base543_", "ctrl_base_", 'all'), ("over the same 543 items", "over the same 431 items")])
    for root, _, _ in os.walk(DEP):
        if root.endswith('__pycache__'): shutil.rmtree(root)
    # shipped scripts: relative paths, public field names, English headers
    S = f'{DEP}/eval/scripts'
    os.rename(f'{S}/score_eval.py', f'{S}/score_eval.py')
    HERE = 'os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", '
    patch(f'{S}/eval_vlm.py', [('import argparse, base64, io, json, re, time', 'import argparse, base64, io, json, os, re, time'),
                               ('default=WORK + "/eval_input.json"', f'default={HERE}"recognition_input.json")')], HEADERS['eval_vlm.py'])
    patch(f'{S}/eval_refusal.py', [('import argparse, base64, io, json, time', 'import argparse, base64, io, json, os, time'),
                                   ('default=WORK + "/refusal_input.json"', f'default={HERE}"refusal_input.json")')], HEADERS['eval_refusal.py'])
    patch(f'{S}/eval_doctrine.py', [('import argparse, base64, io, json, time', 'import argparse, base64, io, json, os, time'),
                                    ('default=WORK + "/doctrine_input.json"', f'default={HERE}"doctrine_open_input.json")')], HEADERS['eval_doctrine.py'])
    patch(f'{S}/score_eval.py', [('P = WORK', 'P = os.environ.get("TC3_VQA_RUNS", ".")  # directory holding eval_recog_<model>.jsonl and refusal_<model>.jsonl')], HEADERS['score_eval.py'])
    patch(f'{S}/score_doctrine.py', [('from tc3_vqa.paths import RELEASE, WORK\n', ''), ('glob.glob(WORK + "/doc_*.jsonl")', 'glob.glob("doc_*.jsonl")'),
                                     ('open(WORK + "/doctrine_scores.json", "w")', 'open("doctrine_scores.json", "w")'),
                                     ('os.environ.get("TC3_VQA_DATA", RELEASE + "/data")', 'os.environ.get("TC3_VQA_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))')])
    patch(f'{DEP}/mcq/build_doctrine_mcq.py', [
        ('import argparse, json, hashlib, re', 'import argparse, json, hashlib, os, re'),
        ('P = WORK', 'P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")  # deposit root'),
        ('    ans = [json.loads(l) for l in open(f"{P}/clinician_review_queue_hybrid.jsonl")\n           if json.loads(l).get("task_type") == "answerable"]\n    meta = {it["id"]: it for it in json.load(open(f"{P}/eval_input.json"))["items"]}',
         '    ans = [r for r in (json.loads(l) for l in open(f"{P}/data/items.jsonl")) if r.get("task_type") == "answerable"]\n    meta = {r["item_id"]: {"frames": [f["frame_id"] for f in r["frame_refs"]], "march": r["march_category"], "safety_critical": r["safety_critical"]} for r in ans}'),
        ('it["matched_concept_id"]', 'it["concept_id"]', 'all'),
        ('"safety_critical": m.get("safety_critical"), "severity": m.get("severity"),', '"safety_critical": m.get("safety_critical"),'),
        ('out = f"{P}/doctrine_mcq_{suffix}.json"', 'out = f"{P}/mcq/doctrine_mcq_{suffix}.json"')])
    # source manifest: merge the useful yt-dlp columns into one CSV
    yt = {r['video_id']: r for r in csv.DictReader(open(f'{REL}/meta/source_videos_ytmeta.csv'))}
    rows = list(csv.DictReader(open(f'{REL}/meta/source_videos.csv')))
    cols = ['video_id', 'source_url', 'license_category', 'channel', 'channel_id', 'channel_category', 'requires_login', 'pixels_released', 'n_items', 'title', 'upload_date', 'duration_s', 'source_status', 'status_checked']
    avail = {}
    if os.path.exists(AVAILABILITY):                       # written by check_source_availability.py
        avail = {a['video_id']: a for a in csv.DictReader(open(AVAILABILITY))}
    checked = CHECK_DATE if avail else ''
    with open(f'{DEP}/meta/source_videos.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            y = yt.get(r['video_id'], {})
            # age-restricted sources cannot be probed without a signed-in session; the authors checked them by hand
            st = avail.get(r['video_id'], {}).get('status', '')
            st = 'available' if st in ('available', 'age_restricted') else ('unavailable' if st else '')
            w.writerow({**{c: r.get(c, '') for c in cols}, 'channel_id': y.get('channel_id', ''), 'upload_date': y.get('upload_date', ''),
                        'duration_s': y.get('duration', ''), 'title': r.get('title') or y.get('title', ''),
                        'source_status': st, 'status_checked': checked if st else ''})
    # field dictionary
    with open(f'{DEP}/meta/FIELDS.md', 'w') as f:
        f.write(f'# Field dictionary — `data/items.jsonl` ({VERSION})\n\nOne JSON object per line, one line per item. '
                'Fields marked "answerable only" / "refusal items" are absent on the other task type.\n\n| field | meaning |\n|---|---|\n')
        for n, d in ANS_FIELDS: f.write(f'| `{n}` | {d} |\n')
        f.write('\n## `questions[]`\n\n| field | meaning |\n|---|---|\n')
        for n, d in Q_FIELDS: f.write(f'| `{n}` | {d} |\n')
        f.write('\n## `data/frame_features.npz`\n\nFrozen image-encoder features for all 903 frames, so items whose pixels are not '
                'shipped can still be evaluated. `frame_ids` holds the frame identifiers and one float16 matrix per encoder holds the '
                'L2-normalized pooled features in the same row order; `data/frame_features.json` names the encoders.\n')
        f.write('\n## `data/doctrine_chunks.jsonl`\n\nOne JSON object per corpus chunk: `citation_id`, `source_id`, `section_path`, `page_start`, '
                '`page_end`, `text`. Every cited answer span is `text[char_start:char_end]` of its chunk.\n')
    write_doctrine_chunks(recs); write_concepts(recs); write_results(); write_citation_and_scripts()
    print(f'clean_release: {len(recs)} records -> {DEP}/data/items.jsonl | detection boxes {n_box} | records with dropped construction fields {n_hist} | frames {len(os.listdir(f"{DEP}/frames"))}')

MARCH = {'tourniquet_application': 'M', 'wound_packing': 'M', 'junctional_hemorrhage': 'M', 'tourniquet_conversion': 'M',
         'nasopharyngeal_airway': 'A', 'surgical_airway_cric': 'A', 'chest_seal': 'R', 'needle_decompression': 'R',
         'iv_io_access': 'C', 'txa_administration': 'C', 'hypothermia_prevention': 'H', 'tccc_documentation': 'process'}
NAMES = {'tourniquet_application': 'Limb tourniquet application', 'wound_packing': 'Wound packing', 'junctional_hemorrhage': 'Junctional hemorrhage control',
         'tourniquet_conversion': 'Tourniquet conversion', 'nasopharyngeal_airway': 'Nasopharyngeal airway (NPA) insertion',
         'surgical_airway_cric': 'Surgical cricothyroidotomy', 'chest_seal': 'Chest seal application', 'needle_decompression': 'Needle chest decompression',
         'iv_io_access': 'IV/IO vascular access', 'txa_administration': 'TXA administration', 'hypothermia_prevention': 'Hypothermia prevention',
         'tccc_documentation': 'TCCC card (DD1380) documentation'}
SAFETY = {'tourniquet_application', 'wound_packing', 'junctional_hemorrhage', 'tourniquet_conversion', 'surgical_airway_cric',
          'chest_seal', 'needle_decompression', 'txa_administration'}   # concept-level stakes flag of the RWHR rubric


def write_doctrine_chunks(recs):
    """Ships the doctrine chunks answers cite: all chunks of the guideline and doctrine sources (public documents), so
    offsets can be verified and doctrine answers scored without rebuilding the corpus."""
    meta = json.load(open(f'{CORPUS}/metadata.json'))
    gold = set(meta['primary_sources']) | set(meta['doctrinal_sources'])
    chunks = {}
    for line in open(f'{CORPUS}/chunks.jsonl'):
        c = json.loads(line)
        if c['source_id'] in gold:
            chunks[c['citation_id']] = {k: c.get(k) for k in ('citation_id', 'source_id', 'section_path', 'page_start', 'page_end', 'text')}
    for r in recs:
        for q in r['questions']:
            p = q.get('provenance')
            if p and p.get('citation_id'):
                assert chunks[p['citation_id']]['text'][p['char_start']:p['char_end']] == q['answer'], q['qid']
    with open(f'{DEP}/data/doctrine_chunks.jsonl', 'w') as f:
        for c in chunks.values(): f.write(json.dumps(c, ensure_ascii=False) + '\n')
    print(f'doctrine chunks: {len(chunks)} from {len(gold)} sources; every cited span verified')


def write_concepts(recs):
    ans = [r for r in recs if r['task_type'] == 'answerable']
    out = [{'concept_id': c, 'name': NAMES[c], 'march_category': MARCH[c], 'safety_critical': c in SAFETY,
            'answerable_items': sum(r['matched_concept_id'] == c for r in ans),
            'refusal_items': sum(r['scene_category'] == c for r in recs if r['task_type'] == 'refusal')} for c in MARCH]
    json.dump(out, open(f'{DEP}/meta/concepts.json', 'w'), indent=1)


def write_results():
    """Reference results behind the paper's tables, restricted to the released set."""
    R = f'{DEP}/results'; os.makedirs(R, exist_ok=True)
    lb = json.load(open(f'{EXP}/consensus_perception/leaderboard.json'))
    json.dump({m: v['3'] for m, v in lb.items()}, open(f'{R}/baselines.json', 'w'), indent=1)
    ctl = json.load(open(f'{EXP}/consensus_perception/controls.json'))
    json.dump(ctl['released'], open(f'{R}/controls.json', 'w'), indent=1)
    doc = json.load(open(f'{EXP}/consensus_perception/doctrine_scores.json'))
    json.dump([{'model': d['model'], **d['3']} for d in doc], open(f'{R}/doctrine_open.json', 'w'), indent=1)
    gate = json.load(open(f'{EXP}/consensus_perception/gate_3of6.json'))
    votes = {}
    for m in gate['voters']:
        for row in map(json.loads, open(f'{EXP}/consensus_perception/{m}.jsonl')):
            votes.setdefault(row['item_id'], {})[m] = row['vote']
    json.dump({'rule': 'an answerable item is released when at least 3 of the 6 models name its concept',
               'models': gate['voters'], 'votes': votes, 'consensus': gate['consensus'],
               'released': gate['keep'], 'removed': gate['drop']},
              open(f'{R}/recognition_consensus.json', 'w'), indent=1)
    shutil.copy2(f'{EXP}/release_audit/release_audit_summary.json', f'{R}/release_audit.json')
    pr = json.load(open(f'{EXP}/conventional_ablation/claude_answer_only_lso/paired_rubric.json')); pr.pop('run', None)
    json.dump(pr, open(f'{R}/reference_vs_model_answers.json', 'w'), indent=1)
    shutil.copy2(FIG, f'{R}/model_authored_qa.json')
    open(f'{R}/README.md', 'w').write(RESULTS_README)


RESULTS_README = """# Results reported in the paper

These files hold the outputs behind the paper's tables and figures, computed on the released items. They can
be used to check the scoring scripts in `eval/scripts/` against a new run.

| file | content |
|---|---|
| `baselines.json` | recognition, refusal, doctrine multiple-choice and RWHR for the five baseline models |
| `doctrine_open.json` | free-form doctrine answers scored by NLI, strict and partial credit |
| `controls.json` | frame-substitution and first-frame-only controls |
| `recognition_consensus.json` | concept named by each of the six recognition models per candidate item, the consensus label, and the released and removed item lists |
| `release_audit.json` | audit of refusal, visual anchoring and reasoning by three models |
| `reference_vs_model_answers.json` | medical-accuracy ratings of reference answers and model answers to the same questions |
| `model_authored_qa.json` | wrong-intervention and clean-QA rates for questions written directly from frames |
"""


def write_citation_and_scripts():
    open(f'{DEP}/CITATION.cff', 'w').write(CITATION)
    open(f'{DEP}/scripts/requirements.txt', 'w').write("yt-dlp\npillow\nvllm\nsentence-transformers\ntransformers\ntorch\n")
    open(f'{DEP}/scripts/verify.py', 'w').write(VERIFY)


CITATION = """cff-version: 1.2.0
message: "If you use this dataset, please cite it as below."
title: "TC3-VQA: a doctrine-grounded visual question answering dataset for Tactical Combat Casualty Care"
version: "1.1"
type: dataset
license: CC-BY-4.0
doi: "10.5281/zenodo.23170287"
authors:
  - family-names: Kim
    given-names: Junseob
  - family-names: Chng
    given-names: Jade
  - family-names: Ali
    given-names: Ayman
  - family-names: Moas
    given-names: Victor
  - family-names: Lee
    given-names: Yichun
  - family-names: Chin
    given-names: Po-Chun
  - family-names: Hwang
    given-names: Sunil
  - family-names: Kamaleswaran
    given-names: Rishikesan
"""

VERIFY = '''# Checks the integrity of a TC3-VQA package: item counts, citation offsets against the shipped doctrine chunks,
# and agreement between the item records, the COCO detection file, the segments and the shipped frames.
import json, os, sys

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
items = [json.loads(l) for l in open(os.path.join(root, "data", "items.jsonl"))]
chunks = {c["citation_id"]: c for c in map(json.loads, open(os.path.join(root, "data", "doctrine_chunks.jsonl")))}
answerable = [r for r in items if r["task_type"] == "answerable"]
refusal = [r for r in items if r["task_type"] == "refusal"]
print(f"items {len(items)}: {len(answerable)} answerable, {len(refusal)} refusal")

bad = 0
n_cited = 0
for r in items:
    for q in r["questions"]:
        p = q.get("provenance")
        if not p:
            continue
        n_cited += 1
        chunk = chunks.get(p["citation_id"])
        if chunk is None or chunk["text"][p["char_start"]:p["char_end"]] != q["answer"]:
            bad += 1
            print("offset mismatch:", q["qid"])
print(f"cited answers {n_cited}, offset mismatches {bad}")

frames = {f["frame_id"] for r in items for f in r["frame_refs"]}
shipped = set(os.listdir(os.path.join(root, "frames")))
coco = json.load(open(os.path.join(root, "data", "detection_coco.json")))
coco_frames = {im["file_name"] for im in coco["images"]}
segments = [json.loads(l) for l in open(os.path.join(root, "data", "segments.jsonl"))]
seg_items = {i for s in segments for i in s["item_ids"]}
print(f"frames referenced {len(frames)}, shipped {len(shipped)} (all referenced: {shipped <= frames})")
print(f"detection frames {len(coco_frames)} (all referenced: {coco_frames <= frames}), boxes {len(coco['annotations'])}")
print(f"segments {len(segments)} covering {len(seg_items)} items (all items: {seg_items == {r['item_id'] for r in items}})")
sys.exit(1 if bad else 0)
'''


if __name__ == '__main__':
    main()
