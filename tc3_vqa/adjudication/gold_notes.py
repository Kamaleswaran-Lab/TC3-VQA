# Adds the raters' free-text notes to the adjudication file that clean_release.py ships in gold/. The raw rating
# files carry names; here each rater is mapped to the anonymized role used in the release and only the note text
# is copied (recognition_note for the label, comment for the answers or the blinded comparison).
from tc3_vqa.paths import WORK
import glob, json, sys

RAW = WORK + '/gold/raters'                              # one JSON per rater as returned by the rating app
GOLD = WORK + '/release/gold/gold_adjudication.json'
ROLES = ['physician_1', 'physician_2', 'student_1', 'student_2']


def match_role(responses, items):
    """the raw file whose ratings coincide with a role's ratings on the shared items belongs to that role"""
    best = None
    for role in ROLES:
        n = sum(responses.get(i, {}).get('recognition') == items[i]['labels'][role]['recognition'] and
                responses.get(i, {}).get('doctrine') == items[i]['labels'][role]['doctrine'] for i in items)
        if best is None or n > best[1]:
            best = (role, n)
    assert best[1] >= 0.9 * len(items), best
    return best[0]


def main():
    g = json.load(open(GOLD))
    items = g['items']
    seen = set()
    for f in sorted(glob.glob(f'{RAW}/*.json')):
        responses = json.load(open(f))['responses']
        role = match_role(responses, items)
        assert role not in seen, role
        seen.add(role)
        n = 0
        for i, it in items.items():
            r = responses.get(i, {})
            lab = it['labels'][role]
            lab.pop('recognition_note', None); lab.pop('comment', None)
            if r.get('rec_note', '').strip():
                lab['recognition_note'] = r['rec_note'].strip(); n += 1
            if r.get('comment', '').strip():
                lab['comment'] = r['comment'].strip(); n += 1
        print(f'{role}: {n} notes')
    assert seen == set(ROLES)
    g['description'] = g['description'].rstrip('. ') + \
        ". Where a rater wrote one, 'recognition_note' explains the recognition rating and 'comment' the answer ratings or the blinded comparison."
    json.dump(g, open(GOLD, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
