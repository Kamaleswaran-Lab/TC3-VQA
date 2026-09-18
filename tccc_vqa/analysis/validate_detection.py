# Statistics of the equipment-detection layer: how proposed boxes fared under verification and review, by class and
# verification tier, and the localisation agreement between proposal and surviving box.
from tccc_vqa.paths import WORK
import json
from collections import Counter, defaultdict

POOL = WORK
ALIAS = {'gloved_hands': 'hand'}   # the class was renamed after proposal time; not a relabel


def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0: return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def norm(fid):
    return fid.replace('/', '__') + ('.jpg' if not fid.endswith('.jpg') else '')


def main():
    prop = json.load(open(f'{POOL}/detect_coco.json'))
    cat = {c['id']: c['name'] for c in prop['categories']}
    img = {i['id']: i for i in prop['images']}
    proposals = []
    for a in prop['annotations']:
        im = img[a['image_id']]; W, H = im['width'], im['height']; x, y, w, h = a['bbox']
        proposals.append({'frame': norm(im.get('frame_id') or im['file_name'].rsplit('.', 1)[0]), 'label': ALIAS.get(cat[a['category_id']], cat[a['category_id']]),
                          'box': [x / W, y / H, (x + w) / W, (y + h) / H],
                          'verify': a.get('verify_internvl3'), 'tier': a.get('tier')})
    print(f'proposed boxes {len(proposals)} over {len({p["frame"] for p in proposals})} frames')
    print('cross-verifier verdict:', dict(Counter(p['verify'] for p in proposals)))
    print('proposal tier        :', dict(Counter(p['tier'] for p in proposals)))

    reviewed = [json.loads(l) for l in open(f'{POOL}/detect_final.jsonl')]
    kept = defaultdict(list)
    for r in reviewed:
        kept[norm(r['frame_id'])].append(r)
    print(f'boxes after review {len(reviewed)} '
          f'(reviewed {sum(1 for r in reviewed if r["source"] == "reviewed")}, '
          f'recovered {sum(1 for r in reviewed if r["source"] == "recovered")})')

    # match each proposal to the surviving box of the same label on the same frame (greedy, best IoU)
    survived, deleted, ious, relabel = Counter(), Counter(), [], 0
    surv_t, del_t = Counter(), Counter()
    used = set()
    for p in proposals:
        cands = [(i, r) for i, r in enumerate(kept[p['frame']])
                 if (p['frame'], i) not in used and r['source'] == 'reviewed']
        same = [(iou(p['box'], r['bbox_norm']), i, r) for i, r in cands if r['label'] == p['label']]
        best = max(same, default=(0, None, None))
        if best[0] >= 0.5:
            used.add((p['frame'], best[1])); survived[p['verify']] += 1; surv_t[p['tier']] += 1; ious.append(best[0])
        else:
            other = [(iou(p['box'], r['bbox_norm']), i, r) for i, r in cands if r['label'] != p['label']]
            b2 = max(other, default=(0, None, None))
            if b2[0] >= 0.5:
                used.add((p['frame'], b2[1])); survived[p['verify']] += 1; surv_t[p['tier']] += 1; ious.append(b2[0]); relabel += 1
            else:
                deleted[p['verify']] += 1; del_t[p['tier']] += 1

    print('\nsurvival of a proposed box through review, by cross-verifier verdict')
    print(f"{'verdict':>10s}{'proposed':>10s}{'kept':>8s}{'deleted':>9s}{'deletion rate':>15s}")
    for v in ('yes', 'no', None):
        n = survived[v] + deleted[v]
        if n:
            print(f'{str(v):>10s}{n:10d}{survived[v]:8d}{deleted[v]:9d}{deleted[v] / n:15.3f}')
    tot_k, tot_d = sum(survived.values()), sum(deleted.values())
    print(f"{'all':>10s}{tot_k + tot_d:10d}{tot_k:8d}{tot_d:9d}{tot_d / (tot_k + tot_d):15.3f}")
    print(f'  boxes kept but relabelled: {relabel}')
    print('\nsame, by proposal tier')
    print(f"{'tier':>12s}{'proposed':>10s}{'kept':>8s}{'deleted':>9s}{'deletion rate':>15s}")
    for t in sorted({p['tier'] for p in proposals}, key=str):
        n = surv_t[t] + del_t[t]
        if n: print(f'{str(t):>12s}{n:10d}{surv_t[t]:8d}{del_t[t]:9d}{del_t[t] / n:15.3f}')

    if ious:
        # review was accept / relabel / delete; box geometry was fixed at SAM-refinement time, so this
        # confirms the reviewer did not redraw boxes rather than measuring localisation agreement.
        print(f'\nbox geometry preserved through review: {sum(1 for i in ious if i >= 0.99) / len(ious):.3f} of kept boxes at IoU >= 0.99 (n={len(ious)})')

    corr = json.load(open(f'{POOL}/detect_coco_corrected.json'))
    print('\nreviewer verdicts on the proposals it examined:', dict(Counter(a.get('claude_verdict') for a in corr['annotations'])))
    print('relabelled from:', dict(Counter(a.get('relabel_from') for a in corr['annotations'] if a.get('relabel_from'))))


if __name__ == '__main__':
    main()
