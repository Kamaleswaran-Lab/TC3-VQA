# Builds a YOLO detection dataset from the released equipment layer with a fixed image-level split.
from tc3_vqa.paths import FRAMES, RELEASE, WORK
import json, os, random, sys
from collections import Counter, defaultdict

DEP = RELEASE
FRAMES = FRAMES      # every released frame, local copy
ROOT = sys.argv[1] if len(sys.argv) > 1 else WORK + '/yolo_release'   # output dataset directory
VAL_FRAC, SEED = 0.2, 0


def main():
    coco = json.load(open(f'{DEP}/data/detection_coco.json'))
    names = [c['name'] for c in coco['categories']]
    cid0 = {c['id']: i for i, c in enumerate(coco['categories'])}
    img = {im['id']: im for im in coco['images']}
    by_img = defaultdict(list)
    for a in coco['annotations']:
        by_img[a['image_id']].append(a)

    ids = sorted(by_img)
    rnd = random.Random(SEED)
    rnd.shuffle(ids)
    n_val = int(len(ids) * VAL_FRAC)
    val = set(ids[:n_val])

    # keep every class represented in train: move an image back if a class would otherwise be train-empty
    cls_train = Counter(cid0[a['category_id']] for i in ids if i not in val for a in by_img[i])
    for k in range(len(names)):
        if cls_train[k] == 0:
            for i in list(val):
                if any(cid0[a['category_id']] == k for a in by_img[i]):
                    val.discard(i)
                    for a in by_img[i]:
                        cls_train[cid0[a['category_id']]] += 1
                    break

    for sub in ('train', 'val'):
        os.makedirs(f'{ROOT}/images/{sub}', exist_ok=True)
        os.makedirs(f'{ROOT}/labels/{sub}', exist_ok=True)

    n, nb = Counter(), Counter()
    for iid in ids:
        im = img[iid]; W, H = im['width'], im['height']
        sub = 'val' if iid in val else 'train'
        stem = im['file_name'][:-4] if im['file_name'].endswith('.jpg') else im['file_name']
        link = f'{ROOT}/images/{sub}/{stem}.jpg'
        src = f'{FRAMES}/{im["file_name"]}'
        assert os.path.exists(src), src
        if not os.path.exists(link):
            os.symlink(src, link)
        lines = []
        for a in by_img[iid]:
            x, y, w, h = a['bbox']
            lines.append(f'{cid0[a["category_id"]]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}')
            nb[sub] += 1
        open(f'{ROOT}/labels/{sub}/{stem}.txt', 'w').write('\n'.join(lines) + '\n')
        n[sub] += 1

    open(f'{ROOT}/data.yaml', 'w').write(
        f'path: {ROOT}\ntrain: images/train\nval: images/val\nnc: {len(names)}\nnames: {names}\n')
    print(f'YOLO dataset from the released layer: train {n["train"]} images / {nb["train"]} boxes, '
          f'val {n["val"]} images / {nb["val"]} boxes, {len(names)} classes')
    for sub in ('train', 'val'):
        c = Counter()
        for f in os.listdir(f'{ROOT}/labels/{sub}'):
            for line in open(f'{ROOT}/labels/{sub}/{f}'):
                if line.strip(): c[names[int(line.split()[0])]] += 1
        print(f'  {sub}: ' + ', '.join(f'{k} {v}' for k, v in c.most_common()))
    print('wrote', f'{ROOT}/data.yaml')


if __name__ == '__main__':
    main()
