# Figure of the model-authored QA comparison: three public-domain examples with the reference QA and the model-written
# QA side by side, and the wrong-intervention panel.
from tc3_vqa.paths import EXPERIMENTS, RELEASE
import json, math, textwrap
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEP = Path(RELEASE)
EXP = Path(EXPERIMENTS + '/conventional_ablation')
OUT = EXP
OURS, CONV = '#1b9e77', '#d95f02'
AXES = ('medical_accuracy', 'protocol_adherence', 'completeness', 'actionability', 'safety')
AXIS_LABEL = ('Accuracy', 'Protocol', 'Completeness', 'Actionability', 'Safety')

# frame index and vertical crop (fractions of height kept, outside the released caption masks)
EXAMPLES = [
    dict(item='ans_00898', frame=0, crop=(0.0, 1.0), title='(a) Wrong intervention, plausible answer',
         ours=[('Recognition', 'Wound packing'),
               ('Q', 'How can the effectiveness of wound packing be improved according to TCCC guidance?'),
               ('A', '"The effectiveness of wound packing may be improved when skin closure over the packing can be achieved..."')],
         conv=[('Recognition', 'Hemostatic Dressing Application'),
               ('Q', 'What is the proper method for applying a tourniquet to control bleeding as shown here?'),
               ('A', 'Place tourniquet 2-3 inches above wound. Tighten until bleeding stops. Mark time...')],
         note='Wound packing is shown, not a tourniquet. The five judges still rated the answer 5, 5, 4, 3, 3 for accuracy.'),
    dict(item='ans_00958', frame=1, crop=(0.0, 0.70), title='(b) Wrong intervention, unsupported answer',
         ours=[('Recognition', 'Surgical cricothyroidotomy'),
               ('Q', 'At what site should the initial vertical skin incision for a surgical cricothyroidotomy be made?'),
               ('A', '"Make vertical incision through the skin over cricothyroid membrane."')],
         conv=[('Recognition', 'Needle Decompression'),
               ('Q', 'What should be done if the needle decompression does not relieve tension pneumothorax?'),
               ('A', 'Perform chest tube insertion immediately.')],
         note='No needle decompression is shown. The five judges rated the answer 1, 3, 3, 1, 2 for accuracy.'),
    dict(item='ans_01089', frame=0, crop=(0.0, 1.0), title='(c) Wrong intervention, off-target care',
         ours=[('Recognition', 'Chest seal application'),
               ('Q', 'After applying a chest seal, what complication must the casualty be monitored for and how is it treated?'),
               ('A', '"Monitor the casualty for the potential development of a subsequent tension pneumothorax..."')],
         conv=[('Recognition', 'Chest Seal Application'),
               ('Q', "Given the visible chest wound and the application of a tourniquet, what is the next step in TCCC for managing this casualty's condition effectively?"),
               ('A', 'The next step is to check for airway obstruction and breathing issues... Ensure the tourniquet is properly secured...')],
         note='A chest seal is shown and no tourniquet is present. The five judges rated the answer 2, 3, 3, 2, 3 for accuracy.'),
]


def trim_black(im, thresh=18):
    """drop letterbox or pillarbox borders (rows or columns that are almost black) left by the source video"""
    import numpy as np
    a = np.asarray(im.convert('L'))
    cols = np.where(a.max(axis=0) > thresh)[0]; rows = np.where(a.max(axis=1) > thresh)[0]
    if len(cols) == 0 or len(rows) == 0:
        return im
    return im.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * p, 100 * (c - h) / d, 100 * (c + h) / d


def text_block(ax, y, header, lines, colour, width=62):
    ax.text(0.0, y, header, color='white', fontsize=8.5, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.25', fc=colour, ec='none'), transform=ax.transAxes)
    y -= 0.085
    for label, s in lines:
        wrapped = textwrap.wrap(f'{label}: {s}', width)
        for w in wrapped:
            ax.text(0.0, y, w, fontsize=7.6, va='top', transform=ax.transAxes)
            y -= 0.058
        y -= 0.012
    return y


def main():
    items = {json.loads(l)['item_id']: json.loads(l) for l in open(DEP / 'data/items.jsonl')}
    full = json.load(open(EXP / 'full/summary.json'))
    lso = json.load(open(EXP / 'full_lso/summary.json'))

    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(15, 10.2))
    top = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.04, wspace=0.12, top=0.95, bottom=0.37)
    bot = fig.add_gridspec(1, 3, wspace=0.42, top=0.29, bottom=0.06, left=0.06, right=0.98)

    for c, ex in enumerate(EXAMPLES):
        it = items[ex['item']]; fr = it['frame_refs'][ex['frame']]
        assert it['pixels_released']
        im = Image.open(DEP / 'frames' / fr['frame_id']).convert('RGB')
        W, H = im.size
        im = trim_black(im.crop((0, int(H * ex['crop'][0]), W, int(H * ex['crop'][1]))))
        ax = fig.add_subplot(top[0, c]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ex['title'], fontsize=11, fontweight='bold', loc='left')
        tx = fig.add_subplot(top[1, c]); tx.axis('off')
        y = text_block(tx, 1.0, 'Released (ours): verbatim, cited', ex['ours'], OURS)
        y = text_block(tx, y - 0.02, 'Conventional: model-written', ex['conv'], CONV)
        for w in textwrap.wrap(ex['note'], 66):
            tx.text(0.0, y - 0.01, w, fontsize=7.6, style='italic', color='#7a2e00', va='top', transform=tx.transAxes)
            y -= 0.058

    # (d) wrong-intervention rates, judge-free
    cc = full['concept_check']
    oa, oc = cc['A']['open_questions'], cc['C']['open_questions']
    ma, mc = cc['A']['mcq'], cc['C']['mcq']
    n_open = full['by_arm_type']['A:open_pooled']['llama70b']['n']          # 1,604 open questions per arm
    bars = [('Open questions about\nanother intervention\n(of all open questions)', wilson(oa['strict_mismatch'], n_open), wilson(oc['strict_mismatch'], n_open)),
            ('Recognition answer\nwrong or non-specific', wilson(543 - ma['strict_correct'], 543), wilson(543 - mc['strict_correct'], 543))]
    ax = fig.add_subplot(bot[0, 0])
    for i, (lab, a, cv) in enumerate(bars):
        for dx, v, col in ((-0.18, a, OURS), (0.18, cv, CONV)):
            ax.bar(i + dx, v[0], 0.34, color=col, yerr=[[v[0] - v[1]], [v[2] - v[0]]], capsize=3, error_kw=dict(lw=0.8))
            ax.text(i + dx, v[2] + 1.2, f'{v[0]:.1f}%', ha='center', fontsize=8)
    ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel('% (Wilson 95% CI)', fontsize=8.5); ax.set_ylim(0, 45)
    ax.set_title('(d) Wrong intervention (judge-free)', fontsize=11, fontweight='bold', loc='left')

    # (e) clean QA per judge
    ax = fig.add_subplot(bot[0, 1])
    rows = [('Llama-3.3-70B\n(source excluded)', lso['clean']['llama70b']), ('MedGemma-27B\n(source excluded)', lso['clean']['medgemma27b']),
            ('Pixtral-Large\n(source included)', full['clean']['pixtral_large'])]
    for i, (lab, cl) in enumerate(rows):
        for dx, arm, col in ((-0.18, 'A', OURS), (0.18, 'C', CONV)):
            r = 100 * cl[arm]['rate']; lo, hi = (100 * v for v in cl[arm]['ci95'])
            ax.bar(i + dx, r, 0.34, color=col, yerr=[[r - lo], [hi - r]], capsize=3, error_kw=dict(lw=0.8))
            ax.text(i + dx, hi + 1.0, f'{r:.1f}', ha='center', fontsize=8)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(50, 105); ax.set_ylabel('% clean QA (video bootstrap 95% CI)', fontsize=8.5)
    ax.set_title('(e) Clean QA', fontsize=11, fontweight='bold', loc='left')

    # (f) TC3-VLM rubric, leave-source-out
    ax = fig.add_subplot(bot[0, 2])
    y = list(range(len(AXES)))[::-1]
    for tag, marker, off in (('llama70b', 'o', 0.12), ('medgemma27b', 's', -0.12)):
        sa = lso['by_arm_type']['A:open_pooled'][tag]; sc = lso['by_arm_type']['C:open_pooled'][tag]
        for yi, a in zip(y, AXES):
            ax.plot([sc[a], sa[a]], [yi + off, yi + off], color='#bbbbbb', lw=1, zorder=1)
            ax.scatter(sa[a], yi + off, color=OURS, marker=marker, s=34, zorder=2)
            ax.scatter(sc[a], yi + off, color=CONV, marker=marker, s=34, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(AXIS_LABEL, fontsize=8.5); ax.set_xlim(3.0, 5.05)
    ax.set_xlabel('mean score (1-5), open answers', fontsize=8.5)
    ax.scatter([], [], color='grey', marker='o', label='Llama-3.3-70B'); ax.scatter([], [], color='grey', marker='s', label='MedGemma-27B')
    ax.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.45, -0.2), ncol=2, frameon=False)
    ax.set_title('(f) LLM-judge rubric (source excluded)', fontsize=11, fontweight='bold', loc='left')

    handles = [plt.Rectangle((0, 0), 1, 1, color=OURS), plt.Rectangle((0, 0), 1, 1, color=CONV)]
    fig.legend(handles, ['Released (correct by construction)', 'Conventional (Qwen2.5-VL-72B alone, model-written)'],
               loc='upper center', ncol=2, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(OUT / 'ablation_figure_draft.png', dpi=220, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / 'ablation_figure_draft.pdf', bbox_inches='tight', facecolor='white')
    print('wrote', OUT / 'ablation_figure_draft.png')


if __name__ == '__main__':
    main()
