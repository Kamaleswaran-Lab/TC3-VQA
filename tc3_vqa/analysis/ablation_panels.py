# Draws Figure 6 of the paper from the released package: three reference pairs next to the pair a model wrote from the
# same frames (a-c), and how often directly generated questions presuppose another intervention (d). Set
# TC3_VQA_RELEASE; the figure goes to TC3_VQA_FIGURES.
from tc3_vqa.paths import RELEASE, FIGURES
from tc3_vqa.analysis import ablation_examples as top
import json, math, os, textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch

DATA = json.load(open(f'{RELEASE}/results/model_authored_qa.json'))

for f in (Path.home() / '.local/share/fonts/freefont').glob('*.ttf'):
    fm.fontManager.addfont(str(f))
FAMILY = 'FreeSans' if any(f.name == 'FreeSans' for f in fm.fontManager.ttflist) else 'Ubuntu Sans'
plt.rcParams.update({
    'font.family': FAMILY, 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False, 'axes.linewidth': 0.8,
    'axes.edgecolor': '#555555', 'xtick.color': '#444444', 'ytick.color': '#333333',
    'xtick.major.width': 0.8, 'xtick.major.size': 3, 'ytick.major.size': 0,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

ARMS = [('released', 'Released', '#0f6b62'),
        ('qwen_concept', 'Qwen + verified concept', '#8c6bb1'),
        ('claude_alone', 'Opus 5 alone', '#d98c3f'),
        ('qwen_alone', 'Qwen alone', '#b84a3c')]
COLOR = {a: c for a, _, c in ARMS}
OPEN_VLM = '#b84a3c'                                   # every open VLM writing alone shares one colour
FIXED_ROWS = ['released', 'qwen_concept', 'claude_alone']
GRID = '#e8e8e8'
INK = '#222222'


def sub_title(ax, text):
    ax.set_title(text, loc='left', fontsize=8, color='#444444', pad=4)


def style_x(ax, lo, hi, ticks, label):
    ax.set_xlim(lo, hi); ax.set_xticks(ticks)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel(label, color='#333333')


def gen_rows():
    G = DATA['generators']
    rest = sorted((k for k in G if k not in FIXED_ROWS), key=lambda k: G[k]['wrong_open']['rate'])
    return [k for k in FIXED_ROWS if k in G] + rest


def draw_d(axes, show_labels=True):
    """left: open questions naming another intervention (frames only, and after the model's own closed-set pick when that
    run exists); right: the same model choosing among the 12 concepts on the same frames"""
    G = DATA['generators']; rows = gen_rows()
    ys = list(range(len(rows)))[::-1]
    fields = ('wrong_open', 'wrong_closed')
    top = max(100 * v['hi'] for k in rows for f in fields for v in [G[k].get(f)] if v)
    xmax = 10 * math.ceil((top + 9) / 10)
    two = False                       # the self-pick arm is reported in the text: three of the rows have no counterpart
    for i, (ax, field, label) in enumerate(zip(axes, fields, ('Open questions', 'Recognition, 12 options'))):
        for y, k in zip(ys, rows):
            v = G[k].get(field); col = COLOR.get(k, OPEN_VLM)
            if k == 'released' and field == 'wrong_closed':
                ax.text(0.8, y, 'reference label', va='center', fontsize=7.5, color='#999999', style='italic')
                continue
            if v is None:
                ax.text(0.8, y, 'n/a', va='center', fontsize=7.5, color='#999999', style='italic')
                continue
            v2 = G[k].get('wrong_open_selfpick') if (field == 'wrong_open' and two) else None
            h, dy = (0.34, 0.19) if v2 else (0.6, 0.0)
            for vv, yy, alpha in ((v, y + dy, 1.0), (v2, y - dy, 0.45)):
                if vv is None:
                    continue
                r, lo, hi = 100 * vv['rate'], 100 * vv['lo'], 100 * vv['hi']
                ax.barh(yy, max(r, 0.25), height=h, color=col, alpha=alpha, lw=0, zorder=2)
                ax.plot([lo, hi], [yy, yy], color='#333333', lw=0.9, zorder=3, solid_capstyle='butt')
                ax.text(max(hi, r) + 1.0, yy, f'{r:.1f}%', va='center', fontsize=7.5 if not v2 else 6.8, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels([G[k]['label'] for k in rows] if (i == 0 and show_labels) else [])
        ax.set_ylim(-0.7, len(rows) - 0.25)                 # headroom so the top row clears the panel title
        style_x(ax, 0, xmax, list(range(0, xmax + 1, 10 if xmax <= 50 else 20)), '% of questions (95% CI)' if i == 0 else '% of items (95% CI)')
        sub_title(ax, label)
    if two:
        axes[0].legend(handles=[Patch(color='#777777', label='frames only'),
                                Patch(color='#777777', alpha=0.45, lw=0, label='after the model\'s own pick from the 12')],
                       loc='upper right', frameon=False, handlelength=1.2, handletextpad=0.4, borderaxespad=0.2)


def figure_examples_d():
    """manuscript figure at print width: photo examples (a-c) above the wrong-intervention panel (d)"""
    from PIL import Image
    n = DATA['n_items']
    W_IN, H_IN, TOP, BOT, IMG = 6.5, 5.7, 0.975, 0.355, 0.42        # IMG: image row as a fraction of the text row
    fig = plt.figure(figsize=(W_IN, H_IN))
    tg = fig.add_gridspec(2, 3, height_ratios=[IMG, 1.0], hspace=0.02, wspace=0.05, top=TOP, bottom=BOT, left=0.01, right=0.995)
    items = {json.loads(l)['item_id']: json.loads(l) for l in open(top.DEP / 'data/items.jsonl')}
    text_row_in = H_IN * (TOP - BOT) / (1 + IMG)
    fs, step = 6.4, 6.4 * 1.22 / 72 / text_row_in                                  # line step in axes fraction of the text row

    def block(ax, y, header, lines, colour, width=51):
        ax.text(0.0, y, header, color=colour, fontsize=8.8, fontweight='bold', va='top', transform=ax.transAxes)
        y -= step * 1.15 * 8.8 / fs
        for label, text in lines:
            for w in textwrap.wrap(f'{label}: {text}', width):
                ax.text(0.0, y, w, fontsize=fs, va='top', transform=ax.transAxes); y -= step
            y -= step * 0.25
        return y

    for c, ex in enumerate(top.EXAMPLES):
        it = items[ex['item']]; fr = it['frame_refs'][ex['frame']]
        im = Image.open(top.DEP / 'frames' / fr['frame_id']).convert('RGB'); W, H = im.size
        im = top.trim_black(im.crop((0, int(H * ex['crop'][0]), W, int(H * ex['crop'][1]))))
        a = fig.add_subplot(tg[0, c]); a.imshow(im); a.set_xticks([]); a.set_yticks([]); a.set_anchor('N')
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color('#bbbbbb')
        a.set_title(ex['title'], fontsize=7.5, loc='left', color=INK, pad=3)
        tx = fig.add_subplot(tg[1, c]); tx.axis('off')
        yy = block(tx, 1.0, 'Ours', ex['ours'], ARMS[0][2])
        yy = block(tx, yy - step * 0.3, 'Conventional', ex['conv'], ARMS[3][2])
        for w in textwrap.wrap(ex['note'], 53):
            tx.text(0.0, yy, w, fontsize=fs, style='italic', color='#6b2f24', va='top', transform=tx.transAxes); yy -= step
    gd = fig.add_gridspec(1, 2, wspace=0.10, top=0.27, bottom=0.055, left=0.19, right=0.965)
    ad = [fig.add_subplot(gd[0, 0]), fig.add_subplot(gd[0, 1])]
    draw_d(ad)
    pos = ad[0].get_position()
    fig.text(pos.x0, pos.y1 + 0.045, f'(d) Questions presupposing another intervention ({n} audit-confirmed candidates)', fontsize=8, color=INK, va='bottom', ha='left')
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(f'{FIGURES}/figure_examples_d.pdf', bbox_inches='tight', facecolor='white')
    print('wrote figure_examples_d.pdf')
    plt.close(fig)


if __name__ == '__main__':
    figure_examples_d()
