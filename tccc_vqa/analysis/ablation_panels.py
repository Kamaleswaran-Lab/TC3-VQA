# Quantitative panels of the model-authored QA comparison from figure_data.json: wrong-intervention rates, clean-QA
# rates per judge and the rubric table. Writes PNG and PDF files.
from tccc_vqa.paths import EXPERIMENTS
import json, math, textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
DATA = json.load(open(Path(EXPERIMENTS) / 'conventional_ablation' / 'figure_data.json'))
OUT = HERE / 'panels'; OUT.mkdir(exist_ok=True)

for f in (Path.home() / '.local/share/fonts/carlito').glob('*.ttf'):
    fm.fontManager.addfont(str(f))
FAMILY = 'Carlito' if any(f.name == 'Carlito' for f in fm.fontManager.ttflist) else 'Ubuntu Sans'
plt.rcParams.update({
    'font.family': FAMILY, 'font.size': 10, 'axes.labelsize': 10,
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
OPEN_VLM = '#b84a3c'                                   # every open VLM writing alone shares the Qwen-alone colour
FIXED_ROWS = ['released', 'qwen_concept', 'claude_alone']
AXES = [('medical_accuracy', 'Accuracy'), ('protocol_adherence', 'Protocol'), ('completeness', 'Completeness'),
        ('actionability', 'Actionability'), ('safety', 'Safety')]
MARKERS = ['o', 's', 'D', '^', 'v', 'P']
GRID = '#e8e8e8'
INK = '#222222'


def group_title(fig, ax, letter, text, dy=0.06):
    pos = ax.get_position()
    fig.text(pos.x0, pos.y1 + dy, letter, fontsize=13, fontweight='bold', color=INK, va='bottom', ha='left')
    fig.text(pos.x0 + 0.22 / fig.get_figwidth(), pos.y1 + dy, text, fontsize=11.5, color=INK, va='bottom', ha='left')   # fixed 0.22 in gap


def sub_title(ax, text):
    ax.set_title(text, loc='left', fontsize=10, color='#444444', pad=6)


def style_x(ax, lo, hi, ticks, label):
    ax.set_xlim(lo, hi); ax.set_xticks(ticks)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel(label, color='#333333')


def gen_rows():
    G = DATA['generators']
    rest = sorted((k for k in G if k not in FIXED_ROWS), key=lambda k: G[k]['wrong_open']['rate'])
    return [k for k in FIXED_ROWS if k in G] + rest


def draw_d(axes, show_labels=True):
    G = DATA['generators']; rows = gen_rows()
    ys = list(range(len(rows)))[::-1]
    top = max(100 * v['hi'] for k in rows for v in (G[k]['wrong_open'], G[k]['wrong_mcq']) if v)
    xmax = 10 * math.ceil((top + 9) / 10)
    for i, (ax, field, label) in enumerate(zip(axes, ('wrong_open', 'wrong_mcq'), ('Open questions', 'Recognition answers'))):
        for y, k in zip(ys, rows):
            v = G[k][field]; col = COLOR.get(k, OPEN_VLM)
            if v is None:
                ax.text(0.8, y, 'n/a (concept given)', va='center', fontsize=9, color='#999999', style='italic')
                continue
            r, lo, hi = 100 * v['rate'], 100 * v['lo'], 100 * v['hi']
            if field == 'wrong_mcq':
                other = 100 * v['rate_other_listed']
                ax.barh(y, max(other, 0.25), height=0.6, color=col, zorder=2)
                ax.barh(y, r - other, left=other, height=0.6, color=col, alpha=0.38, zorder=2, lw=0)
            else:
                ax.barh(y, max(r, 0.25), height=0.6, color=col, zorder=2)
            ax.plot([lo, hi], [y, y], color='#333333', lw=0.9, zorder=3, solid_capstyle='butt')
            ax.text(max(hi, r) + 1.0, y, f'{r:.1f}%', va='center', fontsize=9, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels([G[k]['label'] for k in rows] if (i == 0 and show_labels) else [])
        style_x(ax, 0, xmax, list(range(0, xmax + 1, 10 if xmax <= 50 else 20)), '% of questions (95% CI)')
        sub_title(ax, label)
    axes[1].legend(handles=[Patch(color='#777777', label='another listed intervention'),
                            Patch(color='#777777', alpha=0.38, lw=0, label='none of the 12')],
                   loc='upper right', frameon=False, fontsize=8.5, handlelength=1.2, handletextpad=0.4, borderaxespad=0.2)


def judge_offsets(n):
    span = min(0.3, 0.1 * (n - 1) + 0.05)
    return [0.0] if n == 1 else [span - 2 * span * j / (n - 1) for j in range(n)]


def draw_e(ax, show_labels=True):
    judges = DATA['judges']; ys = list(range(len(ARMS)))[::-1]; offs = judge_offsets(len(judges))
    lows = [100 * DATA['arms'][a][j['key']]['clean']['lo'] for a, _, _ in ARMS for j in judges]
    lo_x = 10 * math.floor(min(lows) / 10)
    for y, (arm, name, col) in zip(ys, ARMS):
        for o, mk, j in zip(offs, MARKERS, judges):
            c = DATA['arms'][arm][j['key']]['clean']; yy = y + o
            ax.plot([100 * c['lo'], 100 * c['hi']], [yy, yy], color=col, lw=1.2, zorder=2, solid_capstyle='butt')
            ax.scatter(100 * c['rate'], yy, s=30, marker=mk, color=col, edgecolor='white', lw=0.7, zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels([a[1] for a in ARMS] if show_labels else [])
    style_x(ax, lo_x, 100.5, list(range(lo_x, 101, 10)), '% clean QA (95% CI)')
    for mk, j in zip(MARKERS, judges):
        ax.scatter([], [], marker=mk, color='#777777', s=26, label=j['label'])
    ax.legend(loc='upper left', frameon=False, fontsize=8.5, handletextpad=0.2, borderaxespad=0.0, labelspacing=0.25)
    sub_title(ax, 'No wrong-intervention question, no contradicted claim')


def draw_rubric(ax, axis, show_labels=True, title=None):
    judges = DATA['judges']; ys = list(range(len(judges)))[::-1]
    offs = [0.27, 0.09, -0.09, -0.27]
    lows = [DATA['arms'][a][j['key']]['rubric_all'][axis]['lo'] for a, _, _ in ARMS for j in judges]
    lo_x = min(3.0, math.floor(2 * min(lows)) / 2)
    for y, j in zip(ys, judges):
        for o, (arm, _, col) in zip(offs, ARMS):
            m = DATA['arms'][arm][j['key']]['rubric_all'][axis]
            ax.plot([m['lo'], m['hi']], [y + o, y + o], color=col, lw=1.2, zorder=2, solid_capstyle='butt')
            ax.scatter(m['mean'], y + o, s=22, color=col, edgecolor='white', lw=0.6, zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels([j['label'] for j in judges] if show_labels else [])
    ticks = [lo_x + 0.5 * i for i in range(int(round((5.0 - lo_x) / 0.5)) + 1)]
    style_x(ax, lo_x, 5.05, ticks, 'mean score (1–5)')
    sub_title(ax, title or dict(AXES)[axis])


def draw_f_table(ax, fontsize=10, w_lab=0.34):
    """medical accuracy (1-5) per arm and judge on all audit-confirmed open questions; best arm per judge in bold"""
    judges = DATA['judges']; ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); T = ax.transAxes
    short = {'llama70b': 'Llama-3.3\n70B', 'medgemma27b': 'MedGemma\n27B', 'phi4': 'Phi-4\n14B', 'deepseek70b': 'DeepSeek-R1\n70B',
             'gptoss120b': 'gpt-oss\n120b', 'pixtral_large': 'Pixtral\nLarge'}
    cols = [short.get(j['key'], j['label']) for j in judges]
    vals = [[DATA['arms'][a][j['key']]['rubric_all']['medical_accuracy']['mean'] for j in judges] for a, _, _ in ARMS]
    best = [max(range(len(ARMS)), key=lambda i: vals[i][c]) for c in range(len(judges))]
    x0 = 0.0; w = (1 - w_lab) / len(judges); row_h = 1 / (len(ARMS) + 1.6)
    y = 1 - row_h * 0.8
    for c, name in enumerate(cols):
        ax.text(x0 + w_lab + w * (c + 0.5), y, name, ha='center', va='center', fontsize=fontsize - 1, color='#444444', transform=T)
    ax.plot([0, 1], [y - row_h * 0.62] * 2, color='#555555', lw=0.8, transform=ax.transAxes, clip_on=False)
    for i, (arm, name, col) in enumerate(ARMS):
        yy = y - row_h * (i + 1)
        ax.add_patch(plt.Rectangle((0, yy - 0.14 * row_h), 0.018, 0.28 * row_h, color=col, transform=ax.transAxes, clip_on=False))
        ax.text(0.03, yy, name, ha='left', va='center', fontsize=fontsize, color=INK, transform=T)
        for c in range(len(judges)):
            ax.text(x0 + w_lab + w * (c + 0.5), yy, f'{vals[i][c]:.2f}', ha='center', va='center', fontsize=fontsize,
                    color=INK, fontweight='bold' if best[c] == i else 'normal', transform=T)
    ax.plot([0, 1], [y - row_h * (len(ARMS) + 0.5)] * 2, color='#555555', lw=0.8, transform=ax.transAxes, clip_on=False)
    with open(OUT / 'panel_f_rubric_table.csv', 'w') as f:
        f.write('arm,' + ','.join(j['label'] for j in judges) + '\n')
        for (arm, name, _), row in zip(ARMS, vals):
            f.write(name + ',' + ','.join(f'{v:.2f}' for v in row) + '\n')


def arm_legend(fig, x, y, ncol=4, fontsize=9.5):
    handles = [plt.Line2D([], [], marker='s', ls='', color=c, markersize=8) for _, _, c in ARMS]
    names = [n for _, n, _ in ARMS][:-1] + ['Open VLM alone (e, f: Qwen2.5-VL)' if len(DATA['generators']) > 4 else ARMS[-1][1]]
    fig.legend(handles, names, loc='upper center', ncol=ncol, frameon=False, fontsize=fontsize,
               bbox_to_anchor=(x, y), handletextpad=0.3, columnspacing=1.8)


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=400, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


E_ARMS = [('released', 'Released\n(verbatim, cited)', '#0f6b62'), ('qwen_concept', 'Qwen2.5-VL-72B,\ntold the concept', '#8c6bb1'),
          ('claude_alone', 'Claude Opus 5\nalone', '#d98c3f'), ('qwen_alone', 'Qwen2.5-VL-72B\nalone', '#b84a3c')]
E_JUDGES = [('llama70b', 'Llama-3.3-70B', 'o'), ('medgemma27b', 'MedGemma-27B', 's'), ('phi4', 'Phi-4', 'D'),
            ('deepseek70b', 'DeepSeek-R1-70B', '^'), ('gptoss120b', 'gpt-oss-120b', 'v')]


def draw_e_v2(fig, ax):
    """clean QA per arm: range across the five judges (band), each judge (open marker), five-judge mean (tick + label)"""
    ys = list(range(len(E_ARMS)))[::-1]
    for y, (arm, name, col) in zip(ys, E_ARMS):
        vals = [100 * DATA['arms'][arm][j]['clean']['rate'] for j, _, _ in E_JUDGES]
        lo, hi, mean = min(vals), max(vals), sum(vals) / len(vals)
        ax.plot([lo, hi], [y, y], color=col, lw=9, alpha=0.2, solid_capstyle='round', zorder=1)
        for (j, jn, mk), v in zip(E_JUDGES, vals):
            ax.scatter(v, y, marker=mk, s=24, facecolor='white', edgecolor='#555555', lw=0.9, zorder=3)
        ax.plot([mean, mean], [y - 0.3, y + 0.3], color=col, lw=2.6, zorder=4, solid_capstyle='butt')
        ax.text(hi + 1.2, y, f'{mean:.1f}%', va='center', fontsize=10.5, color=col, fontweight='bold')
    ax.set_yticks(ys); ax.set_yticklabels([n for _, n, _ in E_ARMS], fontsize=9.8); ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.6, len(E_ARMS) - 0.4)
    ax.set_xlim(68, 104); ax.set_xticks([70, 80, 90, 100]); ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel('% of questions judged clean', color='#333333')
    handles = [plt.Line2D([], [], marker=mk, ls='', markerfacecolor='white', markeredgecolor='#555555', markersize=5.5, label=jn)
               for _, jn, mk in E_JUDGES]
    handles.append(plt.Line2D([], [], color='#777777', lw=2.6, label='mean of five judges'))
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=8.8, bbox_to_anchor=(0.58, 0.02),
               handletextpad=0.3, columnspacing=1.2)


def figure_examples_d():
    """manuscript figure: photo examples (a-c) above the wrong-intervention panel (d)"""
    from tccc_vqa.analysis import ablation_examples as top
    from PIL import Image
    n = DATA['n_items']
    fig = plt.figure(figsize=(15, 11.5))
    tg = fig.add_gridspec(2, 3, height_ratios=[0.58, 1.05], hspace=0.03, wspace=0.12, top=0.97, bottom=0.36)
    items = {json.loads(l)['item_id']: json.loads(l) for l in open(top.DEP / 'data/items.jsonl')}
    for c, ex in enumerate(top.EXAMPLES):
        it = items[ex['item']]; fr = it['frame_refs'][ex['frame']]
        im = Image.open(top.DEP / 'frames' / fr['frame_id']).convert('RGB'); W, H = im.size
        im = top.trim_black(im.crop((0, int(H * ex['crop'][0]), W, int(H * ex['crop'][1]))))
        a = fig.add_subplot(tg[0, c]); a.imshow(im); a.set_xticks([]); a.set_yticks([]); a.set_anchor('S')
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color('#bbbbbb')
        a.set_title(ex['title'], fontsize=11.5, loc='left', fontweight='bold', color=INK)
        tx = fig.add_subplot(tg[1, c]); tx.axis('off')
        yy = top.text_block(tx, 1.0, 'Released: verbatim, cited', ex['ours'], ARMS[0][2])
        yy = top.text_block(tx, yy - 0.02, 'Qwen2.5-VL-72B alone: model-written', ex['conv'], ARMS[3][2])
        for w in textwrap.wrap(ex['note'], 66):
            tx.text(0.0, yy - 0.01, w, fontsize=8, style='italic', color='#6b2f24', va='top', transform=tx.transAxes); yy -= 0.058
    gd = fig.add_gridspec(1, 2, wspace=0.08, top=0.285, bottom=0.05, left=0.20, right=0.93)
    ad = [fig.add_subplot(gd[0, 0]), fig.add_subplot(gd[0, 1])]
    draw_d(ad)
    group_title(fig, ad[0], 'd', f'Questions presupposing another intervention ({n} audit-confirmed candidates)', dy=0.035)
    fig.savefig(HERE / 'figure_examples_d.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(HERE / 'figure_examples_d.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    n = DATA['n_items']; n_gen = len(DATA['generators']); n_j = len(DATA['judges'])
    h_d = 1.3 + 0.28 * n_gen; h_f = 1.5 + 0.42 * n_j

    fig, axes = plt.subplots(1, 2, figsize=(8.4, h_d)); fig.subplots_adjust(wspace=0.1, top=1 - 0.75 / h_d, bottom=0.6 / h_d, left=0.25)
    draw_d(axes); group_title(fig, axes[0], 'd', f'Questions presupposing another intervention ({n} audit-confirmed items)', dy=0.3 / h_d)
    save(fig, 'panel_d_wrong_intervention')

    fig, ax = plt.subplots(figsize=(4.8, 3.3)); fig.subplots_adjust(top=0.8, bottom=0.17, left=0.33)
    draw_e(ax); group_title(fig, ax, 'e', f'Clean QA ({n_j} judges)', dy=0.09)
    save(fig, 'panel_e_clean_qa')

    fig, ax = plt.subplots(figsize=(1.9 + 0.95 * n_j, 2.2)); fig.subplots_adjust(top=0.72, bottom=0.04, left=0.03, right=0.99)
    draw_f_table(ax); group_title(fig, ax, 'f', f'LLM-judge medical accuracy, 1-5 ({n} audit-confirmed items, source excluded)', dy=0.1)
    save(fig, 'panel_f_rubric')

    fig, axs = plt.subplots(1, len(AXES), figsize=(13.5, h_f), sharey=False); fig.subplots_adjust(wspace=0.2, top=1 - 0.8 / h_f, bottom=1.0 / h_f, left=0.12)
    for i, (ax, (key, _)) in enumerate(zip(axs, AXES)):
        draw_rubric(ax, key, show_labels=i == 0)
    group_title(fig, axs[0], 'S', 'LLM-judge rubric, all axes (source excluded, audit-confirmed items)', dy=0.3 / h_f)
    arm_legend(fig, 0.55, 0.3 / h_f)
    save(fig, 'panel_f_rubric_all_axes')

    fig, ax = plt.subplots(figsize=(6.6, 3.0)); fig.subplots_adjust(top=0.97, bottom=0.2, left=0.25)
    draw_e_v2(fig, ax); save(fig, 'panel_e_clean_qa_v2')
    figure_examples_d()

    # composite
    from tccc_vqa.analysis import ablation_examples as top
    from PIL import Image
    fig = plt.figure(figsize=(16.5, 12.5))
    tg = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.04, wspace=0.12, top=0.965, bottom=0.47)
    items = {json.loads(l)['item_id']: json.loads(l) for l in open(top.DEP / 'data/items.jsonl')}
    for c, ex in enumerate(top.EXAMPLES):
        it = items[ex['item']]; fr = it['frame_refs'][ex['frame']]
        im = Image.open(top.DEP / 'frames' / fr['frame_id']).convert('RGB'); W, H = im.size
        im = top.trim_black(im.crop((0, int(H * ex['crop'][0]), W, int(H * ex['crop'][1]))))
        a = fig.add_subplot(tg[0, c]); a.imshow(im); a.set_xticks([]); a.set_yticks([]); a.set_anchor('S')
        for s in a.spines.values():
            s.set_visible(True); s.set_color('#bbbbbb')
        a.set_title(ex['title'], fontsize=11.5, loc='left', fontweight='bold', color=INK)
        tx = fig.add_subplot(tg[1, c]); tx.axis('off')
        yy = top.text_block(tx, 1.0, 'Released: verbatim, cited', ex['ours'], ARMS[0][2])
        yy = top.text_block(tx, yy - 0.02, 'Qwen alone: model-written', ex['conv'], ARMS[3][2])
        for w in textwrap.wrap(ex['note'], 66):
            tx.text(0.0, yy - 0.01, w, fontsize=8, style='italic', color='#6b2f24', va='top', transform=tx.transAxes); yy -= 0.058

    outer = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.1, 1.85], wspace=0.42, top=0.37, bottom=0.08, left=0.13, right=0.985)
    gd = outer[0, 0].subgridspec(1, 2, wspace=0.08)
    ad = [fig.add_subplot(gd[0, 0]), fig.add_subplot(gd[0, 1])]
    ae = fig.add_subplot(outer[0, 1]); af = fig.add_subplot(outer[0, 2])
    draw_d(ad); draw_e(ae); draw_f_table(af, fontsize=8.5, w_lab=0.36)
    group_title(fig, ad[0], 'd', f'Questions presupposing another intervention ({n} audit-confirmed items)', dy=0.035)
    group_title(fig, ae, 'e', 'Clean QA', dy=0.035)
    group_title(fig, af, 'f', 'LLM-judge medical accuracy (1-5)', dy=0.035)
    arm_legend(fig, 0.55, 0.03, fontsize=10.5)
    fig.savefig(HERE / 'ablation_figure_v2.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(HERE / 'ablation_figure_v2.pdf', bbox_inches='tight', facecolor='white')
    print('font', FAMILY, '| generators', n_gen, '| judges', n_j, '| wrote', *(p.name for p in sorted(OUT.iterdir())), 'ablation_figure_v2.png/pdf')


if __name__ == '__main__':
    main()
