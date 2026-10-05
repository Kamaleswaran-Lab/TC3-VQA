# Draws Figures 4 and 5 of the paper from the released package: fig_overview.pdf, the composition of the released
# layers (refusal reasons, answer sources, equipment classes, body regions), and fig_baselines.pdf, recognition against
# refusal and coverage against risk for the five baseline models. Set TC3_VQA_RELEASE; figures go to TC3_VQA_FIGURES.
from tc3_vqa.paths import RELEASE, FIGURES
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path

DEP = RELEASE
OUT = FIGURES
NAME = {'qwen2vl7b': 'Qwen2-VL-7B', 'qwen25vl7b': 'Qwen2.5-VL-7B', 'phi35v': 'Phi-3.5-Vision',
        'internvl3_8b': 'InternVL3-8B', 'internvl3_38b': 'InternVL3-38B'}
COL = {'qwen2vl7b': '#2E75B6', 'qwen25vl7b': '#7FB3E0', 'phi35v': '#C00000',
       'internvl3_8b': '#7DA65B', 'internvl3_38b': '#2F5E23'}
# FreeSans (a Helvetica clone) at print size; falls back to DejaVu Sans when it is not installed
for f in (Path.home() / '.local/share/fonts/freefont').glob('*.ttf'):
    fm.fontManager.addfont(str(f))
FAMILY = 'FreeSans' if any(f.name == 'FreeSans' for f in fm.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family': FAMILY, 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'legend.fontsize': 7.5, 'axes.titlesize': 8.5, 'axes.linewidth': .7, 'xtick.major.width': .7,
                     'ytick.major.width': .7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def board():
    """baseline results in the model order of NAME"""
    b = json.load(open(f'{DEP}/results/baselines.json'))
    return [b[m] for m in NAME]


def fig_baselines(rows):
    """Figure 5: recognition against refusal, and coverage against risk."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3.2))
    # RWHR label position per model, in points from the marker; models are identified by the shared legend
    off = {'qwen2vl7b': (-24, 0, 'right', 'center'), 'qwen25vl7b': (0, -16, 'center', 'top'),
           'phi35v': (0, -22, 'center', 'top'), 'internvl3_8b': (12, 10, 'left', 'bottom'),
           'internvl3_38b': (0, 12, 'center', 'bottom')}
    for r in rows:
        x, y, m = r['rec_acc'], r['refusal_abstain_acc'], r['model']
        lo, hi = r['rec_acc_ci']
        a1.errorbar([x], [y], xerr=[[x - lo], [hi - x]], fmt='none', ecolor=COL[m], elinewidth=.9,
                    capsize=2, alpha=.8, zorder=2)
        a1.scatter([x], [y], s=50 + 520 * r['RWHR_default'] ** 2, color=COL[m], alpha=.72,
                   edgecolor='white', lw=1.0, zorder=3)
        dx, dy, ha, va = off[m]
        a1.annotate(f"RWHR {r['RWHR_default']:.2f}", (x, y), textcoords='offset points',
                    xytext=(dx, dy), ha=ha, va=va, fontsize=7, zorder=4)
    a1.set_xlabel('recognition accuracy (Wilson 95% interval)')
    a1.set_ylabel('refusal accuracy (correct abstention)')
    a1.set_xlim(.55, .95); a1.set_ylim(.05, 1.12)
    a1.set_xticks([.6, .7, .8, .9])
    a1.grid(alpha=.22, lw=.5)
    a1.set_title('(a) Recognition against refusal', loc='left')

    for r in rows:
        a2.plot([r['coverage']] * 2, [r['halluc_rate_unweighted'], r['RWHR_default']],
                color=COL[r['model']], lw=.8, alpha=.45, zorder=2)
        a2.scatter([r['coverage']], [r['RWHR_default']], s=54, color=COL[r['model']],
                   zorder=3, edgecolor='white', lw=.9)
        a2.scatter([r['coverage']], [r['halluc_rate_unweighted']], s=30, facecolor='white',
                   edgecolor=COL[r['model']], lw=.9, zorder=3)
    a2.set_xlabel('coverage (share of items answered)')
    a2.set_ylabel('risk among answered items')
    a2.set_xlim(.58, 1.0); a2.set_ylim(0, 1.12)
    a2.grid(alpha=.22, lw=.5)
    a2.set_title('(b) Coverage against risk', loc='left')
    kind = [plt.Line2D([], [], marker='o', ls='none', color='#666', markeredgecolor='white', markersize=6.5, label='RWHR (severity-weighted)'),
            plt.Line2D([], [], marker='o', ls='none', markerfacecolor='white', markeredgecolor='#666', markersize=5.5, label='unweighted rate')]
    a2.legend(handles=kind, loc='upper left', handletextpad=.3, borderpad=.3)

    models = [plt.Line2D([], [], marker='o', ls='none', color=COL[r['model']], markersize=6, label=NAME[r['model']]) for r in rows]
    fig.legend(handles=models, loc='lower center', ncol=len(models), handletextpad=.3,
               columnspacing=1.4, bbox_to_anchor=(.5, 0))
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(f'{OUT}/fig_baselines.pdf', bbox_inches='tight'); plt.close(fig)
    print('wrote fig_baselines.pdf')


def fig_overview():
    """Figure 4: composition of the released layers."""
    import collections
    items = [json.loads(l) for l in open(f'{DEP}/data/items.jsonl')]
    ans = [x for x in items if x['task_type'] == 'answerable']
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.4)); ax = axs.ravel()
    grey, ink = '#8a8a8a', '#3a3a3a'

    def hbar(a, labels, values, color=grey):
        y = range(len(labels))[::-1]
        a.barh(list(y), values, color=color, height=.7)
        a.set_yticks(list(y)); a.set_yticklabels(labels)
        for yy, v in zip(y, values):
            a.text(v + max(values) * .02, yy, str(v), va='center', fontsize=7, color=ink)
        a.set_xlim(0, max(values) * 1.22)

    # (a) refusal reasons
    names = {'blood_loss_volume': 'blood-loss volume', 'elapsed_time': 'elapsed time', 'vital_signs': 'vital signs',
             'step_outcome_not_shown': 'outcome not shown', 'medication_name_or_dose': 'drug name or dose', 'other': 'other',
             'off_frame_anatomy': 'off-frame anatomy', 'patient_history': 'patient history'}
    c = collections.Counter(x['refusal_reason'] for x in items if x['task_type'] == 'refusal').most_common()
    hbar(ax[0], [names[k] for k, _ in c], [v for _, v in c])
    ax[0].set_title('(a) Refusal questions: information missing', loc='left')

    # (b) answer source by question type
    src_names = [('atp_4_02_11_casualty_response', 'ATP 4-02.11', '#2F5E23'), ('tccc_guidelines_2024_01_25', 'CoTCCC Guidelines 2024', '#7DA65B'),
                 ('tccc_guidelines_medical_personnel', 'CoTCCC Guidelines (medical)', '#b9d3a2'),
                 ('jts_cpgs_a_h', 'JTS CPG', '#2E75B6'), ('jts_top10_operational', 'JTS CPG', '#2E75B6'), ('jts_cpgs_i_z', 'JTS CPG', '#2E75B6')]
    groups = [('doctrine', 'doctrine_scene'), ('reasoning', 'reasoning'), ('how', 'how')]
    left = [0.0] * len(groups); seen = set()
    for sid, label, col in src_names:
        vals = []
        for _, t in groups:
            qs = [q for x in ans for q in x['questions'] if q['type'] == t and q.get('provenance')]
            vals.append(100 * sum(q['provenance']['source_id'] == sid for q in qs) / len(qs))
        ax[1].barh(range(len(groups))[::-1], vals, left=left, color=col, height=.6, label=label if label not in seen else None)
        seen.add(label); left = [l + v for l, v in zip(left, vals)]
    ax[1].set_yticks(range(len(groups))[::-1]); ax[1].set_yticklabels([g for g, _ in groups])
    ax[1].set_xlim(0, 100); ax[1].set_xlabel('% of answers')
    ax[1].legend(loc='lower center', bbox_to_anchor=(.5, -.62), ncol=2, handlelength=1, columnspacing=1)
    ax[1].set_title('(b) Cited source, by question type', loc='left')

    # (c) equipment classes
    c = collections.Counter(b['label'] for x in ans for b in x['detection']).most_common(9)
    hbar(ax[2], [{'iv_io_catheter': 'IV/IO catheter'}.get(k, k.replace('_', ' ')) for k, _ in c], [v for _, v in c])
    ax[2].set_title('(c) Equipment boxes, nine most frequent classes', loc='left')

    # (d) body region
    names = {'lower_extremity': 'lower extremity', 'upper_extremity': 'upper extremity', 'chest': 'chest', 'head_face': 'head or face',
             'pelvis_groin': 'pelvis or groin', 'neck': 'neck', 'abdomen': 'abdomen', 'back': 'back', 'multiple_generic': 'multiple',
             'unclear': 'unclear'}
    c = collections.Counter(x['observed_anatomy']['region'] for x in ans).most_common()
    hbar(ax[3], [names[k] for k, _ in c], [v for _, v in c])
    ax[3].set_title('(d) Body region of the intervention', loc='left')
    fig.tight_layout(w_pad=2.0, h_pad=1.0)
    fig.savefig(f'{OUT}/fig_overview.pdf', bbox_inches='tight'); plt.close(fig)
    print('wrote fig_overview.pdf')


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_baselines(board()); fig_overview()


if __name__ == '__main__':
    main()
