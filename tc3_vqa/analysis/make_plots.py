# Baseline figure of the paper: recognition against abstention with point area proportional to RWHR, and the
# coverage-risk plane. Reads the leaderboard written by score_eval.py.
from tc3_vqa.paths import FIGURES, WORK
import json, math, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Q = WORK
OUT = FIGURES
NAME = {'qwen2vl7b': 'Qwen2-VL-7B', 'qwen25vl7b': 'Qwen2.5-VL-7B', 'phi35v': 'Phi-3.5-Vision',
        'internvl3_8b': 'InternVL3-8B', 'internvl3_38b': 'InternVL3-38B'}
COL = {'qwen2vl7b': '#2E75B6', 'qwen25vl7b': '#7FB3E0', 'phi35v': '#C00000',
       'internvl3_8b': '#7DA65B', 'internvl3_38b': '#2F5E23'}
plt.rcParams.update({'font.size': 8, 'axes.linewidth': .7, 'xtick.major.width': .7,
                     'ytick.major.width': .7, 'legend.frameon': False})


def board():
    return json.load(open(f'{Q}/leaderboard.json'))


def fig_baselines(rows):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.35))
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
                    xytext=(dx, dy), ha=ha, va=va, fontsize=6.4, zorder=4)
    a1.set_xlabel('recognition accuracy (Wilson 95% interval)')
    a1.set_ylabel('refusal accuracy (correct abstention)')
    a1.set_xlim(.55, .95); a1.set_ylim(.05, 1.12)
    a1.set_xticks([.6, .7, .8, .9])
    a1.grid(alpha=.22, lw=.5)
    a1.set_title('(a) the two axes are not aligned', fontsize=8.5, loc='left')

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
    a2.set_title('(b) severity weighting widens the spread', fontsize=8.5, loc='left')
    kind = [plt.Line2D([], [], marker='o', ls='none', color='#666', markeredgecolor='white', markersize=6.5, label='RWHR (severity-weighted)'),
            plt.Line2D([], [], marker='o', ls='none', markerfacecolor='white', markeredgecolor='#666', markersize=5.5, label='unweighted rate')]
    a2.legend(handles=kind, fontsize=6.3, loc='upper left', handletextpad=.3, borderpad=.3)

    models = [plt.Line2D([], [], marker='o', ls='none', color=COL[r['model']], markersize=6, label=NAME[r['model']]) for r in rows]
    fig.legend(handles=models, loc='lower center', ncol=len(models), fontsize=7, handletextpad=.3,
               columnspacing=1.4, bbox_to_anchor=(.5, 0))
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(f'{OUT}/fig_baselines.pdf', bbox_inches='tight'); plt.close(fig)
    print('wrote fig_baselines.pdf')


def fig_funnel():
    # (stage label, unit, kept, dropped, what the filter removes)
    stages = [('collected video', 'clips', 407, 8, 'decode failures'),
              ('informative windows', 'windows', 8012, 9043, 'no action phase'),
              ('answerable candidates', 'items', 5051, 2961, 'routed to refusal'),
              ('after per-video cap', 'items', 1515, 3536, 'source diversity'),
              ('audit usable', 'items', 806, 709, 'concept not visible'),
              ('after refinement and source rights', 'items', 543, 263, 'refinement, source rights'),
              ('released answerable', 'items', 431, 112, 'six-model consensus gate')]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ys = list(range(len(stages)))[::-1]
    XMAX = 4.6
    for y, (lab, unit, kept, dropped, why) in zip(ys, stages):
        lk, ld = math.log10(kept), math.log10(kept + dropped)
        ax.barh(y, ld, height=.6, color='#F4F4F4', edgecolor='#CCCCCC', lw=.6, zorder=1)
        ax.barh(y, lk, height=.6, color='#DEEAF6', edgecolor='#2E75B6', lw=.9, zorder=2)
        ax.text(-.08, y, lab, ha='right', va='center', fontsize=7.4)
        ax.text(lk - .07, y, f'{kept:,}', ha='right', va='center', fontsize=7.2,
                color='#1F3864', zorder=4)
        ax.text(XMAX + .05, y, f'{100 * kept / (kept + dropped):.0f}% kept', ha='left', va='center',
                fontsize=7, color='#1F3864')
        ax.text(XMAX + 1.02, y, f'{dropped:,} {why}', ha='left', va='center', fontsize=6.8, color='#888')
    ax.set_xlim(0, XMAX + 3.1); ax.set_ylim(-.6, len(stages) - .4)
    ax.set_yticks([]); ax.set_xticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title('bar length is $\\log_{10}$ of the count; the unit changes between stages',
                 fontsize=7, color='#555', loc='left')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_funnel.pdf', bbox_inches='tight'); plt.close(fig)
    print('wrote fig_funnel.pdf')


def main():
    rows = board()
    os.makedirs(OUT, exist_ok=True)
    fig_baselines(rows); fig_funnel()


if __name__ == '__main__':
    main()
