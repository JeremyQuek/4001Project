"""Extra figure for the written report (VLM accuracy by predicate and by variant).

Reads outputs/results.json only; nothing is refitted or rescored.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import CANONICAL, OUTPUTS, VARIANTS, read_json

VARIANT_LABELS = {'orig': 'original (true)', 'converse': 'converse (true)',
                  'swap': 'swapped (false)', 'opposite': 'opposite (false)'}


def main():
    results = read_json(OUTPUTS / 'results.json')
    overall = results['cohorts']['all']['unfiltered_vlm_accuracy']
    plt.rcParams.update({'font.size': 9})
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6),
                                gridspec_kw={'width_ratios': [8, 4]})
    panels = (('canonical_predicate', CANONICAL, {name: name for name in CANONICAL}, 'By spatial predicate'),
              ('variant', VARIANTS, VARIANT_LABELS, 'By statement variant'))
    for axis, (key, order, labels, title) in zip(axes, panels):
        groups = results['breakdowns'][key]
        values = [groups[name]['vlm_accuracy'] for name in order]
        positions = range(len(order))
        axis.bar(positions, values, width=0.6, color='tab:blue')
        for position, name, value in zip(positions, order, values):
            axis.annotate(f'{value:.3f}', (position, value), xytext=(0, 3), textcoords='offset points',
                          ha='center', fontsize=8, color='dimgray',
                          bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.6})
        axis.axhline(overall, color='dimgray', linewidth=0.8, linestyle='--', zorder=0)
        axis.axhline(0.5, color='dimgray', linewidth=0.8, zorder=0)
        axis.set_xticks(list(positions), [labels[name] for name in order], rotation=20, ha='right')
        axis.set_ylim(0.4, 0.9)
        axis.set_title(title, loc='left', fontsize=10, fontweight='bold')
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=0.3)
        axis.set_axisbelow(True)
    axes[0].set_ylabel('VLM accuracy on test')
    # Reference-line labels sit in the right margin so they cannot cover a bar.
    for value, label in ((overall, f'all test\nquestions:\n{overall:.3f}'), (0.5, 'guessing:\n0.5')):
        axes[1].annotate(label, (1.02, value), xycoords=('axes fraction', 'data'), va='center', fontsize=8,
                         color='dimgray')
    figure.tight_layout()
    figure.savefig(OUTPUTS / 'figure_vlm_accuracy.png', dpi=200)
    plt.close(figure)
    print('Wrote outputs/figure_vlm_accuracy.png')


if __name__ == '__main__':
    main()
