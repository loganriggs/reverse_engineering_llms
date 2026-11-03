"""Plot replicated regional ablations and opener-edge rescue for the V10 model."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).parent / 'results')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    prefix = 'v10_linked_depth2_seed1_'
    region = [json.loads((args.results_dir / f'{prefix}region_mediation{s}.json').read_text())
              for s in (99007, 99008)]
    rescue = [json.loads((args.results_dir / f'{prefix}opener_rescue{s}.json').read_text())
              for s in (99009, 99010)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout='constrained')
    x = np.arange(3)
    labels = ['All before opener', 'Current clause', 'Earlier clauses']
    native_keys = ['pre_all', 'current_clause', 'prior_clauses']
    control_keys = [k + '_orthogonal' for k in native_keys]
    for i, row in enumerate(region):
        offset = (i - 0.5) * .35
        axes[0].bar(x + offset - .085,
                    [row['summary'][k]['delta_kl'] for k in native_keys],
                    .17, color='#a33a2b', alpha=.72 + i*.28,
                    label='Native edge sum' if i == 0 else None)
        axes[0].bar(x + offset + .085,
                    [row['summary'][k]['delta_kl'] for k in control_keys],
                    .17, color='#647a9e', alpha=.72 + i*.28,
                    label='Equal residual norm, random direction' if i == 0 else None)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel('Bracket oracle KL increase (nats/site)')
    axes[0].set_title('Remove L0H3 edges at closer query\nTwo independent 128-event panels')
    axes[0].legend(frameon=False, fontsize=8, loc='upper right')
    axes[0].grid(axis='y', alpha=.18)

    modes = ['rescue_h0', 'rescue_h1', 'rescue_both']
    labels = ['Restore L1H0\nopener edge', 'Restore L1H1\nopener edge',
              'Restore both']
    for i, row in enumerate(rescue):
        axes[1].bar(x + (i-.5)*.33,
                    [row['rescue_fraction'][m] for m in modes],
                    .31, color=['#668cb3', '#315d8e'][i],
                    label=f'Fresh panel {i+1}')
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Fraction of induced KL damage rescued')
    axes[1].set_title('Restore baseline opener-edge vectors\nafter L0H3 pre-opener removal')
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis='y', alpha=.18)
    for ax in axes:
        ax.set_axisbelow(True)
    output = args.output or args.results_dir / f'{prefix}mediation.png'
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
