"""Scientific scatter of V8 bond-basis causal fidelity versus token loss."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


STYLES = {
    'weight_only': ('#1b9e77', 'Weight-only'),
    'data_no_source_sites': ('#d95f02', 'Unlabeled, no source sites'),
    'data_all_sites': ('#e6ab02', 'Unlabeled, all sites'),
    'source_edit_supervised': ('#7570b3', 'Supervised edit'),
    'hybrid_weight_data': ('#1f78b4', 'Weight/data hybrid'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for axis, path in zip(axes, args.inputs, strict=True):
        data = json.loads(path.read_text())
        seed = Path(data['config']['checkpoint']).parent.name.split('seed')[-1]
        panel = data['results']['syntax1']
        native = panel['native']
        axis.scatter(native['evaluation']['token_nll'],
                     native['edits']['source_immediate'][
                         'mean_model_response'],
                     marker='*', s=200, color='black', label='Native', zorder=5)
        for method, (color, label) in STYLES.items():
            rows = sorted((row for row in panel['rows']
                           if row['basis'] == method and
                           row['rank'] in (16, 32)),
                          key=lambda row: row['rank'])
            axis.plot([row['token_nll'] for row in rows],
                      [row['source_edit']['mean_model_response']
                       for row in rows],
                      color=color, marker='o', linewidth=1.5,
                      label=label)
            for row in rows:
                axis.annotate(str(row['rank']),
                              (row['token_nll'], row['source_edit'][
                                  'mean_model_response']),
                              xytext=(4, -5), textcoords='offset points',
                              fontsize=8)
        axis.set_title(f'Model seed {seed}')
        axis.set_xlabel('Held-out token NLL (lower is better)')
        axis.set_xlim(.7, 3.7)
        axis.set_ylim(-.03, 1.02)
        axis.grid(alpha=.2)
    axes[0].set_ylabel('Immediate-source edit response (higher is better)')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(.5, -.03))
    fig.tight_layout(rect=(0, .12, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches='tight')
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
