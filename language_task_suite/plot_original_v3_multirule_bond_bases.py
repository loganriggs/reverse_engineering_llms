"""Plot rule-specific loss after matched-rank v3 attention-bond projections."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


RULES = ('noun_bigram', 'skip_bigram', 'skip_trigram')
NAMES = ('Noun bigram', 'Skip-bigram', 'Bracket close')
METHODS = ('weight_only', 'all', 'leave_rule_out', 'random_match')
METHOD_NAMES = ('Weight-only', 'All unlabeled data', 'Exclude labeled target sites',
                'Remove same number at random')
COLORS = ('#7285ab', '#3b8d7a', '#b75043', '#a9a9a9')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=Path,
                   default=Path(__file__).parent/'results'/'original_v3_sparse_rule_graphs')
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--checkpoint-kind', choices=('sparse','dense'), default='sparse')
    p.add_argument('--output', type=Path, default=None)
    a = p.parse_args()
    seeds = (95101,95201) if a.checkpoint_kind=='sparse' else (95301,95401)
    infix = '' if a.checkpoint_kind=='sparse' else 'dense_'
    files = [a.results_dir/f'multirule_bond_{infix}seed{s}_{s+1}.json'
             for s in seeds]
    runs = [json.loads(path.read_text()) for path in files]
    fig, axes = plt.subplots(1,2,figsize=(11.4,4.5),sharey=True,layout='constrained')
    for layer,ax in enumerate(axes):
        for m,method in enumerate(METHODS):
            centers = np.arange(3) + (m-1.5)*.19
            vals = []
            for run in runs:
                one=[]
                for rule in RULES:
                    name = ('without_'+rule if method=='leave_rule_out' else
                            'random_without_'+rule if method=='random_match' else method)
                    row = next(r for r in run['rows'] if r['layer']==layer
                               and r['rank']==a.rank and r['basis']==name)
                    one.append(row['rule_scores'][rule]['teacher_kl'])
                vals.append(one)
            vals = np.asarray(vals)
            ax.bar(centers, vals.mean(0), .18, color=COLORS[m],
                   label=METHOD_NAMES[m] if layer==0 else None)
            for seed_values in vals:
                ax.scatter(centers, seed_values, color='#202020',s=10,zorder=3)
        ax.set_xticks(range(3),NAMES)
        ax.set_title(f'Layer {layer} bond, rank {a.rank}/64')
        ax.set_ylabel('KL from native sparse model (nats/site)' if layer==0 else '')
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=4,
               frameon=False,bbox_to_anchor=(.5,1.12),fontsize=8)
    fig.suptitle(f'Original jointly sampled v3, {a.checkpoint_kind} checkpoint: '
                 'two disjoint splits',
                 y=1.22)
    output = a.output or a.results_dir/(
        f'multirule_bond_{a.checkpoint_kind}_rank{a.rank}.png')
    fig.savefig(output,dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
