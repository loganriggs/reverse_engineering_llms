"""Show untouched third-split rank-16 weight/data hybrid comparison."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).parent/'results'/'original_v3_sparse_rule_graphs'
    files={'60%-zero':root/'multirule_bond_seed95501_95502.json',
           'Dense':root/'multirule_bond_dense_seed95601_95602.json'}
    methods=('weight_only','all','hybrid_0.1','hybrid_1')
    names=('Weight-only','Activation PCA','Hybrid α=.1','Hybrid α=1')
    colors=('#7385aa','#3b8d7a','#bd8b42','#814a94')
    rules=('noun_bigram','skip_bigram','skip_trigram')
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.5),sharey=True,layout='constrained')
    for ax,(checkpoint,path) in zip(axes,files.items()):
        data=json.loads(path.read_text())
        for i,method in enumerate(methods):
            row=next(r for r in data['rows'] if r['layer']==1
                     and r['rank']==16 and r['basis']==method)
            values=[row['rule_scores'][rule]['teacher_kl'] for rule in rules]
            ax.bar(np.arange(3)+(i-1.5)*.19,values,.18,color=colors[i],
                   label=names[i] if checkpoint=='60%-zero' else None)
        ax.set_xticks(range(3),('Noun bigram','Skip-bigram','Bracket close'))
        ax.set_title(f'{checkpoint} model, L1 bond rank 16/64')
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('KL from native model (nats/site)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',
               bbox_to_anchor=(.5,1.14),ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Weight/data hybrid on untouched third fit/evaluation splits',
                 y=1.24)
    output=root/'multirule_bond_hybrid_holdout.png'
    fig.savefig(output,dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(output)


if __name__=='__main__':main()
