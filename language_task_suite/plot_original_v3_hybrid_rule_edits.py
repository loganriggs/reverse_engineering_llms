"""Plot supported-rule edit fidelity of rank-16 joint-v3 bond bases."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).parent/'results'/'original_v3_sparse_rule_graphs'
    seeds={'60%-zero':(96101,96201),'Dense':(96301,96401)}
    methods=('weight_only','all','hybrid_0.1','hybrid_1')
    names=('Weight-only','Activation PCA','Hybrid α=.1','Hybrid α=1')
    colors=('#7385aa','#3b8d7a','#bd8b42','#814a94')
    kinds=('noun_switch','skip_switch','bracket_content_control')
    labels=('Noun switch','Skip-rule switch','Bracket-content\ncontrol')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True,layout='constrained')
    for ax,(checkpoint,split_seeds) in zip(axes,seeds.items()):
        runs=[json.loads((root/f'hybrid_edits_{"sparse" if checkpoint=="60%-zero" else "dense"}_seed{s}_{s+1}.json').read_text())
              for s in split_seeds]
        for i,method in enumerate(methods):
            values=np.asarray([[run['scores'][method][kind]['native_effect_l1']
                                for kind in kinds] for run in runs])
            positions=np.arange(3)+(i-1.5)*.19
            ax.bar(positions,values.mean(0),.18,color=colors[i],
                   label=names[i] if checkpoint=='60%-zero' else None)
            for row in values:
                ax.scatter(positions,row,s=10,color='#202020',zorder=3)
        ax.set_xticks(range(3),labels)
        ax.set_title(f'{checkpoint} model, layer-1 rank 16/64')
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('L1 error in edit response vs native model')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',
               bbox_to_anchor=(.5,1.13),ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Supported edits on two fresh panels per checkpoint',y=1.23)
    path=root/'hybrid_rule_edits_rank16.png'
    fig.savefig(path,dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(path)


if __name__=='__main__':main()
