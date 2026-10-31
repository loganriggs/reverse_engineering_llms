"""Plot semantic key alignment versus edit sparsity for exact gauges."""

import json

import matplotlib.pyplot as plt
import numpy as np

from dual_joint_function_canonical import HERE


SOURCE='three_count_nonlinear_gauge.json'
REDUCTIONS={'uniform_table':'Uniform reduction',
            'empirical_table':'Frequency reduction',
            'context_nonlinear_weight_0.1':'Downstream reduction'}
GAUGES={'svd':('SVD','o'),'key_l1':('Key L1','s'),
        'context_edit_sparsity':('Context-edit','^')}


def run():
    rows=json.loads((HERE/SOURCE).read_text())['rows']
    fig,ax=plt.subplots(figsize=(8.2,5.1))
    colors=plt.cm.tab10(np.linspace(0,.4,len(REDUCTIONS)))
    for (reduction,rlabel),color in zip(REDUCTIONS.items(),colors):
        selected=[r for r in rows if r['reduction']==reduction]
        for gauge,(glabel,marker) in GAUGES.items():
            x=np.array([r['gauges'][gauge]['postfit_key_alignment'][
                'mean_distinct_family_cosine'] for r in selected])
            y=np.array([r['gauges'][gauge]['panels']['test'][
                'energy_weighted_effective_axes'] for r in selected])
            ax.scatter(x,y,s=16,color=color,alpha=.22,marker=marker)
            ax.scatter(x.mean(),y.mean(),s=110,color=color,marker=marker,
                       edgecolor='black',linewidth=.7,
                       label=f'{rlabel}: {glabel}',zorder=5)
    ax.set(xlabel='Post-fit alignment with distinct A/B/C key variables (higher is better)',
           ylabel='Held-out effective axes under key edits (lower is sparser)',
           xlim=(.57,1.025),ylim=(1.15,2.07),
           title='Exact gauges trade variable recovery for sparse deletion effects')
    ax.grid(alpha=.25)
    ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False)
    fig.tight_layout()
    path=HERE/'three_count_nonlinear_gauge.png'
    fig.savefig(path,dpi=180,bbox_inches='tight')
    print(path)


if __name__=='__main__':run()
