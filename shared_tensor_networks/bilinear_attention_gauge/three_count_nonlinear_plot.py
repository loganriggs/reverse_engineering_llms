"""Plot ordinary versus rare parity accuracy for the nonlinear toy sweep."""

import json

import matplotlib.pyplot as plt
import numpy as np

from dual_joint_function_canonical import HERE


SOURCE='three_count_nonlinear_metric_s0-1-2-3-4-5-6-7-8-9.json'
CHOSEN={
    'uniform_table':'Uniform table',
    'empirical_table':'Empirical table',
    'context_nonlinear_weight_0':'Natural logits',
    'context_nonlinear_weight_0.05':'Natural + .05 parity',
    'context_nonlinear_weight_0.1':'Natural + .1 parity',
    'context_nonlinear_weight_1_uniform_weight_20':'Natural + parity + 20 uniform',
    'context_nonlinear_weight_1_uniform_weight_100':'Natural + parity + 100 uniform',
    'uniform_table_rank3':'Uniform rank 3',
}


def run():
    data=json.loads((HERE/SOURCE).read_text())
    rows=data['rows']
    fig,ax=plt.subplots(figsize=(8.4,5.4))
    colors=plt.cm.tab10(np.linspace(0,1,len(CHOSEN)))
    for (name,label),color in zip(CHOSEN.items(),colors):
        x=np.array([r['scores'][name]['all']['parity_accuracy'] for r in rows])
        y=np.array([r['scores'][name]['rare_heldout']['parity_accuracy'] for r in rows])
        ax.scatter(x,y,s=18,alpha=.30,color=color)
        ax.scatter(x.mean(),y.mean(),s=110,color=color,edgecolor='black',
                   linewidth=.6,label=label,zorder=5)
    ax.set(xlabel='Full-test parity accuracy',
           ylabel='Withheld rare-combination parity accuracy',
           xlim=(.48,1.025),ylim=(-.04,1.045),
           title='Rank-two attention reductions: downstream fidelity versus rare transfer')
    ax.grid(alpha=.25)
    ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False)
    fig.tight_layout()
    path=HERE/'three_count_nonlinear_metric.png'
    fig.savefig(path,dpi=180,bbox_inches='tight')
    print(path)


if __name__=='__main__':run()
