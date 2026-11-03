"""Plot replicated causal interventions on the trained V10 bracket circuit."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,
                   default=Path('language_task_suite/results'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    heads=[json.loads((a.results/f'v10_linked_depth2_seed1_head_ablation{s}.json').read_text())
           for s in (99003,99004)]
    edges=[json.loads((a.results/f'v10_linked_depth2_seed1_opener_edges{s}.json').read_text())
           for s in (99005,99006)]
    head_labels=[f"L{r['layer']}H{r['head']}" for r in heads[0]['rows']]
    head_vals=np.array([[r['delta_kl']['bracket_close'] for r in d['rows']]
                        for d in heads])
    keys=[(1,0,'opener'),(1,0,'last_content'),
          (1,1,'opener'),(1,1,'last_content'),
          (0,3,'zero_query_write'),(0,3,'zero_opener_write'),
          (0,3,'pre_opener'),(0,3,'span_after_opener'),(0,3,'self')]
    edge_labels=['L1H0 opener edge','L1H0 local edge',
                 'L1H1 opener edge','L1H1 local edge',
                 'L0H3 query write','L0H3 opener write',
                 'L0H3 pre-opener edges','L0H3 bracket-span edges',
                 'L0H3 self edge']
    edge_vals=[]
    for d in edges:
        lookup={(r['layer'],r['head'],r['key']):r['delta_kl'] for r in d['rows']}
        edge_vals.append([lookup[k] for k in keys])
    edge_vals=np.asarray(edge_vals)
    fig,(ax0,ax1)=plt.subplots(1,2,figsize=(13,6.2),layout='constrained')
    xx=np.arange(len(head_labels))
    for i,seed in enumerate((99003,99004)):
        ax0.bar(xx+(i-.5)*.36,head_vals[i],.34,label=f'panel {seed}')
    ax0.set_xticks(xx,head_labels,rotation=50)
    ax0.set_ylabel('Increase in bracket exact-oracle KL (nats)')
    ax0.set_title('Remove one whole head')
    ax0.grid(axis='y',alpha=.2)
    ax0.legend(frameon=False,fontsize=8)
    yy=np.arange(len(keys))
    for i,seed in enumerate((99005,99006)):
        ax1.barh(yy+(i-.5)*.36,edge_vals[i],.34,label=f'panel {seed}')
    ax1.set_yticks(yy,edge_labels)
    ax1.invert_yaxis()
    ax1.set_xlabel('Increase in bracket exact-oracle KL (nats)')
    ax1.set_title('Remove selected edge or position write')
    ax1.grid(axis='x',alpha=.2)
    ax1.legend(frameon=False,fontsize=8)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=180)
    plt.close(fig)
    print('saved',a.output)


if __name__=='__main__':main()
