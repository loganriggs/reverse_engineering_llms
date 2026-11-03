"""Summarize replicated causal edge ablations in the shared v3 model."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def pair_mean_range(rows):
    values=np.asarray(rows,dtype=float)
    return values.mean(0),np.abs(values-values.mean(0)).max(0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results-dir',type=Path,
                   default=Path(__file__).parent/'results')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    edge=[json.loads((a.results_dir/f'v3_shared_bracket_edge_ablation_seed{seed}.json').read_text())
          for seed in (98001,98011)]
    opener=[json.loads((a.results_dir/f'v3_shared_bracket_opener_edge_seed{seed}.json').read_text())
            for seed in (98101,98111)]
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    colors={'dense':'#365c9a','sparse60':'#e28b30'}
    for col,(rule,title) in enumerate((('skip_bigram','Place…adjective skip-bigram'),
                                        ('skip_trigram','Opener…content bracket close'))):
        ax=axes[0,col]
        for offset,checkpoint in enumerate(('dense','sparse60')):
            vals=[[(d['results'][checkpoint][rule][f'H{h}_anchor']['oracle_kl_change'])
                   for h in range(4)] for d in edge]
            mean,err=pair_mean_range(vals)
            xx=np.arange(4)+(-.18 if offset==0 else .18)
            ax.bar(xx,mean,.35,yerr=err,capsize=3,color=colors[checkpoint],
                   label='Dense' if checkpoint=='dense' else '60% zero')
        ax.set_xticks(range(4),[f'H{h}' for h in range(4)])
        ax.set_ylabel('Δ full-oracle KL (nats)')
        ax.set_title(title)
        ax.axhline(0,color='black',linewidth=.7)
        ax.grid(axis='y',alpha=.2)
        if col==0:ax.legend(frameon=False)
    labels=['Native','Remove H0 opener','Remove H1 opener','Remove H0 next word']
    for col,checkpoint in enumerate(('dense','sparse60')):
        ax=axes[1,col]
        vals=[]
        for d in opener:
            r=d['results'][checkpoint]
            vals.append([r['H0_opener']['native']['response'],
                         r['H0_opener']['ablated']['response'],
                         r['H1_opener']['ablated']['response'],
                         r['H0_next_token']['ablated']['response']])
        mean,err=pair_mean_range(vals)
        ax.bar(range(4),mean,yerr=err,capsize=3,
               color=[colors[checkpoint],'#b64e4e','#8363a7','#839797'])
        oracle=np.mean([d['results'][checkpoint]['H0_opener']['native']['oracle_response']
                        for d in opener])
        ax.axhline(oracle,color='black',linestyle='--',linewidth=1,
                   label='Exact oracle')
        ax.set_xticks(range(4),labels,rotation=25,ha='right')
        ax.set_ylabel('P(new closer) after−before opener swap')
        ax.set_title(('Dense' if checkpoint=='dense' else '60% zero')+
                     ' model: supported opener edit')
        ax.set_ylim(0,.75)
        ax.grid(axis='y',alpha=.2)
        if col==0:ax.legend(frameon=False)
    fig.suptitle('Sparse rule paths survive pruning and carry causal rule responses',fontsize=14)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
