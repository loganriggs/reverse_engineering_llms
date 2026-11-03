"""Task loss and exact-edit gate for matched V12 training arms."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_arm(folder):
    training=json.loads((folder/'results.json').read_text())['records']
    panels=[json.loads((folder/f'fresh_panel_{seed}.json').read_text())
            ['panels']['reference_candidate'] for seed in (99701,99711)]
    return training,panels


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ordinary',type=Path,required=True)
    p.add_argument('--weighted',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    arms={'Ordinary CE':load_arm(a.ordinary)}
    if a.weighted:
        arms['100× answer CE']=load_arm(a.weighted)
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    colors={'Ordinary CE':'#2b6cb0','100× answer CE':'#d55e00'}
    for name,(records,panels) in arms.items():
        steps=[row['step'] for row in records]
        answer=[row['scores']['price_answer']['ce'] for row in records]
        axes[0].plot(steps,answer,marker='o',label=name,color=colors[name])
    axes[0].axhline(math.log(6),ls='--',color='gray',label='Six-price guess')
    axes[0].axhline(.8,ls=':',color='black',label='Circuit gate')
    axes[0].set(xlabel='Training updates',ylabel='Diagnostic answer CE (nats)',
                title='Fixed enriched-stream diagnostic')
    axes[0].legend(fontsize=8,frameon=False)
    axes[0].grid(alpha=.2)

    centers=np.arange(2)
    width=.36 if len(arms)>1 else .5
    for j,(name,(_,panels)) in enumerate(arms.items()):
        offset=(j-(len(arms)-1)/2)*width
        ce=[v['per_rule']['price_answer']['ce'] for v in panels]
        axes[1].bar(centers+offset,ce,width,label=name,color=colors[name])
    axes[1].axhline(math.log(6),ls='--',color='gray')
    observed=[]
    for _,(_,panels) in arms.items():
        observed.extend(v['price_answer_baselines'].get(
            'uniform_observed_price_set_ce') for v in panels)
    observed=[v for v in observed if v is not None]
    if observed:
        guess=float(np.mean(observed))
        axes[1].axhline(guess,ls='-.',color='#197a5b')
        axes[1].text(1.95,guess+.025,'Visible-price-set guess',
                     ha='right',va='bottom',fontsize=8,color='#197a5b')
    axes[1].axhline(.8,ls=':',color='black')
    axes[1].set(xticks=centers,xticklabels=['Panel A','Panel B'],
                ylabel='Answer CE (nats)',title='Untouched reference-rate streams')
    axes[1].grid(axis='y',alpha=.2)

    labels=['Fact price','Query item']
    xs=np.arange(2)
    for j,(name,(_,panels)) in enumerate(arms.items()):
        values=[np.mean([v['edits'][kind]['mean_response'] for v in panels])
                for kind in ('fact_edit','key_edit')]
        offset=(j-(len(arms)-1)/2)*width
        bars=axes[2].bar(xs+offset,values,width,label=name,color=colors[name])
        for bar,value in zip(bars,values):
            axes[2].annotate(f'{value:+.2g}',
                (bar.get_x()+bar.get_width()/2,max(value,0)),
                xytext=(0,4),textcoords='offset points',
                ha='center',fontsize=8)
    axes[2].axhline(1,ls='--',color='gray',label='Exact oracle')
    axes[2].axhline(.5,ls=':',color='black',label='Circuit gate')
    axes[2].set(xticks=xs,xticklabels=labels,
                ylabel='New-target probability change',
                title='Supported edits, mean across two panels')
    axes[2].grid(axis='y',alpha=.2)
    fig.suptitle('V12 item–price binding: prediction and causal response',fontsize=13)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=180)
    plt.close(fig)
    print('saved',a.output)


if __name__=='__main__':main()
