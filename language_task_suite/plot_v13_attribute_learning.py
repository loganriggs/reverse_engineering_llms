"""Plot V13 learning and held-out causal effects without hiding failed gates."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--panel-a',type=Path,required=True)
    p.add_argument('--panel-b',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    train=json.loads(args.training.read_text())
    panels=[json.loads(v.read_text())['panels']['reference']
            for v in (args.panel_a,args.panel_b)]
    records=train['records']
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    steps=[r['step'] for r in records]
    for label,color in (('attribute_adjective','#2f6b9a'),
                        ('attribute_object','#c36c32'),
                        ('mixed_verb','#4f9361')):
        axes[0].plot(steps,[r['scores'].get(label,{}).get('ce',np.nan)
                            for r in records],'-o',label=label.replace('_',' '),
                     color=color)
        if label!='mixed_verb':
            entropy=np.mean([v['per_rule'][label]['oracle_entropy']
                             for v in panels])
            axes[0].axhline(entropy,ls=':',lw=1.5,color=color,
                            label=f'{label.replace("_"," ")} oracle entropy')
    axes[0].set(xlabel='Training update',ylabel='Cross-entropy (nats)',
                title='Reference-stream learning')
    axes[0].legend(fontsize=7)
    labels=('attribute_adjective','attribute_object')
    centers=np.arange(2)
    width=.23
    for i,panel in enumerate(panels):
        values=[panel['edits'][k]['mean_model_response'] for k in labels]
        locations=centers+(i-.5)*width
        axes[1].bar(locations,values,width=width,
                    label=f'fresh panel {i+1}')
        for x,value in zip(locations,values):
            axes[1].text(x,.035 if value>=0 else -.035,
                         f'{value:+.4f}',ha='center',
                         va='center',rotation=90,fontsize=8)
    axes[1].scatter(centers,
                    [np.mean([v['edits'][k]['mean_oracle_response']
                              for v in panels]) for k in labels],
                    marker='D',s=42,c='black',label='exact oracle')
    axes[1].axhline(0,color='gray',lw=.7)
    axes[1].set_xticks(centers,['Earlier fact →\nnew adjective',
                               'Earlier fact →\noriginal item'])
    axes[1].set(ylabel='Target-probability change',
                title='Supported earlier-fact edits')
    axes[1].legend(fontsize=8)
    fig.suptitle('V13 attribute binding: prediction and causal response')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180)
    plt.close(fig)
    print('saved',args.output)


if __name__=='__main__':main()
