"""Show local conditional competence beside cross-sentence binding fidelity."""

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
    panels=[json.loads(path.read_text())['panels']['reference']
            for path in (args.panel_a,args.panel_b)]
    records=train['records']
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    steps=[row['step'] for row in records]
    for kind,color in (('conditional_action','#366da0'),
                       ('attribute_object','#c46b36'),
                       ('mixed_verb','#53915a')):
        values=[row['scores'].get(kind,{}).get('ce',np.nan)
                for row in records]
        axes[0].plot(steps,values,'-o',color=color,
                     label=kind.replace('_',' '))
        if kind!='mixed_verb':
            h=np.mean([panel['per_rule'][kind]['oracle_entropy']
                       for panel in panels])
            axes[0].axhline(h,color=color,ls=':',lw=1.4,
                            label=kind.replace('_',' ')+' oracle')
    axes[0].set(xlabel='Training update',ylabel='Cross-entropy (nats)',
                title='Reference-stream learning')
    axes[0].legend(fontsize=7)
    kinds=('conditional_action','attribute_object')
    labels=('Local condition → new action',
            'Earlier fact → original item')
    centers=np.arange(2)
    width=.25
    for i,panel in enumerate(panels):
        values=[panel['edits'][kind]['mean_model_response'] for kind in kinds]
        locations=centers+(i-.5)*width
        axes[1].bar(locations,values,width=width,label=f'fresh panel {i+1}')
        for x,value in zip(locations,values):
            axes[1].text(x,value+(.02 if value>=0 else -.02),
                         f'{value:+.3f}',ha='center',
                         va='bottom' if value>=0 else 'top',fontsize=8)
    oracles=[np.mean([panel['edits'][kind]['mean_oracle_response']
                      for panel in panels]) for kind in kinds]
    axes[1].scatter(centers,oracles,marker='D',s=42,c='black',
                    label='exact oracle')
    axes[1].axhline(0,color='gray',lw=.7)
    axes[1].set_xticks(centers,labels)
    axes[1].set(ylabel='Target-probability change',
                title='Supported edits on fresh reference panels')
    axes[1].legend(fontsize=8)
    fig.suptitle('V14: local action rule versus cross-sentence binding')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180)
    plt.close(fig)
    print('saved',args.output)


if __name__=='__main__':main()
