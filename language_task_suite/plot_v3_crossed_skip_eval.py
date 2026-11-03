"""Plot the planted crossed table, learned competence, and anchor edits."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=json.loads(args.evaluation.read_text())
    audit=json.loads(args.audit.read_text())['audit']
    anchors=audit['table']['anchors']
    triggers=audit['table']['triggers']
    outputs=audit['table']['outputs']
    targets=audit['table']['targets']
    target_index=np.asarray([[outputs.index(target) for target in row]
                             for row in targets])
    probability=np.asarray([[result['pair_table'][f'{a}|{t}']
                             ['mean_target_probability'] for t in triggers]
                            for a in anchors])
    response=np.asarray([[result['anchor_edits']['by_pair'][f'{a}|{t}']
                          ['new_response'] for t in triggers] for a in anchors])
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    axes[0].imshow(target_index,vmin=-.5,vmax=len(outputs)-.5,cmap='tab10')
    a=axes[1].imshow(probability,vmin=0,vmax=1,cmap='viridis')
    b=axes[2].imshow(response,vmin=0,vmax=1,cmap='viridis')
    for ax in axes:
        ax.set_xticks(range(len(triggers)),triggers)
        ax.set_yticks(range(len(anchors)),anchors)
    for i in range(len(anchors)):
        for j in range(len(triggers)):
            axes[0].text(j,i,targets[i][j],ha='center',va='center',
                         color='white',weight='bold')
            axes[1].text(j,i,f'{probability[i,j]:.2f}',ha='center',va='center',
                         color='white' if probability[i,j]<.55 else 'black')
            axes[2].text(j,i,f'{response[i,j]:.2f}',ha='center',va='center',
                         color='white' if response[i,j]<.55 else 'black')
    axes[0].set_title('Planted next token')
    axes[1].set_title('Model P(correct) on fresh streams',fontsize=11)
    axes[2].set_title('Anchor edit: ΔP(new target)',fontsize=11)
    fig.colorbar(a,ax=axes[1],label='Probability')
    fig.colorbar(b,ax=axes[2],label='Response; oracle = 1')
    fig.suptitle('Crossed skip rule: both earlier place and adjacent adjective matter')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
