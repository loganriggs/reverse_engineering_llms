"""Visual comparison of local rank-16 bond priors on crossed skip rules."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHODS=[('weight_only','Weight SVD'),('all','Activation all'),
         ('without_skip_bigram','Activation\nwithout skip sites'),
         ('random_without_skip_bigram','Random matched\nomission'),
         ('hybrid_1','Weight + activation')]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--rank',type=int,default=16)
    p.add_argument('--layer',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    data=json.loads(args.results.read_text())
    rows=[r for r in data['rows'] if r['layer']==args.layer
          and r['rank']==args.rank]
    specs=[('Skip-site teacher KL',lambda m:m['skip_bigram']['teacher_kl']),
           ('Anchor-edit error vs native',lambda m:m['edit_native_mae']),
           ('Anchor-edit error vs oracle',lambda m:m['edit_oracle_mae']),
           ('Bond reconstruction MSE',lambda m:m['bond_reconstruction_mse'])]
    fig,axes=plt.subplots(2,2,figsize=(13,7.5),layout='constrained')
    colors=['#6671a3','#5d9871','#d67555','#8d8d8d','#ad8b45']
    for ax,(title,metric) in zip(axes.ravel(),specs):
        values=[]
        for method,_ in METHODS:
            pair=[metric(row['metrics']) for row in rows if row['method']==method]
            assert len(pair)==2
            values.append(pair)
        means=np.mean(values,axis=1)
        ax.bar(range(len(METHODS)),means,color=colors,width=.7)
        for i,pair in enumerate(values):
            ax.scatter([i-.08,i+.08],pair,c='black',s=13,zorder=3)
        ax.set_xticks(range(len(METHODS)),[name for _,name in METHODS],fontsize=8)
        ax.set_title(title)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(f'Crossed-skip sparse model: layer {args.layer} rank {args.rank} '
                 'pre-W_O projection; two fresh panels')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
