"""Per-mapping causal edge ablations for the jointly sampled sparse v3 model."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_v3_shared_sparse_rule_edges import evaluate
from original_v3_shared_bracket import LanguageV3SharedBracket
from plot_original_v3_sparse_rule_graphs import events
from train_original_v3_shared_bracket import make_batch
from model import BilinearGPT, Config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=97901)
    parser.add_argument('--streams',type=int,default=512)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    torch.set_num_threads(1)
    device=torch.device(args.device)
    gen=LanguageV3SharedBracket(seed=args.seed)
    x,y,labels,_=make_batch(gen,args.streams,129,device)
    found=events(x,y,labels,gen,256)
    payload=torch.load(args.checkpoint,map_location=device,weights_only=True)
    model=BilinearGPT(Config(**payload['config'])).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    rules=list(gen.skip_bigrams.items())
    metrics={}
    matrix=np.full((len(rules),4),np.nan)
    for i,((anchor,trigger),target) in enumerate(rules):
        records=[r for r in found['skip_bigram'] if r[5]==anchor and r[6]==trigger]
        assert records
        label=f'{anchor} … {trigger} → {target}'
        metrics[label]={}
        for head in range(4):
            result=evaluate(model,gen,x,records,'skip_bigram',head,'anchor',32,device)
            metrics[label][f'H{head}']=result
            matrix[i,head]=result['oracle_kl_change']
    args.output_dir.mkdir(parents=True,exist_ok=True)
    result=dict(seed=args.seed,streams=args.streams,checkpoint=str(args.checkpoint),
                metrics=metrics,oracle_kl_change=matrix.tolist(),
                caveat='Layer-1 selected-anchor-edge ablation only, on sampled rule events. '
                       'A near-zero change does not prove absence of another anchor circuit.')
    (args.output_dir/'skip_bigram_per_row_ablation.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained')
    vmax=max(float(np.abs(matrix).max()),1e-6)
    image=ax.imshow(matrix,cmap='coolwarm',vmin=-vmax,vmax=vmax,aspect='auto')
    ax.set_xticks(range(4),[f'H{i}' for i in range(4)])
    ax.set_yticks(range(len(rules)),list(metrics),fontsize=8)
    ax.set_title('Remove selected anchor edge: change in full-oracle KL')
    fig.colorbar(image,ax=ax,label='ΔKL (nats; positive means worse)')
    fig.savefig(args.output_dir/'skip_bigram_per_row_ablation.png',dpi=180)
    plt.close(fig)
    print(json.dumps({k:{h:round(v['oracle_kl_change'],3) for h,v in row.items()}
                      for k,row in metrics.items()},indent=2))


if __name__=='__main__':main()
