"""Per-pair causal ablations of the selected earlier-place attention edge."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

_pre=argparse.ArgumentParser(add_help=False)
_pre.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
_source,_=_pre.parse_known_args()
sys.path.insert(0,_source.source)

from eval_v3_shared_sparse_rule_edges import evaluate
from model import BilinearGPT, Config
from original_v3_crossed_skip import LanguageV3CrossedSkip
from plot_v3_crossed_skip_sparse_paths import records_for_rules
from train_original_v3_shared_bracket import make_batch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dense',type=Path,required=True)
    p.add_argument('--sparse',type=Path,required=True)
    p.add_argument('--seed',type=int,default=99301)
    p.add_argument('--streams',type=int,default=512)
    p.add_argument('--per-pair',type=int,default=32)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    device=torch.device(args.device)
    gen=LanguageV3CrossedSkip(seed=args.seed)
    x,y,labels,_=make_batch(gen,args.streams,129,device)
    grouped=records_for_rules(gen,x,y,labels,args.per_pair)
    order=[(a,t) for a in gen.crossed_skip_anchors
           for t in gen.crossed_skip_triggers]
    outputs=[gen.token2id[w] for w in gen.crossed_skip_outputs]
    matrices={}
    detailed={}
    for stage,path in [('dense',args.dense),('sparse',args.sparse)]:
        payload=torch.load(path,map_location=device,weights_only=True)
        assert payload.get('generator')=='LanguageV3CrossedSkip'
        model=BilinearGPT(Config(**payload['config'])).to(device)
        model.load_state_dict(payload['model'])
        model.eval()
        matrix=np.zeros((16,4))
        by_pair={}
        for i,key in enumerate(order):
            target=gen.token2id[gen.skip_bigrams[key]]
            alternatives=[v for v in outputs if v!=target]
            records=[(b,q,k,tid,alternatives,key[0],key[1])
                     for b,q,k,tid in grouped[key]]
            by_head={}
            for head in range(4):
                metric=evaluate(model,gen,x,records,'skip_bigram',head,
                                'anchor',32,device)
                matrix[i,head]=metric['oracle_kl_change']
                by_head[f'H{head}']=metric
            by_pair[f'{key[0]}|{key[1]}']=by_head
        matrices[stage]=matrix.tolist()
        detailed[stage]=by_pair
    labels=[f'{a} … {t} → {gen.skip_bigrams[a,t]}' for a,t in order]
    vmax=max(np.max(np.abs(matrix)) for matrix in matrices.values())
    fig,axes=plt.subplots(1,2,figsize=(10,7.5),layout='constrained')
    for ax,stage in zip(axes,('dense','sparse')):
        im=ax.imshow(matrices[stage],vmin=-vmax,vmax=vmax,
                     cmap='coolwarm',aspect='auto')
        ax.set_title(stage)
        ax.set_xticks(range(4),[f'H{i}' for i in range(4)])
        ax.set_yticks(range(16),labels,fontsize=7)
    fig.colorbar(im,ax=axes,label='Δ full-oracle KL (positive = worse)')
    fig.suptitle('Remove one selected L1 place→query edge on each crossed-skip event')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output_dir/'crossed_skip_edge_ablations.png',dpi=180)
    plt.close(fig)
    result=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                rule_order=labels,matrices=matrices,detailed=detailed,
                caveat='This removes one layer-1 head edge at the selected '
                       'anchor. It proves local edge necessity on these events '
                       'where KL rises, not uniqueness or complete circuit '
                       'sufficiency. Other positions and heads may compensate.')
    (args.output_dir/'crossed_skip_edge_ablations.json').write_text(
        json.dumps(result,indent=2)+'\n')
    print('max effect', {k:float(np.max(v)) for k,v in matrices.items()},flush=True)


if __name__=='__main__':main()
