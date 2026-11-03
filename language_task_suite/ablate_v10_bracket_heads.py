"""Exact full-model head removal on fresh V10 rule sites."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from surface_mixture_v10_linked_events import SurfaceMixtureV10LinkedEvents
from train_surface_mixture import evaluate,make_eval


RULES=('verb','object','source_influenced_object',
       'pending_claim_matching_object','source_answer','bracket_close')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--streams',type=int,default=256)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--seed',type=int,default=99003)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0,a.source)
    from model import BilinearGPT,Config
    run=json.loads((Path(a.run_dir)/'results.json').read_text())
    assert run['config']['generator']=='v10' and len(run['records'])==1
    record=run['records'][0]
    model=BilinearGPT(Config(**record['model_config'])).to(a.device)
    model.load_state_dict(torch.load(Path(a.run_dir)/f"model_depth{record['depth']}.pt",
                                     map_location=a.device,weights_only=True))
    model.eval()
    gen=SurfaceMixtureV10LinkedEvents(seed=a.seed,**run['generator_kwargs'])
    data=make_eval(gen,a.streams,a.length,a.device)
    baseline=evaluate(model,data)
    rows=[]
    for layer,block in enumerate(model.blocks):
        attn=block.attn
        dim=attn.head_dim
        for head in range(attn.n_head):
            def remove_head(module,inputs,head=head,dim=dim):
                x=inputs[0].clone()
                x[...,head*dim:(head+1)*dim]=0
                return (x,)
            hook=attn.out.register_forward_pre_hook(remove_head)
            try:
                scores=evaluate(model,data)
            finally:
                hook.remove()
            rows.append(dict(layer=layer,head=head,scores=scores,
                delta_kl={rule:scores[rule]['kl']-baseline[rule]['kl']
                          for rule in RULES if rule in scores and rule in baseline}))
    matrix=np.array([[row['delta_kl'].get(rule,np.nan) for rule in RULES]
                     for row in rows])
    vmax=max(.05,float(np.nanquantile(np.abs(matrix),.95)))
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    im=ax.imshow(matrix,cmap='coolwarm',vmin=-vmax,vmax=vmax,aspect='auto')
    ax.set_xticks(range(len(RULES)),[r.replace('_',' ') for r in RULES],rotation=25,ha='right')
    ax.set_yticks(range(len(rows)),[f"L{r['layer']}H{r['head']}" for r in rows])
    ax.set_title('Full-model head removal: change in exact-oracle KL (positive = worse)')
    fig.colorbar(im,ax=ax,label='Δ KL (nats)')
    plot=Path(a.output).with_suffix('.png')
    fig.savefig(plot,dpi=180)
    plt.close(fig)
    result=dict(config=vars(a),baseline=baseline,rows=rows,
                figure=str(plot),caveat='Zeroing a whole head before W_O across '
                'all positions is a causal model intervention. It does not '
                'isolate one token edge or prove a unique minimal circuit. '
                'Negative delta means ablation improves oracle KL.')
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('baseline bracket KL',baseline['bracket_close']['kl'])
    for row in rows:
        print(f"L{row['layer']}H{row['head']}",
              {k:round(row['delta_kl'].get(k,float('nan')),4)
               for k in ('bracket_close','source_answer','pending_claim_matching_object')})


if __name__=='__main__':main()
