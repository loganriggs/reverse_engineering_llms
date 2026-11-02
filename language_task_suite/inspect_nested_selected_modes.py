"""Readable token loadings of causally selected shared score modes."""

import argparse
import json
import sys
from pathlib import Path

import torch

from eval_multi_genre_token_tensor import score_tables
from eval_nested_shared_token_basis import shared_bases,token_marginal
from surface_mixture_v2_nested_varlen import SurfaceMixtureV2NestedVarlen


def profile(vector,probability,generator,orient_token=None):
    vector=vector.detach().float().cpu()
    probability=probability.detach().float().cpu()
    if orient_token is None:
        orient=int(vector.abs().argmax())
    else:
        orient=generator.token2id[orient_token]
    if vector[orient]<0:
        vector=-vector
    raw=vector/probability.sqrt()
    top=lambda arr:[dict(token=generator.id2token[i],
        coefficient=float(arr[i])) for i in arr.abs().topk(20).indices.tolist()]
    return dict(orientation_token=generator.id2token[orient],
        bracket_weighted={token:float(vector[generator.token2id[token]])
                          for token in ('(', '[', ')', ']')},
        bracket_deweighted={token:float(raw[generator.token2id[token]])
                            for token in ('(', '[', ')', ']')},
        top_weighted=top(vector),top_deweighted=top(raw))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--nfit',type=int,default=2048)
    p.add_argument('--fit-seeds',default='20261212,20261215')
    p.add_argument('--length',type=int,default=128)
    p.add_argument('--smoothing',type=float,default=16.)
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,a.source)
    from model import BilinearGPT,Config
    torch.set_num_threads(4)
    meta=json.loads((a.model_dir/'results.json').read_text())
    record=next(row for row in meta['records'] if row['depth']==1)
    model=BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(a.model_dir/'model_depth1.pt',
                                     map_location='cpu',weights_only=True))
    model=model.to(a.device).eval()
    model.blocks[0].attn.rotary.inv_freq=(
        model.blocks[0].attn.rotary.inv_freq.to(a.device))
    tables,_=score_tables(model,a.length)
    profiles={}
    for seed in [int(x) for x in a.fit_seeds.split(',')]:
        gen=SurfaceMixtureV2NestedVarlen(seed=seed,
                                         **meta['generator_kwargs'])
        probability,stats=token_marginal(gen,a.nfit,a.length,gen.vocab,
                                         a.smoothing,a.device)
        query,key=shared_bases(tables,probability)
        profiles[str(seed)]=dict(fit=stats,
            head1_key0=profile(key[1,:,0],probability,gen,'('),
            head0_query0=profile(query[0,:,0],probability,gen))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(config={**vars(a),
        'model_dir':str(a.model_dir),'output':str(a.output)},profiles=profiles,
        caveat='Eigenvector sign is arbitrary; head1 key mode is oriented '
               'so ( is positive. Weighted and deweighted coordinates '
               'answer different questions. Token loadings alone do not '
               'prove a causal mechanism.'),indent=2)+'\n')
    for seed,row in profiles.items():
        print(seed,'bracket weighted',row['head1_key0']['bracket_weighted'])


if __name__=='__main__':
    main()
