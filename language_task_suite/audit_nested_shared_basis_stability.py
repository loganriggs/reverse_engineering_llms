"""Subspace stability and bracket-token leverage of shared score bases."""

import argparse
import json
import sys
from pathlib import Path

import torch

from eval_multi_genre_token_tensor import score_tables
from eval_nested_shared_token_basis import shared_bases,token_marginal
from surface_mixture_v2_nested_varlen import SurfaceMixtureV2NestedVarlen
from surface_mixture_v2_varied import SurfaceMixtureV2Varied


def overlap(left,right,rank):
    a,b=left[:,:,:rank],right[:,:,:rank]
    return ((a.transpose(-1,-2)@b).square().sum(dim=(-1,-2))/rank)


def leverage(basis,indices,rank):
    return basis[:,indices,:rank].square().sum(-1).mean(1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--nfit',type=int,default=2048)
    p.add_argument('--length',type=int,default=128)
    p.add_argument('--smoothing',type=float,default=16.)
    p.add_argument('--seeds',default='20261212,20261215')
    p.add_argument('--ranks',default='4,8,16,32')
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
    target=SurfaceMixtureV2NestedVarlen(seed=0,
                                       **meta['generator_kwargs'])
    vocab=target.vocab
    bracket=[target.token2id[token] for token in ('(', '[', ')', ']')]
    uniform=torch.full((len(vocab),),1/len(vocab),device=a.device)
    weights={'uniform':uniform}
    fit={}
    for seed in [int(x) for x in a.seeds.split(',')]:
        nested=SurfaceMixtureV2NestedVarlen(seed=seed,
                                            **meta['generator_kwargs'])
        base=SurfaceMixtureV2Varied(seed=seed,parenthetical_mass=0.)
        for name,gen in [('nested',nested),('no_bracket',base)]:
            weight,stats=token_marginal(gen,a.nfit,a.length,vocab,
                                        a.smoothing,a.device)
            weights[f'{name}_{seed}']=weight
            fit[f'{name}_{seed}']=stats
    bases={name:shared_bases(tables,weight) for name,weight in weights.items()}
    ranks=[int(x) for x in a.ranks.split(',')]
    diagnostics={}
    for name,(query,key) in bases.items():
        diagnostics[name]={str(rank):dict(
            query_bracket_leverage=[float(v) for v in
                leverage(query,bracket,rank)],
            key_bracket_leverage=[float(v) for v in
                leverage(key,bracket,rank)]) for rank in ranks}
    comparisons={}
    seed0,seed1=[int(x) for x in a.seeds.split(',')]
    for left,right in [(f'nested_{seed0}',f'nested_{seed1}'),
                       (f'no_bracket_{seed0}',f'no_bracket_{seed1}'),
                       (f'nested_{seed0}',f'no_bracket_{seed0}'),
                       ('uniform',f'nested_{seed0}'),
                       ('uniform',f'no_bracket_{seed0}')]:
        comparisons[f'{left}__{right}']={str(rank):dict(
            query=[float(v) for v in overlap(bases[left][0],
                                             bases[right][0],rank)],
            key=[float(v) for v in overlap(bases[left][1],
                                           bases[right][1],rank)])
            for rank in ranks}
    result=dict(config={**vars(a),'model_dir':str(a.model_dir),
                        'output':str(a.output)},fit=fit,
                bracket_token_ids=bracket,diagnostics=diagnostics,
                overlaps=comparisons,
                metric='Mean squared canonical correlation between rank-r '
                       'subspaces; 1 is identical. Token leverage is mean '
                       'row squared norm on the four bracket tokens, per '
                       'head, in the weighted orthonormal coordinates.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('nested fit overlap rank8',comparisons[
        f'nested_{seed0}__nested_{seed1}']['8'])
    for name in ['uniform',f'nested_{seed0}',f'no_bracket_{seed0}']:
        print(name,'rank8 bracket leverage',diagnostics[name]['8'])


if __name__=='__main__':
    main()
