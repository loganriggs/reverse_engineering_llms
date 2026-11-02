"""Does the canonical weighted basis concentrate causal task effects?"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from eval_multi_genre_token_tensor import score_tables
from eval_nested_basis_mode_ablations import (compact,damage,make_panel,
    mode_contribution)
from eval_nested_shared_token_basis import (evaluate,project_tables,
    shared_bases,token_marginal)
from surface_mixture_v2_nested_varlen import SurfaceMixtureV2NestedVarlen


def random_orthogonal(rank,seed,device):
    generator=torch.Generator(device=device).manual_seed(seed)
    matrix=torch.randn((rank,rank),generator=generator,device=device)
    q,r=torch.linalg.qr(matrix)
    return q*torch.sign(torch.diagonal(r))[None,:]


@torch.no_grad()
def rotated_components(core,u,v,scale,head,mode,rotation):
    new_u,new_v=u.clone(),v.clone()
    new_core=core.clone()
    if mode=='key':
        new_v[head]=v[head]@rotation
        new_core[head]=core[head]@rotation
    else:
        new_u[head]=u[head]@rotation
        new_core[head]=rotation.T@core[head]
    return [mode_contribution(new_core,new_u,new_v,scale,head,mode,i)
            for i in range(rotation.shape[0])]


def evaluate_axes(model,baseline,downstream,panel,components,head,mode,
                  baseline_metrics):
    rows=[]
    for i,component in enumerate(components):
        altered=baseline.clone()
        altered[head]-=component
        metrics=compact(evaluate(model,altered,downstream,*panel))
        rows.append(dict(index=i,damage=damage(metrics,baseline_metrics)))
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--nfit',type=int,default=2048)
    p.add_argument('--fit-seed',type=int,default=20261212)
    p.add_argument('--edit-seed',type=int,default=20261303)
    p.add_argument('--verb-seed',type=int,default=20261304)
    p.add_argument('--nstreams',type=int,default=64)
    p.add_argument('--verb-limit',type=int,default=64)
    p.add_argument('--length',type=int,default=128)
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--rotations',type=int,default=16)
    p.add_argument('--rotation-seed',type=int,default=20261305)
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
    gen=SurfaceMixtureV2NestedVarlen(seed=a.fit_seed,
                                     **meta['generator_kwargs'])
    probability,fit_stats=token_marginal(gen,a.nfit,a.length,gen.vocab,
                                         a.smoothing,a.device)
    tables,downstream=score_tables(model,a.length)
    bases=shared_bases(tables,probability)
    u,v=(b[:,:,:a.rank] for b in bases)
    scale=probability.sqrt()
    weighted=tables*scale[None,None,:,None]*scale[None,None,None,:]
    core=torch.einsum('hqi,hlqk,hkj->hlij',u,weighted,v)
    baseline=project_tables(tables,probability,bases,a.rank)
    panel=make_panel(model,meta,a.edit_seed,a.verb_seed,a.nstreams,
                     a.verb_limit,a.length,a.device)
    base_metrics=compact(evaluate(model,baseline,downstream,*panel))
    cases=[(1,'key','outer_edit'),(0,'query','verb_kl')]
    result=dict(config={**vars(a),'model_dir':str(a.model_dir),
                        'output':str(a.output)},fit=fit_stats,
                evaluation=dict(bracket_pairs=len(panel[0]),
                                verb_sites=len(panel[1])),
                baseline=base_metrics,cases={})
    def save():
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
    identity=torch.eye(a.rank,device=a.device)
    for head,mode,target in cases:
        canonical=rotated_components(core,u,v,scale,head,mode,identity)
        canonical_rows=evaluate_axes(model,baseline,downstream,panel,
            canonical,head,mode,base_metrics)
        canonical_max=max(r['damage'][target] for r in canonical_rows)
        random_rows=[]
        for j in range(a.rotations):
            rotation=random_orthogonal(a.rank,a.rotation_seed+j*17+
                                       head*1000,a.device)
            components=rotated_components(core,u,v,scale,head,mode,rotation)
            replay_error=float((sum(components)-baseline[head]).abs().max())
            assert replay_error<1e-3,replay_error
            rows=evaluate_axes(model,baseline,downstream,panel,
                components,head,mode,base_metrics)
            random_rows.append(dict(rotation=j,rows=rows,
                max_target_damage=max(r['damage'][target] for r in rows),
                replay_max_abs=replay_error))
            print(head,mode,'rotation',j,'max',round(
                random_rows[-1]['max_target_damage'],3),flush=True)
            result['cases'][f'{head}_{mode}']=dict(target=target,
                canonical=canonical_rows,canonical_max=canonical_max,
                random=random_rows)
            save()
        maxima=[r['max_target_damage'] for r in random_rows]
        result['cases'][f'{head}_{mode}']['random_summary']=dict(
            mean=float(np.mean(maxima)),minimum=float(np.min(maxima)),
            maximum=float(np.max(maxima)),
            fraction_at_least_canonical=float(np.mean(
                np.array(maxima)>=canonical_max)))
        save()
    result['caveat']='Random orthogonal rotations leave the rank-8 score '
    result['caveat']+='function exactly unchanged before ablation. This '
    result['caveat']+='tests causal-effect concentration of the chosen '
    result['caveat']+='basis on one fixed panel, not uniqueness among all '
    result['caveat']+='possible gauges or transfer across model seeds.'
    save()
    for name,row in result['cases'].items():
        print(name,'canonical max',row['canonical_max'],
              'random summary',row['random_summary'])


if __name__=='__main__':
    main()
