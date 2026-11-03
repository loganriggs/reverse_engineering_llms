"""Compare bond priors on joint rules and exact supported bracket edits."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_pre=argparse.ArgumentParser(add_help=False)
_pre.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
_source,_=_pre.parse_known_args()
sys.path.insert(0,_source.source)

from generator_v3 import make_batch
from model import BilinearGPT,Config
from eval_original_v3_multirule_bond_bases import (activation_grams,
                                                     bases_from_grams,
                                                     evaluate as rule_evaluate)
from eval_original_v3_shared_bracket import selected_pairs,score_pairs
from original_v3_shared_bracket import LanguageV3SharedBracket


@torch.no_grad()
def pair_scores_with_basis(model,gen,pairs,layer,basis,rank,device):
    if basis is None:
        return score_pairs(model,gen,pairs,32,device)
    U=basis[:,:rank]
    def project(module,inputs):
        z=inputs[0]
        return ((z@U)@U.T,)
    handle=model.blocks[layer].attn.out.register_forward_pre_hook(project)
    try:return score_pairs(model,gen,pairs,32,device)
    finally:handle.remove()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,
                   default=Path('/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits'))
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--fit-streams',type=int,default=256)
    p.add_argument('--eval-streams',type=int,default=128)
    p.add_argument('--pair-streams',type=int,default=512)
    p.add_argument('--pair-limit',type=int,default=256)
    p.add_argument('--length',type=int,default=129)
    p.add_argument('--fit-seed',type=int,default=97301)
    p.add_argument('--eval-seed',type=int,default=97302)
    p.add_argument('--ranks',default='16,32,64')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    device=torch.device(a.device)
    payload=torch.load(a.run_dir/'model.pt',map_location=device,weights_only=True)
    model=BilinearGPT(Config(**payload['config'])).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    grams,counts=activation_grams(model,LanguageV3SharedBracket(seed=a.fit_seed),
        a.fit_streams,16,a.length,device,a.fit_seed)
    bases=bases_from_grams(model,grams)
    x,y,labels,_=make_batch(LanguageV3SharedBracket(seed=a.eval_seed),
                            a.eval_streams,a.length,device)
    native_bonds={}
    hooks=[]
    for layer,block in enumerate(model.blocks):
        def capture(module,inputs,layer=layer):native_bonds[layer]=inputs[0].detach()
        hooks.append(block.attn.out.register_forward_pre_hook(capture))
    with torch.no_grad():
        try:native_logp=F.log_softmax(model(x)[0].float(),-1)
        finally:
            for hook in hooks:hook.remove()
        native_ce=-native_logp.gather(-1,y[...,None]).squeeze(-1)
    pairs=selected_pairs(LanguageV3SharedBracket(seed=a.eval_seed+1),
                         a.pair_streams,a.length,a.pair_limit)
    assert len(pairs)==a.pair_limit
    reference=pair_scores_with_basis(model,LanguageV3SharedBracket(seed=a.eval_seed+2),
                                     pairs,0,None,64,device)
    rows=[]
    for layer in range(len(model.blocks)):
        for basis_name in ('weight_only','all','hybrid_0.1','hybrid_1'):
            for rank in (int(x) for x in a.ranks.split(',')):
                basis=bases[layer][basis_name]
                rule=rule_evaluate(model,x,y,labels,layer,basis,rank,
                                   native_logp,native_ce,native_bonds[layer])
                bracket=pair_scores_with_basis(model,
                    LanguageV3SharedBracket(seed=a.eval_seed+2),pairs,
                    layer,basis,rank,device)
                rows.append(dict(layer=layer,basis=basis_name,rank=rank,
                                 rule_scores=rule,bracket_oracle=bracket))
                print(layer,basis_name,rank,
                      'bracketKL',round(bracket['all']['original_kl'],4),
                      'editL1',round(bracket['all']['effect_l1_error'],4),
                      flush=True)
    for row in rows:
        if row['rank']==model.config.n_embd:
            assert max(v['teacher_kl'] for v in row['rule_scores'].values())<1e-5
            assert abs(row['bracket_oracle']['all']['original_kl']-
                       reference['all']['original_kl'])<1e-5
            assert abs(row['bracket_oracle']['all']['new_closer_effect']-
                       reference['all']['new_closer_effect'])<1e-5
    result=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                fit_counts=counts,n_pairs=len(pairs),native_bracket_oracle=reference,
                rows=rows,
                caveat='Unlabeled bond bases fit on separate streams. Rule-site '
                       'teacher KL and sampled CE are separate from full '
                       'bracket-oracle KL and supported opener-edit fidelity. '
                       'One bond projected at a time; this is not a canonical '
                       'tensor decomposition or sparse circuit. Model trained '
                       'on all rule families, and DGP realism remains negative.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
