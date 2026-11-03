"""Whole-kind task-blind bond-basis transfer to V7 conditional action."""

import argparse
import json
import random
from pathlib import Path

import torch

from document_mixture_v7_cross import sample_document
from eval_document_mixture_v7_action_causal import collect as action_cases
from eval_document_mixture_v7_action_causal import score as action_score
from eval_document_world_v5_clause_bond_bases import (
    project_hook,unlabeled_gram,
)
from train_document_mixture_v5 import DeepModel,SPECS,encode,evaluate


def sample(n,seed):
    rng=random.Random(seed)
    return [sample_document(rng) for _ in range(n)]


def bases(weight,data,seed):
    d=weight.shape[1]
    wg=weight.double().T@weight.double()
    grams={'weight_only':wg,'unlabeled_other_kinds':data,
           'hybrid_1':wg/wg.trace()+data/data.trace()}
    result={}
    for name,gram in grams.items():
        _,vectors=torch.linalg.eigh(gram)
        result[name]=vectors.flip(1).float().contiguous()
    result['physical']=torch.eye(d,device=weight.device)
    gen=torch.Generator(device=weight.device).manual_seed(seed)
    result['random']=torch.linalg.qr(torch.randn(
        (d,d),generator=gen,device=weight.device)).Q
    for name,u in result.items():
        assert (u.T@u-torch.eye(d,device=u.device)).abs().max()<1e-4,name
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--layer',type=int,default=2)
    p.add_argument('--fit-streams',type=int,default=1500)
    p.add_argument('--test-streams',type=int,default=800)
    p.add_argument('--action-streams',type=int,default=400)
    p.add_argument('--fit-seed',type=int,default=34991)
    p.add_argument('--test-seed',type=int,default=34992)
    p.add_argument('--ranks',default='8,16,32,64')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    vocab=saved['vocabulary'];n_ctx=saved['n_ctx']
    args=saved['model_kwargs']
    model=DeepModel(len(vocab),args['d_model'],args['n_head'],
                    SPECS[f"block{saved['depth']}"],n_ctx,norm=args['norm'],
                    residual=args['residual'],attention=args['attention']).to(a.device)
    model.load_state_dict(saved['state_dict']);model.eval()
    bond=model.layers[a.layer].o
    fit=[d for d in sample(a.fit_streams,a.fit_seed)
         if d.kind!='narrative' and len(d.tokens)-1<=n_ctx]
    assert len(fit)>=500
    gram,n_bond=unlabeled_gram(model,bond,fit,vocab,n_ctx,a.device,
                               a.batch_size)
    directions=bases(bond.weight.detach(),gram,a.fit_seed+991)
    test=[d for d in sample(a.test_streams,a.test_seed)
          if d.kind=='narrative' and len(d.tokens)-1<=n_ctx]
    assert len(test)>=100
    test_data=encode(test,vocab,n_ctx)
    cases,n_available=action_cases(a.test_seed+1000,a.action_streams,
                                   192,128)
    assert len(cases)==128
    native=evaluate(model,test,test_data,vocab,a.device)
    native_edit=action_score(model,cases,vocab,n_ctx,a.device,a.batch_size)
    ranks=sorted(set(int(v) for v in a.ranks.split(',')))
    assert ranks[-1]==bond.in_features
    rows=[]
    for name,u in directions.items():
        for rank in ranks:
            hook=bond.register_forward_pre_hook(project_hook(u,rank))
            try:
                measured=evaluate(model,test,test_data,vocab,a.device)
                edit=action_score(model,cases,vocab,n_ctx,a.device,
                                  a.batch_size)
            finally:
                hook.remove()
            rows.append(dict(basis=name,rank=rank,
                             token_nll=measured['token_nll'],
                             action=measured['by_site']['conditional_action'],
                             edit=edit))
            if rank==bond.in_features:
                assert abs(measured['token_nll']-native['token_nll'])<1e-5
                assert abs(edit['mean_model_response']-
                           native_edit['mean_model_response'])<1e-5
            print(name,rank,round(measured['token_nll'],3),
                  round(edit['mean_model_response'],3),flush=True)
    out=dict(config={**vars(a),'checkpoint':str(a.checkpoint),
                     'output':str(a.output)},
             fit=dict(kinds=sorted({d.kind for d in fit}),n_documents=len(fit),
                      n_bond_vectors=n_bond,n_narrative_documents=0),
             test=dict(kind='narrative',n_documents=len(test),
                       n_supported_action_edits=len(cases),
                       n_available_edits=n_available),
             native=dict(token_nll=native['token_nll'],
                         action=native['by_site']['conditional_action'],
                         edit=native_edit),rows=rows,
             caveat='Narrative is a whole-kind holdout for unlabeled basis '
                    'fitting, not for model training. Full rank is an exact '
                    'local gauge; lower ranks are lossy pre-W_O projections '
                    'at all positions in one layer. Action edits are on '
                    'fresh V14 narrative samples. This does not prove '
                    'unique circuit recovery or natural-language realism.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
