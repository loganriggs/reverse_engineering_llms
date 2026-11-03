"""Local pre-W_O basis comparison on a held-out document genre.

All model-training genres were seen; summary is withheld only from basis fit.
Weight-only and unlabeled-data bases use no rule labels. The edit-contrast
basis explicitly uses source-edit pairs and is evaluated on disjoint documents.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from eval_document_world_v5_clause_causal import collect,score
from ablate_document_world_v5_clause_heads import load_model
from document_world_v5_clause_realizations import sample_documents
from train_document_world_v5_clause import encode,evaluate


@torch.no_grad()
def unlabeled_gram(model,bond,docs,vocab,n_ctx,device,batch_size):
    gram=torch.zeros((bond.in_features,bond.in_features),dtype=torch.float64,
                     device=device)
    n_vectors=0
    for start in range(0,len(docs),batch_size):
        part=docs[start:start+batch_size]
        x,_=encode(part,vocab,n_ctx)
        lengths=torch.tensor([len(doc.tokens)-1 for doc in part],device=device)
        captured=[]
        hook=bond.register_forward_pre_hook(
            lambda module,inputs:captured.append(inputs[0].detach()))
        try:
            model(x.to(device))
        finally:
            hook.remove()
        z=captured[0]
        valid=torch.arange(n_ctx,device=device)[None,:]<lengths[:,None]
        values=z[valid].double()
        gram+=values.T@values
        n_vectors+=len(values)
    return gram,n_vectors


@torch.no_grad()
def query_vectors(model,bond,entries,vocab,n_ctx,device,batch_size):
    vectors=[]
    for start in range(0,len(entries),batch_size):
        part=entries[start:start+batch_size]
        pad=vocab['<PAD>']
        x=torch.full((len(part),n_ctx),pad,dtype=torch.long,device=device)
        query=torch.tensor([site.index-1 for doc,site in part],device=device)
        for i,(doc,site) in enumerate(part):
            ids=[vocab[token] for token in doc.tokens[:-1]]
            assert len(ids)<=n_ctx and 0<=site.index-1<len(ids)
            x[i,:len(ids)]=torch.tensor(ids,device=device)
        captured=[]
        hook=bond.register_forward_pre_hook(
            lambda module,inputs:captured.append(inputs[0].detach()))
        try:
            model(x)
        finally:
            hook.remove()
        vectors.append(captured[0][torch.arange(len(part),device=device),
                                   query].double())
    return torch.cat(vectors)


def basis_set(weight,unlabeled,contrast,seed):
    d=weight.shape[1]
    weight_gram=weight.double().T@weight.double()
    grams={'weight_only':weight_gram,'unlabeled_data':unlabeled,
           'source_edit_supervised':contrast}
    for alpha in (.1,1.,10.):
        grams[f'hybrid_{alpha:g}']=(unlabeled/unlabeled.trace()
                                  +alpha*weight_gram/weight_gram.trace())
    bases={}
    for key,gram in grams.items():
        _,vectors=torch.linalg.eigh(gram)
        bases[key]=vectors.flip(1).float().contiguous()
    bases['physical']=torch.eye(d,device=weight.device)
    generator=torch.Generator(device=weight.device).manual_seed(seed)
    bases['random']=torch.linalg.qr(torch.randn(
        (d,d),device=weight.device,generator=generator)).Q
    for key,u in bases.items():
        assert (u.T@u-torch.eye(d,device=u.device)).abs().max()<1e-4,key
    return bases


def project_hook(basis,rank):
    u=basis[:,:rank]
    def project(module,inputs):
        z=inputs[0]
        return ((z@u)@u.T,)
    return project


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--fit-streams',type=int,default=1000)
    p.add_argument('--eval-streams',type=int,default=1000)
    p.add_argument('--eval-documents',type=int,default=200)
    p.add_argument('--max-cases',type=int,default=128)
    p.add_argument('--fit-seed',type=int,default=346001)
    p.add_argument('--eval-seed',type=int,default=346002)
    p.add_argument('--layer',type=int,default=2)
    p.add_argument('--ranks',default='4,8,16,32,64')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    model,saved=load_model(a.checkpoint,a.device)
    vocab=saved['vocabulary'];n_ctx=saved['n_ctx']
    kwargs=saved['generator_kwargs']
    assert saved['depth']==2 and a.layer in (0,2)
    bond=model.layers[a.layer].o
    fit_docs=[doc for doc in sample_documents(a.fit_streams,a.fit_seed,**kwargs)
              if doc.genre!='summary' and len(doc.tokens)-1<=n_ctx]
    assert len(fit_docs)>=100
    gram,n_vectors=unlabeled_gram(model,bond,fit_docs,vocab,n_ctx,a.device,
                                  a.batch_size)
    fit_cases,_=collect(a.fit_streams,a.fit_seed+1000,kwargs,n_ctx,10000)
    source_fit=[row for row in fit_cases['source_attribution']
                if row[0].genre!='summary' and
                   len(row[0].tokens)-1<=n_ctx and
                   len(row[2].tokens)-1<=n_ctx]
    source_fit=source_fit[:256]
    assert len(source_fit)>=64
    original=query_vectors(model,bond,[(r[0],r[1]) for r in source_fit],
                           vocab,n_ctx,a.device,a.batch_size)
    edited=query_vectors(model,bond,[(r[2],r[3]) for r in source_fit],
                         vocab,n_ctx,a.device,a.batch_size)
    delta=edited-original
    contrast=delta.T@delta
    bases=basis_set(bond.weight.detach(),gram,contrast,a.fit_seed+4000)
    test_docs=[doc for doc in sample_documents(
        a.eval_streams,a.eval_seed,**kwargs)
        if doc.genre=='summary' and len(doc.tokens)-1<=n_ctx]
    test_docs=test_docs[:a.eval_documents]
    assert len(test_docs)>=50
    test_data=encode(test_docs,vocab,n_ctx)
    eval_cases,_=collect(a.eval_streams,a.eval_seed+1000,kwargs,n_ctx,10000)
    rng=random.Random(a.eval_seed+4000)
    test_cases={}
    for family in ('source_attribution','source_nuisance'):
        rows=[row for row in eval_cases[family]
              if row[0].genre=='summary' and
                 len(row[0].tokens)-1<=n_ctx and
                 len(row[2].tokens)-1<=n_ctx]
        rng.shuffle(rows)
        test_cases[family]=rows[:a.max_cases]
        assert len(test_cases[family])>=32
    native_eval=evaluate(model,test_docs,test_data,vocab,a.device)
    native_edits=score(model,test_cases,vocab,n_ctx,a.device,a.batch_size)
    ranks=sorted(set(int(v) for v in a.ranks.split(',')))
    assert ranks[-1]==bond.in_features and ranks[0]>=1
    rows=[]
    for name,u in bases.items():
        for rank in ranks:
            hook=bond.register_forward_pre_hook(project_hook(u,rank))
            try:
                measured=evaluate(model,test_docs,test_data,vocab,a.device)
                edits=score(model,test_cases,vocab,n_ctx,a.device,a.batch_size)
            finally:
                hook.remove()
            rows.append(dict(basis=name,rank=rank,
                             token_nll=measured['token_nll'],
                             source=measured['by_site']['source_attribution'],
                             source_edit=edits['source_attribution'],
                             nuisance_edit=edits['source_nuisance']))
            if rank==bond.in_features:
                assert abs(measured['token_nll']-native_eval['token_nll'])<1e-5
                assert abs(edits['source_attribution']['mean_model_response']-
                           native_edits['source_attribution']
                           ['mean_model_response'])<1e-5
            print(name,rank,'token NLL',round(measured['token_nll'],4),
                  'source edit',round(edits['source_attribution']
                                      ['mean_model_response'],4),flush=True)
    out=dict(config={**vars(a),'checkpoint':str(a.checkpoint),
                     'output':str(a.output)},
             fit=dict(genres=['story','log'],n_documents=len(fit_docs),
                      n_bond_vectors=n_vectors,n_source_edit_pairs=len(source_fit)),
             test=dict(heldout_basis_genre='summary',n_documents=len(test_docs),
                       causal_cases={k:len(v) for k,v in test_cases.items()}),
             native=dict(evaluation=native_eval,edits=native_edits),rows=rows,
             caveat='Full-rank orthogonal projections are exact local gauges. '
                    'Lower ranks are lossy pre-W_O bond interventions applied '
                    'at all positions in one layer. Summary was seen by model '
                    'training but withheld from basis fit. The edit-contrast '
                    'basis is explicitly supervised; others are not. This '
                    'does not recover a unique global tensor circuit.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
