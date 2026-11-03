"""Fresh visible-fact and nuisance edits for trained variable-clause LMs."""

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

OLD_REPO=Path(__file__).resolve().parents[2]/'reverse_engineer/tensor_language'
sys.path.insert(0,str(OLD_REPO))
from deep_model import DeepModel,SPECS  # noqa: E402
from audit_document_world_v5_clause_realizations import _edited_world  # noqa: E402
from document_world import NAMES  # noqa: E402
from document_world_v5_clause_realizations import (  # noqa: E402
    answer_from_visible_prefix,realize,sample_documents,
)


def _nuisance_source_world(world,plan):
    after,family,obj,attr=plan
    assert family=='source_attribution'
    target=next(j for j in range(after,-1,-1)
                if world['events'][j]['object']==obj and
                world['events'][j]['attribute']==attr)
    others=[j for j in range(after+1) if j!=target]
    if not others:
        return None
    j=others[-1]
    changed=copy.deepcopy(world)
    old=changed['sources'][j]
    changed['sources'][j]=next(name for name in NAMES if name!=old)
    return changed


def _source_is_delayed(world,plan):
    after,family,obj,attr=plan
    assert family=='source_attribution'
    target=next(j for j in range(after,-1,-1)
                if world['events'][j]['object']==obj and
                world['events'][j]['attribute']==attr)
    return target < after


def collect(n,seed,kwargs,n_ctx,max_cases):
    docs=sample_documents(n,seed,**kwargs)
    cases=defaultdict(list);skipped=defaultdict(int)
    for doc in docs:
        for k,(site,plan) in enumerate(zip(doc.sites,
                                            doc.latent['recap_plan'],strict=True)):
            changed=realize(_edited_world(doc.latent,plan),**kwargs)
            altered=changed.sites[k]
            assert altered.family==site.family and altered.answer!=site.answer
            assert answer_from_visible_prefix(doc.tokens[:site.index])==site.answer
            assert answer_from_visible_prefix(
                changed.tokens[:altered.index])==altered.answer
            if site.index>n_ctx or altered.index>n_ctx:
                skipped[site.family+'_length']+=1
                continue
            row=(doc,site,changed,altered,altered.answer)
            cases[site.family].append(row)
            if site.family=='source_attribution':
                suffix='delayed' if _source_is_delayed(doc.latent,plan) else 'immediate'
                cases['source_'+suffix].append(row)
                nuisance=_nuisance_source_world(doc.latent,plan)
                if nuisance is not None:
                    other=realize(nuisance,**kwargs)
                    other_site=other.sites[k]
                    assert other_site.answer==site.answer
                    assert answer_from_visible_prefix(
                        other.tokens[:other_site.index])==site.answer
                    if other_site.index<=n_ctx:
                        nuisance_row=(doc,site,other,other_site,site.answer)
                        cases['source_nuisance'].append(nuisance_row)
                        cases['source_nuisance_'+suffix].append(nuisance_row)
                    else:
                        skipped['source_nuisance_length']+=1
    rng=random.Random(seed+991)
    selected={}
    for family,rows in cases.items():
        rng.shuffle(rows)
        selected[family]=rows[:max_cases]
    return selected,dict(skipped)


@torch.no_grad()
def _probabilities(model,rows,vocab,n_ctx,device,batch_size):
    out=[]
    pad=vocab['<PAD>']
    for start in range(0,len(rows),batch_size):
        part=rows[start:start+batch_size]
        x=torch.full((len(part),n_ctx),pad,dtype=torch.long,device=device)
        query=torch.tensor([site.index-1 for doc,site,target in part],
                           dtype=torch.long,device=device)
        target=torch.tensor([vocab[target] for doc,site,target in part],
                            dtype=torch.long,device=device)
        for i,(doc,_,_) in enumerate(part):
            ids=[vocab[word] for word in doc.tokens[:-1]]
            assert len(ids)<=n_ctx
            x[i,:len(ids)]=torch.tensor(ids,device=device)
        logits=model(x)
        p=logits[torch.arange(len(part),device=device),query].float()
        out.extend(p.softmax(-1).gather(-1,target[:,None]).squeeze(-1)
                   .cpu().tolist())
    return out


def score(model,cases,vocab,n_ctx,device,batch_size):
    rows={}
    for family,part in cases.items():
        before=[(doc,site,target) for doc,site,other,other_site,target in part]
        after=[(other,other_site,target)
               for doc,site,other,other_site,target in part]
        p0=_probabilities(model,before,vocab,n_ctx,device,batch_size)
        p1=_probabilities(model,after,vocab,n_ctx,device,batch_size)
        delta=[b-a for a,b in zip(p0,p1,strict=True)]
        rows[family]=dict(n=len(part),mean_model_response=sum(delta)/len(delta),
                          mean_abs_response=sum(abs(v) for v in delta)/len(delta),
                          mean_target_probability_before=sum(p0)/len(p0),
                          mean_target_probability_after=sum(p1)/len(p1),
                          oracle_response=(0. if family.startswith('source_nuisance') else 1.))
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--seed',type=int,default=341991)
    p.add_argument('--streams',type=int,default=1000)
    p.add_argument('--max-cases',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--style-entropy',type=float,
                   help='Optional evaluation surface entropy override')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    vocab=saved['vocabulary'];n_ctx=saved['n_ctx']
    kwargs=dict(saved['generator_kwargs'])
    if a.style_entropy is not None:
        kwargs['style_entropy']=a.style_entropy
    cases,skipped=collect(a.streams,a.seed,kwargs,n_ctx,a.max_cases)
    model=DeepModel(len(vocab),saved['model_kwargs']['d_model'],
                    saved['model_kwargs']['n_head'],
                    SPECS[f"block{saved['depth']}"],n_ctx,
                    norm=saved['model_kwargs']['norm'],
                    residual=saved['model_kwargs']['residual'],
                    attention=saved['model_kwargs']['attention']).to(a.device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    rows=score(model,cases,vocab,n_ctx,a.device,a.batch_size)
    out=dict(config={**vars(a),'checkpoint':str(a.checkpoint),
                     'output':str(a.output),'generator_kwargs':kwargs},
             skipped=skipped,by_edit=rows,
             caveat='Supported re-rendered upstream edits; probability '
                    'response is not circuit recovery. Source nuisance edits '
                    'change an unrelated visible report while preserving '
                    'the correct source. One model seed.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    for family,row in rows.items():
        print(family,row['n'],round(row['mean_model_response'],3),flush=True)


if __name__=='__main__':main()
