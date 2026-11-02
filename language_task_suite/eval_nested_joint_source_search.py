"""Joint source-mask search for the one-block nested-bracket model.

Coordinate swaps minimize teacher KL either on the original prefix only or
on its valid opener edit as well. The paired arm is an in-sample feasibility
bound; only the original-only arm has an untouched relevant edit.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_nested_source_sufficiency import (logits_from_sources,
    source_vectors, verb_cases)
from eval_surface_mixture_v2_nested_edits import build_pairs
from surface_mixture_v2_nested_varlen import SurfaceMixtureV2NestedVarlen


def score_mask(pattern, k):
    chosen=pattern[0].abs().topk(min(k,pattern.shape[-1]),dim=-1).indices
    mask=torch.zeros_like(pattern[0],dtype=torch.bool)
    mask.scatter_(1,chosen,True)
    return mask


def kl(target, logits):
    logp=F.log_softmax(logits.float(),dim=-1)
    return (target*(target.clamp_min(1e-30).log()-logp)).sum(-1)


@torch.no_grad()
def swap_search(model, residual, vectors, target, initial, sweeps,
                use_pair):
    """Fixed K/head coordinate descent, selecting on original or pair."""
    mask=initial.clone()
    heads,length=mask.shape
    count=int(mask[0].sum())
    assert all(int(row.sum())==count for row in mask)
    selected=(vectors*mask[None,:,:,None]).sum(dim=(1,2))
    batch=2 if use_pair else 1
    def objective():
        x=residual[:batch]+selected[:batch]
        if model.use_rmsnorm or model.use_final_norm:
            x=model.final_norm(x)
        return float(kl(target[:batch],model.lm_head(x)).mean())
    initial_loss=objective()
    for _ in range(sweeps):
        for head in range(heads):
            for old in mask[head].nonzero().flatten().tolist():
                base=selected[:batch]-vectors[:batch,head,old]
                candidates=residual[:batch,None,:]+base[:,None,:]+\
                    vectors[:batch,head]
                if model.use_rmsnorm or model.use_final_norm:
                    candidates=model.final_norm(candidates)
                logits=model.lm_head(candidates)
                scores=kl(target[:batch,None,:],logits).mean(dim=0)
                # Another selected position cannot be selected twice.
                scores[mask[head]]=float('inf')
                scores[old]=kl(target[:batch],
                    logits[:,old]).mean()
                new=int(scores.argmin())
                if new!=old:
                    mask[head,old]=False
                    mask[head,new]=True
                    selected+=vectors[:,head,new]-vectors[:,head,old]
    assert objective()<=initial_loss+1e-4
    assert all(int(row.sum())==count for row in mask)
    return mask


def eval_pair(model, residual, vectors, target, pair, control, mask):
    logits=logits_from_sources(model,residual,vectors,
        mask[None].expand(len(residual),-1,-1))
    pred=F.softmax(logits.float(),dim=-1)
    teacher_kls=kl(target,logits)
    oracle=torch.zeros_like(pred[0])
    oracle[pair['answer_after']]+=1
    oracle[pair['answer_before']]-=1
    return dict(teacher_kl=float(teacher_kls.mean()),
                before_teacher_kl=float(teacher_kls[0]),
                edited_teacher_kl=float(teacher_kls[1]),
                control_teacher_kl=float(teacher_kls[2]),
                teacher_effect_l1=float(((pred[1]-pred[0])-
                                         (target[1]-target[0])).abs().sum()),
                oracle_effect_l1=float(((pred[1]-pred[0])-oracle).abs().sum()),
                control_effect_l1=float((pred[2]-pred[0]).abs().sum()),
                after_correct=float(pred[1].argmax()==pair['answer_after']),
                opener_selected=float(mask[:,len(pair['original'])-
                    pair['opener_distance']].float().mean()))


def summarize(rows):
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row['method'],row['kind'])].append(row)
        grouped[(row['method'],row['kind']+f"@lag{row['lag']}")].append(row)
    keys=('teacher_kl','before_teacher_kl','edited_teacher_kl',
          'control_teacher_kl','teacher_effect_l1','oracle_effect_l1',
          'control_effect_l1',
          'after_correct','opener_selected')
    return [dict(method=method,kind=kind,n=len(items),
                 **{key:float(np.mean([item[key] for item in items]))
                    for key in keys})
            for (method,kind),items in sorted(grouped.items())]


@torch.no_grad()
def evaluate_pairs(model,triples,apply_rotary_emb,k,sweeps,device):
    rows=[]
    selected_lags=defaultdict(Counter)
    replay_error=0.
    for pair,control in triples:
        assert pair['original']==control['original']
        idx=torch.tensor([pair['original'],pair['edited'],
                          control['edited']],dtype=torch.long,device=device)
        residual,pattern,vectors=source_vectors(model,idx,apply_rotary_emb)
        native=model(idx)[0][:,-1]
        replay_error=max(replay_error,float((native-
            logits_from_sources(model,residual,vectors)).abs().max()))
        target=F.softmax(native.float(),dim=-1)
        initial=score_mask(pattern,k)
        masks={
            'score_abs':initial,
            'swap_original':swap_search(model,residual,vectors,target,
                                         initial,sweeps,False),
            'swap_paired':swap_search(model,residual,vectors,target,
                                      initial,sweeps,True),
        }
        for method,mask in masks.items():
            selected_lags[(method,pair['kind'])].update(
                len(pair['original'])-int(pos)
                for pos in mask.nonzero()[:,1])
            rows.append(dict(method=method,kind=pair['kind'],
                lag=pair['opener_distance'],**eval_pair(model,residual,vectors,
                    target,pair,control,mask)))
    return dict(replay_max_abs=replay_error,rows=summarize(rows),
        selected_lags=[dict(method=method,kind=kind,
                            top_lags=counts.most_common(16))
                       for (method,kind),counts in sorted(selected_lags.items())])


@torch.no_grad()
def evaluate_verbs(model,cases,apply_rotary_emb,k,sweeps,device):
    rows=defaultdict(list)
    native_oracle=[]
    for prefix,q in cases:
        idx=torch.tensor([prefix],dtype=torch.long,device=device)
        residual,pattern,vectors=source_vectors(model,idx,apply_rotary_emb)
        target=F.softmax(model(idx)[0][:,-1].float(),dim=-1)
        oracle=torch.zeros_like(target)
        for token,mass in q.items():
            oracle[0,token]=mass
        initial=score_mask(pattern,k)
        masks={'score_abs':initial,
               'swap_original':swap_search(model,residual,vectors,target,
                                            initial,sweeps,False)}
        native_oracle.append(float(kl(oracle,model(idx)[0][:,-1]).mean()))
        for method,mask in masks.items():
            logits=logits_from_sources(model,residual,vectors,mask[None])
            rows[method].append((float(kl(oracle,logits).mean()),
                                 float(kl(target,logits).mean())))
    return dict(n=len(cases),native_oracle_kl=float(np.mean(native_oracle)),
        rows=[dict(method=method,oracle_kl=float(np.mean(vals,axis=0)[0]),
                   teacher_kl=float(np.mean(vals,axis=0)[1]))
              for method,vals in sorted(rows.items())])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--nstreams',type=int,default=128)
    p.add_argument('--verb-streams',type=int,default=64)
    p.add_argument('--verb-limit',type=int,default=128)
    p.add_argument('--length',type=int,default=128)
    p.add_argument('--seed',type=int,default=20261208)
    p.add_argument('--k',type=int,default=4)
    p.add_argument('--sweeps',type=int,default=2)
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,args.source)
    from model import BilinearGPT,Config,apply_rotary_emb
    torch.set_num_threads(4)
    meta=json.loads((args.model_dir/'results.json').read_text())
    record=next(row for row in meta['records'] if row['depth']==1)
    model=BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(args.model_dir/'model_depth1.pt',
                                     map_location='cpu',weights_only=True))
    model=model.to(args.device).eval()
    gen=SurfaceMixtureV2NestedVarlen(seed=args.seed,
                                    **meta['generator_kwargs'])
    pairs=build_pairs(gen,args.nstreams,args.length)
    triples=[]
    for i in range(0,len(pairs),2):
        relevant,control=pairs[i:i+2]
        assert 'relevant' in relevant['kind'] and 'control' in control['kind']
        triples.append((relevant,control))
    result=evaluate_pairs(model,triples,apply_rotary_emb,args.k,
                          args.sweeps,args.device)
    assert result['replay_max_abs']<5e-4
    verb_gen=SurfaceMixtureV2NestedVarlen(seed=args.seed+100,
                                         **meta['generator_kwargs'])
    result['verbs']=evaluate_verbs(model,verb_cases(verb_gen,args.verb_streams,
        args.length,args.verb_limit),apply_rotary_emb,args.k,args.sweeps,
        args.device)
    result['config']={**vars(args),'model_dir':str(args.model_dir),
                      'output':str(args.output)}
    result['pair_counts']={kind:sum(pair['kind']==kind for pair,_ in triples)
                           for kind in set(pair['kind'] for pair,_ in triples)}
    result['caveat']='Paired mask search directly sees the relevant edited '
    result['caveat']+='teacher output; it is an in-sample feasibility probe. '
    result['caveat']+='Original-only search does not see the edits, and its '
    result['caveat']+='edit metrics are counterfactual transfer.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('pairs',len(triples),'replay max abs',result['replay_max_abs'])
    for row in result['rows']:
        if row['kind']=='inner_relevant':
            print(row)


if __name__=='__main__':
    main()
