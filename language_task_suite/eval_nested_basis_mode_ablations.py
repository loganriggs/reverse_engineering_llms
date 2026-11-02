"""Causal functional ablations of shared first-layer token-score modes."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from eval_multi_genre_token_tensor import score_tables
from eval_nested_shared_token_basis import (evaluate,native_probabilities,
    last_logits,project_tables,shared_bases,token_marginal)
from eval_nested_source_sufficiency import verb_cases
from eval_surface_mixture_v2_edits import build_pairs as build_verb_edit_pairs
from eval_surface_mixture_v2_nested_edits import (build_pairs,
    probabilities as native_prefix_probabilities)
from surface_mixture_v2_nested_varlen import SurfaceMixtureV2NestedVarlen


@torch.no_grad()
def mode_contribution(core,u,v,scale,head,mode,index):
    """Contribution of one weighted query or key basis axis, all lags."""
    if mode=='query':
        row=torch.einsum('lj,kj->lk',core[head,:,index,:],v[head])
        weighted=u[head,:,index][None,:,None]*row[:,None,:]
    elif mode=='key':
        column=torch.einsum('qi,li->lq',u[head],core[head,:,:,index])
        weighted=column[:,:,None]*v[head,:,index][None,None,:]
    else:
        raise ValueError(mode)
    return weighted/scale[None,:,None]/scale[None,None,:]


@torch.no_grad()
def make_components(tables,probability,rank):
    bases=shared_bases(tables,probability)
    u,v=(basis[:,:,:rank] for basis in bases)
    scale=probability.sqrt()
    weighted=tables*scale[None,None,:,None]*scale[None,None,None,:]
    core=torch.einsum('hqi,hlqk,hkj->hlij',u,weighted,v)
    full=project_tables(tables,probability,bases,rank)
    components={(head,mode,index):mode_contribution(
        core,u,v,scale,head,mode,index)
        for head in range(tables.shape[0])
        for mode in ('query','key') for index in range(rank)}
    # Query and key decompositions must independently reconstitute the
    # projected score table for each head.
    errors=[]
    for head in range(tables.shape[0]):
        for mode in ('query','key'):
            replay=sum(components[(head,mode,i)] for i in range(rank))
            errors.append(float((replay-full[head]).abs().max()))
    assert max(errors)<1e-3,max(errors)
    return full,components,max(errors)


def make_panel(model,meta,edit_seed,verb_seed,nstreams,verb_limit,length,
               device):
    gen=SurfaceMixtureV2NestedVarlen(seed=edit_seed,
                                     **meta['generator_kwargs'])
    pairs=build_pairs(gen,nstreams,length)
    triples=[]
    for i in range(0,len(pairs),2):
        relevant,control=pairs[i:i+2]
        assert 'relevant' in relevant['kind'] and 'control' in control['kind']
        assert relevant['original']==control['original']
        triples.append((relevant,control))
    verb_gen=SurfaceMixtureV2NestedVarlen(seed=verb_seed,
                                          **meta['generator_kwargs'])
    verbs=verb_cases(verb_gen,nstreams,length,verb_limit)
    native=native_probabilities(model,triples,verbs,device)
    return triples,verbs,native


def compact(metrics):
    return dict(inner=metrics['edits']['inner_relevant'],
                outer=metrics['edits']['outer_relevant'],
                verb=metrics['verbs'])


def damage(row,baseline):
    return dict(outer_edit=row['outer']['oracle_edit_l1']-
                baseline['outer']['oracle_edit_l1'],
                inner_edit=row['inner']['oracle_edit_l1']-
                baseline['inner']['oracle_edit_l1'],
                verb_kl=row['verb']['oracle_kl']-
                baseline['verb']['oracle_kl'])


def make_verb_edit_panel(model,meta,seed,nstreams,length,device):
    generator=SurfaceMixtureV2NestedVarlen(seed=seed,
                                           **meta['generator_kwargs'])
    pairs=build_verb_edit_pairs(generator,nstreams,length-1)
    before=native_prefix_probabilities(model,[p['original'] for p in pairs],
                                        device)
    after=native_prefix_probabilities(model,[p['edited'] for p in pairs],
                                       device)
    return generator,pairs,before,after


@torch.no_grad()
def evaluate_verb_edits(model,tables,downstream,panel):
    generator,pairs,teacher_before,teacher_after=panel
    grouped={}
    for i,pair in enumerate(pairs):
        before=torch.softmax(last_logits(model,pair['original'],tables,
                                          downstream).float(),dim=-1).cpu().numpy()
        after=torch.softmax(last_logits(model,pair['edited'],tables,
                                         downstream).float(),dim=-1).cpu().numpy()
        oracle_before=torch.zeros(len(generator.vocab)).numpy()
        oracle_after=torch.zeros(len(generator.vocab)).numpy()
        for token,mass in pair['q_before'].items():
            oracle_before[generator.token2id[token]]=mass
        for token,mass in pair['q_after'].items():
            oracle_after[generator.token2id[token]]=mass
        delta=after-before
        expected=oracle_after-oracle_before
        teacher=teacher_after[i]-teacher_before[i]
        grouped.setdefault(pair['kind'],[]).append((
            float(abs(delta-expected).sum()),
            float(abs(delta-teacher).sum()),
            float(abs(delta).sum()/2)))
    return {kind:dict(n=len(values),oracle_effect_l1=sum(v[0] for v in values)/len(values),
        teacher_effect_l1=sum(v[1] for v in values)/len(values),
        model_effect_tv=sum(v[2] for v in values)/len(values))
        for kind,values in grouped.items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--nfit',type=int,default=2048)
    p.add_argument('--fit-seed',type=int,default=20261212)
    p.add_argument('--dev-edit-seed',type=int,default=20261221)
    p.add_argument('--dev-verb-seed',type=int,default=20261222)
    p.add_argument('--holdout-edit-seed',type=int,default=20261223)
    p.add_argument('--holdout-verb-seed',type=int,default=20261224)
    p.add_argument('--dev-streams',type=int,default=128)
    p.add_argument('--holdout-streams',type=int,default=256)
    p.add_argument('--dev-verb-limit',type=int,default=128)
    p.add_argument('--holdout-verb-limit',type=int,default=256)
    p.add_argument('--holdout-verb-edit-seed',type=int,default=20261231)
    p.add_argument('--holdout-verb-edit-streams',type=int,default=64)
    p.add_argument('--length',type=int,default=128)
    p.add_argument('--rank',type=int,default=8)
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
    fit=SurfaceMixtureV2NestedVarlen(seed=a.fit_seed,
                                     **meta['generator_kwargs'])
    probability,fit_stats=token_marginal(fit,a.nfit,a.length,fit.vocab,
                                         a.smoothing,a.device)
    tables,downstream=score_tables(model,a.length)
    baseline,components,replay_error=make_components(tables,probability,a.rank)
    dev=make_panel(model,meta,a.dev_edit_seed,a.dev_verb_seed,
                   a.dev_streams,a.dev_verb_limit,a.length,a.device)
    holdout=make_panel(model,meta,a.holdout_edit_seed,a.holdout_verb_seed,
                       a.holdout_streams,a.holdout_verb_limit,a.length,a.device)
    dev_base=compact(evaluate(model,baseline,downstream,*dev))
    holdout_base=compact(evaluate(model,baseline,downstream,*holdout))
    result=dict(config={**vars(a),'model_dir':str(a.model_dir),
                        'output':str(a.output)},fit=fit_stats,
                component_replay_max_abs=replay_error,
                dev=dict(counts=dict(Counter(p['kind'] for p,_ in dev[0])),
                         baseline=dev_base,modes=[]),
                holdout=dict(counts=dict(Counter(p['kind'] for p,_ in
                                                 holdout[0])),
                             baseline=holdout_base,selected=[]))
    def save():
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
    for head,mode,index in components:
        altered=baseline.clone()
        altered[head]-=components[(head,mode,index)]
        metrics=compact(evaluate(model,altered,downstream,*dev))
        row=dict(head=head,mode=mode,index=index,metrics=metrics,
                 damage=damage(metrics,dev_base))
        result['dev']['modes'].append(row)
        print('dev',head,mode,index,'outer damage',round(
              row['damage']['outer_edit'],3),'verb',round(
              row['damage']['verb_kl'],3),flush=True)
        save()
    rows=result['dev']['modes']
    top_outer=max(rows,key=lambda x:x['damage']['outer_edit'])
    top_verb=max(rows,key=lambda x:x['damage']['verb_kl'])
    low_impact=min(rows,key=lambda x:(abs(x['damage']['outer_edit'])+
                 abs(x['damage']['inner_edit'])+
                 abs(x['damage']['verb_kl'])))
    selected=[]
    for row in (top_outer,top_verb,low_impact):
        key=(row['head'],row['mode'],row['index'])
        if key not in selected:
            selected.append(key)
    result['selection_rule']='Largest outer oracle-effect L1 increase; '
    result['selection_rule']+='largest verb oracle-KL increase; smallest '
    result['selection_rule']+='sum of absolute outer/inner edit and verb '
    result['selection_rule']+='KL changes. Deduplicate modes. Selection '
    result['selection_rule']+='uses development panel only.'
    verb_edit_panel=make_verb_edit_panel(model,meta,a.holdout_verb_edit_seed,
        a.holdout_verb_edit_streams,a.length,a.device)
    result['holdout']['verb_edit_counts']=dict(Counter(
        pair['kind'] for pair in verb_edit_panel[1]))
    result['holdout']['baseline_verb_edits']=evaluate_verb_edits(
        model,baseline,downstream,verb_edit_panel)
    for head,mode,index in selected:
        altered=baseline.clone()
        altered[head]-=components[(head,mode,index)]
        metrics=compact(evaluate(model,altered,downstream,*holdout))
        result['holdout']['selected'].append(dict(head=head,mode=mode,
            index=index,metrics=metrics,damage=damage(metrics,holdout_base),
            verb_edits=evaluate_verb_edits(model,altered,downstream,
                                           verb_edit_panel)))
        save()
    if top_outer['mode']=='key':
        head,index=top_outer['head'],top_outer['index']
        component=components[(head,'key',index)]
        opener_ids=[fit.token2id[token] for token in ('(', '[')]
        opener_component=torch.zeros_like(component)
        opener_component[:,:,opener_ids]=component[:,:,opener_ids]
        result['holdout']['outer_key_partition']=[]
        for name,part in [('opener_keys_only',opener_component),
                          ('other_keys_only',component-opener_component)]:
            altered=baseline.clone()
            altered[head]-=part
            metrics=compact(evaluate(model,altered,downstream,*holdout))
            result['holdout']['outer_key_partition'].append(dict(
                name=name,head=head,index=index,metrics=metrics,
                damage=damage(metrics,holdout_base),
                verb_edits=evaluate_verb_edits(model,altered,downstream,
                                               verb_edit_panel)))
            save()
    result['caveat']='Ablating one score-table token mode across all lags '
    result['caveat']+='is a functional intervention, not a native model '
    result['caveat']+='weight edit. Single-mode effects can miss redundant '
    result['caveat']+='groups. The basis was fitted to unlabeled synthetic '
    result['caveat']+='data; mode selection used a separate development '
    result['caveat']+='panel and holdout evaluation was not used to choose '
    result['caveat']+='the modes.'
    save()
    for row in result['holdout']['selected']:
        print('holdout',row['head'],row['mode'],row['index'],row['damage'])


if __name__=='__main__':
    main()
