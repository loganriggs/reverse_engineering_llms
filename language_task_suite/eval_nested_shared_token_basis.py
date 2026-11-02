"""Shared token-pair score bases: weight-only vs distribution-weighted.

One query-token and one key-token basis per head are shared across all
relative lags. The approximation is a two-sided projection of the exact
first-layer score tables; final RMSNorm and LM head remain native.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_multi_genre_token_tensor import score_tables
from eval_nested_source_sufficiency import verb_cases
from eval_surface_mixture_v2_nested_edits import build_pairs
from surface_mixture_v2_nested_varlen import SurfaceMixtureV2NestedVarlen
from surface_mixture_v2_varied import SurfaceMixtureV2Varied


def token_marginal(generator,nstreams,length,target_vocab,smoothing,device):
    counts=Counter()
    for _ in range(nstreams):
        ids,_,_=generator.sample_sequence(length)
        counts.update(generator.id2token[i] for i in ids)
    raw=torch.tensor([counts[token]+smoothing for token in target_vocab],
                     dtype=torch.float32,device=device)
    return raw/raw.sum(),dict(total=sum(counts.values()),
        observed=sum(counts[token]>0 for token in target_vocab),
        bracket_counts={token:counts[token] for token in ('(', '[', ')', ']')})


@torch.no_grad()
def shared_bases(tables,probability):
    """Eigenspaces of both shared token-mode Grams, per head."""
    scale=probability.sqrt()
    weighted=tables.float()*scale[None,None,:,None]*scale[None,None,None,:]
    query_gram=torch.einsum('hlij,hlkj->hik',weighted,weighted)
    key_gram=torch.einsum('hlij,hlik->hjk',weighted,weighted)
    _,query=torch.linalg.eigh(query_gram)
    _,key=torch.linalg.eigh(key_gram)
    return query.flip(-1),key.flip(-1)


@torch.no_grad()
def project_tables(tables,probability,bases,rank):
    scale=probability.sqrt()
    weighted=tables.float()*scale[None,None,:,None]*scale[None,None,None,:]
    query,key=bases
    u,v=query[:,:,:rank],key[:,:,:rank]
    core=torch.einsum('hqi,hlqk,hkj->hlij',u,weighted,v)
    reconstructed=torch.einsum('hqi,hlij,hkj->hlqk',u,core,v)
    return reconstructed/scale[None,None,:,None]/scale[None,None,None,:]


@torch.no_grad()
def projection_check(device):
    generator=torch.Generator(device=device).manual_seed(1729)
    x=torch.randn((2,3,7,7),device=device,generator=generator)
    probability=torch.rand((7,),device=device,generator=generator)
    probability=probability/probability.sum()
    bases=shared_bases(x,probability)
    scale=probability.sqrt()
    errors=[]
    for rank in (1,2,3,7):
        approximation=project_tables(x,probability,bases,rank)
        weighted_error=((x-approximation)*scale[None,None,:,None]*
                        scale[None,None,None,:]).square().sum()
        errors.append(float(weighted_error))
    full_error=float((x-project_tables(x,probability,bases,7)).abs().max())
    assert full_error<1e-5 and all(a>=b-1e-6 for a,b in
                                    zip(errors,errors[1:]))
    return dict(full_rank_max_abs=full_error,weighted_errors=errors)


@torch.no_grad()
def last_logits(model,ids,tables,downstream):
    idx=torch.tensor(ids,dtype=torch.long,device=tables.device)
    length=len(ids)
    lags=torch.arange(length-1,-1,-1,device=tables.device)
    scores=tables[:,lags,int(idx[-1]),idx]
    values=downstream[:,idx]
    attention=torch.einsum('ht,htd->d',scores,values)
    residual=model.embed(idx[-1])+attention
    if model.use_rmsnorm or model.use_final_norm:
        residual=model.final_norm(residual)
    return model.lm_head(residual)


def kl_to(p,logits):
    logp=F.log_softmax(logits.float(),dim=-1)
    return float((p*(p.clamp_min(1e-30).log()-logp)).sum())


@torch.no_grad()
def evaluate(model,tables,downstream,triples,verbs,native):
    rows=defaultdict(list)
    for i,(pair,control) in enumerate(triples):
        ids=(pair['original'],pair['edited'],control['edited'])
        logits=[last_logits(model,x,tables,downstream) for x in ids]
        p=[F.softmax(z.float(),dim=-1) for z in logits]
        oracle=torch.zeros_like(p[0])
        oracle[pair['answer_after']]+=1
        oracle[pair['answer_before']]-=1
        row=dict(teacher_kl=float(np.mean([kl_to(ref,z) for ref,z in
            zip(native['triples'][i],logits)])),
            oracle_edit_l1=float(((p[1]-p[0])-oracle).abs().sum()),
            teacher_edit_l1=float(((p[1]-p[0])-(native['triples'][i][1]-
                native['triples'][i][0])).abs().sum()),
            control_edit_l1=float((p[2]-p[0]).abs().sum()),
            edited_correct=float(p[1].argmax()==pair['answer_after']))
        rows[pair['kind']].append(row)
        rows[pair['kind']+f"@lag{pair['opener_distance']}"].append(row)
    edits={kind:dict(n=len(items),**{key:float(np.mean([r[key] for r in items]))
           for key in items[0]}) for kind,items in rows.items()}
    verb_kl=[]
    verb_teacher_kl=[]
    for i,(prefix,oracle_dict) in enumerate(verbs):
        logits=last_logits(model,prefix,tables,downstream)
        oracle=torch.zeros_like(logits)
        for token,mass in oracle_dict.items():
            oracle[token]=mass
        verb_kl.append(kl_to(oracle,logits))
        verb_teacher_kl.append(kl_to(native['verbs'][i],logits))
    return dict(edits=edits,verbs=dict(n=len(verbs),
                oracle_kl=float(np.mean(verb_kl)),
                teacher_kl=float(np.mean(verb_teacher_kl))))


@torch.no_grad()
def native_probabilities(model,triples,verbs,device):
    all_prefix=[]
    for pair,control in triples:
        all_prefix.extend((pair['original'],pair['edited'],control['edited']))
    all_prefix.extend(prefix for prefix,_ in verbs)
    output=[]
    for ids in all_prefix:
        idx=torch.tensor([ids],dtype=torch.long,device=device)
        output.append(F.softmax(model(idx)[0][0,-1].float(),dim=-1))
    split=3*len(triples)
    return dict(triples=[output[i:i+3] for i in range(0,split,3)],
                verbs=output[split:])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--nfit',type=int,default=2048)
    p.add_argument('--nedit-streams',type=int,default=256)
    p.add_argument('--verb-streams',type=int,default=128)
    p.add_argument('--verb-limit',type=int,default=256)
    p.add_argument('--length',type=int,default=128)
    p.add_argument('--ranks',default='4,8,16,32,64')
    p.add_argument('--smoothing',type=float,default=16.)
    p.add_argument('--fit-seed',type=int,default=20261212)
    p.add_argument('--edit-seed',type=int,default=20261213)
    p.add_argument('--verb-seed',type=int,default=20261214)
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,args.source)
    from model import BilinearGPT,Config
    torch.set_num_threads(4)
    check=projection_check(args.device)
    meta=json.loads((args.model_dir/'results.json').read_text())
    record=next(row for row in meta['records'] if row['depth']==1)
    model=BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(args.model_dir/'model_depth1.pt',
                                     map_location='cpu',weights_only=True))
    model=model.to(args.device).eval()
    model.blocks[0].attn.rotary.inv_freq=(
        model.blocks[0].attn.rotary.inv_freq.to(args.device))
    nested_fit=SurfaceMixtureV2NestedVarlen(seed=args.fit_seed,
                                            **meta['generator_kwargs'])
    base_fit=SurfaceMixtureV2Varied(seed=args.fit_seed,
                                    parenthetical_mass=0.)
    vocab=nested_fit.vocab
    assert len(vocab)==model.config.vocab_size
    uniform=torch.full((len(vocab),),1/len(vocab),device=args.device)
    nested_weight,nested_stats=token_marginal(nested_fit,args.nfit,
        args.length,vocab,args.smoothing,args.device)
    base_weight,base_stats=token_marginal(base_fit,args.nfit,
        args.length,vocab,args.smoothing,args.device)
    weights={'uniform':uniform,'nested_unlabeled':nested_weight,
             'no_bracket_fit':base_weight}
    tables,downstream=score_tables(model,args.length)
    gen=SurfaceMixtureV2NestedVarlen(seed=args.edit_seed,
                                     **meta['generator_kwargs'])
    pairs=build_pairs(gen,args.nedit_streams,args.length)
    triples=[]
    for i in range(0,len(pairs),2):
        relevant,control=pairs[i:i+2]
        assert 'relevant' in relevant['kind'] and 'control' in control['kind']
        assert relevant['original']==control['original']
        triples.append((relevant,control))
    verb_gen=SurfaceMixtureV2NestedVarlen(seed=args.verb_seed,
                                          **meta['generator_kwargs'])
    verbs=verb_cases(verb_gen,args.verb_streams,args.length,args.verb_limit)
    native=native_probabilities(model,triples,verbs,args.device)
    result=dict(config={**vars(args),'model_dir':str(args.model_dir),
                        'output':str(args.output)},
                projection_check=check,
                fit=dict(nested=nested_stats,no_bracket=base_stats,
                         marginal='smoothed single-token frequency'),
                evaluation=dict(n_edit_pairs=len(triples),n_verbs=len(verbs),
                    edit_counts=dict(Counter(p['kind'] for p,_ in triples))),
                rows=[])
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    full=evaluate(model,tables,downstream,triples,verbs,native)
    # Exact weight-derived token score table has only bfloat16 RoPE phase
    # rounding error from its relative-lag representation.
    result['full_table']=full
    assert full['verbs']['teacher_kl']<1e-4
    assert max(row['teacher_kl'] for row in full['edits'].values())<1e-4
    result['native_verb_kl']=float(np.mean([
        sum(mass*(np.log(mass)-np.log(max(float(ref[token]),1e-30)))
            for token,mass in oracle.items())
        for (_,oracle),ref in zip(verbs,native['verbs'])]))
    save()
    ranks=[int(x) for x in args.ranks.split(',')]
    for name,prob in weights.items():
        bases=shared_bases(tables,prob)
        for rank in ranks:
            approximate=project_tables(tables,prob,bases,rank)
            metrics=evaluate(model,approximate,downstream,triples,verbs,native)
            result['rows'].append(dict(method=name,rank=rank,**metrics))
            print(name,rank,'inner edit L1',round(metrics['edits'][
                'inner_relevant']['oracle_edit_l1'],3),'outer',round(
                metrics['edits']['outer_relevant']['oracle_edit_l1'],3),
                'verb KL',round(metrics['verbs']['oracle_kl'],3),flush=True)
            save()
    result['caveat']='Shared query/key token bases are per head and fixed '
    result['caveat']+='across relative lags. Data-aware fits use unlabeled '
    result['caveat']+='generator token frequencies, not task labels. The '
    result['caveat']+='no-bracket arm holds bracket examples out of basis '
    result['caveat']+='weight fitting, not out of model training. Relative-'
    result['caveat']+='lag tables approximate bfloat16 RoPE rounding.'
    save()


if __name__=='__main__':
    main()
