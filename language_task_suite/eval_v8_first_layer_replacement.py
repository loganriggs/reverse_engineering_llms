"""Insert fixed-offset token-score reductions into the full v8 model.

The layer-0 attention head score at every causal pair with a chosen offset
is replaced by a table learned at one absolute position pair. All other
attention contributions and weights remain intact. Score full-model KL and
exact-support section edits on fresh generated streams.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_surface_mixture_v7_scope import build as scope_build
from surface_mixture_v8 import SurfaceMixtureV8
from train_surface_mixture import make_eval


def tables_for_methods(score, factors, ranks):
    result={'exact_base_table':score,'zero_offset':torch.zeros_like(score)}
    for rank in ranks:
        for method in ('score_svd','output_svd','data_prior0.1'):
            heads=[]
            for head in range(score.shape[0]):
                left,right=factors[f'head{head}_rank{rank}'][method]
                heads.append(left@right.T)
            result[f'{method}_rank{rank}']=torch.stack(heads)
    return result


@torch.no_grad()
def patch_delta(model, ids, score_table, ov, offset, apply_rotary_emb):
    """Replacement minus actual head contributions, after W_O."""
    block=model.blocks[0]
    attn=block.attn
    x=block.norm_attn(model.embed(ids))
    batch,length,_=x.shape
    if offset>=length:
        return torch.zeros_like(x),dict(n_pairs=0,relative_contribution_l2=0.)
    heads,head_dim=attn.n_head,attn.head_dim
    q1=attn.q1(x).view(batch,length,heads,head_dim)
    k1=attn.k1(x).view(batch,length,heads,head_dim)
    q2=attn.q2(x).view(batch,length,heads,head_dim)
    k2=attn.k2(x).view(batch,length,heads,head_dim)
    cos,sin=attn.rotary(q1)
    q1,k1=apply_rotary_emb(q1,cos,sin),apply_rotary_emb(k1,cos,sin)
    q2,k2=apply_rotary_emb(q2,cos,sin),apply_rotary_emb(k2,cos,sin)
    if attn.qk_norm:
        q1,k1,q2,k2=(F.normalize(z,dim=-1) for z in (q1,k1,q2,k2))
    first=(q1[:,offset:]*k1[:,:-offset]).sum(-1)/head_dim
    second=(q2[:,offset:]*k2[:,:-offset]).sum(-1)/head_dim
    actual=first*second
    query_tokens=ids[:,offset:]
    key_tokens=ids[:,:-offset]
    replacement=score_table[:,query_tokens,key_tokens].permute(1,2,0)
    values=ov[:,key_tokens].permute(1,2,0,3)
    native_contribution=(actual[...,None]*values).sum(2)
    replacement_contribution=(replacement[...,None]*values).sum(2)
    change=replacement_contribution-native_contribution
    delta=torch.zeros_like(x)
    delta[:,offset:]=change
    relative=float((change.square().sum()/native_contribution.square().sum()
                    .clamp_min(1e-30)).sqrt())
    return delta,dict(n_pairs=batch*(length-offset),
                      relative_contribution_l2=relative,
                      mean_change_l2=float(change.square().sum(-1).sqrt().mean()))


@torch.no_grad()
def logits(model, ids, score_table, ov, offset, apply_rotary_emb):
    if score_table is None:
        return model(ids)[0],dict(n_pairs=0,relative_contribution_l2=0.)
    delta,local=patch_delta(model,ids,score_table,ov,offset,apply_rotary_emb)
    hook=model.blocks[0].attn.out.register_forward_hook(
        lambda _module,_inputs,output:output+delta)
    try:
        result=model(ids)[0]
    finally:
        hook.remove()
    return result,local


def masked_mean(values, mask):
    count=int(mask.sum())
    return dict(n=count,mean=float(values[mask].mean()) if count else None)


@torch.no_grad()
def stream_scores(model, data, methods, ov, offset, apply_rotary_emb, batch_size):
    labels=('all','verb','obj','answer','source_answer','scope_answer')
    teacher_parts=[]
    patched_parts={name:[] for name in methods}
    local_parts={name:[] for name in methods}
    for start in range(0,len(data['x']),batch_size):
        ids=data['x'][start:start+batch_size]
        teacher,_=logits(model,ids,None,ov,offset,apply_rotary_emb)
        teacher_parts.append(teacher)
        for name,table in methods.items():
            out,local=logits(model,ids,table,ov,offset,apply_rotary_emb)
            patched_parts[name].append(out)
            local_parts[name].append(local)
    teacher=torch.cat(teacher_parts)
    p_teacher=teacher.softmax(-1)
    log_teacher=teacher.log_softmax(-1)
    masks={name:(torch.ones_like(data['verb']) if name=='all'
                 else data[name]) for name in labels}
    out={}
    for name in methods:
        patched=torch.cat(patched_parts[name])
        log_patched=patched.log_softmax(-1)
        teacher_kl=(p_teacher*(log_teacher-log_patched)).sum(-1).clamp_min(0)
        oracle_kl=(data['q']*(data['q'].clamp_min(1e-30).log()-log_patched))
        oracle_kl=oracle_kl.sum(-1).clamp_min(0)
        out[name]=dict(teacher_kl={label:masked_mean(teacher_kl,mask)
                                   for label,mask in masks.items()},
                       oracle_kl={label:masked_mean(oracle_kl,mask)
                                  for label,mask in masks.items()},
                       local_relative_contribution_l2=float(
                           sum(part['relative_contribution_l2']*part['n_pairs']
                               for part in local_parts[name])/
                           max(1,sum(part['n_pairs'] for part in local_parts[name]))),
                       local_mean_change_l2=float(
                           sum(part['mean_change_l2']*part['n_pairs']
                               for part in local_parts[name])/
                           max(1,sum(part['n_pairs'] for part in local_parts[name]))))
    baseline_oracle=(data['q']*(data['q'].clamp_min(1e-30).log()-log_teacher))
    baseline_oracle=baseline_oracle.sum(-1).clamp_min(0)
    return out,{label:masked_mean(baseline_oracle,mask) for label,mask in masks.items()}


def pad_prefixes(prefixes, pad_id, device):
    length=max(len(prefix) for prefix in prefixes)
    x=torch.full((len(prefixes),length),pad_id,dtype=torch.long,device=device)
    lengths=[]
    for i,prefix in enumerate(prefixes):
        x[i,:len(prefix)]=torch.tensor(prefix,dtype=torch.long,device=device)
        lengths.append(len(prefix))
    return x,torch.tensor(lengths,device=device)-1


@torch.no_grad()
def predict_prefixes(model, prefixes, table, ov, offset, apply_rotary_emb,
                     pad_id, device, batch_size):
    results=[]
    for start in range(0,len(prefixes),batch_size):
        group=prefixes[start:start+batch_size]
        ids,positions=pad_prefixes(group,pad_id,device)
        output,_=logits(model,ids,table,ov,offset,apply_rotary_emb)
        chosen=output[torch.arange(len(group),device=device),positions]
        results.append(chosen.softmax(-1).cpu())
    return torch.cat(results)


def causal_scores(model,pairs,methods,ov,offset,apply_rotary_emb,pad_id,
                  device,batch_size,vocab):
    prefixes=[]
    index={}
    for pair in pairs:
        for field in ('original','edited'):
            key=tuple(pair[field])
            if key not in index:
                index[key]=len(prefixes)
                prefixes.append(pair[field])
    def indices(field):
        return torch.tensor([index[tuple(pair[field])] for pair in pairs])
    before_index,after_index=indices('original'),indices('edited')
    teacher=predict_prefixes(model,prefixes,None,ov,offset,apply_rotary_emb,
                             pad_id,device,batch_size)
    native=teacher[after_index]-teacher[before_index]
    oracle=torch.zeros_like(native)
    for i,pair in enumerate(pairs):
        for token,mass in pair['q_after'].items(): oracle[i,vocab[token]]+=mass
        for token,mass in pair['q_before'].items(): oracle[i,vocab[token]]-=mass
    result={}
    for name,table in methods.items():
        probs=predict_prefixes(model,prefixes,table,ov,offset,apply_rotary_emb,
                               pad_id,device,batch_size)
        effect=probs[after_index]-probs[before_index]
        groups={}
        for kind in sorted(set(pair['kind'] for pair in pairs)):
            mask=torch.tensor([pair['kind']==kind for pair in pairs])
            mean_l1=lambda x:float(x[mask].abs().sum(-1).mean())
            groups[kind]=dict(n=int(mask.sum()),
                              native_effect_l1=mean_l1(native),
                              oracle_effect_l1=mean_l1(oracle),
                              effect_to_teacher_l1=mean_l1(effect-native),
                              effect_to_oracle_l1=mean_l1(effect-oracle),
                              model_effect_l1=mean_l1(effect))
        result[name]=groups
    return result,dict(n_unique_prefixes=len(prefixes),
                       pair_counts=dict(Counter(pair['kind'] for pair in pairs)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir',required=True)
    p.add_argument('--tables',required=True)
    p.add_argument('--factors',required=True)
    p.add_argument('--ranks',default='8,16')
    p.add_argument('--stream-n',type=int,default=256)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--stream-seed',type=int,default=65001)
    p.add_argument('--causal-streams',type=int,default=400)
    p.add_argument('--causal-cap',type=int,default=100)
    p.add_argument('--causal-seed',type=int,default=66001)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--device',default='cpu')
    p.add_argument('--torch-threads',type=int,default=1)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.torch_threads<1:p.error('--torch-threads must be positive')
    torch.set_num_threads(a.torch_threads)
    sys.path.insert(0,a.source)
    from model import BilinearGPT,Config,apply_rotary_emb
    folder=Path(a.model_dir)
    saved=json.loads((folder/'results.json').read_text())
    table_meta=json.loads(Path(a.tables).with_suffix('.json').read_text())
    factor_meta=json.loads(Path(a.factors).with_suffix('.json').read_text())
    assert table_meta['config']['model_dir']==str(folder)
    assert factor_meta['config']['tables']==a.tables
    query=table_meta['config']['query_position']
    key=table_meta['config']['key_position']
    offset=query-key
    assert 0<offset<a.length
    record=saved['records'][0]
    device=torch.device(a.device)
    model=BilinearGPT(Config(**record['model_config'])).to(device)
    model.load_state_dict(torch.load(folder/f"model_depth{record['depth']}.pt",
                                     map_location=device,weights_only=True))
    model.eval()
    tables=torch.load(a.tables,map_location=device,weights_only=True)
    raw_factors=torch.load(a.factors,map_location=device,weights_only=True)
    methods=tables_for_methods(tables['score'],raw_factors,
                               [int(rank) for rank in a.ranks.split(',')])
    gen=SurfaceMixtureV8(seed=a.stream_seed,**saved['generator_kwargs'])
    data=make_eval(gen,a.stream_n,a.length,device)
    # At the table's original absolute pair, its score must match the model.
    base_ids=data['x'][:min(8,len(data['x'])),:query+1]
    base_delta,_=patch_delta(model,base_ids,tables['score'],tables['ov'],
                             offset,apply_rotary_emb)
    base_pair_max_abs=float(base_delta[:,query].abs().max())
    if base_pair_max_abs>1e-5:
        raise AssertionError(f'Exact base-position table mismatch: {base_pair_max_abs}')
    stream,baseline_oracle=stream_scores(model,data,methods,tables['ov'],
                                         offset,apply_rotary_emb,a.batch_size)
    causal_gen=SurfaceMixtureV8(seed=a.causal_seed,**saved['generator_kwargs'])
    _,pairs,attempted=scope_build(causal_gen,a.causal_streams,a.length,a.causal_cap)
    assert pairs and set(pair['kind'] for pair in pairs)=={
        'section_key','relevant_color','irrelevant_color'}
    causal,causal_info=causal_scores(model,pairs,methods,tables['ov'],offset,
                                     apply_rotary_emb,causal_gen.token2id['.'],
                                     device,a.batch_size,causal_gen.token2id)
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    result=dict(config=vars(a),offset=offset,
                base_pair_max_abs_replay_error=base_pair_max_abs,
                stream_baseline_oracle_kl=baseline_oracle,
                stream=stream,causal=causal,causal_panel=causal_info,
                attempted=dict(attempted),
                caveat='Replaces only layer-0 scores at one relative offset, '
                       'using a table fitted at one absolute position pair. '
                       'The exact-base-table control includes RoPE translation '
                       'error. Other offsets, layers and model weights are '
                       'unchanged. This is downstream fidelity for one offset, '
                       'not whole-attention compression.')
    path.write_text(json.dumps(result,indent=2)+'\n')
    for name in methods:
        print(name,'all KL',round(stream[name]['teacher_kl']['all']['mean'],7),
              'scope KL',stream[name]['teacher_kl']['scope_answer']['mean'],
              'key effect err',round(causal[name]['section_key']['effect_to_teacher_l1'],5),
              flush=True)


if __name__=='__main__':main()
