"""End-to-end selected-edge ablations for learned joint-v3 skip rules."""

import argparse
import json
from pathlib import Path

import torch

from train_original_v3_shared_bracket import make_batch
from original_v3_shared_bracket import (LanguageV3SharedBracket,
                                         bracket_next_distribution)
from plot_original_v3_sparse_rule_graphs import events, exact_edge_contrasts
from model import BilinearGPT, Config, apply_rotary_emb


def oracle_for_record(gen, row, record, rule):
    _,q,k,target,_,opener,_=record
    if rule=='skip_bigram':
        return {gen.id2token[target]:1.}
    content=[gen.id2token[int(v)] for v in row[k+1:q+1]]
    return bracket_next_distribution(gen,opener,content)


@torch.no_grad()
def ablated_logits(model, x, query, key, head, layer=1):
    attn=model.blocks[layer].attn
    D=attn.head_dim
    assert not attn.qk_norm
    assert (key<=query).all()
    def remove_edge(module, inputs, output):
        residual=inputs[0]
        batch=torch.arange(len(residual),device=residual.device)
        B,T,C=residual.shape
        def shaped(linear):
            return linear(residual).reshape(B,T,attn.n_head,D)
        q1,k1,q2,k2,v=[shaped(w) for w in
                        (attn.q1,attn.k1,attn.q2,attn.k2,attn.v)]
        cos,sin=attn.rotary(q1)
        q1,k1=apply_rotary_emb(q1,cos,sin),apply_rotary_emb(k1,cos,sin)
        q2,k2=apply_rotary_emb(q2,cos,sin),apply_rotary_emb(k2,cos,sin)
        score1=(q1[batch,query,head]*k1[batch,key,head]).sum(-1)/D
        score2=(q2[batch,query,head]*k2[batch,key,head]).sum(-1)/D
        value=v[batch,key,head]
        contribution=(score1*score2)[:,None]*(
            value @ attn.out.weight[:,head*D:(head+1)*D].T)
        changed=output.clone()
        changed[batch,query]-=contribution
        return changed
    hook=attn.register_forward_hook(remove_edge)
    try:
        return model(x)[0]
    finally:
        hook.remove()


@torch.no_grad()
def evaluate(model,gen,x,records,rule,head,control,batch_size,device):
    vocab=gen.vocab
    target_ids=[]
    oracle=[]
    all_native=[]
    all_changed=[]
    for start in range(0,len(records),batch_size):
        subset=records[start:start+batch_size]
        inputs=torch.stack([x[b] for b,*_ in subset]).to(device)
        q=torch.tensor([row[1] for row in subset],device=device)
        k=torch.tensor([row[2]+(control=='next_token') for row in subset],device=device)
        assert (k<=q).all()
        native=model(inputs)[0][torch.arange(len(subset),device=device),q].float()
        changed=ablated_logits(model,inputs,q,k,head)[
            torch.arange(len(subset),device=device),q].float()
        all_native.append(native)
        all_changed.append(changed)
        for record in subset:
            b=record[0]
            target_ids.append(record[3])
            dist=oracle_for_record(gen,x[b],record,rule)
            oracle.append([dist.get(token,0.) for token in vocab])
    native=torch.cat(all_native)
    changed=torch.cat(all_changed)
    qdist=torch.tensor(oracle,dtype=torch.float32,device=device)
    target=torch.tensor(target_ids,device=device)
    p0=native.softmax(-1)
    p1=changed.softmax(-1)
    l0=native.log_softmax(-1)
    l1=changed.log_softmax(-1)
    idx=torch.arange(len(target),device=device)
    oracle_log=qdist.clamp_min(1e-30).log()
    kl0=(qdist*(oracle_log-l0)).sum(-1)
    kl1=(qdist*(oracle_log-l1)).sum(-1)
    target_delta=p1[idx,target]-p0[idx,target]
    return dict(n=len(records),mean_native_target_probability=float(p0[idx,target].mean()),
                mean_ablated_target_probability=float(p1[idx,target].mean()),
                mean_target_probability_change=float(target_delta.mean()),
                fraction_target_probability_decreased=float((target_delta<0).float().mean()),
                native_oracle_kl=float(kl0.mean()),ablated_oracle_kl=float(kl1.mean()),
                oracle_kl_change=float((kl1-kl0).mean()),
                mean_native_to_ablated_kl=float((p0*(l0-l1)).sum(-1).mean()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sparse-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=98001)
    p.add_argument('--streams',type=int,default=512)
    p.add_argument('--max-per-rule',type=int,default=256)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    torch.set_num_threads(1)
    device=torch.device(a.device)
    gen=LanguageV3SharedBracket(seed=a.seed)
    x,y,labels,_=make_batch(gen,a.streams,129,device)
    found=events(x,y,labels,gen,a.max_per_rule)
    assert all(len(rows)==a.max_per_rule for rows in found.values())
    checkpoint_paths={'dense':a.sparse_run/'dense.pt',
                      'sparse60':a.sparse_run/'final_no_l1.pt'}
    result={}
    identity_errors={}
    for checkpoint_name,path in checkpoint_paths.items():
        payload=torch.load(path,map_location=device,weights_only=True)
        model=BilinearGPT(Config(**payload['config'])).to(device)
        model.load_state_dict(payload['model'])
        model.eval()
        result[checkpoint_name]={}
        identity_errors[checkpoint_name]={}
        for rule,records in found.items():
            result[checkpoint_name][rule]={}
            b,q,k,target,alternatives,*rest=records[0]
            one=x[b:b+1]
            local,checks=exact_edge_contrasts(model,one,
                {rule:[(0,q,k,target,alternatives,*rest)]})
            baseline=model(one)[0][0,q]
            errors=[]
            for head in range(4):
                edited=ablated_logits(model,one,torch.tensor([q],device=device),
                                      torch.tensor([k],device=device),head)[0,q]
                observed=((baseline[target]-baseline[alternatives].mean())-
                          (edited[target]-edited[alternatives].mean()))
                expected=local[rule][1][0,head]
                errors.append(abs(float(observed.detach())-float(expected)))
            identity_errors[checkpoint_name][rule]=max(errors)
            assert max(errors)<1e-4 and max(checks)<1e-4
            for head in range(4):
                for control in ('anchor','next_token'):
                    metric=evaluate(model,gen,x,records,rule,head,control,32,device)
                    result[checkpoint_name][rule][f'H{head}_{control}']=metric
                    print(checkpoint_name,rule,head,control,
                          'Δoracle KL',round(metric['oracle_kl_change'],4),
                          'Δtarget p',round(metric['mean_target_probability_change'],4),
                          flush=True)
    output=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                results=result,edge_identity_max_error=identity_errors,
                caveat='Only the specified layer-1 head and one query-key '
                       'edge are removed. next_token control removes the '
                       'first content token after the anchor. Events are '
                       'selected from fresh sampled rule labels; bracket '
                       'events condition on observed closure, but KL uses '
                       'the full visible-prefix oracle. Two pruning runs '
                       'share the same dense teacher. These effects identify '
                       'edge necessity on the sampled panel, not uniqueness '
                       'or complete circuit sufficiency.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
