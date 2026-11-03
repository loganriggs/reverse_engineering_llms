"""Remove only a selected key→query edge from a trained bilinear head."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from surface_mixture_v10_linked_events import SurfaceMixtureV10LinkedEvents
from train_surface_mixture import make_eval


def pick_events(gen,data,limit):
    x=data['x'].cpu()
    events=[]
    for b,t in torch.nonzero(data['bracket_close'],as_tuple=False).tolist():
        words=[gen.id2token[int(i)] for i in x[b,:t+1]]
        keys=[k for k in range(t,-1,-1) if words[k] in ('(','[')
              and '.' not in words[k:t+1]]
        if keys and t-keys[0]>=2:
            events.append((b,t,keys[0],t-1))
        if len(events)>=limit:break
    return events


@torch.no_grad()
def chosen_metrics(model,gen,data,events,layer,head,key_kind,device):
    from model import apply_rotary_emb
    q=data['q']
    ce_sum=0.;ent_sum=0.;closer_sum=0.
    for start in range(0,len(events),32):
        part=events[start:start+32]
        b=torch.tensor([e[0] for e in part],device=device)
        pos=torch.tensor([e[1] for e in part],device=device)
        key=torch.tensor([e[2] if key_kind in ('opener','zero_opener_write')
                          else e[1] if key_kind=='self' else e[3]
                          for e in part],device=device)
        xx=data['x'][b]
        qq=q[b,pos]
        attn=model.blocks[layer].attn
        dim=attn.head_dim
        edge=[]
        def capture(module,inputs):
            z=inputs[0]
            B,T,C=z.shape; H=attn.n_head; D=dim
            def projection(linear):return linear(z).reshape(B,T,H,D)
            q1,k1,q2,k2,v=[projection(w) for w in
                            (attn.q1,attn.k1,attn.q2,attn.k2,attn.v)]
            cos,sin=attn.rotary(q1)
            q1,k1=apply_rotary_emb(q1,cos,sin),apply_rotary_emb(k1,cos,sin)
            q2,k2=apply_rotary_emb(q2,cos,sin),apply_rotary_emb(k2,cos,sin)
            assert not attn.qk_norm
            ii=torch.arange(B,device=z.device)
            if key_kind in ('span_after_opener','pre_opener','all_query_edges'):
                p1=torch.einsum('bd,btd->bt',q1[ii,pos,head],k1[:,:,head])/D
                p2=torch.einsum('bd,btd->bt',q2[ii,pos,head],k2[:,:,head])/D
                positions=torch.arange(T,device=z.device)[None,:]
                opener=torch.tensor([e[2] for e in part],device=z.device)[:,None]
                query=pos[:,None]
                if key_kind=='span_after_opener':
                    mask=(positions>opener)&(positions<=query)
                elif key_kind=='pre_opener':
                    mask=positions<opener
                else:
                    mask=positions<=query
                edge.append(torch.einsum('bt,btd->bd',
                            p1*p2*mask,v[:,:,head]))
            else:
                p1=(q1[ii,pos,head]*k1[ii,key,head]).sum(-1)/D
                p2=(q2[ii,pos,head]*k2[ii,key,head]).sum(-1)/D
                edge.append((p1*p2)[:,None]*v[ii,key,head])
        def remove(module,inputs):
            y=inputs[0].clone()
            ii=torch.arange(len(y),device=y.device)
            y[ii,pos,head*dim:(head+1)*dim]-=edge.pop()
            return (y,)
        def zero_write(module,inputs):
            y=inputs[0].clone()
            ii=torch.arange(len(y),device=y.device)
            where=(key if key_kind in ('zero_opener_write',
                                      'zero_last_content_write') else pos)
            y[ii,where,head*dim:(head+1)*dim]=0
            return (y,)
        hooks=[]
        if key_kind.startswith('zero_'):
            hooks=[attn.out.register_forward_pre_hook(zero_write)]
        elif key_kind!='none':
            hooks=[attn.register_forward_pre_hook(capture),
                   attn.out.register_forward_pre_hook(remove)]
        try:
            logits=model(xx)[0][torch.arange(len(xx),device=device),pos].float()
        finally:
            for h in hooks:h.remove()
        logp=F.log_softmax(logits,-1)
        ce_sum+=float(-(qq*logp).sum())
        ent_sum+=float(-(qq*qq.clamp_min(1e-30).log()).sum())
        # Only one close token is supported at each event.
        # Use generator IDs directly to avoid assuming vocabulary order.
        close_ids=torch.tensor([gen.token2id[')' if int(data['x'][bb,kk])==gen.token2id['(']
                                         else ']'] for bb,_,kk,_ in part],device=device)
        closer_sum+=float(logp.exp().gather(-1,close_ids[:,None]).sum())
    return dict(n=len(events),kl=(ce_sum-ent_sum)/len(events),
                closer_probability=closer_sum/len(events))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--streams',type=int,default=256)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--seed',type=int,default=99005)
    p.add_argument('--limit',type=int,default=128)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0,a.source)
    from model import BilinearGPT,Config
    run=json.loads((Path(a.run_dir)/'results.json').read_text())
    record=run['records'][0]
    model=BilinearGPT(Config(**record['model_config'])).to(a.device)
    model.load_state_dict(torch.load(Path(a.run_dir)/f"model_depth{record['depth']}.pt",
                                     map_location=a.device,weights_only=True))
    model.eval()
    gen=SurfaceMixtureV10LinkedEvents(seed=a.seed,**run['generator_kwargs'])
    data=make_eval(gen,a.streams,a.length,a.device)
    events=pick_events(gen,data,a.limit)
    assert events
    baseline=chosen_metrics(model,gen,data,events,0,0,'none',a.device)
    rows=[]
    for layer,head in ((0,3),(1,0),(1,1),(0,0),(1,2)):
        for kind in ('opener','last_content'):
            metrics=chosen_metrics(model,gen,data,events,layer,head,kind,a.device)
            rows.append(dict(layer=layer,head=head,key=kind,metrics=metrics,
                             delta_kl=metrics['kl']-baseline['kl'],
                             closer_probability_drop=(baseline['closer_probability']-
                                                     metrics['closer_probability'])))
    for kind in ('zero_opener_write','zero_last_content_write','zero_query_write'):
        metrics=chosen_metrics(model,gen,data,events,0,3,kind,a.device)
        rows.append(dict(layer=0,head=3,key=kind,metrics=metrics,
                         delta_kl=metrics['kl']-baseline['kl'],
                         closer_probability_drop=(baseline['closer_probability']-
                                                 metrics['closer_probability'])))
    for kind in ('self','span_after_opener','pre_opener','all_query_edges'):
        metrics=chosen_metrics(model,gen,data,events,0,3,kind,a.device)
        rows.append(dict(layer=0,head=3,key=kind,metrics=metrics,
                         delta_kl=metrics['kl']-baseline['kl'],
                         closer_probability_drop=(baseline['closer_probability']-
                                                 metrics['closer_probability'])))
    lookup={row['key']:row['metrics'] for row in rows if row['layer']==0
            and row['head']==3}
    assert abs(lookup['all_query_edges']['kl']-
               lookup['zero_query_write']['kl'])<1e-5
    assert abs(lookup['all_query_edges']['closer_probability']-
               lookup['zero_query_write']['closer_probability'])<1e-5
    result=dict(config=vars(a),baseline=baseline,rows=rows,
                caveat='Exact selected edge subtraction before W_O for every '
                       'chosen bracket-close query. This is a causal edge '
                       'intervention on a sampled prefix; it does not make '
                       'the full head or a unique minimal circuit. Last-content '
                       'key is a local-position control, not magnitude matched.')
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('baseline',baseline)
    for row in rows:
        print(f"L{row['layer']}H{row['head']}",row['key'],
              round(row['delta_kl'],4),round(row['closer_probability_drop'],4))


if __name__=='__main__':main()
