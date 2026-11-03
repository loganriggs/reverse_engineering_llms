"""Decompose L0H3 pre-opener edges and read L1 opener scores after removal."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_v10_bracket_opener_edges import pick_events
from surface_mixture_v10_linked_events import SurfaceMixtureV10LinkedEvents
from train_surface_mixture import make_eval


def clause_start(gen, x, opener):
    period=gen.token2id['.']
    before=torch.where(x[:opener]==period)[0]
    return int(before[-1])+1 if len(before) else 0


@torch.no_grad()
def run_condition(model,gen,data,events,kind,device):
    from model import apply_rotary_emb
    rows=[]
    l0=model.blocks[0].attn
    l1=model.blocks[1].attn
    head=3; D=l0.head_dim
    for start in range(0,len(events),32):
        part=events[start:start+32]
        b=torch.tensor([e[0] for e in part],device=device)
        pos=torch.tensor([e[1] for e in part],device=device)
        opener=torch.tensor([e[2] for e in part],device=device)
        clause=torch.tensor([clause_start(gen,data['x'][e[0]],e[2])
                             for e in part],device=device)
        xx=data['x'][b]
        target=data['q'][b,pos]
        edge=[]; edge_norm=[]; residual_norm=[]; scores=[]
        def l0_capture(module,inputs):
            z=inputs[0]
            B,T,C=z.shape; H=l0.n_head
            def projection(linear):return linear(z).reshape(B,T,H,D)
            q1,k1,q2,k2,v=[projection(w) for w in
                            (l0.q1,l0.k1,l0.q2,l0.k2,l0.v)]
            cos,sin=l0.rotary(q1)
            q1,k1=apply_rotary_emb(q1,cos,sin),apply_rotary_emb(k1,cos,sin)
            q2,k2=apply_rotary_emb(q2,cos,sin),apply_rotary_emb(k2,cos,sin)
            assert not l0.qk_norm
            ii=torch.arange(B,device=z.device)
            p1=torch.einsum('bd,btd->bt',q1[ii,pos,head],k1[:,:,head])/D
            p2=torch.einsum('bd,btd->bt',q2[ii,pos,head],k2[:,:,head])/D
            key=torch.arange(T,device=z.device)[None,:]
            if kind.startswith('current_clause'):
                mask=(key>=clause[:,None])&(key<opener[:,None])
            elif kind.startswith('prior_clauses'):
                mask=key<clause[:,None]
            else:
                mask=key<opener[:,None]
            raw=torch.einsum('bt,btd->bd',p1*p2*mask,v[:,:,head])
            if kind.endswith('orthogonal'):
                # Match the actual residual write norm after W_O while
                # changing the direction in the head's coordinate space.
                rng=torch.Generator().manual_seed(812001+start)
                noise=torch.randn(raw.shape,generator=rng).to(raw.device)
                noise=noise-(noise*raw).sum(-1,keepdim=True)*raw/(
                    raw.square().sum(-1,keepdim=True)+1e-12)
                W=l0.out.weight[:,head*D:(head+1)*D]
                target_norm=(raw@W.T).norm(dim=-1,keepdim=True)
                noise_norm=(noise@W.T).norm(dim=-1,keepdim=True)
                raw=noise*target_norm/(noise_norm+1e-12)
            edge.append(raw)
            edge_norm.extend(raw.norm(dim=-1).cpu().tolist())
            W=l0.out.weight[:,head*D:(head+1)*D]
            residual_norm.extend((raw@W.T).norm(dim=-1).cpu().tolist())
        def l0_remove(module,inputs):
            y=inputs[0].clone()
            ii=torch.arange(len(y),device=y.device)
            y[ii,pos,head*D:(head+1)*D]-=edge.pop()
            return (y,)
        def l1_read(module,inputs):
            z=inputs[0]
            B,T,C=z.shape; H=l1.n_head; d=l1.head_dim
            def projection(linear):return linear(z).reshape(B,T,H,d)
            q1,k1,q2,k2=[projection(w) for w in
                          (l1.q1,l1.k1,l1.q2,l1.k2)]
            cos,sin=l1.rotary(q1)
            q1,k1=apply_rotary_emb(q1,cos,sin),apply_rotary_emb(k1,cos,sin)
            q2,k2=apply_rotary_emb(q2,cos,sin),apply_rotary_emb(k2,cos,sin)
            ii=torch.arange(B,device=z.device)
            first=(q1[ii,pos,:2]*k1[ii,opener,:2]).sum(-1)/d
            second=(q2[ii,pos,:2]*k2[ii,opener,:2]).sum(-1)/d
            scores.extend((first*second).cpu().tolist())
        hooks=[]
        if kind!='baseline':
            hooks.extend((l0.register_forward_pre_hook(l0_capture),
                          l0.out.register_forward_pre_hook(l0_remove)))
        hooks.append(l1.register_forward_pre_hook(l1_read))
        try:
            logits=model(xx)[0][torch.arange(len(xx),device=device),pos].float()
        finally:
            for h in hooks:h.remove()
        logp=F.log_softmax(logits,-1)
        entropy=-(target*target.clamp_min(1e-30).log()).sum(-1)
        kl=-(target*logp).sum(-1)-entropy
        close=torch.tensor([gen.token2id[')' if int(data['x'][e[0],e[2]])
                                ==gen.token2id['('] else ']'] for e in part],
                           device=device)
        prob=logp.exp().gather(-1,close[:,None]).squeeze(-1)
        rows.extend(dict(kl=float(kl[i]),closer_probability=float(prob[i]),
                         opener_scores=scores[i])
                    for i in range(len(part)))
    return dict(n=len(rows),kl=sum(r['kl'] for r in rows)/len(rows),
                closer_probability=sum(r['closer_probability'] for r in rows)/len(rows),
                mean_opener_scores=[sum(r['opener_scores'][h] for r in rows)/len(rows)
                                    for h in (0,1)],
                mean_head_vector_norm=(sum(edge_norm)/len(edge_norm) if edge_norm else None),
                mean_residual_write_norm=(sum(residual_norm)/len(residual_norm)
                                          if residual_norm else None),
                per_event=rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--streams',type=int,default=256)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--seed',type=int,default=99007)
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
    kinds=('baseline','pre_all','current_clause','prior_clauses',
           'pre_all_orthogonal','current_clause_orthogonal',
           'prior_clauses_orthogonal')
    scores={kind:run_condition(model,gen,data,events,kind,a.device)
            for kind in kinds}
    base=scores['baseline']
    summary={kind:dict(n=s['n'],delta_kl=s['kl']-base['kl'],
           closer_probability_drop=base['closer_probability']-s['closer_probability'],
           opener_score_change=[s['mean_opener_scores'][h]-
                                base['mean_opener_scores'][h] for h in (0,1)],
           mean_residual_write_norm=s['mean_residual_write_norm'])
           for kind,s in scores.items()}
    result=dict(config=vars(a),baseline=base,summary=summary,conditions=scores,
                caveat='Regional edge subtraction is exact for the selected '
                       'head and query. Orthogonal controls have matched '
                       'post-W_O residual norm but are off-manifold random '
                       'directions; they test magnitude specificity only. '
                       'Layer-1 opener scores are the signed QK1×QK2 '
                       'pattern, not output-logit contributions.')
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('baseline',base['kl'],base['mean_opener_scores'])
    for kind,row in summary.items():print(kind,row)


if __name__=='__main__':main()
