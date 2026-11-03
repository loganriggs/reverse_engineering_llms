"""Patch baseline L1 opener-edge vectors into L0H3 pre-opener-ablation runs."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_v10_bracket_opener_edges import pick_events
from surface_mixture_v10_linked_events import SurfaceMixtureV10LinkedEvents
from train_surface_mixture import make_eval


@torch.no_grad()
def evaluate_condition(model,gen,data,events,mode,device):
    from model import apply_rotary_emb
    l0=model.blocks[0].attn
    l1=model.blocks[1].attn
    total_kl=0.;total_prob=0.;total_change=0.
    for start in range(0,len(events),32):
        part=events[start:start+32]
        b=torch.tensor([e[0] for e in part],device=device)
        pos=torch.tensor([e[1] for e in part],device=device)
        opener=torch.tensor([e[2] for e in part],device=device)
        xx=data['x'][b]
        target=data['q'][b,pos]
        batch=torch.arange(len(part),device=device)

        def opener_edges(z):
            B,T,C=z.shape; H=l1.n_head; D=l1.head_dim
            def projected(linear):return linear(z).reshape(B,T,H,D)
            q1,k1,q2,k2,v=[projected(w) for w in
                            (l1.q1,l1.k1,l1.q2,l1.k2,l1.v)]
            cos,sin=l1.rotary(q1)
            q1,k1=apply_rotary_emb(q1,cos,sin),apply_rotary_emb(k1,cos,sin)
            q2,k2=apply_rotary_emb(q2,cos,sin),apply_rotary_emb(k2,cos,sin)
            p1=(q1[batch,pos,:2]*k1[batch,opener,:2]).sum(-1)/D
            p2=(q2[batch,pos,:2]*k2[batch,opener,:2]).sum(-1)/D
            return (p1*p2)[...,None]*v[batch,opener,:2]

        baseline_edges=[]
        def baseline_capture(module,inputs):
            baseline_edges.append(opener_edges(inputs[0]))
        hook=l1.register_forward_pre_hook(baseline_capture)
        try:
            baseline_logits=model(xx)[0][batch,pos].float()
        finally:hook.remove()
        assert len(baseline_edges)==1
        original=baseline_edges[0]

        if mode=='baseline':
            logits=baseline_logits
        else:
            pre_vector=[];current=[]
            def l0_capture(module,inputs):
                z=inputs[0]
                B,T,C=z.shape;H=l0.n_head;D=l0.head_dim;h=3
                def projected(linear):return linear(z).reshape(B,T,H,D)
                q1,k1,q2,k2,v=[projected(w) for w in
                                (l0.q1,l0.k1,l0.q2,l0.k2,l0.v)]
                cos,sin=l0.rotary(q1)
                q1,k1=apply_rotary_emb(q1,cos,sin),apply_rotary_emb(k1,cos,sin)
                q2,k2=apply_rotary_emb(q2,cos,sin),apply_rotary_emb(k2,cos,sin)
                p1=torch.einsum('bd,btd->bt',q1[batch,pos,h],k1[:,:,h])/D
                p2=torch.einsum('bd,btd->bt',q2[batch,pos,h],k2[:,:,h])/D
                mask=torch.arange(T,device=z.device)[None,:]<opener[:,None]
                pre_vector.append(torch.einsum('bt,btd->bd',p1*p2*mask,v[:,:,h]))
            def l0_remove(module,inputs):
                y=inputs[0].clone()
                h=3;D=l0.head_dim
                y[batch,pos,h*D:(h+1)*D]-=pre_vector.pop()
                return (y,)
            def l1_capture(module,inputs):
                current.append(opener_edges(inputs[0]))
            def l1_patch(module,inputs):
                y=inputs[0].clone()
                now=current.pop()
                D=l1.head_dim
                for h in ((0,1) if mode=='rescue_both' else
                          (0,) if mode=='rescue_h0' else
                          (1,) if mode=='rescue_h1' else ()):
                    y[batch,pos,h*D:(h+1)*D]+=original[:,h]-now[:,h]
                return (y,)
            hooks=[l0.register_forward_pre_hook(l0_capture),
                   l0.out.register_forward_pre_hook(l0_remove),
                   l1.register_forward_pre_hook(l1_capture),
                   l1.out.register_forward_pre_hook(l1_patch)]
            try:
                logits=model(xx)[0][batch,pos].float()
            finally:
                for hook in hooks:hook.remove()
        logp=F.log_softmax(logits,-1)
        entropy=-(target*target.clamp_min(1e-30).log()).sum(-1)
        total_kl+=float((-(target*logp).sum(-1)-entropy).sum())
        close=torch.tensor([gen.token2id[')' if int(data['x'][e[0],e[2]])
                                ==gen.token2id['('] else ']'] for e in part],
                           device=device)
        total_prob+=float(logp.exp().gather(-1,close[:,None]).sum())
        total_change+=float((logits-baseline_logits).norm(dim=-1).sum())
    return dict(n=len(events),kl=total_kl/len(events),
                closer_probability=total_prob/len(events),
                mean_logit_change_norm=total_change/len(events))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--streams',type=int,default=256)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--seed',type=int,default=99009)
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
    scores={mode:evaluate_condition(model,gen,data,events,mode,a.device)
            for mode in ('baseline','remove_pre','rescue_h0','rescue_h1','rescue_both')}
    base=scores['baseline']['kl'];removed=scores['remove_pre']['kl']
    rescued={mode:(removed-scores[mode]['kl'])/(removed-base)
             for mode in ('rescue_h0','rescue_h1','rescue_both')}
    result=dict(config=vars(a),scores=scores,rescue_fraction=rescued,
                caveat='Restores only the baseline L1 opener-edge vector(s) '
                       'at selected query sites after removing L0H3 pre-opener '
                       'edges. Rescue estimates causal mediation for this '
                       'intervention; it does not prove that this is the '
                       'only native circuit or that the whole DGP is learned.')
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(scores)
    print('rescue fractions',rescued)


if __name__=='__main__':main()
