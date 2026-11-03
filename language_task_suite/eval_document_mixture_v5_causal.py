"""Fresh paired causal panels for V5 scoped-record and score tasks."""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from document_mixture_v5 import (CATEGORIES, draw_record, draw_score,
                                 realize_record, realize_score)

OLD_REPO = Path(__file__).resolve().parents[2] / 'reverse_engineer/tensor_language'
sys.path.insert(0, str(OLD_REPO))
from deep_model import DeepModel, SPECS  # noqa: E402


def load(checkpoint, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    kwargs = saved['model_kwargs']
    model = DeepModel(len(saved['vocabulary']), kwargs['d_model'],
                      kwargs['n_head'], SPECS[f"block{saved['depth']}"],
                      saved['n_ctx'], norm=kwargs['norm'],
                      residual=kwargs['residual'],
                      attention=kwargs['attention']).to(device)
    model.load_state_dict(saved['state_dict'])
    return model.eval(), saved['vocabulary'], saved['n_ctx']


@torch.no_grad()
def probs(model, vocab, n_ctx, device, doc, candidates):
    site = doc.sites[0]
    ids = [vocab[t] for t in doc.tokens[:site.index]]
    assert len(ids) <= n_ctx and all(c in vocab for c in candidates)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(x)[0,len(ids)-1]
    select = torch.tensor([vocab[c] for c in candidates],device=device)
    return F.softmax(logits[select],dim=-1).cpu()


def tv(a,b):
    return (a-b).abs().sum().item()/2


def panel(checkpoint, n, seed, device):
    model,vocab,n_ctx=load(checkpoint,device)
    rng=random.Random(seed)
    rows=collections.defaultdict(list)
    for _ in range(n):
        w=draw_record(rng)
        base=realize_record(w)
        edited_world=dict(w)
        edited_world['query_ship']=next(x for x in w['ships']
                                        if x!=w['query_ship'])
        changed=realize_record(edited_world)
        reordered_world=dict(w)
        reordered_world['order']=tuple(reversed(w['order']))
        reordered=realize_record(reordered_world)
        candidates=tuple(c for c in CATEGORIES if c in vocab)
        p=probs(model,vocab,n_ctx,device,base,candidates)
        q=probs(model,vocab,n_ctx,device,changed,candidates)
        r=probs(model,vocab,n_ctx,device,reordered,candidates)
        group=('record_query_last' if w['ships'][w['order'][-1]]==w['query_ship']
               else 'record_query_nonlast')
        rows[group].append(dict(base_gold=p[candidates.index(base.sites[0].target)].item(),
                                edited_gold=q[candidates.index(changed.sites[0].target)].item(),
                                base_hit=candidates[p.argmax().item()]==base.sites[0].target,
                                edited_hit=candidates[q.argmax().item()]==changed.sites[0].target,
                                key_tv=tv(p,q),order_tv=tv(p,r)))
    for _ in range(n):
        w=draw_score(rng)
        w['headline_leak']=False
        base=realize_score(w)
        changed_world=dict(w)
        changed_world['increment']=1 if w['increment']!=1 else 2
        changed=realize_score(changed_world)
        leaked_world=dict(w)
        leaked_world['headline_leak']=True
        leaked=realize_score(leaked_world)
        candidates=tuple(str(i) for i in range(1,9) if str(i) in vocab)
        p=probs(model,vocab,n_ctx,device,base,candidates)
        q=probs(model,vocab,n_ctx,device,changed,candidates)
        r=probs(model,vocab,n_ctx,device,leaked,candidates)
        rows['score_unleaked'].append(dict(
            base_gold=p[candidates.index(base.sites[0].target)].item(),
            edited_gold=q[candidates.index(changed.sites[0].target)].item(),
            leaked_gold=r[candidates.index(leaked.sites[0].target)].item(),
            base_hit=candidates[p.argmax().item()]==base.sites[0].target,
            edited_hit=candidates[q.argmax().item()]==changed.sites[0].target,
            leaked_hit=candidates[r.argmax().item()]==leaked.sites[0].target,
            event_tv=tv(p,q),leak_tv=tv(p,r)))
        if w['initial'][0] <= 4:
            four_world = dict(w)
            four_world['increment'] = 4  # Unseen increment; output still in 1..8.
            four = realize_score(four_world)
            s = probs(model,vocab,n_ctx,device,four,candidates)
            rows['score_increment4_ood'].append(dict(
                gold=s[candidates.index(four.sites[0].target)].item(),
                hit=candidates[s.argmax().item()]==four.sites[0].target,
                base_to_four_tv=tv(p,s)))
        right_world=dict(w)
        right_world['query_team']=w['teams'][1]  # Never queried in training.
        right=realize_score(right_world)
        right_edited_world=dict(right_world)
        right_edited_world['increment']=changed_world['increment']
        right_edited=realize_score(right_edited_world)
        u=probs(model,vocab,n_ctx,device,right,candidates)
        v=probs(model,vocab,n_ctx,device,right_edited,candidates)
        rows['score_right_team_ood'].append(dict(
            gold=u[candidates.index(right.sites[0].target)].item(),
            hit=candidates[u.argmax().item()]==right.sites[0].target,
            irrelevant_event_tv=tv(u,v)))
    return dict(checkpoint=str(checkpoint),n_per_task=n,seed=seed,
                candidate_softmax='renormalized over task values',
                rows={group:dict(n=len(values),
                                 means={key:sum(float(v[key]) for v in values)/len(values)
                                        for key in values[0]})
                      for group,values in rows.items()},
                oracle=dict(record_key_tv=1,record_order_tv=0,
                            score_event_tv=1,score_leak_tv=0,
                            score_base_to_increment4_tv=1,
                            score_right_irrelevant_event_tv=0),
                caveat='One trained model and one fresh paired panel; candidate-normalized distributions.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--n',type=int,default=200)
    p.add_argument('--seed',type=int,default=23031)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    result=panel(a.checkpoint,a.n,a.seed,a.device)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    for group,row in result['rows'].items():print(group,row,flush=True)
