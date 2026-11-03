"""Paired V6 causal panels on train-fold and unseen-combination worlds."""

import argparse
import collections
import json
import random
from pathlib import Path

import torch

from document_mixture_v5 import CATEGORIES, realize_record, realize_score
from document_mixture_v6 import draw_record_v6, draw_score_v6
from eval_document_mixture_v5_causal import load, probs, tv


def summarize(rows):
    return {group:dict(n=len(values),
                       means={key:sum(float(v[key]) for v in values)/len(values)
                              for key in values[0]})
            for group, values in rows.items()}


def panel(checkpoint, n, seed, device):
    model,vocab,n_ctx=load(checkpoint,device)
    rng=random.Random(seed)
    result={}
    for split in ('iid','composition'):
        rows=collections.defaultdict(list)
        for _ in range(n):
            w=draw_record_v6(rng,split)
            doc=realize_record(w)
            edited_world=dict(w)
            edited_world['query_ship']=next(ship for ship in w['ships']
                                             if ship!=w['query_ship'])
            edited=realize_record(edited_world)
            order_world=dict(w)
            order_world['order']=tuple(reversed(w['order']))
            reordered=realize_record(order_world)
            candidates=tuple(c for c in CATEGORIES if c in vocab)
            p=probs(model,vocab,n_ctx,device,doc,candidates)
            q=probs(model,vocab,n_ctx,device,edited,candidates)
            r=probs(model,vocab,n_ctx,device,reordered,candidates)
            group=('record_last' if w['ships'][w['order'][-1]]==w['query_ship']
                   else 'record_nonlast')
            rows[group].append(dict(
                gold=p[candidates.index(doc.sites[0].target)].item(),
                hit=candidates[p.argmax().item()]==doc.sites[0].target,
                edited_gold=q[candidates.index(edited.sites[0].target)].item(),
                edited_hit=candidates[q.argmax().item()]==edited.sites[0].target,
                key_tv=tv(p,q),order_tv=tv(p,r)))
        for _ in range(n):
            w=draw_score_v6(rng,split)
            w['headline_leak']=False
            doc=realize_score(w)
            changed_world=dict(w)
            changed_world['increment']=1 if w['increment']!=1 else 2
            changed=realize_score(changed_world)
            leak_world=dict(w)
            leak_world['headline_leak']=True
            leaked=realize_score(leak_world)
            candidates=tuple(str(i) for i in range(1,10) if str(i) in vocab)
            p=probs(model,vocab,n_ctx,device,doc,candidates)
            q=probs(model,vocab,n_ctx,device,changed,candidates)
            r=probs(model,vocab,n_ctx,device,leaked,candidates)
            side='left' if w['query_team']==w['teams'][0] else 'right'
            rows['score_'+side].append(dict(
                gold=p[candidates.index(doc.sites[0].target)].item(),
                hit=candidates[p.argmax().item()]==doc.sites[0].target,
                edited_gold=q[candidates.index(changed.sites[0].target)].item(),
                edited_hit=candidates[q.argmax().item()]==changed.sites[0].target,
                event_tv=tv(p,q),leak_tv=tv(p,r)))
        result[split]=summarize(rows)
    return dict(checkpoint=str(checkpoint),n_per_task_per_split=n,seed=seed,
                candidate_softmax='renormalized over task values',
                splits=result,
                oracle=dict(record_key_tv=1,record_order_tv=0,
                            score_left_event_tv=1,score_right_event_tv=0,
                            score_leak_tv=0),
                caveat='One model and fresh paired panel per split; composition '
                       'holds out record assignments and score number pairs.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--n',type=int,default=200)
    p.add_argument('--seed',type=int,default=23041)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    result=panel(a.checkpoint,a.n,a.seed,a.device)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    for split,rows in result['splits'].items():
        print(split,{k:(v['n'],v['means'].get('hit'),
                         v['means'].get('key_tv',v['means'].get('event_tv')))
                     for k,v in rows.items()},flush=True)
