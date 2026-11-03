"""Fresh held-out V13 prediction and supported attribute-binding edits."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
from model import BilinearGPT, Config
from audit_v13_attributes import attribute_positions
from surface_mixture_v2 import _split
from surface_mixture_v4 import _content
from surface_mixture_v13_attributes import (
    ATTRIBUTES, SurfaceMixtureV13Attributes, attribute_state,
)
from train_v13_attributes import COMMON, REFERENCE, DIAGNOSTIC


def collect(seed, nstreams, length, kwargs, max_edits):
    gen=SurfaceMixtureV13Attributes(seed=seed,**COMMON,**kwargs)
    streams=[]; edits=[]
    for _ in range(nstreams):
        ids,labels,qs=gen.sample_sequence(length)
        words=[gen.id2token[v] for v in ids]
        streams.append((ids,labels,qs))
        positions=attribute_positions(words)
        for t,label in enumerate(labels):
            if label not in ('attribute_adjective','attribute_object'):
                continue
            history,raw=_split(words[:t])
            content=_content(raw)
            state=attribute_state(history)
            pending=[(item,attribute) for item,attribute in
                     state['facts'].items() if item not in state['mentioned']]
            if not pending:
                continue
            if label=='attribute_adjective':
                item,old=pending[0]
            else:
                matches=[(item,attribute) for item,attribute in pending
                         if content[-1]==attribute]
                assert len(matches)==1
                item,old=matches[0]
            replacement=next((v for v in ATTRIBUTES
                              if v not in state['facts'].values()),None)
            if replacement is None:
                continue
            pos,check=positions[item]
            assert check==old and pos<t
            altered=words[:t].copy();altered[pos]=replacement
            for u in range(pos+1,t):
                assert gen.next_distribution(altered[:u]).get(altered[u],0)>0
            edited_q=gen.next_distribution(altered)
            target=replacement if label=='attribute_adjective' else item
            oracle_response=edited_q.get(target,0.)-qs[t].get(target,0.)
            assert oracle_response>0 if label=='attribute_adjective' else oracle_response<0
            edits.append(dict(kind=label,x=ids[:-1],query=t-1,
                              position=pos,target=target,
                              new=replacement,old=old,item=item,
                              pending=len(pending),
                              latest_pending_item=pending[-1][0],
                              oracle_response=oracle_response))
    rng=random.Random(seed+991)
    selected=[]
    for kind in ('attribute_adjective','attribute_object'):
        rows=[e for e in edits if e['kind']==kind]
        multi=[e for e in rows if e['pending']>1]
        single=[e for e in rows if e['pending']==1]
        rng.shuffle(multi);rng.shuffle(single)
        chosen=multi[:max_edits//2]+single[:max_edits//2]
        remaining=[e for e in multi[max_edits//2:]+single[max_edits//2:]]
        rng.shuffle(remaining)
        selected.extend((chosen+remaining)[:max_edits])
    return gen,streams,selected


@torch.no_grad()
def score(model,gen,streams,edits,device,batch_size):
    totals=defaultdict(list);oracle_entropies=defaultdict(list)
    model.eval()
    for start in range(0,len(streams),batch_size):
        part=streams[start:start+batch_size]
        ids=torch.tensor([r[0] for r in part],device=device)
        logp=model(ids[:,:-1])[0].float().log_softmax(-1)
        ce=-logp.gather(-1,ids[:,1:,None]).squeeze(-1).cpu().numpy()
        for b,(_,labels,qs) in enumerate(part):
            for t,label in enumerate(labels[1:],start=1):
                totals[label].append(float(ce[b,t-1]))
                q=qs[t]
                oracle_entropies[label].append(-sum(p*np.log(p)
                                                 for p in q.values() if p>0))
    responses=defaultdict(list)
    for start in range(0,len(edits),batch_size):
        part=edits[start:start+batch_size]
        base=torch.tensor([v['x'] for v in part],device=device)
        changed=base.clone()
        for i,v in enumerate(part):
            changed[i,v['position']]=gen.token2id[v['new']]
        query=torch.tensor([v['query'] for v in part],device=device)
        rows=torch.arange(len(part),device=device)
        before=model(base)[0].float().softmax(-1)[rows,query]
        after=model(changed)[0].float().softmax(-1)[rows,query]
        for i,v in enumerate(part):
            token_id=gen.token2id[v['target']]
            responses[v['kind']].append(dict(
                response=float(after[i,token_id]-before[i,token_id]),
                oracle_response=v['oracle_response'],
                before=float(before[i,token_id]),
                after=float(after[i,token_id]),
                pending=v['pending'],
                latest_pending_correct=v['latest_pending_item']==v['item']))
    per_rule={label:dict(n=len(values),ce=float(np.mean(values)),
                         oracle_entropy=float(np.mean(oracle_entropies[label])))
              for label,values in totals.items()}
    edit_summary={}
    for kind,rows in responses.items():
        edit_summary[kind]=dict(
            n=len(rows),mean_model_response=float(np.mean([r['response'] for r in rows])),
            mean_oracle_response=float(np.mean([r['oracle_response'] for r in rows])),
            mean_abs_effect_error=float(np.mean([abs(r['response']-r['oracle_response'])
                                                 for r in rows])),
            multi_pending_n=sum(r['pending']>1 for r in rows),
            latest_pending_item_accuracy=float(np.mean([
                r['latest_pending_correct'] for r in rows])))
    return dict(per_rule=per_rule,edits=edit_summary)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--seed',type=int,default=99721)
    p.add_argument('--streams',type=int,default=512)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--max-edits',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    saved=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
    assert saved['generator']=='SurfaceMixtureV13Attributes'
    model=BilinearGPT(Config(**saved['config'])).to(args.device)
    model.load_state_dict(saved['model'])
    panels={}
    for name,kwargs in (('reference',REFERENCE),('diagnostic',DIAGNOSTIC)):
        gen,streams,edits=collect(args.seed,args.streams,args.length,
                                  kwargs,args.max_edits)
        panels[name]=score(model,gen,streams,edits,
                           torch.device(args.device),args.batch_size)
        print(name,'adjective',panels[name]['per_rule'].get('attribute_adjective'),
              'object',panels[name]['per_rule'].get('attribute_object'),
              'edits',panels[name]['edits'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(
        config={**vars(args),'checkpoint':str(args.checkpoint),
                'output':str(args.output)},panels=panels,
        caveat='Response is change in model probability of the selected '
               'target after one supported earlier fact edit. The oracle '
               'response is computed from the two exact prefix distributions. '
               'This measures task prediction and causal sensitivity, not '
               'circuit recovery or natural-language realism.'),indent=2)+'\n')


if __name__=='__main__':main()
