"""Fresh pairwise and anchor-edit evaluation for the crossed skip DGP."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

_pre=argparse.ArgumentParser(add_help=False)
_pre.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
_source,_=_pre.parse_known_args()
sys.path.insert(0,_source.source)

from model import BilinearGPT, Config
from original_v3_crossed_skip import LanguageV3CrossedSkip, nearest_skip_anchor


def collect(gen,nstreams,length,edits_per_pair):
    events=[]
    edits=[]
    edit_counts=Counter()
    anchors=gen.crossed_skip_anchors
    for _ in range(nstreams):
        ids,labels=gen.sample_sequence(length)
        words=[gen.id2token[i] for i in ids]
        for t in range(1,length):
            if labels[t]!='skip_bigram':continue
            found=nearest_skip_anchor(gen,words[:t])
            assert found is not None
            k,anchor=found
            trigger=words[t-1]
            target=gen.skip_bigrams[anchor,trigger]
            assert words[t]==target
            item=dict(x=ids[:t],query=t-1,anchor=anchor,trigger=trigger,
                      target=target,distance=t-k)
            events.append(item)
            key=(anchor,trigger)
            if (t-k!=2 or labels[k]!='cat_bigram'
                    or labels[t-1]!='cat_bigram'
                    or edit_counts[key]>=edits_per_pair):
                continue
            alternative=anchors[(anchors.index(anchor)+1)%len(anchors)]
            changed=words[:t].copy();changed[k]=alternative
            new_target=gen.skip_bigrams[alternative,trigger]
            assert new_target!=target
            assert gen._sample_normal_token(changed,{})==(new_target,'skip_bigram')
            edits.append(dict(**item,edited_anchor=alternative,
                              edited_x=[gen.token2id[w] for w in changed],
                              new_target=new_target,anchor_position=k))
            edit_counts[key]+=1
    assert min(edit_counts.values())>=edits_per_pair,edit_counts
    return events,edits


@torch.no_grad()
def score(model,gen,events,edits,length,device,batch_size):
    pair_prob=defaultdict(list)
    pair_ce=defaultdict(list)
    pair_top=defaultdict(list)
    outputs=gen.crossed_skip_outputs
    out_ids=[gen.token2id[w] for w in outputs]
    for start in range(0,len(events),batch_size):
        part=events[start:start+batch_size]
        x=torch.zeros(len(part),length-1,dtype=torch.long,device=device)
        for i,row in enumerate(part):x[i,:len(row['x'])]=torch.tensor(row['x'],device=device)
        q=torch.tensor([row['query'] for row in part],device=device)
        logits=model(x)[0][torch.arange(len(part),device=device),q].float()
        probs=logits.softmax(-1)
        for i,row in enumerate(part):
            key=(row['anchor'],row['trigger'])
            target=gen.token2id[row['target']]
            p=float(probs[i,target])
            pair_prob[key].append(p)
            pair_ce[key].append(-float(probs[i,target].clamp_min(1e-30).log()))
            pair_top[key].append(outputs[int(probs[i,out_ids].argmax())])
    pair_table={f'{a}|{t}':dict(n=len(pair_prob[a,t]),
                    target=gen.skip_bigrams[a,t],
                    mean_target_probability=float(np.mean(pair_prob[a,t])),
                    mean_target_ce=float(np.mean(pair_ce[a,t])),
                    selected_output_accuracy=float(np.mean([
                        x==gen.skip_bigrams[a,t] for x in pair_top[a,t]])))
                for a in gen.crossed_skip_anchors
                for t in gen.crossed_skip_triggers}
    edit_rows=[]
    for start in range(0,len(edits),batch_size):
        part=edits[start:start+batch_size]
        original=torch.zeros(len(part),length-1,dtype=torch.long,device=device)
        changed=original.clone()
        for i,row in enumerate(part):
            original[i,:len(row['x'])]=torch.tensor(row['x'],device=device)
            changed[i,:len(row['edited_x'])]=torch.tensor(row['edited_x'],device=device)
        q=torch.tensor([row['query'] for row in part],device=device)
        both=torch.cat((original,changed))
        p=model(both)[0].float().softmax(-1)[
            torch.arange(2*len(part),device=device),torch.cat((q,q))]
        for i,row in enumerate(part):
            old=gen.token2id[row['target']]
            new=gen.token2id[row['new_target']]
            edit_rows.append(dict(anchor=row['anchor'],trigger=row['trigger'],
                new_response=float(p[len(part)+i,new]-p[i,new]),
                old_response=float(p[len(part)+i,old]-p[i,old]),
                original_target_probability=float(p[i,old]),
                edited_target_probability=float(p[len(part)+i,new]),
                edited_accuracy=int(int(p[len(part)+i].argmax())==new),
                distance=row['distance']))
    by_pair={}
    for a in gen.crossed_skip_anchors:
        for t in gen.crossed_skip_triggers:
            group=[row for row in edit_rows if row['anchor']==a and row['trigger']==t]
            by_pair[f'{a}|{t}']=dict(n=len(group),
                new_response=float(np.mean([r['new_response'] for r in group])),
                edited_target_probability=float(np.mean([
                    r['edited_target_probability'] for r in group])),
                edited_accuracy=float(np.mean([r['edited_accuracy'] for r in group])))
    return dict(pair_table=pair_table,anchor_edits=dict(
        n=len(edit_rows),oracle_new_response=1.,
        mean_new_response=float(np.mean([r['new_response'] for r in edit_rows])),
        mean_old_response=float(np.mean([r['old_response'] for r in edit_rows])),
        mean_original_target_probability=float(np.mean([
            r['original_target_probability'] for r in edit_rows])),
        mean_edited_target_probability=float(np.mean([
            r['edited_target_probability'] for r in edit_rows])),
        edited_accuracy=float(np.mean([r['edited_accuracy'] for r in edit_rows])),
        by_pair=by_pair))


def trigger_only_baseline(gen,audit_counts,events):
    counts={tuple(key.split('|')):value for key,value in audit_counts.items()}
    total=sum(counts[a,t] for a in gen.crossed_skip_anchors
              for t in gen.crossed_skip_triggers)
    assert total>0
    ce=[];correct=[]
    for row in events:
        trigger=row['trigger']
        denom=sum(counts[a,trigger] for a in gen.crossed_skip_anchors)
        matched=sum(counts[a,trigger] for a in gen.crossed_skip_anchors
                    if gen.skip_bigrams[a,trigger]==row['target'])
        prob=matched/denom
        ce.append(-np.log(prob))
        best=max(gen.crossed_skip_anchors,key=lambda a:counts[a,trigger])
        correct.append(gen.skip_bigrams[best,trigger]==row['target'])
    return dict(training_audit_streams=4000,site_conditioned_ce=float(np.mean(ce)),
                site_conditioned_accuracy=float(np.mean(correct)),
                caveat='Optimistic trigger-only baseline conditioned on '
                       'skip-labeled sites, using an independent 4000-stream '
                       'panel to estimate anchor frequencies.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--audit-counts',type=Path,required=True)
    p.add_argument('--seed',type=int,default=99201)
    p.add_argument('--streams',type=int,default=1024)
    p.add_argument('--length',type=int,default=129)
    p.add_argument('--edits-per-pair',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    gen=LanguageV3CrossedSkip(seed=args.seed)
    events,edits=collect(gen,args.streams,args.length,args.edits_per_pair)
    payload=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
    assert payload['generator']=='LanguageV3CrossedSkip'
    model=BilinearGPT(Config(**payload.get('config',payload.get('model_config')))).to(args.device)
    model.load_state_dict(payload['model'])
    model.eval()
    result=score(model,gen,events,edits,args.length,args.device,args.batch_size)
    audit=json.loads(args.audit_counts.read_text())['audit']['pair_counts']
    result['trigger_only_baseline']=trigger_only_baseline(gen,audit,events)
    result['config']={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    result['checkpoint_step']=payload.get('step')
    result['sampled_skip_events']=len(events)
    result['caveat']=('All events are sampled from the crossed joint DGP. '
                     'The edit changes a same-category anchor at distance 2, '
                     'preserves prefix support through a positive trigger '
                     'boost, and changes the exact deterministic next token. '
                     'This is a diagnostic rule, not a naturalistic language '
                     'or full-vocabulary oracle comparison.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('skip events',len(events),'pairs',len(edits),'p(correct)',
          round(np.mean([v['mean_target_probability'] for v in
                         result['pair_table'].values()]),3),'edit response',
          round(result['anchor_edits']['mean_new_response'],3),'baseline acc',
          round(result['trigger_only_baseline']['site_conditioned_accuracy'],3),
          flush=True)


if __name__=='__main__':main()
