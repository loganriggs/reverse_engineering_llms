"""Train a small bilinear LM on joint v3 rules with editable bracket content."""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

_pre=argparse.ArgumentParser(add_help=False)
_pre.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
_source,_=_pre.parse_known_args()
sys.path.insert(0,_source.source)

from generator_v3 import make_batch
from model import BilinearGPT,Config
from original_v3_shared_bracket import (LanguageV3SharedBracket,
                                        bracket_close_distribution)


@torch.no_grad()
def evaluate(model,gen,nstreams,length,batch_size,device,limit):
    model.eval()
    by_rule=defaultdict(lambda:[0.,0])
    edits=[]
    examples=[]
    for start in range(0,nstreams,batch_size):
        x,y,labels,_=make_batch(gen,min(batch_size,nstreams-start),length,device)
        logits=model(x)[0].float()
        ce=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none')
        ce=ce.reshape_as(y)
        for b,row in enumerate(labels):
            for q,name in enumerate(row):
                by_rule[name][0]+=float(ce[b,q])
                by_rule[name][1]+=1
                if name!='skip_trigram' or len(edits)>=limit:
                    continue
                words=[gen.id2token[int(token)] for token in x[b,:q+1]]
                candidates=[k for k in range(q,-1,-1) if words[k] in ('(','[')
                            and not any(w in (')',']') for w in words[k+1:])]
                if not candidates:continue
                opener_pos=candidates[0]
                opener=words[opener_pos]
                count=q-opener_pos
                if count not in (2,3,4):continue
                old=')' if opener=='(' else ']'
                new=']' if opener=='(' else ')'
                mass=next(iter(bracket_close_distribution(opener,count).values()))
                changed=x[b].clone()
                changed[opener_pos]=gen.token2id['[' if opener=='(' else '(']
                edits.append((x[b].clone(),changed,q,gen.token2id[old],
                              gen.token2id[new],mass,count))
                if len(examples)<4:
                    examples.append(dict(prefix=' '.join(words[opener_pos:]),
                        changed_opener='[' if opener=='(' else '(',
                        original_closer=old,new_closer=new,oracle_closer_mass=mass))
    if not edits:
        raise RuntimeError('No bracket edits selected')
    before_new=[];after_new=[];before_old=[];after_old=[]
    for start in range(0,len(edits),batch_size):
        part=edits[start:start+batch_size]
        original=torch.stack([e[0] for e in part])
        changed=torch.stack([e[1] for e in part])
        q=torch.tensor([e[2] for e in part],device=device)
        old=torch.tensor([e[3] for e in part],device=device)
        new=torch.tensor([e[4] for e in part],device=device)
        both=torch.cat((original,changed))
        p=model(both)[0].float().softmax(-1)
        selected=p[torch.arange(len(both),device=device),torch.cat((q,q))]
        n=len(part)
        before_new.extend(selected[:n].gather(-1,new[:,None]).flatten().cpu().tolist())
        after_new.extend(selected[n:].gather(-1,new[:,None]).flatten().cpu().tolist())
        before_old.extend(selected[:n].gather(-1,old[:,None]).flatten().cpu().tolist())
        after_old.extend(selected[n:].gather(-1,old[:,None]).flatten().cpu().tolist())
    avg=lambda values:sum(values)/len(values)
    result=dict(per_rule_ce={rule:total/count for rule,(total,count) in by_rule.items()},
                per_rule_n={rule:count for rule,(_,count) in by_rule.items()},
                bracket_edit=dict(n=len(edits),new_before=avg(before_new),
                    new_after=avg(after_new),old_before=avg(before_old),
                    old_after=avg(after_old),
                    new_effect=avg([b-a for a,b in zip(before_new,after_new)]),
                    oracle_new_effect=avg([e[5] for e in edits]),
                    by_content_length={str(k):sum(e[6]==k for e in edits)
                                       for k in (2,3,4)}),
                examples=examples)
    model.train()
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,
                   default=Path('/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits'))
    p.add_argument('--depth',type=int,default=2)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--length',type=int,default=129)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--eval-every',type=int,default=250)
    p.add_argument('--eval-streams',type=int,default=256)
    p.add_argument('--edit-limit',type=int,default=128)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    device=torch.device(a.device)
    config=Config(vocab_size=54,n_layer=a.depth,n_head=a.heads,
                  n_embd=a.width,seq_len=a.length-1,
                  block_has_mlp=(False,)*a.depth,
                  use_rmsnorm=False,use_qk_norm=False,use_final_norm=False)
    model=BilinearGPT(config).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.)
    train_gen=LanguageV3SharedBracket(seed=a.seed+1000)
    records=[]
    start_time=time.time()
    def snapshot(step):
        scores=evaluate(model,LanguageV3SharedBracket(seed=a.seed+2000),
            a.eval_streams,a.length,32,device,a.edit_limit)
        records.append(dict(step=step,elapsed_seconds=time.time()-start_time,
                            scores=scores))
        print(step,'ce',round(sum(scores['per_rule_ce'].values())/
                     len(scores['per_rule_ce']),3),
              'noun',round(scores['per_rule_ce'].get('noun_bigram',math.nan),3),
              'skip',round(scores['per_rule_ce'].get('skip_bigram',math.nan),3),
              'bracket',round(scores['per_rule_ce'].get('skip_trigram',math.nan),3),
              'opener effect',round(scores['bracket_edit']['new_effect'],3),
              'oracle',round(scores['bracket_edit']['oracle_new_effect'],3),
              flush=True)
    snapshot(0)
    model.train()
    for step in range(1,a.steps+1):
        x,y,_,_=make_batch(train_gen,a.batch_size,a.length,device)
        logits=model(x)[0]
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step%a.eval_every==0 or step==a.steps:
            snapshot(step)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),config=vars(config),
                    generator='LanguageV3SharedBracket'),a.output_dir/'model.pt')
    result=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                model_config=vars(config),records=records,
                caveat='Same joint-rule DGP at train and evaluation, disjoint '
                       'random streams. No whole-task training holdout. '
                       'Bracket edit changes only opener in a shared-content '
                       'span; model competence must be judged against oracle '
                       'effect and per-rule CE separately. Surface naturalness '
                       'is not established.')
    (a.output_dir/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
