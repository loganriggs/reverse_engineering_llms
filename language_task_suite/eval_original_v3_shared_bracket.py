"""Fresh per-rule CE and full-oracle opener-edit fidelity for shared v3."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_pre=argparse.ArgumentParser(add_help=False)
_pre.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
_source,_=_pre.parse_known_args()
sys.path.insert(0,_source.source)

from model import BilinearGPT,Config
from original_v3_shared_bracket import LanguageV3SharedBracket,bracket_next_distribution
from train_original_v3_shared_bracket import evaluate


def selected_pairs(gen,nstreams,length,limit):
    result=[]
    for _ in range(nstreams):
        ids,labels=gen.sample_sequence(length)
        words=[gen.id2token[i] for i in ids]
        for t,label in enumerate(labels):
            if label!='skip_trigram' or t>=length-1:
                continue
            q=t-1
            keys=[k for k in range(q,-1,-1) if words[k] in ('(','[')
                  and not any(w in (')',']') for w in words[k+1:t])]
            if not keys:continue
            k=keys[0]
            opener=words[k]
            content=words[k+1:t]
            if len(content) not in (2,3,4):continue
            old=bracket_next_distribution(gen,opener,content)
            replacement='[' if opener=='(' else '('
            new=bracket_next_distribution(gen,replacement,content)
            assert old.get(words[t],0)>0
            assert abs(sum(old.values())-1)<1e-10
            assert abs(sum(new.values())-1)<1e-10
            for word in set(old)|set(new):
                if word not in (')',']'):
                    assert abs(old.get(word,0)-new.get(word,0))<1e-12
            changed=ids[:-1].copy();changed[k]=gen.token2id[replacement]
            result.append(dict(original=ids[:-1],edited=changed,query=q,
                               opener=opener,content_count=len(content),
                               old_oracle=old,new_oracle=new,
                               old_closer=')' if opener=='(' else ']',
                               new_closer=']' if opener=='(' else ')'))
            if len(result)>=limit:return result
    return result


@torch.no_grad()
def score_pairs(model,gen,pairs,batch_size,device):
    vocab=gen.vocab
    rows=[]
    for start in range(0,len(pairs),batch_size):
        part=pairs[start:start+batch_size]
        old=torch.tensor([p['original'] for p in part],device=device)
        new=torch.tensor([p['edited'] for p in part],device=device)
        q=torch.tensor([p['query'] for p in part],device=device)
        combined=torch.cat((old,new))
        logits=model(combined)[0].float()
        selected=logits[torch.arange(len(combined),device=device),
                        torch.cat((q,q))]
        predicted=selected.softmax(-1)
        logp=selected.log_softmax(-1)
        n=len(part)
        for i,pair in enumerate(part):
            q0=torch.tensor([pair['old_oracle'].get(token,0.) for token in vocab],
                            device=device)
            q1=torch.tensor([pair['new_oracle'].get(token,0.) for token in vocab],
                            device=device)
            p0,p1=predicted[i],predicted[n+i]
            lp0,lp1=logp[i],logp[n+i]
            def kl(target,lp):
                return float((target*(target.clamp_min(1e-30).log()-lp)).sum())
            target=gen.token2id[pair['new_closer']]
            rows.append(dict(content_count=pair['content_count'],
                original_kl=kl(q0,lp0),edited_kl=kl(q1,lp1),
                new_closer_effect=float(p1[target]-p0[target]),
                oracle_new_effect=float(q1[target]-q0[target]),
                effect_l1_error=float(((p1-p0)-(q1-q0)).abs().sum()),
                original_old_closer_probability=float(p0[gen.token2id[pair['old_closer']]]),
                edited_new_closer_probability=float(p1[target])))
    def summarize(items):
        return dict(n=len(items),**{key:sum(row[key] for row in items)/len(items)
                for key in items[0] if key!='content_count'})
    return dict(all=summarize(rows),by_content_count={str(k):summarize(
        [row for row in rows if row['content_count']==k]) for k in (2,3,4)
        if any(row['content_count']==k for row in rows)})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,
                   default=Path('/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits'))
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--streams',type=int,default=512)
    p.add_argument('--length',type=int,default=129)
    p.add_argument('--limit',type=int,default=256)
    p.add_argument('--seed',type=int,default=97101)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    device=torch.device(a.device)
    payload=torch.load(a.run_dir/'model.pt',map_location=device,weights_only=True)
    model=BilinearGPT(Config(**payload['config'])).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    ordinary=evaluate(model,LanguageV3SharedBracket(seed=a.seed),a.streams,
                      a.length,32,device,a.limit)
    pairs=selected_pairs(LanguageV3SharedBracket(seed=a.seed+1),a.streams,
                         a.length,a.limit)
    assert len(pairs)==a.limit
    causal=score_pairs(model,LanguageV3SharedBracket(seed=a.seed+2),pairs,32,device)
    result=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                ordinary=ordinary,full_bracket_oracle=causal,
                caveat='Fresh streams from the same joint generator as training. '
                       'Full bracket oracle includes closure probability and '
                       'continuation-word probabilities. Opener flip retains '
                       'the complete visible content prefix in support. No '
                       'whole-task model-training holdout or naturalness claim.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('CE', {k:round(ordinary['per_rule_ce'].get(k,float('nan')),3)
                 for k in ('noun_bigram','skip_bigram','skip_trigram','induction')})
    print('full oracle',causal)


if __name__=='__main__':main()
