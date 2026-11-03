"""Fresh supported causal edits for the V7 cross-section task."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from audit_document_mixture_v7_cross import edits
from document_mixture_v7_cross import draw_cross,realize_cross,site_oracle
from train_document_mixture_v5 import DeepModel,SPECS


def cross_site(doc):
    return next(s for s in doc.sites if s.family=='cross_source_record')


def collect(n,seed,split,n_ctx):
    rng=random.Random(seed)
    cases=defaultdict(list)
    skipped=0
    for _ in range(n):
        world=draw_cross(rng,split)
        original=realize_cross(world)
        site=cross_site(original)
        assert site_oracle(original,site)=={site.target:1.}
        if len(original.tokens)-1>n_ctx:
            skipped+=1;continue
        cases['baseline'].append((original,site))
        for name,changed,flip in edits(world):
            other=realize_cross(changed)
            other_site=cross_site(other)
            assert site_oracle(other,other_site)=={other_site.target:1.}
            assert (other_site.target!=site.target)==flip
            if len(other.tokens)-1<=n_ctx:
                cases[name].append((original,site,other,other_site,flip))
    return cases,skipped


@torch.no_grad()
def predict(model,entries,vocab,n_ctx,device,batch_size):
    rows=[]
    categories=[vocab[token] for token in ('11B','9C','7A','6D','4B','3C')]
    for start in range(0,len(entries),batch_size):
        part=entries[start:start+batch_size]
        x=torch.full((len(part),n_ctx),vocab['<PAD>'],
                     dtype=torch.long,device=device)
        position=torch.tensor([site.index-1 for doc,site in part],
                              device=device)
        for i,(doc,site) in enumerate(part):
            ids=[vocab[token] for token in doc.tokens[:-1]]
            assert len(ids)<=n_ctx
            x[i,:len(ids)]=torch.tensor(ids,device=device)
        logits=model(x)[torch.arange(len(part),device=device),position]
        full=logits.float().softmax(-1)
        candidate=logits[:,categories].float().softmax(-1)
        for i in range(len(part)):
            rows.append(dict(full={token:float(full[i,vocab[token]])
                                   for token in ('11B','9C','7A','6D','4B','3C')},
                             candidate={token:float(candidate[i,j])
                                        for j,token in enumerate(
                                            ('11B','9C','7A','6D','4B','3C'))},
                             full_argmax=int(logits[i].argmax())))
    return rows


def measure(model,cases,vocab,n_ctx,device,batch_size):
    out={}
    base=cases['baseline']
    p=predict(model,base,vocab,n_ctx,device,batch_size)
    out['baseline']=dict(n=len(base),full_vocab_accuracy=sum(
        row['full_argmax']==vocab[site.target]
        for row,(_,site) in zip(p,base,strict=True))/len(base),
        candidate_accuracy=sum(max(row['candidate'],
                                   key=row['candidate'].get)==site.target
                               for row,(_,site) in zip(p,base,strict=True))
                           /len(base),
        mean_correct_probability=sum(row['full'][site.target]
                                     for row,(_,site) in zip(p,base,strict=True))
                                 /len(base))
    for name,rows in cases.items():
        if name=='baseline':continue
        before=predict(model,[(a,s) for a,s,b,t,flip in rows],
                       vocab,n_ctx,device,batch_size)
        after=predict(model,[(b,t) for a,s,b,t,flip in rows],
                      vocab,n_ctx,device,batch_size)
        delta=[];target_accuracy=[]
        for row,(a,s,b,t,flip) in enumerate(rows):
            target=t.target if flip else s.target
            delta.append(after[row]['full'][target]-before[row]['full'][target])
            target_accuracy.append(after[row]['full_argmax']==vocab[t.target])
        out[name]=dict(n=len(rows),oracle_answer_changes=rows[0][-1],
                       mean_target_probability_response=sum(delta)/len(delta),
                       mean_abs_response=sum(abs(v) for v in delta)/len(delta),
                       edited_full_vocab_accuracy=sum(target_accuracy)
                                                   /len(target_accuracy))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--n',type=int,default=200)
    p.add_argument('--seed',type=int,default=34901)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    vocab=saved['vocabulary'];n_ctx=saved['n_ctx']
    args=saved['model_kwargs']
    model=DeepModel(len(vocab),args['d_model'],args['n_head'],
                    SPECS[f"block{saved['depth']}"],n_ctx,norm=args['norm'],
                    residual=args['residual'],attention=args['attention']).to(a.device)
    model.load_state_dict(saved['state_dict']);model.eval()
    panels={};skipped={}
    for split in ('train','composition'):
        cases,n_skip=collect(a.n,a.seed+(0 if split=='train' else 100000),
                             split,n_ctx)
        assert len(cases['baseline'])>=a.n*.9
        panels[split]=measure(model,cases,vocab,n_ctx,a.device,a.batch_size)
        skipped[split]=n_skip
        print(split,'n',len(cases['baseline']),
              'accuracy',round(panels[split]['baseline']
                               ['full_vocab_accuracy'],3),
              'source edit',round(panels[split]['relevant_source']
                                  ['mean_target_probability_response'],3),
              flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(config={**vars(a),
        'checkpoint':str(a.checkpoint),'output':str(a.output)},
        skipped=skipped,panels=panels,
        caveat='Full-vocabulary source/record edit responses on newly '
               'sampled documents. This is a behavioral gate, not evidence '
               'of a recovered internal algorithm. The train-split panel '
               'uses fresh combinations from training support.'),indent=2)+'\n')


if __name__=='__main__':main()
