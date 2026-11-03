"""Full next-token CE baseline on variable-clause world-state documents."""

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

OLD_REPO=Path(__file__).resolve().parents[2]/'reverse_engineer/tensor_language'
if not (OLD_REPO/'deep_model.py').is_file():
    raise FileNotFoundError(OLD_REPO/'deep_model.py')
sys.path.insert(0,str(OLD_REPO))
from deep_model import DeepModel,SPECS  # noqa: E402
from document_world_v5_clause_realizations import (  # noqa: E402
    draw_varied,realize,sample_documents,
)


def encode(docs,vocab,n_ctx):
    x=torch.full((len(docs),n_ctx),vocab['<PAD>'],dtype=torch.long)
    y=torch.full((len(docs),n_ctx),-100,dtype=torch.long)
    for i,doc in enumerate(docs):
        ids=[vocab[word] for word in doc.tokens]
        assert len(ids)-1<=n_ctx
        x[i,:len(ids)-1]=torch.tensor(ids[:-1])
        y[i,:len(ids)-1]=torch.tensor(ids[1:])
    return x,y


@torch.no_grad()
def evaluate(model,docs,data,vocab,device,batch_size=64):
    model.eval()
    totals=collections.defaultdict(lambda:[0.,0,0])
    losses=collections.defaultdict(lambda:[0.,0])
    for start in range(0,len(docs),batch_size):
        end=min(start+batch_size,len(docs))
        x,y=(a[start:end].to(device) for a in data)
        logits=model(x)
        per=F.cross_entropy(logits.flatten(0,1),y.flatten(),
                            reduction='none',ignore_index=-100).view_as(y)
        for j,doc in enumerate(docs[start:end]):
            valid=len(doc.tokens)-1
            losses[doc.genre][0]+=per[j,:valid].sum().item()
            losses[doc.genre][1]+=valid
            for site in doc.sites:
                pos=site.index-1
                assert pos>=0 and y[j,pos].item()==vocab[site.answer]
                scores=logits[j,pos]
                nll=-F.log_softmax(scores,-1)[vocab[site.answer]].item()
                hit=int(scores.argmax().item()==vocab[site.answer])
                for key in (site.family,f'{doc.genre}:{site.family}'):
                    totals[key][0]+=nll
                    totals[key][1]+=hit
                    totals[key][2]+=1
    return dict(token_nll=sum(v for v,n in losses.values())/
                          sum(n for v,n in losses.values()),
                by_genre={k:dict(token_nll=v/n,tokens=n)
                          for k,(v,n) in losses.items()},
                by_site={k:dict(n=row[2],target_nll=row[0]/row[2],
                                full_vocab_accuracy=row[1]/row[2])
                         for k,row in totals.items()})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--depths',default='1,2')
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--train-n',type=int,default=8000)
    p.add_argument('--test-n',type=int,default=1000)
    p.add_argument('--vocab-calibration-n',type=int,default=20000)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--d-model',type=int,default=64)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=34001)
    p.add_argument('--style-entropy',type=float,default=.7)
    p.add_argument('--aside-rate',type=float,default=.3)
    p.add_argument('--paragraph-rate',type=float,default=.22)
    p.add_argument('--header-rate',type=float,default=.5)
    p.add_argument('--source-delay-rate',type=float,default=0.)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--save-models',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(4)
    kwargs=dict(style_entropy=a.style_entropy,aside_rate=a.aside_rate,
                paragraph_rate=a.paragraph_rate,header_rate=a.header_rate,
                source_delay_rate=a.source_delay_rate)
    train=sample_documents(a.train_n,a.seed,**kwargs)
    test=sample_documents(a.test_n,a.seed+100000,**kwargs)
    shifted=[realize(doc.latent,1.,a.aside_rate,a.paragraph_rate,
                     a.header_rate) for doc in test]
    train_texts={doc.tokens for doc in train}
    assert not any(doc.tokens in train_texts for doc in test)
    calibration_rng=random.Random(a.seed+200000)
    words={'<PAD>'}|{word for doc in train for word in doc.tokens}
    for _ in range(a.vocab_calibration_n):
        words.update(draw_varied(calibration_rng,**kwargs).tokens)
    vocab={word:i for i,word in enumerate(sorted(words))}
    unseen={word for doc in test+shifted for word in doc.tokens if word not in vocab}
    if unseen:
        raise ValueError(f'Test tokens absent from train vocabulary: {sorted(unseen)}')
    n_ctx=max(len(doc.tokens)-1 for doc in train+test+shifted)
    train_data=encode(train,vocab,n_ctx)
    test_data=encode(test,vocab,n_ctx)
    shifted_data=encode(shifted,vocab,n_ctx)
    output=dict(config={**vars(a),'output':str(a.output)},
                n_vocab=len(vocab),n_ctx=n_ctx,
                train_documents=len(train),test_documents=len(test),
                exact_document_overlap=0,
                objective='ordinary full next-token CE; no task labels in training',
                caveat='Structural synthetic DGP pilot, not FineWeb-like. '
                       'One train seed and no whole-task holdout.',results={})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for depth in (int(v) for v in a.depths.split(',')):
        assert 1<=depth<=4
        torch.manual_seed(a.seed+depth)
        model=DeepModel(len(vocab),a.d_model,a.heads,SPECS[f'block{depth}'],
                        n_ctx,norm='rms',residual='lerp',
                        attention='bilinear').to(a.device)
        opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
        rng=random.Random(a.seed+300000+depth)
        losses=[];clock=time.time()
        for step in range(a.steps):
            idx=torch.tensor(rng.sample(range(a.train_n),a.batch_size))
            x,y=(v[idx].to(a.device) for v in train_data)
            opt.zero_grad(set_to_none=True)
            logits=model(x)
            loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),
                                 ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite depth{depth} step{step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            losses.append(loss.item())
            if (step+1)%100==0:
                print(depth,step+1,round(loss.item(),4),flush=True)
        result=dict(parameters=model.n_params,
                    train_loss_last50=sum(losses[-50:])/len(losses[-50:]),
                    test=evaluate(model,test,test_data,vocab,a.device),
                    style_shift=evaluate(model,shifted,shifted_data,
                                         vocab,a.device),
                    seconds=round(time.time()-clock,2))
        output['results'][str(depth)]=result
        a.output.write_text(json.dumps(output,indent=2)+'\n')
        if a.save_models:
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in
                                        model.state_dict().items()},
                            vocabulary=vocab,n_ctx=n_ctx,depth=depth,
                            model_kwargs=dict(d_model=a.d_model,n_head=a.heads,
                                              norm='rms',residual='lerp',
                                              attention='bilinear'),
                            generator_kwargs=kwargs,config=output['config'],
                            evaluation=result),
                       a.output.with_name(a.output.stem+f'_block{depth}.pt'))
    print('saved',a.output,flush=True)


if __name__=='__main__':main()
