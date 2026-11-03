"""Next-token bilinear pilot on the V7 joint task-rich document sampler."""

import argparse
import collections
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from document_mixture_v7_cross import (TASK_RICH_WEIGHTS,KINDS,
                                       sample_document)
from train_document_mixture_v5 import DeepModel,SPECS,encode,evaluate


def sample(n,seed,split,weights):
    rng=random.Random(seed)
    return [sample_document(rng,split=split,weights=weights) for _ in range(n)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--depths',default='2,4')
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--train-n',type=int,default=10000)
    p.add_argument('--test-n',type=int,default=1000)
    p.add_argument('--vocab-calibration-n',type=int,default=10000)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--d-model',type=int,default=64)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=34801)
    p.add_argument('--weights',default=','.join(str(v) for v in
                                               TASK_RICH_WEIGHTS))
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--save-models',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(4)
    weights=tuple(float(v) for v in a.weights.split(','))
    assert len(weights)==len(KINDS) and all(v>=0 for v in weights)
    assert sum(weights)>0
    train=sample(a.train_n,a.seed,'train',weights)
    test=sample(a.test_n,a.seed+100000,'iid',weights)
    composition=sample(a.test_n,a.seed+200000,'composition',weights)
    calibration=sample(a.vocab_calibration_n,a.seed+300000,'train',weights)
    train_text={doc.tokens for doc in train}
    overlap={panel:sum(doc.tokens in train_text for doc in docs)
             for panel,docs in (('iid',test),('composition',composition))}
    vocab={word:i for i,word in enumerate(sorted(
        {'<PAD>'}|{token for doc in train+calibration for token in doc.tokens}))}
    unknown={panel:sorted({token for doc in docs for token in doc.tokens}
                          -set(vocab)) for panel,docs in
             (('iid',test),('composition',composition))}
    assert not any(unknown.values()),unknown
    n_ctx=max(len(doc.tokens)-1 for doc in train+calibration+test+composition)
    train_data=encode(train,vocab,n_ctx)
    test_data=encode(test,vocab,n_ctx)
    composition_data=encode(composition,vocab,n_ctx)
    output=dict(config={**vars(a),'output':str(a.output)},
                generator=dict(weights=weights,kinds=KINDS),
                n_vocab=len(vocab),n_ctx=n_ctx,
                train_documents=len(train),iid_documents=len(test),
                composition_documents=len(composition),
                train_kinds=dict(collections.Counter(d.kind for d in train)),
                iid_kinds=dict(collections.Counter(d.kind for d in test)),
                composition_kinds=dict(collections.Counter(
                    d.kind for d in composition)),
                exact_document_overlap=overlap,
                objective='ordinary full next-token CE across all task kinds',
                caveat='Task-rich mixture remains far from FineWeb form. '
                       'Composition panel withholds record category pairs '
                       'and score combinations from training; other kinds '
                       'are independent draws. Rule labels are evaluation only.',
                results={})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for depth in [int(v) for v in a.depths.split(',')]:
        assert 1<=depth<=4
        torch.manual_seed(a.seed+depth)
        model=DeepModel(len(vocab),a.d_model,a.heads,SPECS[f'block{depth}'],
                        n_ctx,norm='rms',residual='lerp',
                        attention='bilinear').to(a.device)
        opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
        rng=random.Random(a.seed+100+depth)
        recent=[];clock=time.time();model.train()
        for step in range(a.steps):
            index=torch.tensor(rng.sample(range(len(train)),a.batch_size))
            x,y=(data[index].to(a.device) for data in train_data)
            opt.zero_grad(set_to_none=True)
            logits=model(x)
            loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),
                                 ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite loss depth={depth} step={step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            recent.append(loss.item())
            if (step+1)%100==0:
                print(depth,step+1,round(loss.item(),4),flush=True)
        result=dict(parameters=model.n_params,
                    train_loss_last50=sum(recent[-50:])/len(recent[-50:]),
                    iid=evaluate(model,test,test_data,vocab,a.device),
                    composition=evaluate(model,composition,
                                         composition_data,vocab,a.device),
                    seconds=round(time.time()-clock,2))
        output['results'][str(depth)]=result
        a.output.write_text(json.dumps(output,indent=2)+'\n')
        if a.save_models:
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in
                                        model.state_dict().items()},
                            vocabulary=vocab,n_ctx=n_ctx,depth=depth,
                            model_kwargs=dict(d_model=a.d_model,
                                              n_head=a.heads,norm='rms',
                                              residual='lerp',
                                              attention='bilinear'),
                            generator=dict(weights=weights,kinds=KINDS),
                            config={**vars(a),'output':str(a.output)},
                            evaluation=result),
                       a.output.with_name(a.output.stem+f'_block{depth}.pt'))
    print('saved',a.output,flush=True)


if __name__=='__main__':main()
