"""Train 1–4 bilinear blocks on V6 with IID and composition evaluations."""

import argparse
import collections
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from document_mixture_v6 import sample_document
from train_document_mixture_v5 import DeepModel, SPECS, encode, evaluate


def sample(n, seed, split):
    rng = random.Random(seed)
    return [sample_document(rng, split=split) for _ in range(n)]


def exact_overlap(train, test):
    seen = {doc.tokens for doc in train}
    return {kind:dict(n=sum(doc.kind == kind for doc in test),
                      exact=sum(doc.kind == kind and doc.tokens in seen
                                for doc in test))
            for kind in sorted({doc.kind for doc in test})}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--depths',default='1,2,3,4')
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--train-n',type=int,default=20000)
    p.add_argument('--test-n',type=int,default=1000)
    p.add_argument('--composition-n',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--d-model',type=int,default=64)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=23012)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',required=True)
    p.add_argument('--save-models',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(4)
    device=torch.device(a.device)
    train=sample(a.train_n,a.seed,'train')
    iid=sample(a.test_n,a.seed+1,'iid')
    composition=sample(a.composition_n,a.seed+2,'composition')
    splits={'iid':iid,'composition':composition}
    vocab={word:i for i,word in enumerate(sorted(
        {'<PAD>'}|{w for doc in train+iid+composition for w in doc.tokens}))}
    n_ctx=max(len(doc.tokens)-1 for doc in train+iid+composition)
    train_data=encode(train,vocab,n_ctx)
    eval_data={name:encode(docs,vocab,n_ctx) for name,docs in splits.items()}
    output=dict(config=vars(a),n_vocab=len(vocab),n_ctx=n_ctx,
                train_kinds=dict(collections.Counter(doc.kind for doc in train)),
                eval_kinds={name:dict(collections.Counter(doc.kind for doc in docs))
                            for name,docs in splits.items()},
                exact_document_overlap={name:exact_overlap(train,docs)
                                        for name,docs in splits.items()},
                objective='full next-token CE, token-weighted across V6 documents',
                split_rule='record ship/category assignments and score '
                           'initial-left/increment combinations disjoint in composition',
                caveat='Task-rich synthetic pilot; natural lexical/form gate still fails.',
                results={})
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    for depth in [int(x) for x in a.depths.split(',')]:
        assert 1<=depth<=4
        torch.manual_seed(a.seed+depth)
        model=DeepModel(len(vocab),a.d_model,a.heads,
                        SPECS[f'block{depth}'],n_ctx,norm='rms',
                        residual='lerp',attention='bilinear').to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
        rng=random.Random(a.seed+100+depth)
        losses=[]
        start=time.time()
        model.train()
        for step in range(a.steps):
            idx=torch.tensor(rng.sample(range(len(train)),a.batch_size),
                             dtype=torch.long)
            x,y=(array[idx].to(device) for array in train_data)
            opt.zero_grad(set_to_none=True)
            logits=model(x)
            loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite depth={depth} step={step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            losses.append(loss.item())
            if (step+1)%200==0:
                print(depth,step+1,round(loss.item(),4),flush=True)
        result=dict(parameters=model.n_params,
                    train_loss_last50=sum(losses[-50:])/len(losses[-50:]),
                    evaluations={name:evaluate(model,docs,eval_data[name],vocab,device)
                                 for name,docs in splits.items()},
                    seconds=round(time.time()-start,2))
        output['results'][str(depth)]=result
        path.write_text(json.dumps(output,indent=2)+'\n')
        if a.save_models:
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in
                                        model.state_dict().items()},
                            vocabulary=vocab,n_ctx=n_ctx,depth=depth,
                            model_kwargs=dict(d_model=a.d_model,n_head=a.heads,
                                              norm='rms',residual='lerp',
                                              attention='bilinear'),
                            config=vars(a),evaluations=result['evaluations'],
                            document_version='v6'),
                       path.with_name(path.stem+f'_block{depth}.pt'))
    print('saved',path,flush=True)


if __name__=='__main__':main()
