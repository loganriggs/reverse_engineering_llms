"""Full next-token CE baseline on the task-rich V5 document mixture."""

import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

OLD_REPO = Path(__file__).resolve().parents[2] / 'reverse_engineer/tensor_language'
if not (OLD_REPO / 'deep_model.py').is_file():
    raise FileNotFoundError(f'{OLD_REPO}/deep_model.py')
sys.path.insert(0, str(OLD_REPO))
from deep_model import DeepModel, SPECS  # noqa: E402
from document_mixture_v5 import sample_document  # noqa: E402


def sample(n, seed):
    rng = random.Random(seed)
    return [sample_document(rng) for _ in range(n)]


def encode(docs, vocab, n_ctx):
    x = torch.full((len(docs), n_ctx), vocab['<PAD>'], dtype=torch.long)
    y = torch.full((len(docs), n_ctx), -100, dtype=torch.long)
    for i, doc in enumerate(docs):
        ids = [vocab[t] for t in doc.tokens]
        assert len(ids) - 1 <= n_ctx
        x[i, :len(ids)-1] = torch.tensor(ids[:-1])
        y[i, :len(ids)-1] = torch.tensor(ids[1:])
    return x, y


@torch.no_grad()
def evaluate(model, docs, data, vocab, device, batch_size=64):
    model.eval()
    losses = collections.defaultdict(list)
    sites = collections.defaultdict(list)
    n_tokens = 0
    total_loss = 0.
    for start in range(0, len(docs), batch_size):
        end = min(start + batch_size, len(docs))
        x, y = (arr[start:end].to(device) for arr in data)
        logits = model(x)
        per = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                              reduction='none', ignore_index=-100).view_as(y)
        for j, doc in enumerate(docs[start:end]):
            valid = len(doc.tokens)-1
            lm_nll = per[j, :valid].sum().item()
            losses[doc.kind].append((lm_nll, valid))
            total_loss += lm_nll
            n_tokens += valid
            for site in doc.sites:
                pos = site.index - 1
                assert pos >= 0 and y[j, pos].item() == vocab[site.target]
                scores = logits[j, pos]
                nll = -F.log_softmax(scores, -1)[vocab[site.target]].item()
                entropy = -sum(p*math.log(p) for p in site.distribution.values()
                               if p > 0)
                correct = (scores.argmax().item() == vocab[site.target])
                sites[site.family].append((nll, entropy, int(correct),
                                           len(site.distribution) == 1))
    return dict(token_nll=total_loss/n_tokens,
                by_kind={kind:dict(token_nll=sum(v for v,_ in rows)/
                                         sum(n for _,n in rows),
                                   documents=len(rows))
                         for kind,rows in losses.items()},
                by_site={family:dict(n=len(rows),
                                     observed_target_nll=sum(r[0] for r in rows)/len(rows),
                                     oracle_entropy=sum(r[1] for r in rows)/len(rows),
                                     deterministic_n=sum(r[3] for r in rows),
                                     deterministic_full_vocab_accuracy=(
                                         sum(r[2] for r in rows if r[3]) /
                                         sum(r[3] for r in rows))
                                     if any(r[3] for r in rows) else None)
                         for family,rows in sites.items()})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--depths',default='1,2')
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--train-n',type=int,default=2000)
    p.add_argument('--test-n',type=int,default=400)
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
    train=sample(a.train_n,a.seed)
    test=sample(a.test_n,a.seed+1)
    train_texts={doc.tokens for doc in train}
    test_overlap={kind:dict(n=sum(d.kind==kind for d in test),
                            exact_document_overlap=sum(d.kind==kind and
                                                       d.tokens in train_texts
                                                       for d in test))
                  for kind in sorted({d.kind for d in test})}
    vocab={word:i for i,word in enumerate(sorted(
        {'<PAD>'} | {word for d in train+test for word in d.tokens}))}
    n_ctx=max(len(d.tokens)-1 for d in train+test)
    train_data=encode(train,vocab,n_ctx)
    test_data=encode(test,vocab,n_ctx)
    output=dict(config=vars(a),n_vocab=len(vocab),n_ctx=n_ctx,
                train_kinds=dict(collections.Counter(d.kind for d in train)),
                test_kinds=dict(collections.Counter(d.kind for d in test)),
                test_exact_document_overlap=test_overlap,
                objective='full next-token CE, token-weighted across sampled documents',
                caveat='Task-rich synthetic pilot; FineWeb lexical/form gate fails.',
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
        start=time.time()
        model.train()
        recent=[]
        for step in range(a.steps):
            idx=torch.tensor(rng.sample(range(len(train)),a.batch_size),
                             dtype=torch.long)
            x,y=(arr[idx].to(device) for arr in train_data)
            opt.zero_grad(set_to_none=True)
            logits=model(x)
            loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite depth={depth} step={step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            recent.append(loss.item())
            if (step+1)%100==0:
                print(depth,step+1,round(loss.item(),4),flush=True)
        result=dict(parameters=model.n_params,
                    train_loss_last50=sum(recent[-50:])/len(recent[-50:]),
                    evaluation=evaluate(model,test,test_data,vocab,device),
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
                            config=vars(a),evaluation=result['evaluation']),
                       path.with_name(path.stem+f'_block{depth}.pt'))
    print('saved',path,flush=True)


if __name__=='__main__':main()
