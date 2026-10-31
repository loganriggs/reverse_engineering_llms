"""Train four-rule bilinear attention before a controlled weak-rule stress."""

import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dual_joint_function_canonical import HERE
from train_three_count_language import MAX_KEYS, PAD, ThreeCountAttention


class FourCountAttention(ThreeCountAttention):
    def __init__(self):
        super().__init__(width=32, heads=4, qk_width=4)
        self.read = nn.Linear(4, 4, bias=True)


def examples(n, seed, holdout, rate=.75):
    rng = np.random.default_rng(seed)
    selected = []
    queries = []
    remaining = n
    while remaining:
        candidate = rng.poisson(rate, size=(min(250000,max(8192,remaining*9)),8))
        contrast = candidate[:,1::2] - candidate[:,::2]
        rare = rng.random(len(candidate)) < .15
        valid = (candidate.sum(1)<=MAX_KEYS)&np.all(contrast!=0,axis=1)
        withheld = rare&(contrast[:,0]<0)&(contrast[:,1]>0)&(contrast[:,2]<0)
        if holdout:
            valid &= ~withheld
        take = candidate[valid][:remaining]
        selected.append(take)
        queries.append(rare[valid][:remaining])
        remaining -= len(take)
    counts = np.concatenate(selected)
    rare = np.concatenate(queries)
    bits = counts[:,1::2]>counts[:,::2]
    targets = np.where(rare[:,None],bits[:,[1,2,0,3]],bits).astype(np.float32)
    tokens = np.full((n,MAX_KEYS+1),PAD,dtype=np.int64)
    tokens[:,-1] = 8+rare
    ids = np.arange(8)
    for row,count in enumerate(counts):
        keys = np.repeat(ids,count)
        rng.shuffle(keys)
        tokens[row,:len(keys)] = keys
    attrs = dict(rare=rare,heldout=rare&~bits[:,0]&bits[:,1]&~bits[:,2],
                 bits=bits,total_keys=counts.sum(1))
    return torch.from_numpy(tokens),torch.from_numpy(targets),attrs


def evaluate(model,tokens,targets,attrs):
    model.eval()
    with torch.no_grad():
        logits = model(tokens)
        predicted = logits>0
        correct = (predicted==targets.bool()).all(1)
        loss = F.binary_cross_entropy_with_logits(logits,targets,reduction='none').mean(1)
    masks = dict(all=np.ones(len(tokens),dtype=bool),common=~attrs['rare'],
                 rare_seen=attrs['rare']&~attrs['heldout'],
                 rare_heldout=attrs['heldout'])
    return {name:dict(n=int(mask.sum()),
               joint_accuracy=float(correct[torch.from_numpy(mask)].float().mean()),
               bit_accuracy=float((predicted[torch.from_numpy(mask)]==
                                   targets[torch.from_numpy(mask)].bool()).float().mean()),
               bce=float(loss[torch.from_numpy(mask)].mean()))
            for name,mask in masks.items()}


def train(seed,steps=2000,batch_size=512,ntrain=100000,nval=10000,ntest=30000):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    train_x,train_y,_=examples(ntrain,10000+seed,True)
    val_x,val_y,val_attrs=examples(nval,20000+seed,True)
    test_x,test_y,test_attrs=examples(ntest,30000+seed,False)
    assert not np.any(val_attrs['heldout'])
    assert np.sum(test_attrs['heldout'])>100
    model=FourCountAttention()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    rng=np.random.default_rng(40000+seed)
    history=[]
    for step in range(1,steps+1):
        model.train()
        indices=torch.from_numpy(rng.integers(0,ntrain,batch_size))
        output=model(train_x[indices])
        loss=F.binary_cross_entropy_with_logits(output,train_y[indices])
        optimizer.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step==1 or step%250==0 or step==steps:
            validation=evaluate(model,val_x,val_y,val_attrs)
            test=evaluate(model,test_x,test_y,test_attrs)
            history.append(dict(step=step,train_bce=float(loss.detach()),
                                validation=validation,test=test))
            print(seed,step,'val',validation['all']['joint_accuracy'],
                  'rare',test['rare_heldout']['joint_accuracy'],flush=True)
    prefix=HERE/f'four_count_weak_seed{seed}'
    torch.save(model.state_dict(),str(prefix)+'.pt')
    result=dict(seed=seed,steps=steps,ntrain=ntrain,nval=nval,ntest=ntest,
        architecture='four-head bilinear attention, four count-sign logits',
        task='A/B/C/D count-sign bits; rare query rotates A/B/C outputs to B/C/A; D is unchanged',
        holdout='rare query with A=0,B=1,C=0, both D values',
        post_training_calibration='none; weak-rule scaling applied only during analysis',
        history=history)
    (HERE/f'four_count_weak_seed{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=2000)
    args=parser.parse_args()
    train(args.seed,args.steps)
