"""Shared-product attention task with overlapping 3-bit token attributes."""

import argparse
import numpy as np
import torch

from train_three_count_language import MAX_KEYS,PAD
from train_shared_pair_count import train


TOKEN_SIGNS=np.array([[1 if (token>>bit)&1 else -1 for bit in range(3)]
                      for token in range(8)],dtype=np.int64)


def examples_dense(n,seed,holdout,rate=.75):
    rng=np.random.default_rng(seed)
    chosen=[];queries=[];remaining=n
    while remaining:
        candidate=rng.poisson(rate,size=(min(250000,max(8192,remaining*7)),8))
        contrast=candidate@TOKEN_SIGNS
        rare=rng.random(len(candidate))<.15
        valid=(candidate.sum(1)<=MAX_KEYS)&np.all(contrast!=0,axis=1)
        withheld=rare&(contrast[:,0]<0)&(contrast[:,1]>0)&(contrast[:,2]<0)
        if holdout:valid &= ~withheld
        selected=candidate[valid][:remaining]
        chosen.append(selected);queries.append(rare[valid][:remaining])
        remaining-=len(selected)
    counts=np.concatenate(chosen)
    rare=np.concatenate(queries)
    bits=counts@TOKEN_SIGNS>0
    targets=np.where(rare[:,None],bits[:,[1,2,0]],bits).astype(np.float32)
    tokens=np.full((n,MAX_KEYS+1),PAD,dtype=np.int64)
    tokens[:,-1]=8+rare
    for row,count in enumerate(counts):
        keys=np.repeat(np.arange(8),count)
        rng.shuffle(keys)
        tokens[row,:len(keys)]=keys
    attrs=dict(rare=rare,heldout=rare&~bits[:,0]&bits[:,1]&~bits[:,2],
               bits=bits,total_keys=counts.sum(1))
    return torch.from_numpy(tokens),torch.from_numpy(targets),attrs


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=3000)
    a=p.parse_args()
    train(a.seed,a.steps,example_source=examples_dense,name='shared_pair_dense',
          task_description='three overlapping token attributes and two shared quadratic parity rules')
