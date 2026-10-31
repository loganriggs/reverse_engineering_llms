"""Train three-head bilinear attention on three variable-count bit rules."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dual_joint_function_canonical import HERE


MAX_KEYS=12
PAD=10
VOCAB=11


class ThreeCountAttention(nn.Module):
    def __init__(self,width=24,heads=3,qk_width=4):
        super().__init__()
        self.heads=heads
        self.qk_width=qk_width
        self.embed=nn.Embedding(VOCAB,width)
        for name in ("q1","k1","q2","k2"):
            setattr(self,name,nn.Linear(width,heads*qk_width,bias=False))
        self.v=nn.Linear(width,heads,bias=False)
        self.read=nn.Linear(heads,3,bias=True)

    def forward(self,tokens,return_heads=False):
        z=self.embed(tokens)
        def projected(name):
            raw=getattr(self,name)(z).reshape(len(tokens),MAX_KEYS+1,
                                               self.heads,self.qk_width)
            return F.rms_norm(raw,(self.qk_width,))
        q1,k1,q2,k2=[projected(name) for name in ("q1","k1","q2","k2")]
        s1=torch.einsum("bhd,bshd->bsh",q1[:,-1],k1[:,:-1])/self.qk_width
        s2=torch.einsum("bhd,bshd->bsh",q2[:,-1],k2[:,:-1])/self.qk_width
        values=self.v(z)[:,:-1]
        valid=(tokens[:,:-1]!=PAD).unsqueeze(-1)
        heads=(s1*s2*values*valid).sum(1)
        output=self.read(heads)
        return (output,heads) if return_heads else output


def examples(n,seed,holdout,rate=.75):
    rng=np.random.default_rng(seed)
    chosen=[];queries=[];remaining=n
    while remaining:
        candidate=rng.poisson(rate,size=(min(250000,max(8192,remaining*7)),8))
        contrast=candidate[:,1:6:2]-candidate[:,0:6:2]
        rare=rng.random(len(candidate))<.15
        valid=(candidate.sum(1)<=MAX_KEYS)&np.all(contrast!=0,axis=1)
        withheld=rare&(contrast[:,0]<0)&(contrast[:,1]>0)&(contrast[:,2]<0)
        if holdout:valid &= ~withheld
        selected=candidate[valid][:remaining]
        chosen.append(selected);queries.append(rare[valid][:remaining])
        remaining-=len(selected)
    counts=np.concatenate(chosen)
    rare=np.concatenate(queries)
    bits=counts[:,1:6:2]>counts[:,0:6:2]
    targets=np.where(rare[:,None],bits[:,[1,2,0]],bits).astype(np.float32)
    tokens=np.full((n,MAX_KEYS+1),PAD,dtype=np.int64)
    tokens[:,-1]=8+rare
    ids=np.arange(8)
    for row,count in enumerate(counts):
        keys=np.repeat(ids,count)
        rng.shuffle(keys)
        tokens[row,:len(keys)]=keys
    attrs=dict(rare=rare,heldout=rare&~bits[:,0]&bits[:,1]&~bits[:,2],
               bits=bits,total_keys=counts.sum(1))
    return torch.from_numpy(tokens),torch.from_numpy(targets),attrs


def evaluate(model,tokens,targets,attrs):
    model.eval()
    with torch.no_grad():
        output=model(tokens)
        pred=output>0
        correct=(pred==targets.bool()).all(1)
        loss=F.binary_cross_entropy_with_logits(output,targets,reduction="none").mean(1)
    masks={"all":np.ones(len(tokens),dtype=bool),
           "common":~attrs["rare"],
           "rare_seen":attrs["rare"]&~attrs["heldout"],
           "rare_heldout":attrs["heldout"]}
    result={}
    for name,mask in masks.items():
        selected=torch.from_numpy(mask)
        result[name]=dict(n=int(selected.sum()),
            joint_accuracy=float(correct[selected].float().mean()),
            bit_accuracy=float((pred[selected]==targets[selected].bool()).float().mean()),
            bce=float(loss[selected].mean()))
    return result


def train(seed,steps=1500,batch_size=512,ntrain=100000,nval=20000,ntest=30000):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    xtrain,ytrain,_=examples(ntrain,10000+seed,True)
    xval,yval,aval=examples(nval,20000+seed,True)
    xtest,ytest,atest=examples(ntest,30000+seed,False)
    assert not np.any(aval["heldout"])
    assert np.sum(atest["heldout"])>100
    model=ThreeCountAttention()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    rng=np.random.default_rng(40000+seed)
    history=[]
    for step in range(1,steps+1):
        model.train()
        indices=torch.from_numpy(rng.integers(0,ntrain,batch_size))
        output=model(xtrain[indices])
        loss=F.binary_cross_entropy_with_logits(output,ytrain[indices])
        optimizer.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step==1 or step%250==0 or step==steps:
            val=evaluate(model,xval,yval,aval)
            test=evaluate(model,xtest,ytest,atest)
            row=dict(step=step,train_bce=float(loss.detach()),validation=val,test=test)
            history.append(row)
            print(seed,step,row["train_bce"],val["all"]["joint_accuracy"],
                  test["rare_heldout"]["joint_accuracy"],flush=True)
    prefix=HERE/f"three_count_seed{seed}"
    torch.save(model.state_dict(),str(prefix)+".pt")
    result=dict(seed=seed,steps=steps,batch_size=batch_size,ntrain=ntrain,
        nval=nval,ntest=ntest,
        task="A/B/C count-sign bits; rare query rotates outputs B/C/A",
        count_law="Poisson(.75) over eight tokens, conditioned on nonzero A/B/C contrasts and <=12 keys",
        holdout="rare query with A=0, B=1, C=0",
        history=history)
    (HERE/f"three_count_seed{seed}.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--steps",type=int,default=1500)
    p.add_argument("--batch-size",type=int,default=512)
    args=p.parse_args()
    train(args.seed,args.steps,args.batch_size)
