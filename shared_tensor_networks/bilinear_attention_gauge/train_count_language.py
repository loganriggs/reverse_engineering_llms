"""Train two-head bilinear attention on variable-count A/B language."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HERE=Path(__file__).parent
MAX_KEYS=8
PAD=8
VOCAB=9


class CountAttention(nn.Module):
    def __init__(self,width=16,heads=2,qk_width=4):
        super().__init__()
        self.heads=heads
        self.qk_width=qk_width
        self.embed=nn.Embedding(VOCAB,width)
        for name in ("q1","k1","q2","k2"):
            setattr(self,name,nn.Linear(width,heads*qk_width,bias=False))
        self.v=nn.Linear(width,heads,bias=False)
        self.read=nn.Linear(heads,2,bias=True)

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
        head_outputs=(s1*s2*values*valid).sum(dim=1)
        output=self.read(head_outputs)
        return (output,head_outputs) if return_heads else output


def examples(n,seed,holdout,rate=.75):
    """Independent Poisson token counts, rejecting A/B ties and >8 keys."""
    rng=np.random.default_rng(seed)
    chosen=[];swaps=[]
    remaining=n
    while remaining:
        candidate=rng.poisson(rate,size=(min(200000,max(4096,remaining*5)),6))
        a=candidate[:,1]-candidate[:,0]
        b=candidate[:,3]-candidate[:,2]
        swap=rng.random(len(candidate))<.15
        valid=(candidate.sum(1)<=MAX_KEYS)&(a!=0)&(b!=0)
        if holdout:
            valid &= ~(swap&(a<0)&(b>0))
        selected=candidate[valid][:remaining]
        chosen.append(selected)
        swaps.append(swap[valid][:remaining])
        remaining-=len(selected)
    counts=np.concatenate(chosen)
    swap=np.concatenate(swaps)
    a_bit=counts[:,1]>counts[:,0]
    b_bit=counts[:,3]>counts[:,2]
    tokens=np.full((n,MAX_KEYS+1),PAD,dtype=np.int64)
    tokens[:,-1]=6+swap
    ids=np.arange(6)
    for row,count in enumerate(counts):
        keys=np.repeat(ids,count)
        rng.shuffle(keys)
        tokens[row,:len(keys)]=keys
    targets=np.stack((np.where(swap,b_bit,a_bit),
                      np.where(swap,a_bit,b_bit)),axis=1).astype(np.float32)
    attrs=dict(swap=swap,heldout=swap&~a_bit&b_bit,
               a_bit=a_bit,b_bit=b_bit,total_keys=counts.sum(1))
    return torch.from_numpy(tokens),torch.from_numpy(targets),attrs


def evaluate(model,tokens,targets,attrs):
    model.eval()
    with torch.no_grad():
        output=model(tokens)
        prediction=output>0
        correct=(prediction==targets.bool()).all(1)
        bce=F.binary_cross_entropy_with_logits(output,targets,reduction="none").mean(1)
    masks={"all":np.ones(len(tokens),dtype=bool),
           "common":~attrs["swap"],
           "rare_seen":attrs["swap"]&~attrs["heldout"],
           "rare_heldout":attrs["heldout"]}
    result={}
    for name,mask in masks.items():
        selected=torch.from_numpy(mask)
        result[name]=dict(n=int(selected.sum()),
            joint_accuracy=float(correct[selected].float().mean()),
            bit_accuracy=float((prediction[selected]==targets[selected].bool()).float().mean()),
            bce=float(bce[selected].mean()))
    return result


def train(seed,steps=2500,batch_size=512,ntrain=50000,nval=10000,ntest=20000):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    xtrain,ytrain,_=examples(ntrain,1000+seed,holdout=True)
    xval,yval,aval=examples(nval,2000+seed,holdout=True)
    xtest,ytest,atest=examples(ntest,3000+seed,holdout=False)
    assert not np.any(aval["heldout"])
    assert atest["heldout"].sum()>100
    model=CountAttention()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    rng=np.random.default_rng(4000+seed)
    history=[]
    for step in range(1,steps+1):
        model.train()
        indices=torch.from_numpy(rng.integers(0,ntrain,batch_size))
        logits=model(xtrain[indices])
        loss=F.binary_cross_entropy_with_logits(logits,ytrain[indices])
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
    prefix=HERE/f"count_language_seed{seed}"
    torch.save(model.state_dict(),str(prefix)+".pt")
    result=dict(seed=seed,steps=steps,batch_size=batch_size,ntrain=ntrain,
        nval=nval,ntest=ntest,task="two independent count-contrast bits; rare query swaps output order",
        count_law="independent Poisson(.75) token counts, A/B ties and >8 keys rejected",
        holdout="rare query with A count0>count1 and B count1>count0",
        history=history)
    (prefix.with_suffix(".json")).write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--steps",type=int,default=2500)
    parser.add_argument("--batch-size",type=int,default=512)
    args=parser.parse_args()
    train(args.seed,args.steps,args.batch_size)
