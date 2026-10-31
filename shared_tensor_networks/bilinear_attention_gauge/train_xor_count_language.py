"""Train bilinear attention plus bilinear MLP on count bits and their XOR."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dual_joint_function_canonical import HERE
from train_count_language import MAX_KEYS,PAD,VOCAB,examples as count_examples


class XorCountAttention(nn.Module):
    def __init__(self,width=16,heads=2,qk_width=4,mlp_width=16):
        super().__init__()
        self.heads=heads
        self.qk_width=qk_width
        self.embed=nn.Embedding(VOCAB,width)
        for name in ("q1","k1","q2","k2"):
            setattr(self,name,nn.Linear(width,heads*qk_width,bias=False))
        self.v=nn.Linear(width,heads,bias=False)
        self.mlp_left=nn.Linear(heads+1,mlp_width,bias=False)
        self.mlp_right=nn.Linear(heads+1,mlp_width,bias=False)
        self.read=nn.Linear(mlp_width,3,bias=True)

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
        head_outputs=(s1*s2*values*valid).sum(1)
        hidden=torch.cat((head_outputs,torch.ones_like(head_outputs[:,:1])),dim=1)
        output=self.read(self.mlp_left(hidden)*self.mlp_right(hidden))
        return (output,head_outputs) if return_heads else output


def examples(n,seed,holdout,rate=.75):
    tokens,first_two,attrs=count_examples(n,seed,holdout,rate)
    xor=np.logical_xor(attrs["a_bit"],attrs["b_bit"]).astype(np.float32)
    targets=torch.cat((first_two,torch.from_numpy(xor[:,None])),dim=1)
    return tokens,targets,attrs


def evaluate(model,tokens,targets,attrs):
    model.eval()
    with torch.no_grad():
        output=model(tokens)
        pred=output>0
        correct=(pred==targets.bool()).all(1)
        loss=F.binary_cross_entropy_with_logits(output,targets,reduction="none").mean(1)
    masks={"all":np.ones(len(tokens),dtype=bool),
           "common":~attrs["swap"],
           "rare_seen":attrs["swap"]&~attrs["heldout"],
           "rare_heldout":attrs["heldout"]}
    result={}
    for name,mask in masks.items():
        selected=torch.from_numpy(mask)
        result[name]=dict(n=int(selected.sum()),
            joint_accuracy=float(correct[selected].float().mean()),
            bit_accuracy=float((pred[selected]==targets[selected].bool()).float().mean()),
            xor_accuracy=float((pred[selected,2]==targets[selected,2].bool()).float().mean()),
            bce=float(loss[selected].mean()))
    return result


def train(seed,steps=2500,batch_size=512,ntrain=50000,nval=10000,ntest=20000):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    xtrain,ytrain,_=examples(ntrain,1000+seed,True)
    xval,yval,aval=examples(nval,2000+seed,True)
    xtest,ytest,atest=examples(ntest,3000+seed,False)
    assert np.sum(atest["heldout"])>100
    model=XorCountAttention()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    rng=np.random.default_rng(4000+seed)
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
            validation=evaluate(model,xval,yval,aval)
            test=evaluate(model,xtest,ytest,atest)
            history.append(dict(step=step,train_bce=float(loss.detach()),
                                validation=validation,test=test))
            print(seed,step,float(loss.detach()),
                  validation["all"]["joint_accuracy"],
                  test["rare_heldout"]["joint_accuracy"],flush=True)
    prefix=HERE/f"xor_count_seed{seed}"
    torch.save(model.state_dict(),str(prefix)+".pt")
    result=dict(seed=seed,steps=steps,ntrain=ntrain,nval=nval,ntest=ntest,
        task="count-sign A/B plus XOR(A,B); rare query swaps A/B outputs",
        architecture="two-head bilinear attention, then bilinear MLP on head outputs and constant",
        holdout="rare query with A=0 B=1",history=history)
    (HERE/f"xor_count_seed{seed}.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--steps",type=int,default=2500)
    args=p.parse_args()
    train(args.seed,args.steps)
