"""Train a two-head bilinear-attention toy language on CPU.

Input: a permutation of A{bit}, B{bit}, F{bit}, then QA or QB.
Target: the bit carried by the requested class. QB is rare in training and
QB with A1,B1 is entirely held out, while all tokens are trained elsewhere.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


TOKENS=("A0","A1","B0","B1","F0","F1","QA","QB")
LENGTH=4
NHEAD=2
HD=4
WIDTH=16


class BilinearAttentionToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(len(TOKENS),WIDTH)
        self.q1=nn.Linear(WIDTH,NHEAD*HD,bias=False)
        self.k1=nn.Linear(WIDTH,NHEAD*HD,bias=False)
        self.q2=nn.Linear(WIDTH,NHEAD*HD,bias=False)
        self.k2=nn.Linear(WIDTH,NHEAD*HD,bias=False)
        self.v=nn.Linear(WIDTH,NHEAD*HD,bias=False)
        self.o=nn.Linear(NHEAD*HD,WIDTH,bias=False)
        self.read=nn.Linear(WIDTH,2,bias=False)

    def factors(self,tokens):
        z=self.embed(tokens)
        def qk(layer):
            return F.rms_norm(layer(z).reshape(*z.shape[:2],NHEAD,HD),(HD,))
        return qk(self.q1),qk(self.k1),qk(self.q2),qk(self.k2),\
               self.v(z).reshape(*z.shape[:2],NHEAD,HD)

    def forward(self,tokens,return_heads=False):
        q1,k1,q2,k2,value=self.factors(tokens)
        s1=torch.einsum("bhd,bshd->bsh",q1[:,-1],k1[:,:-1])/HD
        s2=torch.einsum("bhd,bshd->bsh",q2[:,-1],k2[:,:-1])/HD
        heads=torch.einsum("bsh,bshd->bhd",s1*s2,value[:,:-1])
        logits=self.read(self.o(heads.reshape(len(tokens),NHEAD*HD)))
        return (logits,heads,s1,s2) if return_heads else logits


def batch(n, generator, rare_probability=.05, holdout=True):
    bits=torch.randint(0,2,(n,3),generator=generator)
    ask_b=torch.rand(n,generator=generator)<rare_probability
    if holdout:
        excluded=ask_b & (bits[:,0]==1) & (bits[:,1]==1)
        bits[excluded,0]=0
    keys=torch.stack((bits[:,0],bits[:,1]+2,bits[:,2]+4),dim=1)
    perm=torch.rand(n,3,generator=generator).argsort(dim=1)
    keys=keys.gather(1,perm)
    tokens=torch.cat((keys,(6+ask_b.long())[:,None]),dim=1)
    labels=torch.where(ask_b,bits[:,1],bits[:,0])
    return tokens,labels


def exhaustive():
    rows=[]
    labels=[]
    attrs=[]
    for abit,bbit,fbit,qbit in itertools.product(range(2),repeat=4):
        for perm in itertools.permutations(range(3)):
            keys=[abit,2+bbit,4+fbit]
            rows.append([keys[i] for i in perm]+[6+qbit])
            labels.append(bbit if qbit else abit)
            attrs.append(dict(A=abit,B=bbit,F=fbit,Q=qbit,
                              heldout=bool(qbit and abit and bbit)))
    return torch.tensor(rows),torch.tensor(labels),attrs


def evaluate(model):
    tokens,labels,attrs=exhaustive()
    model.eval()
    with torch.no_grad():
        logits=model(tokens)
        pred=logits.argmax(-1)
        ce=F.cross_entropy(logits,labels,reduction="none")
    def score(mask):
        idx=torch.tensor(mask,dtype=torch.bool)
        return dict(n=int(idx.sum()),accuracy=float((pred[idx]==labels[idx]).float().mean()),
                    ce=float(ce[idx].mean()))
    return dict(all=score([True]*len(attrs)),
                common=score([a["Q"]==0 for a in attrs]),
                rare_seen=score([a["Q"]==1 and not a["heldout"] for a in attrs]),
                rare_heldout=score([a["heldout"] for a in attrs]))


def train(seed=0,steps=3000,batch_size=512,lr=.003):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    generator=torch.Generator().manual_seed(seed+1000)
    model=BilinearAttentionToy()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    history=[]
    for step in range(1,steps+1):
        model.train()
        tokens,labels=batch(batch_size,generator)
        logits=model(tokens)
        loss=F.cross_entropy(logits,labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step==1 or step%250==0 or step==steps:
            row=dict(step=step,train_ce=float(loss.detach()),**evaluate(model))
            history.append(row)
            print(seed,step,row["train_ce"],row["all"]["accuracy"],
                  row["rare_heldout"]["accuracy"],flush=True)
    return model,history


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--steps",type=int,default=3000)
    p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--output-prefix",type=Path,default=Path(__file__).with_name("toy_language"))
    args=p.parse_args()
    model,history=train(args.seed,args.steps,args.batch_size)
    prefix=Path(str(args.output_prefix)+f"_seed{args.seed}")
    torch.save(model.state_dict(),str(prefix)+".pt")
    Path(str(prefix)+".json").write_text(json.dumps(dict(seed=args.seed,steps=args.steps,
        batch_size=args.batch_size,tokens=TOKENS,architecture="two-head bilinear attention",
        holdout="QB with A1 and B1; all tokens seen in other combinations",
        history=history),indent=2)+"\n")
