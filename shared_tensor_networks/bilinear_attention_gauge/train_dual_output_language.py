"""Two-head bilinear retrieval where both independent output bits are needed."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_toy_language import TOKENS


class DualOutputAttention(nn.Module):
    def __init__(self, width=16, heads=2, qk_width=4):
        super().__init__()
        self.heads=heads
        self.qk_width=qk_width
        self.embed=nn.Embedding(len(TOKENS),width)
        for name in ("q1","k1","q2","k2"):
            setattr(self,name,nn.Linear(width,heads*qk_width,bias=False))
        self.v=nn.Linear(width,heads,bias=False)
        self.read=nn.Linear(heads,2,bias=True)

    def forward(self,tokens,return_heads=False):
        z=self.embed(tokens)
        def projected(name):
            raw=getattr(self,name)(z).reshape(len(tokens),4,self.heads,self.qk_width)
            return F.rms_norm(raw,(self.qk_width,))
        q1,k1,q2,k2=[projected(name) for name in ("q1","k1","q2","k2")]
        s1=torch.einsum("bhd,bshd->bsh",q1[:,-1],k1[:,:-1])/self.qk_width
        s2=torch.einsum("bhd,bshd->bsh",q2[:,-1],k2[:,:-1])/self.qk_width
        values=self.v(z)[:,:-1]
        head_outputs=torch.einsum("bsh,bsh->bh",s1*s2,values)
        logits=self.read(head_outputs)
        return (logits,head_outputs,s1,s2) if return_heads else logits


def batch(n,generator,rare_probability=.05,holdout=True):
    bits=torch.randint(0,2,(n,3),generator=generator)
    swap=torch.rand(n,generator=generator)<rare_probability
    if holdout:
        # Rare swap of A0/B1 is absent; both tokens and the other swap cases
        # occur in training, so this is a combination holdout.
        excluded=swap & (bits[:,0]==0) & (bits[:,1]==1)
        bits[excluded,0]=1
    keys=torch.stack((bits[:,0],bits[:,1]+2,bits[:,2]+4),dim=1)
    perm=torch.rand(n,3,generator=generator).argsort(dim=1)
    seq=torch.cat((keys.gather(1,perm),(6+swap.long())[:,None]),dim=1)
    a,b=bits[:,0],bits[:,1]
    targets=torch.stack((torch.where(swap,b,a),torch.where(swap,a,b)),dim=1).float()
    return seq,targets


def exhaustive():
    seq=[];targets=[];attrs=[]
    for a,b,f,swap in itertools.product(range(2),repeat=4):
        for perm in itertools.permutations(range(3)):
            key=(a,2+b,4+f)
            seq.append([key[i] for i in perm]+[6+swap])
            targets.append([b,a] if swap else [a,b])
            attrs.append(dict(A=a,B=b,F=f,swap=swap,
                              heldout=bool(swap and a==0 and b==1)))
    return torch.tensor(seq),torch.tensor(targets,dtype=torch.float32),attrs


def evaluate(model):
    seq,targets,attrs=exhaustive()
    model.eval()
    with torch.no_grad():
        logits=model(seq)
    prediction=(logits>0).float()
    loss=F.binary_cross_entropy_with_logits(logits,targets,reduction="none").mean(1)
    joint=(prediction==targets).all(1)
    def score(mask):
        selected=torch.tensor(mask,dtype=torch.bool)
        return dict(n=int(selected.sum()),joint_accuracy=float(joint[selected].float().mean()),
                    bit_accuracy=float((prediction[selected]==targets[selected]).float().mean()),
                    bce=float(loss[selected].mean()))
    return dict(all=score([True]*len(attrs)),
                common=score([not a["swap"] for a in attrs]),
                rare_seen=score([a["swap"] and not a["heldout"] for a in attrs]),
                rare_heldout=score([a["heldout"] for a in attrs]))


def train(seed=0,steps=2500,batch_size=512,lr=.003):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    generator=torch.Generator().manual_seed(1000+seed)
    model=DualOutputAttention()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    history=[]
    for step in range(1,steps+1):
        model.train()
        seq,targets=batch(batch_size,generator)
        logits=model(seq)
        loss=F.binary_cross_entropy_with_logits(logits,targets)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step==1 or step%250==0 or step==steps:
            row=dict(step=step,train_bce=float(loss.detach()),**evaluate(model))
            history.append(row)
            print(seed,step,row["train_bce"],row["all"]["joint_accuracy"],
                  row["rare_heldout"]["joint_accuracy"],flush=True)
    return model,history


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--steps",type=int,default=2500)
    p.add_argument("--batch-size",type=int,default=512)
    args=p.parse_args()
    model,history=train(args.seed,args.steps,args.batch_size)
    prefix=Path(__file__).with_name(f"dual_output_seed{args.seed}")
    torch.save(model.state_dict(),str(prefix)+".pt")
    Path(str(prefix)+".json").write_text(json.dumps(dict(seed=args.seed,steps=args.steps,
        batch_size=args.batch_size,task="two independent bits; rare swap of outputs",
        holdout="rare swap with A0/B1",history=history),indent=2)+"\n")
