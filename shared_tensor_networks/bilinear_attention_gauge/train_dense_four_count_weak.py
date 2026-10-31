"""Four-rule bilinear attention with dense, correlated token attributes."""

import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dual_joint_function_canonical import HERE


NKEYS=7
NTOKENS=16
QUERY_IDS=(16,17)
SIGNS=np.array([[1 if (token>>bit)&1 else -1 for bit in range(4)]
                for token in range(NTOKENS)],dtype=np.float64)
MIX=np.array([[1.,.25,0.,0.],
              [0.,1.,.25,0.],
              [.25,0.,1.,0.],
              [.45,.35,0.,1.]],dtype=np.float64)
FEATURES=SIGNS@MIX.T


class DenseFourCountAttention(nn.Module):
    def __init__(self,width=32,heads=4,qk_width=4):
        super().__init__()
        self.heads=heads
        self.qk_width=qk_width
        self.embed=nn.Embedding(NTOKENS+2,width)
        for name in ('q1','k1','q2','k2'):
            setattr(self,name,nn.Linear(width,heads*qk_width,bias=False))
        self.v=nn.Linear(width,heads,bias=False)
        self.read=nn.Linear(heads,4,bias=True)

    def forward(self,tokens):
        z=self.embed(tokens)
        def projected(name):
            raw=getattr(self,name)(z).reshape(len(tokens),NKEYS+1,
                                               self.heads,self.qk_width)
            return F.rms_norm(raw,(self.qk_width,))
        q1,k1,q2,k2=[projected(name) for name in ('q1','k1','q2','k2')]
        s1=torch.einsum('bhd,bshd->bsh',q1[:,-1],k1[:,:-1])/self.qk_width
        s2=torch.einsum('bhd,bshd->bsh',q2[:,-1],k2[:,:-1])/self.qk_width
        values=self.v(z)[:,:-1]
        head_outputs=(s1*s2*values).sum(1)
        return self.read(head_outputs)


def examples(n,seed,holdout,rare_probability=.15):
    rng=np.random.default_rng(seed)
    selected=[];queries=[];remaining=n
    while remaining:
        keys=rng.integers(0,NTOKENS,size=(max(8192,remaining*2),NKEYS))
        scores=FEATURES[keys].sum(axis=1)
        rare=rng.random(len(keys))<rare_probability
        bits=scores>0
        valid=np.all(np.abs(scores)>.15,axis=1)
        withheld=rare&~bits[:,0]&bits[:,1]&~bits[:,2]
        if holdout:valid &= ~withheld
        take=keys[valid][:remaining]
        selected.append(take);queries.append(rare[valid][:remaining])
        remaining-=len(take)
    keys=np.concatenate(selected)
    rare=np.concatenate(queries)
    scores=FEATURES[keys].sum(axis=1)
    bits=scores>0
    target=np.where(rare[:,None],bits[:,[1,2,0,3]],bits).astype(np.float32)
    tokens=np.concatenate((keys,(16+rare.astype(np.int64))[:,None]),axis=1)
    attrs=dict(rare=rare,heldout=rare&~bits[:,0]&bits[:,1]&~bits[:,2],
               bits=bits,scores=scores)
    return torch.from_numpy(tokens),torch.from_numpy(target),attrs


def evaluate(model,tokens,target,attrs):
    model.eval()
    with torch.no_grad():
        logits=model(tokens)
        correct=((logits>0)==target.bool()).all(1)
        loss=F.binary_cross_entropy_with_logits(logits,target,reduction='none').mean(1)
    masks=dict(all=np.ones(len(tokens),dtype=bool),
               common=~attrs['rare'],rare_seen=attrs['rare']&~attrs['heldout'],
               rare_heldout=attrs['heldout'])
    return {name:dict(n=int(mask.sum()),
                      joint_accuracy=float(correct[torch.from_numpy(mask)].float().mean()),
                      bce=float(loss[torch.from_numpy(mask)].mean()))
            for name,mask in masks.items()}


def train(seed,steps=2500,batch_size=512,ntrain=100000,nval=10000,ntest=30000):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    train_x,train_y,_=examples(ntrain,10000+seed,True)
    val_x,val_y,val_attrs=examples(nval,20000+seed,True)
    test_x,test_y,test_attrs=examples(ntest,30000+seed,False)
    assert not np.any(val_attrs['heldout'])
    assert np.sum(test_attrs['heldout'])>100
    model=DenseFourCountAttention()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    rng=np.random.default_rng(40000+seed)
    history=[];best_validation_bce=float('inf');selected_step=None;selected_state=None
    for step in range(1,steps+1):
        model.train()
        indices=torch.from_numpy(rng.integers(0,ntrain,batch_size))
        logits=model(train_x[indices])
        loss=F.binary_cross_entropy_with_logits(logits,train_y[indices])
        optimizer.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step==1 or step%250==0 or step==steps:
            validation=evaluate(model,val_x,val_y,val_attrs)
            test=evaluate(model,test_x,test_y,test_attrs)
            history.append(dict(step=step,train_bce=float(loss.detach()),
                                validation=validation,test=test))
            if validation['all']['bce']<best_validation_bce:
                best_validation_bce=validation['all']['bce']
                selected_step=step
                selected_state={name:value.detach().clone()
                                for name,value in model.state_dict().items()}
            print(seed,step,'val',validation['all']['joint_accuracy'],
                  'rare',test['rare_heldout']['joint_accuracy'],flush=True)
    assert selected_state is not None
    model.load_state_dict(selected_state)
    selected_validation=evaluate(model,val_x,val_y,val_attrs)
    selected_test=evaluate(model,test_x,test_y,test_attrs)
    prefix=HERE/f'dense_four_count_seed{seed}'
    torch.save(model.state_dict(),str(prefix)+'.pt')
    (HERE/f'dense_four_count_seed{seed}.json').write_text(json.dumps(dict(
        seed=seed,steps=steps,ntrain=ntrain,nval=nval,ntest=ntest,
        feature_mix=MIX.tolist(),
        task='four correlated dense-coded token attributes, rare B/C/A/D query',
        holdout='rare query with A=0 B=1 C=0, both D values',
        checkpoint_selection='minimum validation BCE among logged steps; rare heldout test not used',
        selected_step=selected_step,selected_validation=selected_validation,
        selected_test=selected_test,history=history),indent=2)+'\n')
    return model,history


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=2500)
    args=parser.parse_args()
    train(args.seed,args.steps)
