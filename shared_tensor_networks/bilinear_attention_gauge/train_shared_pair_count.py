"""Train three-head attention with two quadratic rules sharing a count bit."""

import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dual_joint_function_canonical import HERE
from train_three_count_language import ThreeCountAttention,examples


class SharedPairCountModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=ThreeCountAttention()
        self.register_buffer('logit_scale',torch.ones(3))
        self.quadratic=nn.Parameter(torch.zeros(2,3,3))
        self.parity_bias=nn.Parameter(torch.zeros(2))

    def forward(self,tokens):
        bits=self.backbone(tokens)
        z=bits/self.logit_scale
        q=.5*(self.quadratic+self.quadratic.transpose(-1,-2))
        parity=torch.einsum('bi,pij,bj->bp',z,q,z)+self.parity_bias
        return torch.cat((bits,parity),dim=1)


def targets(bits):
    parity=torch.stack((torch.logical_xor(bits[:,0].bool(),bits[:,1].bool()),
                        torch.logical_xor(bits[:,1].bool(),bits[:,2].bool())),dim=1)
    return torch.cat((bits,parity.float()),dim=1)


def evaluate(model,tokens,target,attrs):
    model.eval()
    with torch.no_grad():
        logits=model(tokens)
        correct=logits>0
        losses=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    out={}
    for name,mask in (('all',np.ones(len(tokens),dtype=bool)),
                      ('rare_heldout',attrs['heldout'])):
        idx=torch.from_numpy(mask)
        out[name]=dict(n=int(mask.sum()),
            five_output_accuracy=float(torch.all(correct[idx]==target[idx].bool(),dim=1).float().mean()),
            bits_accuracy=float(torch.all(correct[idx,:3]==target[idx,:3].bool(),dim=1).float().mean()),
            pair_accuracy=[float((correct[idx,3+j]==target[idx,3+j].bool()).float().mean())
                           for j in range(2)],
            objective=float(losses[idx,:3].mean()+losses[idx,3:].mean()))
    return out


def train(seed,steps=3000,ntrain=100000,nval=20000,ntest=30000,
          example_source=examples,name='shared_pair_count',
          task_description='paired-token count A/B/C bits plus A XOR B and B XOR C'):
    torch.set_num_threads(1)
    torch.manual_seed(500000+seed)
    train_x,train_bits,_=example_source(ntrain,510000+seed,True)
    val_x,val_bits,val_attrs=example_source(nval,520000+seed,True)
    test_x,test_bits,test_attrs=example_source(ntest,530000+seed,False)
    assert not val_attrs['heldout'].any() and test_attrs['heldout'].sum()>100
    train_y=targets(train_bits);val_y=targets(val_bits);test_y=targets(test_bits)
    model=SharedPairCountModel()
    with torch.no_grad():nn.init.normal_(model.quadratic,std=.05)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    rng=np.random.default_rng(540000+seed)
    best=None;history=[]
    for step in range(1,steps+1):
        idx=torch.from_numpy(rng.integers(0,ntrain,512))
        pred=model(train_x[idx]);target=train_y[idx]
        loss=(F.binary_cross_entropy_with_logits(pred[:,:3],target[:,:3])+
              F.binary_cross_entropy_with_logits(pred[:,3:],target[:,3:]))
        optimizer.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step==1 or step%250==0 or step==steps:
            val=evaluate(model,val_x,val_y,val_attrs)
            objective=val['all']['objective']
            history.append(dict(step=step,validation=val,train_loss=float(loss.detach())))
            if best is None or objective<best['objective']:
                best=dict(step=step,objective=objective,
                          state={k:v.detach().clone() for k,v in model.state_dict().items()})
            print(seed,step,round(objective,5),
                  round(val['all']['five_output_accuracy'],4),flush=True)
    model.load_state_dict(best['state'])
    test=evaluate(model,test_x,test_y,test_attrs)
    result=dict(seed=seed,task=task_description,
        holdout='rare-query A=0 B=1 C=0 absent from train and validation',
        train_sentences=ntrain,validation_sentences=nval,test_sentences=ntest,
        selected_step=best['step'],test=test,history=history)
    torch.save(model.state_dict(),HERE/f'{name}_seed{seed}.pt')
    (HERE/f'{name}_seed{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('selected',seed,best['step'],test['all']['five_output_accuracy'],
          test['rare_heldout']['five_output_accuracy'],flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=3000)
    a=p.parse_args();train(a.seed,a.steps)
