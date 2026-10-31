"""Train a bilinear readout on three-rule attention, then fine-tune jointly."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dual_joint_function_canonical import HERE
from train_three_count_language import ThreeCountAttention,examples


class ComposedCountModel(nn.Module):
    def __init__(self,scale=None):
        super().__init__()
        self.backbone=ThreeCountAttention()
        self.register_buffer('logit_scale',torch.ones(3) if scale is None else scale)
        self.quadratic=nn.Parameter(torch.zeros(3,3))
        self.parity_bias=nn.Parameter(torch.tensor(0.))

    def readout(self,bits):
        z=bits/self.logit_scale
        symmetric=.5*(self.quadratic+self.quadratic.T)
        return torch.einsum('bi,ij,bj->b',z,symmetric,z)+self.parity_bias

    def forward(self,tokens):
        bits=self.backbone(tokens)
        return torch.cat((bits,self.readout(bits)[:,None]),dim=1)


def targets_with_parity(bits):
    parity=torch.logical_xor(bits[:,0].bool(),bits[:,1].bool()).float()
    return torch.cat((bits,parity[:,None]),dim=1)


def evaluation(model,tokens,target,attrs):
    model.eval()
    with torch.no_grad():
        logits=model(tokens)
        pred=logits>0
        truth=target.bool()
        bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    result={}
    for name,mask in (('all',np.ones(len(tokens),dtype=bool)),
                      ('common',~attrs['rare']),
                      ('rare_seen',attrs['rare']&~attrs['heldout']),
                      ('rare_heldout',attrs['heldout'])):
        if not mask.any():continue
        idx=torch.from_numpy(mask)
        result[name]=dict(n=int(idx.sum()),
            bit_joint_accuracy=float(torch.all(pred[idx,:3]==truth[idx,:3],dim=1).float().mean()),
            parity_accuracy=float((pred[idx,3]==truth[idx,3]).float().mean()),
            all_four_accuracy=float(torch.all(pred[idx]==truth[idx],dim=1).float().mean()),
            bit_bce=float(bce[idx,:3].mean()),
            parity_bce=float(bce[idx,3].mean()))
    return result


def train(seed,readout_steps=1200,joint_steps=400,ntrain=100000,nval=20000,ntest=30000):
    torch.set_num_threads(1)
    torch.manual_seed(80000+seed)
    base=ThreeCountAttention()
    base.load_state_dict(torch.load(HERE/f'three_count_seed{seed}.pt',
                                    map_location='cpu',weights_only=True))
    base.eval()
    train_x,train_bits,_=examples(ntrain,10000+seed,True)
    val_x,val_bits,val_attrs=examples(nval,20000+seed,True)
    test_x,test_bits,test_attrs=examples(ntest,30000+seed,False)
    assert not val_attrs['heldout'].any() and test_attrs['heldout'].sum()>100
    train_y=targets_with_parity(train_bits)
    val_y=targets_with_parity(val_bits)
    test_y=targets_with_parity(test_bits)
    with torch.no_grad():
        train_base=torch.cat([base(train_x[i:i+4096])
                              for i in range(0,ntrain,4096)])
        val_base=torch.cat([base(val_x[i:i+4096])
                            for i in range(0,nval,4096)])
    scale=train_base.std(0).clamp_min(1.)
    model=ComposedCountModel(scale)
    model.backbone.load_state_dict(base.state_dict())
    rng=np.random.default_rng(90000+seed)
    readout_opt=torch.optim.Adam((model.quadratic,model.parity_bias),lr=.01)
    best=None;trace=[]
    for step in range(1,readout_steps+1):
        idx=torch.from_numpy(rng.integers(0,ntrain,1024))
        p=model.readout(train_base[idx])
        loss=F.binary_cross_entropy_with_logits(p,train_y[idx,3])
        readout_opt.zero_grad();loss.backward();readout_opt.step()
        if step==1 or step%100==0 or step==readout_steps:
            with torch.no_grad():
                val_p=model.readout(val_base)
                val_loss=float(F.binary_cross_entropy_with_logits(val_p,val_y[:,3]))
            trace.append(dict(phase='readout',step=step,validation_parity_bce=val_loss))
            if best is None or val_loss<best['loss']:
                best=dict(loss=val_loss,step=step,
                          quadratic=model.quadratic.detach().clone(),
                          bias=model.parity_bias.detach().clone())
    with torch.no_grad():
        model.quadratic.copy_(best['quadratic'])
        model.parity_bias.copy_(best['bias'])
    frozen_validation=evaluation(model,val_x,val_y,val_attrs)
    frozen_test=evaluation(model,test_x,test_y,test_attrs)
    torch.save(model.state_dict(),
               HERE/f'three_count_composed_frozen_seed{seed}.pt')

    joint_opt=torch.optim.AdamW(({'params':model.backbone.parameters(),'lr':.0001},
                                 {'params':[model.quadratic,model.parity_bias],'lr':.001}),
                                weight_decay=1e-5)
    selected=None
    for step in range(1,joint_steps+1):
        idx=torch.from_numpy(rng.integers(0,ntrain,512))
        pred=model(train_x[idx]);target=train_y[idx]
        loss=(F.binary_cross_entropy_with_logits(pred[:,:3],target[:,:3])+
              F.binary_cross_entropy_with_logits(pred[:,3],target[:,3]))
        joint_opt.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        joint_opt.step()
        if step==1 or step%100==0 or step==joint_steps:
            val=evaluation(model,val_x,val_y,val_attrs)
            objective=val['all']['bit_bce']+val['all']['parity_bce']
            trace.append(dict(phase='joint',step=step,validation=val,
                              validation_objective=objective))
            if selected is None or objective<selected['objective']:
                selected=dict(step=step,objective=objective,
                              state={k:v.detach().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(selected['state'])
    validation=evaluation(model,val_x,val_y,val_attrs)
    test=evaluation(model,test_x,test_y,test_attrs)
    symmetric=.5*(model.quadratic.detach()+model.quadratic.detach().T)
    output=dict(seed=seed,readout_steps=readout_steps,joint_steps=joint_steps,
        ntrain=ntrain,nval=nval,ntest=ntest,
        initialization='three-rule trained attention; zero quadratic readout',
        task='three count-sign bits plus XOR of first two output positions',
        training_holdout='rare query A=0 B=1 C=0 excluded from train and validation',
        selected_readout_step=best['step'],selected_joint_step=selected['step'],
        saved_frozen_readout_checkpoint=f'three_count_composed_frozen_seed{seed}.pt',
        frozen_readout_validation=frozen_validation,
        frozen_readout_test=frozen_test,
        validation=validation,test=test,
        symmetric_quadratic=symmetric.numpy().tolist(),
        logit_scale=model.logit_scale.detach().numpy().tolist(),
        trace=trace)
    stem=HERE/f'three_count_composed_seed{seed}'
    torch.save(model.state_dict(),str(stem)+'.pt')
    stem.with_suffix('.json').write_text(json.dumps(output,indent=2)+'\n')
    print(seed,'frozen',frozen_test['all']['parity_accuracy'],
          frozen_test['rare_heldout']['parity_accuracy'],
          'joint',test['all']['all_four_accuracy'],
          test['rare_heldout']['all_four_accuracy'],flush=True)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--readout-steps',type=int,default=1200)
    p.add_argument('--joint-steps',type=int,default=400)
    a=p.parse_args()
    train(a.seed,a.readout_steps,a.joint_steps)
