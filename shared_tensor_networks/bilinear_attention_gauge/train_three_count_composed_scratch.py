"""Jointly train bilinear attention and quadratic readout from random weights."""

import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dual_joint_function_canonical import HERE
from train_three_count_composed import ComposedCountModel,evaluation,targets_with_parity
from train_three_count_language import examples


def train(seed,steps=3000,batch_size=512,ntrain=100000,nval=20000,ntest=30000):
    torch.set_num_threads(1)
    torch.manual_seed(300000+seed)
    train_x,bits,_=examples(ntrain,110000+seed,True)
    val_x,val_bits,val_attrs=examples(nval,120000+seed,True)
    test_x,test_bits,test_attrs=examples(ntest,130000+seed,False)
    assert not val_attrs['heldout'].any() and test_attrs['heldout'].sum()>100
    train_y=targets_with_parity(bits)
    val_y=targets_with_parity(val_bits)
    test_y=targets_with_parity(test_bits)
    model=ComposedCountModel()
    with torch.no_grad():
        nn.init.normal_(model.quadratic,std=.05)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    rng=np.random.default_rng(400000+seed)
    best=None;history=[]
    for step in range(1,steps+1):
        idx=torch.from_numpy(rng.integers(0,ntrain,batch_size))
        pred=model(train_x[idx]);target=train_y[idx]
        bit=F.binary_cross_entropy_with_logits(pred[:,:3],target[:,:3])
        parity=F.binary_cross_entropy_with_logits(pred[:,3],target[:,3])
        loss=bit+parity
        optimizer.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step==1 or step%250==0 or step==steps:
            val=evaluation(model,val_x,val_y,val_attrs)
            objective=val['all']['bit_bce']+val['all']['parity_bce']
            history.append(dict(step=step,validation=val,
                                validation_objective=objective,
                                train_loss=float(loss.detach())))
            if best is None or objective<best['objective']:
                best=dict(step=step,objective=objective,
                          state={k:v.detach().clone() for k,v in model.state_dict().items()})
            print(seed,step,round(objective,4),
                  round(val['all']['all_four_accuracy'],4),flush=True)
    model.load_state_dict(best['state'])
    validation=evaluation(model,val_x,val_y,val_attrs)
    test=evaluation(model,test_x,test_y,test_attrs)
    output=dict(seed=seed,steps=steps,batch_size=batch_size,ntrain=ntrain,
        nval=nval,ntest=ntest,initialization='all parameters random',
        task='A/B/C count-sign bits and XOR of first two outputs',
        holdout='rare query A=0 B=1 C=0 excluded from train and validation',
        selected_step=best['step'],validation=validation,test=test,
        history=history)
    stem=HERE/f'three_count_composed_scratch_seed{seed}'
    torch.save(model.state_dict(),str(stem)+'.pt')
    stem.with_suffix('.json').write_text(json.dumps(output,indent=2)+'\n')
    print('selected',seed,best['step'],test['all']['all_four_accuracy'],
          test['rare_heldout']['all_four_accuracy'],flush=True)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=3000)
    a=p.parse_args()
    train(a.seed,a.steps)
