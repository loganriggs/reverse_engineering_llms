"""Supervised cross-entropy rank-two function fit on three-bit attention."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.nn import functional as F

from analyze_three_count_language import head_tables,flatten,unflatten,logits,score
from dual_joint_function_canonical import HERE
from train_three_count_language import ThreeCountAttention,PAD,examples


def count_features(tokens):
    keys=tokens[:,:-1]
    return np.stack([np.sum(keys==k,axis=1) for k in range(8)],axis=1)


def fit_rank2(matrix,bias,train,val,seed,steps=3000,starts=3):
    torch.set_num_threads(1)
    train_tokens,train_targets=train
    val_tokens,val_targets=val
    counts=torch.from_numpy(count_features(train_tokens)).double()
    q=torch.from_numpy(train_tokens[:,-1]-8)
    targets=torch.from_numpy(train_targets).double()
    val_counts=torch.from_numpy(count_features(val_tokens)).double()
    val_q=torch.from_numpy(val_tokens[:,-1]-8)
    val_y=torch.from_numpy(val_targets).double()
    read_bias=torch.from_numpy(bias).double()
    u,s,v=np.linalg.svd(matrix,full_matrices=False)
    base_left=u[:,:2]*np.sqrt(s[:2])
    base_right=np.sqrt(s[:2,None])*v[:2]
    rng=np.random.default_rng(500000+seed)
    best=None
    summaries=[]
    for start in range(starts):
        perturb=.0 if start==0 else .2
        left=torch.nn.Parameter(torch.from_numpy(base_left+
            perturb*rng.standard_normal(base_left.shape)).double())
        right=torch.nn.Parameter(torch.from_numpy(base_right+
            perturb*rng.standard_normal(base_right.shape)).double())
        optimizer=torch.optim.Adam((left,right),lr=.01)
        best_start=None
        for step in range(1,steps+1):
            indices=torch.from_numpy(rng.integers(0,len(q),1024))
            table=(left@right).reshape(2,3,8)
            output=read_bias+torch.einsum("bk,bok->bo",counts[indices],table[q[indices]])
            loss=F.binary_cross_entropy_with_logits(output,targets[indices])
            optimizer.zero_grad();loss.backward()
            optimizer.step()
            if step==1 or step%100==0 or step==steps:
                with torch.no_grad():
                    table=(left@right).reshape(2,3,8)
                    pred=read_bias+torch.einsum("bk,bok->bo",val_counts,table[val_q])
                    validation=float(F.binary_cross_entropy_with_logits(pred,val_y))
                if best_start is None or validation<best_start["validation_bce"]:
                    best_start=dict(validation_bce=validation,step=step,
                                    matrix=(left@right).detach().numpy().copy())
        summaries.append(dict(start=start,validation_bce=best_start["validation_bce"],
                              selected_step=best_start["step"]))
        if best is None or best_start["validation_bce"]<best["validation_bce"]:
            best=dict(start=start,**best_start)
    return best,summaries


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(10):
        model=ThreeCountAttention().double()
        model.load_state_dict(torch.load(HERE/f"three_count_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        model.eval()
        tables=head_tables(model)
        matrix=flatten(sum(tables))
        bias=model.read.bias.detach().numpy()
        train_t,train_y,_=examples(100000,10000+seed,True)
        val_t,val_y,_=examples(20000,20000+seed,True)
        fit,summaries=fit_rank2(matrix,bias,
            (train_t.numpy(),train_y.numpy().astype(float)),
            (val_t.numpy(),val_y.numpy().astype(float)),seed)
        test_t,test_y,attrs=examples(30000,30000+seed,False)
        test=test_t.numpy();targets=test_y.numpy().astype(float)
        reference=logits(tables,test,bias)
        reduced=unflatten(fit["matrix"])
        scored=score([reduced],test,targets,attrs,bias,reference)
        rows.append(dict(seed=seed,selected_start=fit["start"],
            selected_step=fit["step"],validation_bce=fit["validation_bce"],
            start_summaries=summaries,matrix=fit["matrix"].tolist(),
            relative_uniform_table_mse=float(np.sum((fit["matrix"]-matrix)**2)/
                                             np.sum(matrix**2)),scores=scored))
        print(seed,scored["all"]["joint_accuracy"],
              scored["rare_heldout"]["joint_accuracy"],flush=True)
    output=dict(n_seeds=10,n_train=100000,n_validation=20000,n_test=30000,
        fit="rank-two 6x8 factorization optimized for BCE on training labels; three starts, validation BCE selection",
        holdout="rare query A0/B1/C0 absent from both training and validation",
        distinction="supervised task fitting may sacrifice teacher-function fidelity; not unsupervised canonicalization",
        rows=rows)
    (HERE/"three_count_task_fit.json").write_text(json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    run()
