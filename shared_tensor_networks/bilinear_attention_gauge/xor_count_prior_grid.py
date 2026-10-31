"""Compare three exact-gauge legibility priors on the nonlinear attention toy."""

from __future__ import annotations

import json

import numpy as np
import torch

from analyze_xor_count_language import attention_table,attention_from_table,downstream,flatten,unflatten
from dual_joint_function_canonical import HERE,matched_cosines
from train_xor_count_language import XorCountAttention,examples
from xor_count_sparse import accuracy


def rotate(row,dictionary,angle):
    c=np.cos(angle);s=np.sin(angle)
    rotation=np.array([[c,-s],[s,c]])
    return row@rotation.T,rotation@dictionary


def best_angle(array,angles):
    best=(np.inf,0.)
    for begin in range(0,len(angles),128):
        subset=angles[begin:begin+128]
        c=np.cos(subset);s=np.sin(subset)
        first=c[:,None]*array[None,0]-s[:,None]*array[None,1]
        second=s[:,None]*array[None,0]+c[:,None]*array[None,1]
        values=np.abs(first).sum(1)+np.abs(second).sum(1)
        index=int(np.argmin(values))
        if values[index]<best[0]:best=(float(values[index]),float(subset[index]))
    return best


def counts(tokens):
    out=np.zeros((len(tokens),6),dtype=float)
    for key in range(6):out[:,key]=np.sum(tokens[:,:-1]==key,axis=1)
    return out


def one_seed(seed):
    model=XorCountAttention().double()
    model.load_state_dict(torch.load(HERE/f"xor_count_seed{seed}.pt",
                                     map_location="cpu",weights_only=True))
    model.eval()
    u,s,v=np.linalg.svd(flatten(attention_table(model)),full_matrices=False)
    dictionary=v[:2]
    row=u[:,:2]*s[:2]
    fit_tokens,_,_=examples(10000,1000+seed,True)
    activations=counts(fit_tokens.numpy())@dictionary.T
    angles=np.linspace(0,np.pi,4096,endpoint=False)
    objectives={"svd":(0.,0.),
                "key_l1":best_angle(dictionary,angles),
                "row_l1":best_angle(row.T,angles),
                "activation_l1":best_angle(activations.T,angles)}
    test_tokens_t,targets_t,attrs=examples(30000,6000+seed,False)
    tokens=test_tokens_t.numpy();targets=targets_t.numpy().astype(bool)
    result={}
    for name,(objective,angle) in objectives.items():
        rotated_row,rotated_dict=rotate(row,dictionary,angle)
        assert np.max(np.abs(rotated_row@rotated_dict-row@dictionary))<1e-12
        alignment,match=matched_cosines(rotated_dict)
        baseline=downstream(model,attention_from_table(
            unflatten(rotated_row@rotated_dict),tokens))
        deleted={}
        for label,index in match.items():
            reduced=rotated_row@rotated_dict-np.outer(rotated_row[:,index],
                                                       rotated_dict[index])
            logits=downstream(model,attention_from_table(unflatten(reduced),tokens))
            deleted[label]=accuracy(logits,targets,attrs)
        result[name]=dict(objective=objective,angle=angle,
            alignment=alignment,key_l1=float(np.abs(rotated_dict).sum()),
            row_l1=float(np.abs(rotated_row).sum()),
            activation_l1=float(np.abs(counts(fit_tokens.numpy())@
                                             rotated_dict.T).sum()),
            baseline=accuracy(baseline,targets,attrs),deletion=deleted,
            dictionary=rotated_dict.tolist())
    return dict(seed=seed,fit_n=10000,test_n=30000,results=result)


def run():
    torch.set_num_threads(1)
    rows=[one_seed(seed) for seed in range(10)]
    output=dict(fit="exact rank-two weight-only SVD; then orthogonal gauge rotation",
        objectives="key coefficient L1; output/head-factor L1; unlabeled train-sequence activation L1; SVD control",
        evaluation="independent test sequences with whole rare query/bit combination held out from activation fit",
        rows=rows)
    (HERE/"xor_count_prior_grid.json").write_text(json.dumps(output,indent=2)+"\n")
    for name in rows[0]["results"]:
        selected=[r["results"][name] for r in rows]
        print(name,"alignment",np.mean([x["alignment"] for x in selected]),
              "A deletion",{k:np.mean([x["deletion"]["A"][k]
                                       for x in selected]) for k in ("A","B","XOR")},
              "B deletion",{k:np.mean([x["deletion"]["B"][k]
                                       for x in selected]) for k in ("A","B","XOR")},
              flush=True)
    return output


if __name__=="__main__":run()
