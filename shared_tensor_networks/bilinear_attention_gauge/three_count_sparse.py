"""Label-free sparse orientation of the three-bit rank-three token function."""

from __future__ import annotations

import json

import numpy as np
import scipy.optimize as so
import torch

from analyze_three_count_language import (head_tables,flatten,unflatten,logits,
                                          semantic_accuracy,score)
from dual_intervention_sparse_basis import random_rotation,sparse_rotation
from dual_joint_function_canonical import HERE
from train_three_count_language import ThreeCountAttention,examples


TRUTH=np.zeros((3,8))
for family in range(3):
    TRUTH[family,2*family]=-1/np.sqrt(2)
    TRUTH[family,2*family+1]=1/np.sqrt(2)


def alignment(dictionary):
    similarity=np.abs(dictionary@TRUTH.T)
    similarity/=np.linalg.norm(dictionary,axis=1,keepdims=True)
    row,col=so.linear_sum_assignment(-similarity)
    return float(np.mean(similarity[row,col])),\
        {"ABC"[label]:int(component) for component,label in zip(row,col)}


def effects(left,dictionary):
    assignment=alignment(dictionary)[1]
    outputs={}
    for bit,index in assignment.items():
        table=unflatten(np.outer(left[:,index],dictionary[index]))
        energy={name:float(np.sum((table[:,2*f+1,:]-table[:,2*f,:])**2))
                for f,name in enumerate("ABCF")}
        outputs[bit]=dict(component=index,energy=energy,
            intended_fraction=energy[bit]/max(sum(energy.values()),1e-30))
    return outputs


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(10):
        model=ThreeCountAttention().double()
        model.load_state_dict(torch.load(HERE/f"three_count_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        model.eval()
        total=sum(head_tables(model))
        matrix=flatten(total)
        bias=model.read.bias.detach().numpy()
        u,s,v=np.linalg.svd(matrix,full_matrices=False)
        left=u[:,:3]*np.sqrt(s[:3])
        dictionary=np.sqrt(s[:3,None])*v[:3]
        reduced=left@dictionary
        rotation,sparse_d,optimization=sparse_rotation(dictionary,seed+10000)
        sparse_l=left@rotation.T
        assert np.max(np.abs(sparse_l@sparse_d-reduced))<1e-10
        gauges={}
        for name,(l,d) in {"svd":(left,dictionary),
                            "sparse":(sparse_l,sparse_d)}.items():
            gauges[name]=dict(matched_A_B_C_cosine=alignment(d)[0],
                key_l1=float(np.abs(d).sum()),dictionary=d.tolist(),
                row_factors=l.tolist(),effects=effects(l,d))
        rng=np.random.default_rng(600000+seed)
        random_alignment=[];random_purity=[]
        for _ in range(128):
            r=random_rotation(3,rng)
            d=r@dictionary;l=left@r.T
            random_alignment.append(alignment(d)[0])
            random_purity.extend(e["intended_fraction"]
                                 for e in effects(l,d).values())
        tests={}
        for rate in (.35,.75,1.2):
            token_t,target_t,attrs=examples(10000,80000+seed,False,rate)
            tokens=token_t.numpy();targets=target_t.numpy().astype(float)
            reference=logits([total],tokens,bias)
            base=score([unflatten(reduced)],tokens,targets,attrs,bias,reference)
            deletions={}
            for bit,e in gauges["sparse"]["effects"].items():
                component=unflatten(np.outer(sparse_l[:,e["component"]],
                                             sparse_d[e["component"]]))
                edited=unflatten(reduced)-component
                output=logits([edited],tokens,bias)
                deletions[bit]=dict(
                    semantic_accuracy=semantic_accuracy(output,targets,attrs),
                    joint_accuracy=score([edited],tokens,targets,attrs,bias,
                                         reference)["all"]["joint_accuracy"])
            tests[str(rate)]=dict(reduced=base,sparse_component_deletions=deletions)
        rows.append(dict(seed=seed,rank3_relative_table_mse=float(
            np.sum((reduced-matrix)**2)/np.sum(matrix**2)),
            optimization=optimization,gauges=gauges,
            random_alignment_mean=float(np.mean(random_alignment)),
            random_intended_effect_mean=float(np.mean(random_purity)),
            shifted_count_tests=tests))
        print(seed,gauges["sparse"]["matched_A_B_C_cosine"],
              np.mean([e["intended_fraction"] for e in gauges["sparse"]["effects"].values()]),
              flush=True)
    output=dict(n_seeds=10,
        fitting="weight-only combined 6x8 table rank-three SVD, then label-free key-L1 orthogonal rotation",
        label_use="A/B/C contrasts and bit-flip/deletion scores only after orientation selected",
        tests="independent variable-count sentences at Poisson rates .35, .75, 1.2",
        rows=rows)
    (HERE/"three_count_sparse.json").write_text(json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    run()
