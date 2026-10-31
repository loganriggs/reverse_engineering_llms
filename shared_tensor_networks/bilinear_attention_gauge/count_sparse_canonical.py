"""Sparse, label-free rank-two canonical function on variable-count language."""

from __future__ import annotations

import json

import numpy as np
import torch

from analyze_count_language import logits,score
from analyze_dual_output_language import factors
from analyze_toy_language import pair_table
from dual_joint_function_canonical import HERE,flatten,unflatten,rotations,matched_cosines
from train_count_language import CountAttention,examples


def semantic_bit_accuracy(table,tokens,targets,attrs,bias):
    prediction=logits([table],tokens,bias)>0
    rows=np.arange(len(tokens))
    a_col=attrs["swap"].astype(int)
    b_col=1-a_col
    return dict(A=float(np.mean(prediction[rows,a_col]==targets[rows,a_col])),
                B=float(np.mean(prediction[rows,b_col]==targets[rows,b_col])))


def effects(row_factors,dictionary):
    result={}
    assignment=matched_cosines(dictionary)[1]
    for bit,index in assignment.items():
        component=unflatten(np.outer(row_factors[:,index],dictionary[index]))
        energy={"A":float(np.sum((component[:,1,:]-component[:,0,:])**2)),
                "B":float(np.sum((component[:,3,:]-component[:,2,:])**2)),
                "F":float(np.sum((component[:,5,:]-component[:,4,:])**2))}
        result[bit]=dict(component=index,energy=energy,
            intended_fraction=energy[bit]/max(sum(energy.values()),1e-30))
    return result


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(10):
        model=CountAttention().double()
        model.load_state_dict(torch.load(HERE/f"count_language_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        heads=[factors(model,h) for h in range(2)]
        total=sum(pair_table(h,np.eye(h["rank"])) for h in heads)
        bias=model.read.bias.detach().numpy()
        matrix=flatten(total)
        u,s,v=np.linalg.svd(matrix,full_matrices=False)
        left=u[:,:2]*np.sqrt(s[:2])
        dictionary=np.sqrt(s[:2,None])*v[:2]
        rank2=left@dictionary
        rotations_all,angles=rotations()
        rotated=np.einsum("tij,jk->tik",rotations_all,dictionary)
        index=int(np.argmin(np.sum(np.abs(rotated),axis=(1,2))))
        sparse_rotation=rotations_all[index]
        sparse_d=sparse_rotation@dictionary
        sparse_l=left@sparse_rotation.T
        assert np.max(np.abs(sparse_l@sparse_d-rank2))<1e-12
        rng=np.random.default_rng(600000+seed)
        random_align=[];random_purity=[]
        for angle in rng.uniform(0,np.pi,128):
            rotation=np.array([[np.cos(angle),-np.sin(angle)],
                               [np.sin(angle),np.cos(angle)]])
            d=rotation@dictionary;l=left@rotation.T
            random_align.append(matched_cosines(d)[0])
            random_purity.extend(e["intended_fraction"]
                                 for e in effects(l,d).values())
        variants={"svd":(left,dictionary),"sparse":(sparse_l,sparse_d)}
        gauges={}
        for name,(l,d) in variants.items():
            gauges[name]=dict(matched_A_B_cosine=matched_cosines(d)[0],
                key_l1=float(np.abs(d).sum()),dictionary=d.tolist(),
                row_factors=l.tolist(),effects=effects(l,d))
        tests={}
        for rate in (.35,.75,1.2):
            tokens_t,targets_t,attrs=examples(5000,8000+seed,False,rate)
            tokens=tokens_t.numpy();targets=targets_t.numpy().astype(float)
            original=score([total],tokens,targets,attrs,bias)
            reduced=score([unflatten(rank2)],tokens,targets,attrs,bias)
            ablations={}
            for bit,e in gauges["sparse"]["effects"].items():
                component=unflatten(np.outer(sparse_l[:,e["component"]],
                                             sparse_d[e["component"]]))
                ablated=unflatten(rank2)-component
                ablations[bit]=dict(
                    semantic_accuracy=semantic_bit_accuracy(ablated,tokens,targets,attrs,bias),
                    joint_accuracy=score([ablated],tokens,targets,attrs,bias)["all"]["joint_accuracy"])
            tests[str(rate)]=dict(original=original,reduced=reduced,
                                  sparse_component_ablations=ablations)
        rows.append(dict(seed=seed,rank2_relative_table_mse=float(
            np.sum((rank2-matrix)**2)/np.sum(matrix**2)),
            selected_angle=float(angles[index]),gauges=gauges,
            random_rotation_alignment_mean=float(np.mean(random_align)),
            random_rotation_intended_effect_mean=float(np.mean(random_purity)),
            shifted_count_tests=tests))
        print(seed,gauges["sparse"]["matched_A_B_cosine"],
              np.mean([e["intended_fraction"] for e in gauges["sparse"]["effects"].values()]),
              flush=True)
    output=dict(n_seeds=10,
        fit="exact weight-only combined 4x6 token table rank-two SVD; minimize key L1 over 4096 orthogonal rotations",
        label_use="A/B contrast vectors and bit-flip effects only after fitting",
        tests="independent variable-count samples at Poisson rates .35, .75, and 1.2",
        rows=rows)
    (HERE/"count_sparse_canonical.json").write_text(json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    run()
