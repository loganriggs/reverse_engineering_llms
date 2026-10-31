"""Label-free sparse gauge and component deletions for the nonlinear count toy."""

from __future__ import annotations

import json

import numpy as np
import torch

from analyze_xor_count_language import (attention_from_table,attention_table,
    downstream,flatten,unflatten)
from dual_intervention_sparse_basis import sparse_rotation
from dual_joint_function_canonical import HERE,matched_cosines
from train_xor_count_language import XorCountAttention,examples


def accuracy(logits,targets,attrs):
    prediction=logits>0
    swap=attrs["swap"].astype(int)
    rows=np.arange(len(logits))
    return dict(A=float(np.mean(prediction[rows,swap]==targets[rows,swap])),
                B=float(np.mean(prediction[rows,1-swap]==targets[rows,1-swap])),
                XOR=float(np.mean(prediction[:,2]==targets[:,2])),
                joint=float(np.mean(np.all(prediction==targets,axis=1))))


def one_seed(seed):
    model=XorCountAttention().double()
    model.load_state_dict(torch.load(HERE/f"xor_count_seed{seed}.pt",
                                     map_location="cpu",weights_only=True))
    model.eval()
    matrix=flatten(attention_table(model))
    u,s,v=np.linalg.svd(matrix,full_matrices=False)
    row=u[:,:2]*np.sqrt(s[:2])
    dictionary=np.sqrt(s[:2,None])*v[:2]
    # Rotate only within the fixed rank-two function. Normalizing the rows
    # first makes key L1 an orientation objective rather than a scale choice.
    norms=np.linalg.norm(dictionary,axis=1)
    normalized=dictionary/norms[:,None]
    rotations,_,optimization=sparse_rotation(normalized,seed,starts=16,
                                              sweeps=12,grid=4096)
    sparse=rotations@normalized
    row_sparse=(row*norms[None,:])@rotations.T
    assert np.max(np.abs(row_sparse@sparse-row@dictionary))<1e-12
    svd_axes=normalized
    cosine,match=matched_cosines(sparse)
    tokens_t,targets_t,attrs=examples(30000,6000+seed,False)
    tokens=tokens_t.numpy();targets=targets_t.numpy().astype(bool)
    baseline=downstream(model,attention_from_table(unflatten(row_sparse@sparse),tokens))
    masks={"all":np.ones(len(tokens),dtype=bool),"rare_heldout":attrs["heldout"]}
    effects={}
    for name,index in match.items():
        component=unflatten(np.outer(row_sparse[:,index],sparse[index]))
        deleted=unflatten(row_sparse@sparse)-component
        logits=downstream(model,attention_from_table(deleted,tokens))
        effects[name]={}
        for split,mask in masks.items():
            split_attrs={key:value[mask] for key,value in attrs.items()}
            effects[name][split]=accuracy(logits[mask],targets[mask],split_attrs)
        effects[name]["rms_logit_edit"]=float(np.sqrt(np.mean((logits-baseline)**2)))
    full={split:accuracy(baseline[mask],targets[mask],
            {key:value[mask] for key,value in attrs.items()})
          for split,mask in masks.items()}
    # These contrasts are evaluation labels, never inputs to orientation fit.
    random=np.random.default_rng(7000+seed)
    random_cosines=[]
    for _ in range(128):
        theta=random.uniform(0,np.pi)
        rotation=np.array([[np.cos(theta),-np.sin(theta)],
                           [np.sin(theta),np.cos(theta)]])
        random_cosines.append(matched_cosines(rotation@svd_axes)[0])
    return dict(seed=seed,singular_values=s.tolist(),
        sparse_key_l1=float(np.abs(sparse).sum()),
        svd_key_l1=float(np.abs(svd_axes).sum()),
        sparse_axis_cosine=cosine,
        svd_axis_cosine=matched_cosines(svd_axes)[0],
        random_axis_cosine_mean=float(np.mean(random_cosines)),
        random_axis_cosine_p95=float(np.quantile(random_cosines,.95)),
        match=match,dictionary=sparse.tolist(),row_factors=row_sparse.tolist(),
        optimization=optimization,baseline=full,component_deletion=effects)


def run():
    torch.set_num_threads(1)
    rows=[one_seed(seed) for seed in range(10)]
    output=dict(n_seeds=10,n_test_per_seed=30000,
        fit="exact weight-only rank-2 head table, key-L1 orthogonal rotation; no task labels",
        evaluation="post-fit known A/B axis alignment and deletion of function components before original nonlinear MLP",
        rows=rows)
    (HERE/"xor_count_sparse.json").write_text(json.dumps(output,indent=2)+"\n")
    for row in rows:
        print(row["seed"],round(row["sparse_axis_cosine"],5),
              row["baseline"]["all"],
              row["component_deletion"]["A"]["all"],
              row["component_deletion"]["B"]["all"],flush=True)
    return output


if __name__=="__main__":
    run()
