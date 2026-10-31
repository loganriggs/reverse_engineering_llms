"""Group sparse key atoms by exact query/output response, without sentence data."""

from __future__ import annotations

import json

import numpy as np
import torch

from dual_intervention_dictionary import tables_for_model
from dual_intervention_sparse_basis import random_rotation
from dual_joint_function_canonical import HERE,TRUTH
from dual_unsupervised_grouping import group_effects
from train_dual_output_language import DualOutputAttention


def connected_groups(similarity,threshold):
    adjacency=similarity>threshold
    np.fill_diagonal(adjacency,False)
    seen=set();groups=[]
    for first in range(len(similarity)):
        if first in seen:continue
        pending=[first];group=[]
        while pending:
            node=pending.pop()
            if node in seen:continue
            seen.add(node);group.append(node)
            pending.extend(int(i) for i in np.flatnonzero(adjacency[node])
                           if i not in seen)
        groups.append(sorted(group))
    return groups


def evaluate(original_heads,dictionary,threshold):
    # Query/output response of each atom in the combined original function.
    combined=original_heads[:4]+original_heads[4:]
    response=combined@dictionary.T
    response/=np.linalg.norm(response,axis=0,keepdims=True)
    similarity=np.abs(response.T@response)
    groups=connected_groups(similarity,threshold)
    effects=group_effects(original_heads,dictionary,groups)
    coverage=[dict(A=float(np.linalg.norm(TRUTH[0]@
        dictionary[g].T@dictionary[g])),
        B=float(np.linalg.norm(TRUTH[1]@
        dictionary[g].T@dictionary[g]))) for g in groups]
    complete=all(any(e["dominant"]==bit and e["purity"]>.9 and c[bit]>.9
                     for e,c in zip(effects,coverage)) for bit in ("A","B"))
    return dict(groups=groups,n_groups=len(groups),similarity=similarity.tolist(),
                group_effects=effects,group_contrast_coverage=coverage,
                A_B_complete_groups_found=complete)


def run():
    torch.set_num_threads(1)
    source=json.loads((HERE/"dual_intervention_sparse_basis.json").read_text())
    thresholds=(.6,.7,.8,.9)
    records=[]
    for seed in range(10):
        model=DualOutputAttention().double()
        model.load_state_dict(torch.load(HERE/f"dual_output_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        original=tables_for_model(model)["original"]
        found=next(r for r in source["rows"] if r["seed"]==seed and
                   r["configuration"]=="four_rank5")
        dictionaries={name:np.array(found["results"][name]["dictionary"])
                      for name in ("svd","sparse")}
        for threshold in thresholds:
            variants={name:evaluate(original,d,threshold)
                      for name,d in dictionaries.items()}
            rng=np.random.default_rng(800000+10000*seed+int(100*threshold))
            random_success=[evaluate(original,random_rotation(5,rng)@
                dictionaries["svd"],threshold)["A_B_complete_groups_found"]
                for _ in range(128)]
            records.append(dict(seed=seed,threshold=threshold,variants=variants,
                random_complete_group_rate=float(np.mean(random_success))))
    output=dict(scope="exact original function response of rank-five intervention-aware key atoms",
        grouping="absolute cosine of 4D query/output response vectors; connected components; no sentences or task labels",
        evaluation="post-fit >.9 A/B effect purity and >.9 signed contrast coverage in a group for each bit",
        records=records)
    (HERE/"dual_functional_grouping.json").write_text(json.dumps(output,indent=2)+"\n")
    for threshold in thresholds:
        selected=[r for r in records if r["threshold"]==threshold]
        print(threshold,*[sum(r["variants"][name]["A_B_complete_groups_found"]
                              for r in selected) for name in ("svd","sparse")],
              float(np.mean([r["random_complete_group_rate"] for r in selected])),
              flush=True)
    return output


if __name__=="__main__":
    run()
