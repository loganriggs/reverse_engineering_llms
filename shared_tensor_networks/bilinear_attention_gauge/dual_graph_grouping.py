"""Infer variable-size groups of sparse key atoms from unlabeled co-occurrence."""

from __future__ import annotations

import json

import numpy as np
import torch

from dual_intervention_dictionary import tables_for_model
from dual_intervention_sparse_basis import random_rotation
from dual_joint_function_canonical import HERE
from dual_unsupervised_grouping import group_ablation,group_effects
from train_dual_output_language import DualOutputAttention,batch,exhaustive


def connected_groups(dictionary,sequences,threshold):
    activations=dictionary[:,sequences[:,:3]].sum(axis=2).T
    correlation=np.corrcoef(activations.T)
    adjacency=np.abs(correlation)>threshold
    np.fill_diagonal(adjacency,False)
    seen=set();groups=[]
    for first in range(len(dictionary)):
        if first in seen: continue
        pending=[first];group=[]
        while pending:
            node=pending.pop()
            if node in seen: continue
            seen.add(node);group.append(node)
            pending.extend(int(i) for i in np.flatnonzero(adjacency[node])
                           if i not in seen)
        groups.append(sorted(group))
    return groups,correlation


def evaluate(original,dictionary,sequences,threshold,eval_context=None):
    groups,correlation=connected_groups(dictionary,sequences,threshold)
    effects=group_effects(original,dictionary,groups)
    recovered=all(any(e["dominant"]==bit and e["purity"]>.9
                      for e in effects) for bit in ("A","B"))
    result=dict(groups=groups,n_groups=len(groups),
                correlation=correlation.tolist(),group_effects=effects,
                A_B_groups_recovered=recovered)
    if eval_context is not None:
        result["group_ablation"]=group_ablation(original,dictionary,groups,
                                                 effects,eval_context)
    return result


def run():
    torch.set_num_threads(1)
    source=json.loads((HERE/"dual_intervention_sparse_basis.json").read_text())
    rows=[]
    thresholds=(.6,.7,.8,.9)
    sample_sizes=(32,128,512)
    seq_t,target_t,attrs=exhaustive()
    seq=seq_t.numpy();target=target_t.numpy().astype(bool)
    for seed in range(10):
        model=DualOutputAttention().double()
        model.load_state_dict(torch.load(HERE/f"dual_output_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        original=tables_for_model(model)["original"]
        found=next(r for r in source["rows"] if r["seed"]==seed and
                   r["configuration"]=="four_rank5")
        dictionaries={name:np.array(found["results"][name]["dictionary"])
                      for name in ("svd","sparse")}
        context=(seq,target,attrs,model.read.bias.detach().numpy())
        for n in sample_sizes:
            sampled,_=batch(n,torch.Generator().manual_seed(300000+10000*seed+n))
            sentences=sampled.numpy()
            for threshold in thresholds:
                variants={name:evaluate(original,d,sentences,threshold,context)
                          for name,d in dictionaries.items()}
                rng=np.random.default_rng(900000+10000*seed+n+int(100*threshold))
                random_recovered=[evaluate(original,
                    random_rotation(5,rng)@dictionaries["svd"],sentences,threshold)
                    ["A_B_groups_recovered"] for _ in range(128)]
                rows.append(dict(seed=seed,n_unlabeled_sentences=n,
                    correlation_threshold=threshold,variants=variants,
                    random_recovery_rate=float(np.mean(random_recovered))))
    output=dict(task="infer connected groups from unlabeled sparse key activations",
        grouping="edge if absolute Pearson correlation exceeds threshold; connected components; no group count or size supplied",
        fitting="no token-family labels, output targets, or bit-flip responses used for grouping",
        evaluation="post-fit A/B bit-flip purity above .9 for at least one group per bit; additional groups allowed",
        rows=rows)
    (HERE/"dual_graph_grouping.json").write_text(json.dumps(output,indent=2)+"\n")
    for n in sample_sizes:
        for threshold in thresholds:
            selected=[r for r in rows if r["n_unlabeled_sentences"]==n and
                      r["correlation_threshold"]==threshold]
            print(n,threshold,*[round(float(np.mean([r["variants"][name]
                ["A_B_groups_recovered"] for r in selected])),2)
                for name in ("svd","sparse")],
                round(float(np.mean([r["random_recovery_rate"] for r in selected])),2),
                flush=True)
    return output


if __name__=="__main__":
    run()
