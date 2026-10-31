"""Stress-test unlabeled sparse-atom grouping under variable key-count laws."""

from __future__ import annotations

import json

import numpy as np
import torch

from dual_intervention_dictionary import tables_for_model
from dual_joint_function_canonical import HERE,TRUTH
from dual_unsupervised_grouping import group_effects
from train_dual_output_language import DualOutputAttention


def sample_counts(n,law,rng):
    counts=np.zeros((n,6),dtype=np.int64)
    if law=="exact_one":
        for family in range(3):
            bit=rng.integers(0,2,n)
            counts[np.arange(n),2*family+bit]+=1
    elif law=="family_dropout":
        for family in range(3):
            present=rng.random(n)<.7
            bit=rng.integers(0,2,n)
            counts[np.arange(n)[present],(2*family+bit)[present]]+=1
    elif law=="variable_two":
        for family in range(3):
            multiplicity=rng.integers(0,3,n)
            for occurrence in range(2):
                present=multiplicity>occurrence
                bit=rng.integers(0,2,n)
                counts[np.arange(n)[present],(2*family+bit)[present]]+=1
    elif law=="independent_poisson":
        counts=rng.poisson(.75,(n,6))
    elif law=="poisson_filler":
        for family in (0,1):
            bit=rng.integers(0,2,n)
            counts[np.arange(n),2*family+bit]+=1
        counts[:,4:6]=rng.poisson(1.5,(n,2))
    else:
        raise ValueError(law)
    return counts


def groups_from_counts(dictionary,counts,threshold):
    activations=counts@dictionary.T
    correlation=np.corrcoef(activations.T)
    adjacency=np.abs(correlation)>threshold
    np.fill_diagonal(adjacency,False)
    seen=set();groups=[]
    for first in range(len(dictionary)):
        if first in seen:continue
        pending=[first];group=[]
        while pending:
            index=pending.pop()
            if index in seen:continue
            seen.add(index);group.append(index)
            pending.extend(int(i) for i in np.flatnonzero(adjacency[index])
                           if i not in seen)
        groups.append(sorted(group))
    return groups,correlation


def run():
    torch.set_num_threads(1)
    source=json.loads((HERE/"dual_intervention_sparse_basis.json").read_text())
    laws=("exact_one","family_dropout","variable_two",
          "independent_poisson","poisson_filler")
    sample_sizes=(32,512,4096)
    thresholds=(.4,.6,.8)
    records=[]
    for seed in range(10):
        model=DualOutputAttention().double()
        model.load_state_dict(torch.load(HERE/f"dual_output_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        original=tables_for_model(model)["original"]
        found=next(r for r in source["rows"] if r["seed"]==seed and
                   r["configuration"]=="four_rank5")
        dictionary=np.array(found["results"]["sparse"]["dictionary"])
        for law in laws:
            for n in sample_sizes:
                rng=np.random.default_rng(700000+10000*seed+100*n+laws.index(law))
                counts=sample_counts(n,law,rng)
                for threshold in thresholds:
                    groups,correlation=groups_from_counts(dictionary,counts,threshold)
                    effects=group_effects(original,dictionary,groups)
                    coverage=[dict(A=float(np.linalg.norm(TRUTH[0]@
                        dictionary[group].T@dictionary[group])),
                        B=float(np.linalg.norm(TRUTH[1]@
                        dictionary[group].T@dictionary[group])))
                        for group in groups]
                    effect_only=all(any(e["dominant"]==bit and e["purity"]>.9
                        for e in effects) for bit in ("A","B"))
                    complete=all(any(e["dominant"]==bit and e["purity"]>.9
                        and c[bit]>.9 for e,c in zip(effects,coverage))
                        for bit in ("A","B"))
                    records.append(dict(seed=seed,law=law,n_unlabeled_bags=n,
                        threshold=threshold,groups=groups,n_groups=len(groups),
                        A_B_effect_groups_found=effect_only,
                        A_B_complete_groups_found=complete,
                        group_contrast_coverage=coverage,group_effects=effects,
                        correlation=correlation.tolist(),
                        within_family_count_correlations=[float(np.corrcoef(
                            counts[:,2*f],counts[:,2*f+1])[0,1])
                            for f in range(3)]))
    output=dict(scope="same trained rank-five sparse dictionaries; only unlabeled key-count law changes",
        fitting="correlation graph on summed key coordinates, no family labels or task targets",
        evaluation="post-fit selective A/B group effect plus >.9 coverage of each complete signed bit contrast",
        laws=laws,sample_sizes=sample_sizes,thresholds=thresholds,records=records)
    (HERE/"dual_grouping_shift.json").write_text(json.dumps(output,indent=2)+"\n")
    for law in laws:
        for threshold in thresholds:
            selected=[r for r in records if r["law"]==law and
                      r["n_unlabeled_bags"]==4096 and r["threshold"]==threshold]
            print(law,threshold,sum(r["A_B_complete_groups_found"] for r in selected),
                  float(np.mean([r["n_groups"] for r in selected])),flush=True)
    return output


if __name__=="__main__":
    run()
