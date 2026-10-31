"""Compare the largest corpus-response group across exact gauges and starts."""

from __future__ import annotations

import json

import numpy as np
import torch

from dual_functional_grouping import connected_groups
from flagship_matrix_free import SNAPSHOT
from flagship_prerope_response_grouping import component_signature
from flagship_rank96_function_grouping import combine,ids_and_projections
from flagship_transfer import ART


def run():
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    q,k,value_sqrt,_=ids_and_projections(state,128,8192)
    cut=np.load(ART/"head0_rope_metric_factorial_seed41.npz")["mixed__pre"]
    left,s,right=np.linalg.svd(cut,full_matrices=False)
    a=left[:,:96]*np.sqrt(s[:96]);b=right[:96].T*np.sqrt(s[:96])
    base=np.load(ART/"head0_prerope_gauge_sparsity_seed41_s200.npz")
    extra=np.load(ART/"head0_prerope_gauge_sparsity_seed41_s200_random_start.npz")
    rotations={f"identity_{i}":base[f"factor_weight_{i}"] for i in range(3)}
    rotations.update({f"randomstart_{i}":extra[f"factor_weight_{i}"] for i in range(3)})
    rotations.update({f"random_{i}":base[f"random_{i}"] for i in range(3)})
    panel=json.load(open(ART/"head0_prerope_response_grouping_seed41_s200_q128_k8192_both_starts.json"))
    threshold=panel["null_thresholds"]["corpus"]
    groups=[];projectors={}
    for name,rotation in rotations.items():
        sig=component_signature(q["fit"],k["fit"]["corpus"],
            value_sqrt,a,b,rotation,7,8)
        normalized=combine({"corpus":sig},"corpus")
        cos=np.abs(normalized@normalized.T)
        components=sorted(connected_groups(cos,threshold),key=len,reverse=True)
        group=components[0]
        directions=rotation[:,group]
        projector=directions@directions.T
        projectors[name]=projector
        energy=np.sum(sig**2,axis=1)
        groups.append(dict(gauge=name,size=len(group),indices=group,
            n_edges=int(np.triu(cos>threshold,1).sum()),
            component_energy_fraction=float(energy[group].sum()/energy.sum())))
    overlaps=[]
    names=list(rotations)
    for i,name in enumerate(names):
        for other in names[i+1:]:
            pi,pj=projectors[name],projectors[other]
            ki=int(round(np.trace(pi)));kj=int(round(np.trace(pj)))
            overlaps.append(dict(gauges=[name,other],sizes=[ki,kj],
                normalized_projector_overlap=float(np.trace(pi@pj)/min(ki,kj)),
                random_subspace_expected=float(max(ki,kj)/96)))
    output=dict(layer=0,head=0,rank=96,
        threshold=threshold,law="corpus",
        n_query_anchors=256,n_key_draws=8192,
        group_rule="largest connected component of absolute response-cosine graph",
        scope="group projector stability and energy; no semantic or causal validation",
        groups=groups,overlaps=overlaps)
    (ART/"head0_prerope_group_projector.json").write_text(
        json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    run()
