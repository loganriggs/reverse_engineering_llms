"""Exact multi-head bilinear-attention counterexample for DAG reuse recovery.

Two graphs compute the same function on *all continuous inputs*: one computes
a key feature once and fans it out to both heads; the other computes two
independent occurrences with identical weights. External data, including
input interventions, cannot distinguish them. An internal node intervention
can, because it has different downstream consumers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parameters(seed=0,d=6):
    rng=np.random.default_rng(seed)
    detector=rng.normal(size=d)/np.sqrt(d)
    return dict(
        a=rng.normal(size=(2,d,d))/np.sqrt(d),
        b=rng.normal(size=(2,d,d))/np.sqrt(d),
        c=rng.normal(size=(2,d))/np.sqrt(d),
        shared=detector.copy(),
        duplicate=np.stack((detector.copy(),detector.copy())))


def attention_factors(q,k,p):
    qk1=np.einsum("nd,hde,ne->nh",q,p["a"],k)
    qk2=np.einsum("nd,hde,ne->nh",q,p["b"],k)
    ov=np.einsum("ne,he->nh",k,p["c"])
    return qk1*qk2*ov


def forward(q,k,p,graph="shared",intervention=None):
    factors=attention_factors(q,k,p)
    shared=k@p["shared"]
    if graph=="shared":
        detector=np.stack((shared,shared),axis=1)
        if intervention=="first_detector_zero":
            detector[:]=0
        elif intervention=="first_detector_half":
            detector[:]*=.5
    elif graph=="duplicate":
        # These are two distinct computations and parameter occurrences,
        # despite the equality of their numerical parameter values.
        detector=np.stack((k@p["duplicate"][0],k@p["duplicate"][1]),axis=1)
        if intervention=="first_detector_zero":
            detector[:,0]=0
        elif intervention=="first_detector_half":
            detector[:,0]*=.5
    else:
        raise ValueError(graph)
    return factors*detector


def run(seed=0,n=8192,d=6):
    rng=np.random.default_rng(seed+100)
    p=parameters(seed,d)
    domains={
        "gaussian":(rng.normal(size=(n,d)),rng.normal(size=(n,d))),
        "shifted_covariance":(rng.normal(size=(n,d))*np.geomspace(.3,3,d),
                              rng.normal(size=(n,d))*np.geomspace(3,.3,d)),
        "sparse_keys":(rng.normal(size=(n,d)),
                       rng.normal(size=(n,d))*(rng.random((n,d))<.35))}
    records={}
    for name,(q,k) in domains.items():
        shared=forward(q,k,p,"shared")
        duplicate=forward(q,k,p,"duplicate")
        assert np.max(np.abs(shared-duplicate))<1e-12
        # Removing one external key coordinate is also observational: both
        # graphs must respond identically to any such input intervention.
        altered=k.copy();altered[:,0]=0
        input_shared=forward(q,altered,p,"shared")
        input_duplicate=forward(q,altered,p,"duplicate")
        assert np.max(np.abs(input_shared-input_duplicate))<1e-12
        internal_shared=forward(q,k,p,"shared","first_detector_zero")
        internal_duplicate=forward(q,k,p,"duplicate","first_detector_zero")
        assert np.max(np.abs(internal_shared))==0
        assert np.max(np.abs(internal_duplicate[:,0]))==0
        assert np.max(np.abs(internal_duplicate[:,1]-shared[:,1]))<1e-12
        discrepancy=float(np.mean((internal_shared-internal_duplicate)**2))
        # In the duplicated graph the second head is unaffected; in the
        # shared graph both heads are silenced. The discrepancy is material.
        assert discrepancy>1e-3
        records[name]=dict(
            external_max_abs=float(np.max(np.abs(shared-duplicate))),
            input_intervention_max_abs=float(np.max(np.abs(input_shared-input_duplicate))),
            shared_teacher_internal_mse=dict(shared_candidate=0.,
                                             duplicate_candidate=discrepancy),
            duplicate_teacher_internal_mse=dict(shared_candidate=discrepancy,
                                                duplicate_candidate=0.),
            internal_candidate_discrepancy_mse=discrepancy,
            second_head_energy_fraction=float(np.sum(shared[:,1]**2)/np.sum(shared**2)))
    result=dict(seed=seed,n=n,d=d,
                distinct_detector_parameter_counts=dict(shared=d,duplicate=2*d),
                observational_fit_mse=dict(shared=0.,duplicate=0.),
                minimum_detector_parameter_choice="shared",
                attention_graph="two bilinear QK products and OV values; key detector fans out or is duplicated",
                conclusion="external function is identical on all inputs; one internal-node intervention distinguishes computation graphs",
                domains=records)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--n",type=int,default=8192)
    parser.add_argument("--d",type=int,default=6)
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name(
        "shared_computation_identifiability.json"))
    args=parser.parse_args()
    result=run(args.seed,args.n,args.d)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
