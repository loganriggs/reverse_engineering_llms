"""Shared key subspace learned from ordinary and intervened toy head tables."""

from __future__ import annotations

import json

import numpy as np
import torch

from analyze_dual_output_language import factors
from analyze_toy_language import pair_table
from dual_joint_function_canonical import HERE,flatten
from train_dual_output_language import DualOutputAttention


def tables_for_model(model):
    rows={"original":[],"qk1_constant":[],"qk2_constant":[],
          "both_constant":[]}
    for index in range(2):
        h=factors(model,index)
        rows["original"].append(flatten(pair_table(h,np.eye(h["rank"]))))
        qk1=dict(h,u=np.ones((2,1)),v=np.ones((6,1)),rank=1)
        rows["qk1_constant"].append(flatten(pair_table(qk1,np.eye(1))))
        qk2=dict(h,b=np.ones_like(h["b"]))
        rows["qk2_constant"].append(flatten(pair_table(qk2,np.eye(h["rank"]))))
        both=dict(qk1,b=np.ones_like(h["b"]))
        rows["both_constant"].append(flatten(pair_table(both,np.eye(1))))
    return {name:np.vstack(tables) for name,tables in rows.items()}


def run():
    torch.set_num_threads(1)
    conditions={"original_only":["original"],
                "original_plus_qk1":["original","qk1_constant"],
                "original_plus_qk2":["original","qk2_constant"],
                "all_three":["original","qk1_constant","qk2_constant"],
                "all_four":["original","qk1_constant","qk2_constant",
                            "both_constant"]}
    rows=[]
    for seed in range(10):
        model=DualOutputAttention().double()
        model.load_state_dict(torch.load(HERE/f"dual_output_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        model.eval()
        tables=tables_for_model(model)
        for name,seen in conditions.items():
            # Normalize each condition's energy so a large branch edit cannot
            # dominate merely because its output scale changes.
            scaled=[tables[k]/np.linalg.norm(tables[k]) for k in seen]
            stack=np.vstack(scaled)
            _,s,v=np.linalg.svd(stack,full_matrices=False)
            for rank in range(1,7):
                projector=v[:rank].T@v[:rank]
                errors={k:float(np.sum((t-t@projector)**2)/np.sum(t**2))
                        for k,t in tables.items()}
                rows.append(dict(seed=seed,fit_conditions=name,rank=rank,
                                 singular_values=s.tolist(),relative_mse=errors))
    result=dict(n_seeds=10,n_physical_heads=2,
        training="exact weight-derived role-valid token tables; no task labels",
        fitting="SVD of stacked, individually unit-Frobenius tables; held-out branch edit excluded from fitted stack",
        conditions=conditions,rows=rows)
    (HERE/"dual_intervention_dictionary.json").write_text(json.dumps(result,indent=2)+"\n")
    for name in conditions:
        for rank in (2,3,4,5,6):
            sub=[r for r in rows if r["fit_conditions"]==name and r["rank"]==rank]
            print(name,rank,*[round(float(np.mean([r["relative_mse"][k]
                           for r in sub])),5) for k in tables],flush=True)
    return result


if __name__=="__main__":
    run()
