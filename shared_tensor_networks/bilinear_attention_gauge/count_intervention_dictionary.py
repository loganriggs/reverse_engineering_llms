"""Branch-edit fidelity of shared key spaces in variable-count attention toys."""

from __future__ import annotations

import json

import numpy as np
import torch

from dual_intervention_dictionary import tables_for_model
from dual_joint_function_canonical import HERE
from train_count_language import CountAttention


def run():
    torch.set_num_threads(1)
    conditions={"original_only":["original"],
                "original_plus_qk1":["original","qk1_constant"],
                "original_plus_qk2":["original","qk2_constant"],
                "three_conditions":["original","qk1_constant","qk2_constant"],
                "all_four":["original","qk1_constant","qk2_constant",
                            "both_constant"]}
    rows=[]
    for seed in range(10):
        model=CountAttention().double()
        model.load_state_dict(torch.load(HERE/f"count_language_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        tables=tables_for_model(model)
        for name,seen in conditions.items():
            stack=np.vstack([tables[key]/np.linalg.norm(tables[key])
                             for key in seen])
            _,singular,v=np.linalg.svd(stack,full_matrices=False)
            for rank in (2,3,4,5,6):
                projector=v[:rank].T@v[:rank]
                errors={key:float(np.sum((table-table@projector)**2)/
                                  np.sum(table**2))
                        for key,table in tables.items()}
                rows.append(dict(seed=seed,fit_conditions=name,rank=rank,
                    singular_values=singular.tolist(),relative_mse=errors))
    output=dict(n_seeds=10,
        fit="SVD of head-preserving 8x6 exact token tables, each condition unit-Frobenius normalized; no task labels",
        tests="original, QK1 constant, QK2 constant, and both constant; unseen conditions excluded from fitted stack",
        conditions=conditions,rows=rows)
    (HERE/"count_intervention_dictionary.json").write_text(json.dumps(output,indent=2)+"\n")
    for name in ("original_only","three_conditions","all_four"):
        for rank in (2,3,4,5):
            selected=[r for r in rows if r["fit_conditions"]==name and r["rank"]==rank]
            print(name,rank,*[round(float(np.mean([r["relative_mse"][k]
                for r in selected])),4) for k in tables],flush=True)
    return output


if __name__=="__main__":
    run()
