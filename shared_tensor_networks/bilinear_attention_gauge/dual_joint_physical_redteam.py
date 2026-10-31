"""Can the joint sparse key dictionary explain actual trained head contributions?"""

from __future__ import annotations

import json

import numpy as np
import torch

from analyze_dual_output_language import factors,logits
from analyze_toy_language import pair_table
from dual_joint_function_canonical import HERE,flatten,unflatten,truncate
from train_dual_output_language import DualOutputAttention,exhaustive


def project(matrix,dictionary):
    # Head-specific 4x2 coefficients, fixed 2x6 key dictionary.
    coefficients=matrix@dictionary.T@np.linalg.inv(dictionary@dictionary.T)
    return coefficients@dictionary


def relative_error(approx,truth):
    return float(np.sum((approx-truth)**2)/np.sum(truth**2))


def bit_energy(table):
    return dict(A=float(np.sum((table[:,1,:]-table[:,0,:])**2)),
                B=float(np.sum((table[:,3,:]-table[:,2,:])**2)),
                F=float(np.sum((table[:,5,:]-table[:,4,:])**2)))


def run():
    torch.set_num_threads(1)
    joint=json.loads((HERE/"dual_joint_function_canonical.json").read_text())
    seq_t,_,_=exhaustive()
    seq=seq_t.numpy()
    rows=[]
    for seed in range(10):
        model=DualOutputAttention().double()
        model.load_state_dict(torch.load(HERE/f"dual_output_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        model.eval()
        heads=[factors(model,h) for h in range(2)]
        tables=[pair_table(h,np.eye(h["rank"])) for h in heads]
        bias=model.read.bias.detach().numpy()
        direct,head_outputs,_,_=model(seq_t,return_heads=True)
        direct=direct.detach().numpy()
        head_outputs=head_outputs.detach().numpy()
        assert np.max(np.abs(direct-logits(tables,seq,bias)))<1e-10
        dictionary=np.array(joint["records"][seed]["gauges"]["key_sparse"]["dictionary"])
        rng=np.random.default_rng(9000+seed)
        random_dictionary=rng.standard_normal((128,2,6))
        for index,(head,table) in enumerate(zip(heads,tables)):
            truth=flatten(table)
            approx=project(truth,dictionary)
            random_error=[relative_error(project(truth,d),truth)
                          for d in random_dictionary]
            # Removing this physical head is the same as deleting its
            # weighted head-output contribution before the final readout.
            actual_ablation=direct-head_outputs[:,index,None]*\
                model.read.weight.detach().numpy()[:,index]
            exact_ablation=logits([tables[1-index]],seq,bias)
            assert np.max(np.abs(actual_ablation-exact_ablation))<1e-10
            predicted_ablation=logits([tables[1-index],
                                       table-unflatten(approx)],seq,bias)
            # Probe a structural edit never used to choose the dictionary:
            # replace the second QK branch by a constant.
            changed=dict(head,b=np.ones_like(head["b"]))
            changed_table=pair_table(changed,np.eye(head["rank"]))
            changed_matrix=flatten(changed_table)
            changed_approx=project(changed_matrix,dictionary)
            changed_best_rank2,_=truncate(changed_matrix,2)
            qk1_constant=dict(head,u=np.ones((2,1)),v=np.ones((6,1)),rank=1)
            qk1_table=pair_table(qk1_constant,np.eye(1))
            qk1_matrix=flatten(qk1_table)
            qk1_approx=project(qk1_matrix,dictionary)
            qk1_best_rank2,_=truncate(qk1_matrix,2)
            rows.append(dict(seed=seed,head=index,
                original_head_table_relative_mse=relative_error(approx,truth),
                random_dictionary_relative_mse_mean=float(np.mean(random_error)),
                random_dictionary_relative_mse_p05=float(np.quantile(random_error,.05)),
                head_ablation_logit_relative_mse=float(
                    np.sum((predicted_ablation-actual_ablation)**2)/
                    np.sum((actual_ablation-direct)**2)),
                original_bit_flip_energy=bit_energy(table),
                projected_bit_flip_energy=bit_energy(unflatten(approx)),
                qk2_constant_table_relative_mse=relative_error(changed_approx,changed_matrix),
                qk2_constant_best_rank2_relative_mse=relative_error(changed_best_rank2,changed_matrix),
                qk2_constant_unprojected_bit_flip_energy=bit_energy(changed_table),
                qk2_constant_projected_bit_flip_energy=bit_energy(unflatten(changed_approx)),
                qk1_constant_table_relative_mse=relative_error(qk1_approx,qk1_matrix),
                qk1_constant_best_rank2_relative_mse=relative_error(qk1_best_rank2,qk1_matrix)))
    result=dict(scope="ten trained two-head toys; learned joint dictionary projected onto each physical head",
        fitting="dictionary selected from combined original function without task labels; head coefficients fitted from each head's exact token table",
        intervention="physical head deletion checked against direct model output; QK1-constant and QK2-constant edits held out from dictionary selection",
        random_baseline="128 Gaussian two-dimensional key dictionaries per seed",
        rows=rows)
    (HERE/"dual_joint_physical_redteam.json").write_text(json.dumps(result,indent=2)+"\n")
    for key in ("original_head_table_relative_mse",
                "random_dictionary_relative_mse_mean",
                "head_ablation_logit_relative_mse",
                "qk2_constant_table_relative_mse",
                "qk2_constant_best_rank2_relative_mse",
                "qk1_constant_table_relative_mse",
                "qk1_constant_best_rank2_relative_mse"):
        print(key,float(np.mean([r[key] for r in rows])),flush=True)
    return result


if __name__=="__main__":
    run()
