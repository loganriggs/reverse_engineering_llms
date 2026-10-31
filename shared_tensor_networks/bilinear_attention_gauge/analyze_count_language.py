"""Exact token function and joint rank reductions for variable-count toys."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.nn import functional as F

from analyze_dual_output_language import factors
from analyze_toy_language import pair_table
from dual_joint_function_canonical import HERE,flatten,unflatten,truncate
from train_count_language import CountAttention,PAD,examples


def logits(tables,tokens,bias):
    q=tokens[:,-1]-6
    keys=tokens[:,:-1]
    valid=keys!=PAD
    safe_keys=np.where(valid,keys,0)
    return bias+sum(np.sum(t[q[:,None],safe_keys]*valid[:,:,None],axis=1)
                    for t in tables)


def score(tables,tokens,targets,attrs,bias,reference=None):
    output=logits(tables,tokens,bias)
    prediction=output>0
    target=targets.astype(bool)
    ce=F.binary_cross_entropy_with_logits(torch.from_numpy(output),
                                           torch.from_numpy(targets),
                                           reduction="none").numpy().mean(1)
    correct=np.all(prediction==target,axis=1)
    masks={"all":np.ones(len(tokens),dtype=bool),
           "common":~attrs["swap"],
           "rare_seen":attrs["swap"]&~attrs["heldout"],
           "rare_heldout":attrs["heldout"]}
    results={}
    for name,mask in masks.items():
        results[name]=dict(n=int(np.sum(mask)),
            joint_accuracy=float(np.mean(correct[mask])),
            bit_accuracy=float(np.mean((prediction==target)[mask])),
            bce=float(np.mean(ce[mask])))
    if reference is not None:
        results["relative_logit_mse"]=float(np.sum((output-reference)**2)/
                                             np.sum(reference**2))
    return results


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(10):
        model=CountAttention().double()
        model.load_state_dict(torch.load(HERE/f"count_language_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        model.eval()
        heads=[factors(model,index) for index in range(2)]
        tables=[pair_table(h,np.eye(h["rank"])) for h in heads]
        bias=model.read.bias.detach().numpy()
        tokens,targets,attrs=examples(20000,3000+seed,holdout=False)
        direct=model(tokens).detach().numpy()
        x=tokens.numpy();y=targets.numpy().astype(float)
        gate=float(np.max(np.abs(direct-logits(tables,x,bias))))
        assert gate<1e-10,gate
        total=sum(tables)
        matrix=flatten(total)
        singular=np.linalg.svd(matrix,compute_uv=False)
        variants={"original":tables}
        for rank in (1,2,3,4):
            reduced,_=truncate(matrix,rank)
            variants[f"joint_rank{rank}"]=[unflatten(reduced)]
        for rank in (1,2):
            head_tables=[]
            for h in tables:
                reduced,_=truncate(flatten(h),rank)
                head_tables.append(unflatten(reduced))
            variants[f"headwise_rank{rank}"]=head_tables
        scored={name:score(value,x,y,attrs,bias,direct)
                for name,value in variants.items()}
        head_deletions=[score([tables[1-index]],x,y,attrs,bias,direct)
                        for index in range(2)]
        errors={name:float(np.sum((sum(value)-total)**2)/np.sum(total**2))
                for name,value in variants.items()}
        rows.append(dict(seed=seed,direct_table_gate=gate,
            singular_values=singular.tolist(),
            head_qk_quotient_ranks=[h["rank"] for h in heads],
            relative_uniform_table_mse=errors,scores=scored,
            head_deletions=head_deletions))
        print(seed,scored["original"]["rare_heldout"]["joint_accuracy"],
              scored["joint_rank2"]["rare_heldout"]["joint_accuracy"],
              scored["joint_rank3"]["rare_heldout"]["joint_accuracy"],flush=True)
    result=dict(n_seeds=10,n_test=20000,
        extraction="exact all 2x6 role-valid query-key pair table from each checkpoint; pad masked",
        fitting="weight-only SVD of combined 4x6 function table or separate head tables; no sampled inputs or labels",
        evaluation="independent variable-count Poisson(.75) test sequences, including rare held-out swap/bit combination",
        rows=rows)
    (HERE/"count_language_analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    run()
