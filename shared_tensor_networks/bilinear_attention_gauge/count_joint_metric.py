"""Uniform versus training-frequency joint function reduction on count language."""

from __future__ import annotations

import json

import numpy as np
import torch

from analyze_count_language import logits,score
from analyze_dual_output_language import factors
from analyze_toy_language import pair_table
from dual_joint_function_canonical import HERE,flatten,unflatten
from dual_joint_metric import weighted_fit
from train_count_language import CountAttention,PAD,examples


def pair_frequency(tokens):
    q=tokens[:,-1]-6
    keys=tokens[:,:-1]
    valid=keys!=PAD
    counts=np.zeros((2,6),dtype=float)
    np.add.at(counts,(np.broadcast_to(q[:,None],keys.shape)[valid],keys[valid]),1)
    assert np.all(counts>0)
    weights=np.repeat(counts,2,axis=0)
    return weights/weights.sum()


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(10):
        model=CountAttention().double()
        model.load_state_dict(torch.load(HERE/f"count_language_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        heads=[factors(model,h) for h in range(2)]
        total=sum(pair_table(h,np.eye(h["rank"])) for h in heads)
        matrix=flatten(total)
        bias=model.read.bias.detach().numpy()
        train_t,_,_=examples(50000,1000+seed,True)
        empirical=pair_frequency(train_t.numpy())
        uniform=np.ones_like(empirical)/empirical.size
        weights={"uniform":uniform,"empirical":empirical,
                 "half_uniform_empirical":.5*(uniform+empirical)}
        test_t,target_t,attrs=examples(20000,3000+seed,False)
        test=test_t.numpy();target=target_t.numpy().astype(float)
        original=logits([total],test,bias)
        for rank in (1,2):
            for name,weight in weights.items():
                reduced,fit=weighted_fit(matrix,weight,rank)
                metrics=score([unflatten(reduced)],test,target,attrs,bias,original)
                rows.append(dict(seed=seed,rank=rank,measure=name,fit=fit,
                    relative_uniform_table_mse=float(np.sum((reduced-matrix)**2)/
                                                     np.sum(matrix**2)),
                    scores=metrics))
    output=dict(n_seeds=10,n_train_frequency_samples=50000,n_test=20000,
        fit="uniform, training-frequency, or half mixture weighted low-rank combined token table; no task labels",
        holdout="rare query with A=0 B=1 omitted from frequency sample, included in independent test",
        rows=rows)
    (HERE/"count_joint_metric.json").write_text(json.dumps(output,indent=2)+"\n")
    for rank in (1,2):
        for name in ("uniform","empirical","half_uniform_empirical"):
            selected=[r for r in rows if r["rank"]==rank and r["measure"]==name]
            print(rank,name,*[round(float(np.mean([r["scores"][split]
                ["joint_accuracy"] for r in selected])),5)
                for split in ("all","rare_heldout")],flush=True)
    return output


if __name__=="__main__":
    run()
