"""Physical QK-branch red team of nonlinear-toy function dictionaries."""

from __future__ import annotations

import json

import numpy as np
import torch

from analyze_toy_language import rms
from analyze_xor_count_language import attention_from_table,downstream,flatten,unflatten
from dual_joint_function_canonical import HERE
from train_xor_count_language import XorCountAttention,examples


def tables(model):
    embedding=model.embed.weight.detach().numpy()
    query=embedding[6:8];key=embedding[:6]
    variants={name:[] for name in ("original","qk1_constant",
                                    "qk2_constant","both_constant")}
    for h in range(2):
        sl=slice(h*model.qk_width,(h+1)*model.qk_width)
        def projection(name,inputs):
            w=getattr(model,name).weight.detach().numpy()[sl]
            return rms(inputs@w.T)
        a=projection("q1",query)@projection("k1",key).T/model.qk_width
        b=projection("q2",query)@projection("k2",key).T/model.qk_width
        value=key@model.v.weight.detach().numpy()[h]
        for name,left,right in (("original",a,b),
                                ("qk1_constant",np.ones_like(a),b),
                                ("qk2_constant",a,np.ones_like(b)),
                                ("both_constant",np.ones_like(a),np.ones_like(b))):
            variants[name].append(left*right*value[None,:])
    return {name:flatten(np.stack(v,axis=-1)) for name,v in variants.items()}


def run():
    torch.set_num_threads(1)
    configurations={"original_only":["original"],
        "single_edits":["original","qk1_constant","qk2_constant"],
        "all_four":["original","qk1_constant","qk2_constant","both_constant"]}
    rows=[]
    for seed in range(10):
        model=XorCountAttention().double()
        model.load_state_dict(torch.load(HERE/f"xor_count_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        model.eval()
        variants=tables(model)
        test_t,_,_=examples(20000,3000+seed,False)
        tokens=test_t.numpy()
        for name,seen in configurations.items():
            stack=np.vstack([variants[k]/np.linalg.norm(variants[k]) for k in seen])
            _,singular,v=np.linalg.svd(stack,full_matrices=False)
            for rank in (2,3,4,5,6):
                projector=v[:rank].T@v[:rank]
                scores={}
                for condition,original in variants.items():
                    reduced=original@projector
                    head=attention_from_table(unflatten(original),tokens)
                    head_reduced=attention_from_table(unflatten(reduced),tokens)
                    logits=downstream(model,head)
                    logits_reduced=downstream(model,head_reduced)
                    scores[condition]=dict(
                        table_relative_mse=float(np.sum((reduced-original)**2)/
                                                 np.sum(original**2)),
                        head_relative_mse=float(np.sum((head_reduced-head)**2)/
                                                np.sum(head**2)),
                        logit_relative_mse=float(np.sum((logits_reduced-logits)**2)/
                                                 np.sum(logits**2)))
                rows.append(dict(seed=seed,fit_conditions=name,rank=rank,
                    singular_values=singular.tolist(),scores=scores))
        print("seed",seed,flush=True)
    output=dict(n_seeds=10,n_test=20000,
        fit="SVD key spaces of exact head-preserving weight tables; each condition unit-Frobenius normalized; no task labels or sampled inputs",
        conditions=configurations,
        evaluation="original and QK1/QK2/both constant branch interventions; independent test sequences passed through original bilinear MLP",
        rows=rows)
    (HERE/"xor_count_intervention.json").write_text(json.dumps(output,indent=2)+"\n")
    for name in configurations:
        for rank in (2,4,5):
            selected=[r for r in rows if r["fit_conditions"]==name and r["rank"]==rank]
            print(name,rank,{k:round(float(np.mean([r["scores"][k]
                ["table_relative_mse"] for r in selected])),4)
                for k in variants},flush=True)
    return output


if __name__=="__main__":run()
