"""Exact attention table and downstream behavior of bilinear XOR count toy."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.nn import functional as F

from analyze_toy_language import rms
from dual_joint_function_canonical import HERE
from train_xor_count_language import XorCountAttention,examples
from train_count_language import PAD


def attention_table(model):
    embedding=model.embed.weight.detach().numpy()
    query=embedding[6:8]
    key=embedding[:6]
    width=model.qk_width
    tables=[]
    for head in range(model.heads):
        sl=slice(head*width,(head+1)*width)
        def projection(name,inputs):
            w=getattr(model,name).weight.detach().numpy()[sl]
            return rms(inputs@w.T)
        q1=projection("q1",query);k1=projection("k1",key)
        q2=projection("q2",query);k2=projection("k2",key)
        value=key@model.v.weight.detach().numpy()[head]
        tables.append((q1@k1.T/width)*(q2@k2.T/width)*value[None,:])
    return np.stack(tables,axis=-1)


def flatten(table):
    return table.transpose(0,2,1).reshape(4,6)


def unflatten(matrix):
    return matrix.reshape(2,2,6).transpose(0,2,1)


def attention_from_table(table,tokens):
    q=tokens[:,-1]-6
    keys=tokens[:,:-1]
    valid=keys!=PAD
    safe=np.where(valid,keys,0)
    return np.sum(table[q[:,None],safe]*valid[:,:,None],axis=1)


def downstream(model,heads):
    with torch.no_grad():
        h=torch.from_numpy(heads)
        hidden=torch.cat((h,torch.ones_like(h[:,:1])),dim=1)
        output=model.read(model.mlp_left(hidden)*model.mlp_right(hidden))
    return output.detach().numpy()


def semantic_accuracy(output,targets,attrs):
    prediction=output>0
    rows=np.arange(len(output))
    a_col=attrs["swap"].astype(int)
    b_col=1-a_col
    return dict(A=float(np.mean(prediction[rows,a_col]==targets[rows,a_col])),
                B=float(np.mean(prediction[rows,b_col]==targets[rows,b_col])),
                XOR=float(np.mean(prediction[:,2]==targets[:,2])))


def score(model,table,tokens,targets,attrs,reference_logits,reference_heads):
    heads=attention_from_table(table,tokens)
    output=downstream(model,heads)
    prediction=output>0
    correct=np.all(prediction==targets.astype(bool),axis=1)
    ce=F.binary_cross_entropy_with_logits(torch.from_numpy(output),
        torch.from_numpy(targets),reduction="none").numpy().mean(1)
    masks={"all":np.ones(len(tokens),dtype=bool),
           "common":~attrs["swap"],
           "rare_seen":attrs["swap"]&~attrs["heldout"],
           "rare_heldout":attrs["heldout"]}
    result={}
    for name,mask in masks.items():
        result[name]=dict(n=int(np.sum(mask)),
            joint_accuracy=float(np.mean(correct[mask])),
            bit_accuracy=float(np.mean((prediction==targets.astype(bool))[mask])),
            bce=float(np.mean(ce[mask])),
            semantic_accuracy=semantic_accuracy(output[mask],targets[mask],
                {k:v[mask] for k,v in attrs.items() if isinstance(v,np.ndarray)}))
    result["relative_head_mse"]=float(np.sum((heads-reference_heads)**2)/
                                     np.sum(reference_heads**2))
    result["relative_logit_mse"]=float(np.sum((output-reference_logits)**2)/
                                      np.sum(reference_logits**2))
    return result


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(10):
        model=XorCountAttention().double()
        model.load_state_dict(torch.load(HERE/f"xor_count_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        model.eval()
        table=attention_table(model)
        token_t,target_t,attrs=examples(20000,3000+seed,False)
        tokens=token_t.numpy();targets=target_t.numpy().astype(float)
        direct_logits,direct_heads=model(token_t,return_heads=True)
        direct_logits=direct_logits.detach().numpy()
        direct_heads=direct_heads.detach().numpy()
        table_heads=attention_from_table(table,tokens)
        gate_heads=float(np.max(np.abs(table_heads-direct_heads)))
        gate_logits=float(np.max(np.abs(downstream(model,table_heads)-direct_logits)))
        assert gate_heads<1e-10 and gate_logits<1e-10,(gate_heads,gate_logits)
        matrix=flatten(table)
        u,s,v=np.linalg.svd(matrix,full_matrices=False)
        variants={"original":table}
        for rank in (1,2,3,4):
            variants[f"rank{rank}"]=unflatten((u[:,:rank]*s[:rank])@v[:rank])
        scored={name:score(model,t,tokens,targets,attrs,direct_logits,direct_heads)
                for name,t in variants.items()}
        errors={name:float(np.sum((t-table)**2)/np.sum(table**2))
                for name,t in variants.items()}
        rows.append(dict(seed=seed,direct_head_gate=gate_heads,
            direct_logit_gate=gate_logits,singular_values=s.tolist(),
            relative_uniform_table_mse=errors,scores=scored))
        print(seed,scored["original"]["rare_heldout"]["joint_accuracy"],
              scored["rank2"]["all"]["joint_accuracy"],
              scored["rank2"]["rare_heldout"]["joint_accuracy"],flush=True)
    output=dict(n_seeds=10,n_test=20000,
        extraction="exact first-layer 2x6x2 attention-head table, followed by original bilinear MLP",
        fit="weight-only SVD of 4x6 query/head-by-key table; no sampled inputs or labels",
        test="independent variable-count A/B/XOR examples including rare A0/B1 query holdout",
        rows=rows)
    (HERE/"xor_count_analysis.json").write_text(json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    run()
