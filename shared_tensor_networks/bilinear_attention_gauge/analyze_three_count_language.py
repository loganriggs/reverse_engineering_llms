"""Exact token table and rank bottlenecks for trained three-bit attention."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.nn import functional as F

from analyze_toy_language import rms
from dual_joint_function_canonical import HERE
from train_three_count_language import ThreeCountAttention,PAD,examples


def head_tables(model):
    embedding=model.embed.weight.detach().numpy()
    query=embedding[8:10]
    key=embedding[:8]
    width=model.qk_width
    def projected(name,inputs,head):
        w=getattr(model,name).weight.detach().numpy()
        sl=slice(head*width,(head+1)*width)
        return rms(inputs@w[sl].T)
    tables=[]
    for head in range(model.heads):
        q1=projected("q1",query,head)
        k1=projected("k1",key,head)
        q2=projected("q2",query,head)
        k2=projected("k2",key,head)
        first=q1@k1.T/width
        second=q2@k2.T/width
        value=key@model.v.weight.detach().numpy()[head]
        read=model.read.weight.detach().numpy()[:,head]
        tables.append((first*second)[:,:,None]*value[None,:,None]*read[None,None,:])
    return tables


def flatten(table):
    return table.transpose(0,2,1).reshape(6,8)


def unflatten(matrix):
    return matrix.reshape(2,3,8).transpose(0,2,1)


def logits(tables,tokens,bias):
    q=tokens[:,-1]-8
    keys=tokens[:,:-1]
    valid=keys!=PAD
    safe=np.where(valid,keys,0)
    return bias+sum(np.sum(table[q[:,None],safe]*valid[:,:,None],axis=1)
                    for table in tables)


def semantic_accuracy(output,targets,attrs):
    prediction=output>0
    rows=np.arange(len(output))
    # Common output order A/B/C; rare output order B/C/A.
    columns=np.stack((np.where(attrs["rare"],2,0),
                      np.where(attrs["rare"],0,1),
                      np.where(attrs["rare"],1,2)),axis=1)
    return {bit:float(np.mean(prediction[rows,columns[:,index]]==
                           targets[rows,columns[:,index]]))
            for index,bit in enumerate("ABC")}


def score(tables,tokens,targets,attrs,bias,reference):
    output=logits(tables,tokens,bias)
    prediction=output>0
    correct=np.all(prediction==targets.astype(bool),axis=1)
    ce=F.binary_cross_entropy_with_logits(torch.from_numpy(output),
        torch.from_numpy(targets),reduction="none").numpy().mean(1)
    masks={"all":np.ones(len(tokens),dtype=bool),
           "common":~attrs["rare"],
           "rare_seen":attrs["rare"]&~attrs["heldout"],
           "rare_heldout":attrs["heldout"]}
    result={}
    for name,mask in masks.items():
        result[name]=dict(n=int(np.sum(mask)),
            joint_accuracy=float(np.mean(correct[mask])),
            bit_accuracy=float(np.mean((prediction==targets.astype(bool))[mask])),
            bce=float(np.mean(ce[mask])),
            semantic_bit_accuracy=semantic_accuracy(output[mask],targets[mask],
                {k:v[mask] for k,v in attrs.items() if isinstance(v,np.ndarray)}))
    result["relative_logit_mse"]=float(np.sum((output-reference)**2)/
                                       np.sum(reference**2))
    return result


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(10):
        model=ThreeCountAttention().double()
        model.load_state_dict(torch.load(HERE/f"three_count_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        model.eval()
        tables=head_tables(model)
        total=sum(tables)
        matrix=flatten(total)
        u,s,v=np.linalg.svd(matrix,full_matrices=False)
        test_t,target_t,attrs=examples(30000,30000+seed,False)
        test=test_t.numpy();targets=target_t.numpy().astype(float)
        common_bits=attrs["bits"][~attrs["rare"]]
        observed_patterns=np.unique(common_bits@np.array([4,2,1]))
        assert np.array_equal(observed_patterns,np.arange(8)),observed_patterns
        direct=model(test_t).detach().numpy()
        bias=model.read.bias.detach().numpy()
        gate=float(np.max(np.abs(direct-logits(tables,test,bias))))
        assert gate<1e-10,gate
        variants={"original":tables}
        for rank in range(1,7):
            reduced=(u[:,:rank]*s[:rank])@v[:rank]
            variants[f"joint_rank{rank}"]=[unflatten(reduced)]
        for rank in (1,2):
            approximated=[]
            for table in tables:
                hu,hs,hv=np.linalg.svd(flatten(table),full_matrices=False)
                approximated.append(unflatten((hu[:,:rank]*hs[:rank])@hv[:rank]))
            variants[f"headwise_rank{rank}"]=approximated
        scored={name:score(t,test,targets,attrs,bias,direct)
                for name,t in variants.items()}
        errors={name:float(np.sum((sum(t)-total)**2)/np.sum(total**2))
                for name,t in variants.items()}
        head_deletions=[score([t for j,t in enumerate(tables) if j!=head],
                              test,targets,attrs,bias,direct)
                        for head in range(3)]
        rows.append(dict(seed=seed,direct_table_gate=gate,
            observed_common_bit_patterns=observed_patterns.tolist(),
            singular_values=s.tolist(),relative_uniform_table_mse=errors,
            scores=scored,head_deletions=head_deletions))
        print(seed,scored["original"]["rare_heldout"]["joint_accuracy"],
              scored["joint_rank2"]["all"]["joint_accuracy"],
              scored["joint_rank3"]["all"]["joint_accuracy"],flush=True)
    output=dict(n_seeds=10,n_test=30000,
        fit="exact weight-only combined 6x8 query/output-by-key table SVD; no sampled inputs or task labels",
        test="independent variable-count A/B/C sequences including rare A0/B1/C0 query holdout",
        rows=rows)
    (HERE/"three_count_analysis.json").write_text(json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    run()
