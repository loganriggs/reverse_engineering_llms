"""Data-free joint function reduction and sparse key gauge on trained dual-head toys.

This canonicalizes the *combined effective token function*, not either
physical QK bond. Known A/B labels are used only after fitting for evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.optimize as so
import torch

from analyze_dual_output_language import factors,logits,score
from analyze_toy_language import pair_table
from train_dual_output_language import DualOutputAttention,exhaustive


HERE=Path(__file__).parent
TRUTH=np.array([[-1.,1.,0.,0.,0.,0.],
                [0.,0.,-1.,1.,0.,0.]])
TRUTH/=np.linalg.norm(TRUTH,axis=1,keepdims=True)
N_ANGLES=4096


def flatten(table):
    return table.transpose(0,2,1).reshape(4,6)


def unflatten(matrix):
    return matrix.reshape(2,2,6).transpose(0,2,1)


def truncate(matrix,rank):
    u,s,v=np.linalg.svd(matrix,full_matrices=False)
    return (u[:,:rank]*s[:rank])@v[:rank],s


def rotations(n=N_ANGLES):
    theta=np.linspace(0,np.pi,n,endpoint=False)
    c=np.cos(theta);s=np.sin(theta)
    return np.stack((c,-s,s,c),axis=1).reshape(-1,2,2),theta


def matched_cosines(dictionary):
    normalized=dictionary/np.linalg.norm(dictionary,axis=1,keepdims=True)
    correlation=np.abs(normalized@TRUTH.T)
    row,column=so.linear_sum_assignment(-correlation)
    return float(np.mean(correlation[row,column])),\
           {("A" if label==0 else "B"):int(component)
            for component,label in zip(row,column)}


def causal_metrics(row_factors,dictionary,seq,targets,attrs,bias):
    labels=matched_cosines(dictionary)[1]
    metrics={}
    for named_bit,component in labels.items():
        contribution=np.outer(row_factors[:,component],dictionary[component])
        table=unflatten(contribution)
        a_effect=np.sum((table[:,1,:]-table[:,0,:])**2)
        b_effect=np.sum((table[:,3,:]-table[:,2,:])**2)
        f_effect=np.sum((table[:,5,:]-table[:,4,:])**2)
        selectivity=float(max(a_effect,b_effect)/max(a_effect+b_effect,1e-30))
        intended=(a_effect if named_bit=="A" else b_effect)
        intended_fraction=float(intended/max(a_effect+b_effect+f_effect,1e-30))
        full=unflatten(row_factors@dictionary)
        ablated_logits=logits([full-table],seq,bias)
        predictions=ablated_logits>0
        bit_results={}
        for heldout in (False,True):
            mask=np.array([a["heldout"] for a in attrs]) if heldout else np.ones(len(attrs),dtype=bool)
            a_coord=np.array([1 if a["swap"] else 0 for a in attrs])
            b_coord=1-a_coord
            ar=np.mean(predictions[np.arange(len(seq))[mask],a_coord[mask]]==
                       targets[np.arange(len(seq))[mask],a_coord[mask]])
            br=np.mean(predictions[np.arange(len(seq))[mask],b_coord[mask]]==
                       targets[np.arange(len(seq))[mask],b_coord[mask]])
            bit_results["heldout" if heldout else "all"]=dict(
                A_accuracy=float(ar),B_accuracy=float(br))
        metrics[named_bit]=dict(component=component,
            bit_flip_energy=dict(A=float(a_effect),B=float(b_effect),F=float(f_effect)),
            bit_flip_selectivity=selectivity,
            intended_bit_energy_fraction_including_filler=intended_fraction,
            after_component_removal=bit_results)
    return metrics


def one_seed(seed,seq,targets,attrs):
    model=DualOutputAttention().double()
    model.load_state_dict(torch.load(HERE/f"dual_output_seed{seed}.pt",
                                     map_location="cpu",weights_only=True))
    model.eval()
    heads=[factors(model,h) for h in range(2)]
    tables=[pair_table(h,np.eye(h["rank"])) for h in heads]
    bias=model.read.bias.detach().numpy()
    direct=model(torch.from_numpy(seq)).detach().numpy()
    original_logits=logits(tables,seq,bias)
    gate=float(np.max(np.abs(direct-original_logits)))
    assert gate<1e-10,gate
    total=sum(tables)
    matrix=flatten(total)
    u,s,v=np.linalg.svd(matrix,full_matrices=False)
    left=u[:,:2]*np.sqrt(s[:2])
    dictionary=np.sqrt(s[:2,None])*v[:2]
    rank2=left@dictionary
    rank1,_=truncate(matrix,1)
    headwise=[]
    for table in tables:
        approx,_=truncate(flatten(table),1)
        headwise.append(unflatten(approx))
    joint_preserving=np.concatenate([flatten(table) for table in tables],axis=0)
    joint_rank2,head_s=truncate(joint_preserving,2)
    head_tables=[unflatten(joint_rank2[:4]),unflatten(joint_rank2[4:])]
    variants={"original":tables,"collapsed_rank1":[unflatten(rank1)],
              "collapsed_rank2":[unflatten(rank2)],
              "headwise_rank1":headwise,
              "joint_heads_rank2":head_tables}
    scores={name:score(t,seq,targets,bias,attrs)[0]
            for name,t in variants.items()}
    table_error={name:float(np.sum((sum(t)-total)**2)/np.sum(total**2))
                 for name,t in variants.items()}
    assert scores["original"]["joint_accuracy"]==1.
    assert scores["collapsed_rank2"]["joint_accuracy"]==1.
    # A key-dictionary L1 prior, chosen without task/bit labels.
    all_rot,angles=rotations()
    candidate_d=np.einsum("tij,jk->tik",all_rot,dictionary)
    key_l1=np.sum(np.abs(candidate_d),axis=(1,2))
    selected=int(np.argmin(key_l1))
    sparse_rot=all_rot[selected]
    sparse_d=sparse_rot@dictionary
    sparse_left=left@sparse_rot.T
    assert np.max(np.abs(sparse_left@sparse_d-rank2))<1e-12
    factor_l1=(np.abs(candidate_d).sum((1,2))/np.linalg.norm(dictionary)+
               np.abs(np.einsum("ij,tjk->tik",left,all_rot.transpose(0,2,1))).sum((1,2))
               /np.linalg.norm(left))
    factor_index=int(np.argmin(factor_l1))
    factor_rot=all_rot[factor_index]
    factor_d=factor_rot@dictionary
    factor_left=left@factor_rot.T
    rng=np.random.default_rng(1000+seed)
    random_angles=rng.uniform(0,np.pi,128)
    random_align=[];random_selectivity=[]
    for angle in random_angles:
        r=np.array([[np.cos(angle),-np.sin(angle)],
                    [np.sin(angle),np.cos(angle)]])
        d=r@dictionary;l=left@r.T
        random_align.append(matched_cosines(d)[0])
        cm=causal_metrics(l,d,seq,targets,attrs,bias)
        random_selectivity.append(np.mean([z["bit_flip_selectivity"]
                                            for z in cm.values()]))
    gauges={"svd":(left,dictionary),"key_sparse":(sparse_left,sparse_d),
            "factor_sparse":(factor_left,factor_d)}
    gauge_results={}
    for name,(l,d) in gauges.items():
        alignment,assignment=matched_cosines(d)
        gauge_results[name]=dict(
            matched_A_B_cosine=alignment,
            component_assignment=assignment,
            key_dictionary_l1=float(np.sum(np.abs(d))),
            dictionary=d.tolist(),
            row_factors=l.tolist(),
            causal=causal_metrics(l,d,seq,targets,attrs,bias))
    sparse_alignment=gauge_results["key_sparse"]["matched_A_B_cosine"]
    sparse_selectivity=float(np.mean([
        value["bit_flip_selectivity"]
        for value in gauge_results["key_sparse"]["causal"].values()]))
    oracle_alignment=max(matched_cosines(d)[0] for d in candidate_d)
    return dict(seed=seed,direct_table_gate=gate,
        singular_values=s.tolist(),
        head_preserving_singular_values=head_s.tolist(),
        scores=scores,relative_table_mse=table_error,
        head_preserving_rank2_relative_mse=float(
            np.sum((joint_rank2-joint_preserving)**2)/np.sum(joint_preserving**2)),
        key_sparse_angle=float(angles[selected]),
        factor_sparse_angle=float(angles[factor_index]),
        random_rotation_alignment_mean=float(np.mean(random_align)),
        random_rotation_alignment_p95=float(np.quantile(random_align,.95)),
        random_alignment_exceed_sparse_fraction=float(
            np.mean(np.asarray(random_align)>=sparse_alignment)),
        label_oracle_rotation_alignment=float(oracle_alignment),
        random_rotation_selectivity_mean=float(np.mean(random_selectivity)),
        random_selectivity_exceed_sparse_fraction=float(
            np.mean(np.asarray(random_selectivity)>=sparse_selectivity)),
        gauges=gauge_results)


def run():
    torch.set_num_threads(1)
    seq_t,targets_t,attrs=exhaustive()
    seq=seq_t.numpy();targets=targets_t.numpy().astype(np.float64)
    observed=np.array([not a["heldout"] for a in attrs])
    coverage={(q,k):bool(np.any((seq[observed,-1]-6==q)&
                                 np.any(seq[observed,:-1]==k,axis=1)))
              for q in range(2) for k in range(6)}
    assert all(coverage.values()),coverage
    records=[one_seed(seed,seq,targets,attrs) for seed in range(10)]
    result=dict(seeds=list(range(10)),role_valid_query_key_pairs=12,
        pair_cells_covered_by_nonholdout_sequences=len(coverage),
        sequence_holdout="rare query with A0/B1; 12 of 96 exhaustive sequences",
        collapsed_rank2_raw_parameters=20,
        collapsed_rank2_effective_degrees=16,
        headwise_rank1_raw_parameters=20,
        headwise_rank1_effective_degrees=18,
        sparse_objective="minimize L1 of 2x6 key dictionary over 4096 orthogonal rotations; no task labels",
        ground_truth_use="A/B bit detector vectors only in post-fit alignment and causal evaluation",
        scope="exact role-valid token function of ten trained two-head toys; not physical head/circuit recovery",
        records=records)
    (HERE/"dual_joint_function_canonical.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    x=run()
    for r in x["records"]:
        print(r["seed"],r["scores"]["collapsed_rank2"]["joint_accuracy"],
              r["gauges"]["key_sparse"]["matched_A_B_cosine"],flush=True)
