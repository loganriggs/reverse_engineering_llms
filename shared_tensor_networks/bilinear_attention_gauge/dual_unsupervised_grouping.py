"""Group sparse key atoms using unlabeled sentence co-occurrence."""

from __future__ import annotations

import json

import numpy as np
import torch

from dual_intervention_dictionary import tables_for_model
from dual_intervention_sparse_basis import random_rotation
from dual_joint_function_canonical import HERE,unflatten
from analyze_dual_output_language import logits
from train_dual_output_language import DualOutputAttention,batch,exhaustive


PAIRINGS=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]


def grouping(dictionary,sequences):
    # Summed key coordinates for each sentence. Query and target labels are
    # unused; only co-occurrence of key activations is visible.
    activations=dictionary[:,sequences[:,:3]].sum(axis=2).T
    correlation=np.corrcoef(activations.T)
    strengths=[sum(abs(correlation[a,b]) for a,b in pairs)
               for pairs in PAIRINGS]
    selected=int(np.argmax(strengths))
    return PAIRINGS[selected],dict(strength=float(strengths[selected]),
        runner_up=float(sorted(strengths)[-2]),correlation=correlation.tolist())


def group_effects(original_heads,dictionary,pairs):
    # The group ablation is the sum of its component contributions to the
    # *projected* original function; this remains a function-level edit.
    coefficients=original_heads@dictionary.T
    merged=coefficients[:4]+coefficients[4:]
    outputs=[]
    for pair in pairs:
        matrix=sum(np.outer(merged[:,index],dictionary[index]) for index in pair)
        table=unflatten(matrix)
        energy={"A":float(np.sum((table[:,1,:]-table[:,0,:])**2)),
                "B":float(np.sum((table[:,3,:]-table[:,2,:])**2)),
                "F":float(np.sum((table[:,5,:]-table[:,4,:])**2))}
        dominant=max(energy,key=energy.get)
        outputs.append(dict(dominant=dominant,energy=energy,
                            purity=energy[dominant]/max(sum(energy.values()),1e-30)))
    return outputs


def group_ablation(original_heads,dictionary,pairs,effects,eval_context):
    seq,targets,attrs,bias=eval_context
    coefficients=original_heads@dictionary.T
    merged=coefficients[:4]+coefficients[4:]
    full=merged@dictionary
    baseline=logits([unflatten(full)],seq,bias)>0
    assert np.all(baseline==targets)
    a_col=np.array([int(a["swap"]) for a in attrs])
    b_col=1-a_col
    row=np.arange(len(seq))
    outputs=[]
    for pair,effect in zip(pairs,effects):
        removed=sum(np.outer(merged[:,i],dictionary[i]) for i in pair)
        prediction=logits([unflatten(full-removed)],seq,bias)>0
        if effect["dominant"] not in ("A","B"):
            outputs.append(None)
            continue
        target_col=a_col if effect["dominant"]=="A" else b_col
        other_col=b_col if effect["dominant"]=="A" else a_col
        outputs.append(dict(affected_accuracy=float(np.mean(
            prediction[row,target_col]==targets[row,target_col])),
            unaffected_accuracy=float(np.mean(
                prediction[row,other_col]==targets[row,other_col]))))
    return outputs


def evaluate(original_heads,dictionary,sequences,eval_context=None):
    groups,pairing=grouping(dictionary,sequences)
    effects=group_effects(original_heads,dictionary,groups)
    correct=(set(e["dominant"] for e in effects)=={"A","B"})
    selective=(correct and all(e["purity"]>.9 for e in effects))
    result=dict(groups=[list(p) for p in groups],pairing=pairing,
                group_effects=effects,opposite_A_B=correct,
                opposite_A_B_above_90pct=selective,
                mean_group_purity=float(np.mean([e["purity"] for e in effects])))
    if eval_context is not None:
        result["group_ablation"]=group_ablation(original_heads,dictionary,groups,
                                                effects,eval_context)
    return result


def run():
    torch.set_num_threads(1)
    sparse=json.loads((HERE/"dual_intervention_sparse_basis.json").read_text())
    configurations=("three_rank4","four_rank4")
    sample_sizes=(32,128,512,2048)
    seq_t,target_t,attrs=exhaustive()
    eval_seq=seq_t.numpy();eval_target=target_t.numpy().astype(bool)
    records=[]
    for seed in range(10):
        model=DualOutputAttention().double()
        model.load_state_dict(torch.load(HERE/f"dual_output_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        original=tables_for_model(model)["original"]
        eval_context=(eval_seq,eval_target,attrs,model.read.bias.detach().numpy())
        for configuration in configurations:
            source=next(r for r in sparse["rows"]
                        if r["seed"]==seed and r["configuration"]==configuration)
            dictionaries={name:np.array(source["results"][name]["dictionary"])
                          for name in ("svd","sparse")}
            for n in sample_sizes:
                sampled,_=batch(n,torch.Generator().manual_seed(300000+10000*seed+n))
                sequences=sampled.numpy()
                variants={name:evaluate(original,d,sequences,eval_context)
                          for name,d in dictionaries.items()}
                rng=np.random.default_rng(500000+10000*seed+n)
                random_results=[evaluate(original,random_rotation(4,rng)@
                                         dictionaries["svd"],sequences)
                                for _ in range(128)]
                records.append(dict(seed=seed,configuration=configuration,
                    n_unlabeled_sentences=n,variants=variants,
                    random_baseline=dict(
                        opposite_A_B_rate=float(np.mean([z["opposite_A_B"]
                                                        for z in random_results])),
                        selective_rate=float(np.mean([z["opposite_A_B_above_90pct"]
                                                     for z in random_results])),
                        mean_group_purity=float(np.mean([z["mean_group_purity"]
                                                        for z in random_results])))))
    result=dict(task="recover two groups of two sparse atoms from unlabeled key co-occurrence",
        grouping_prior="four components are paired into two groups; maximize sum of absolute within-pair Pearson correlations",
        fitting="no token-family labels, output targets, or bit-flip responses used for grouping",
        evaluation="post-fit bit-flip response of each group on original projected function",
        records=records)
    (HERE/"dual_unsupervised_grouping.json").write_text(json.dumps(result,indent=2)+"\n")
    for configuration in configurations:
        for n in sample_sizes:
            sub=[r for r in records if r["configuration"]==configuration
                 and r["n_unlabeled_sentences"]==n]
            print(configuration,n,*[round(float(np.mean([
                r["variants"][v]["opposite_A_B_above_90pct"] for r in sub])),2)
                for v in ("svd","sparse")],round(float(np.mean([
                r["random_baseline"]["selective_rate"] for r in sub])),2),flush=True)
    return result


if __name__=="__main__":
    run()
