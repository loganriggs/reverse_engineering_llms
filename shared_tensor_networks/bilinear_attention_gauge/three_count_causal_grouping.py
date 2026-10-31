"""Unsupervised grouping of intervention-aware key atoms in a trained three-rule toy."""

from __future__ import annotations

import json
import argparse

import numpy as np
import scipy.optimize as so
import torch

from analyze_three_count_language import flatten,head_tables,logits,rms,unflatten
from dual_functional_grouping import connected_groups
from dual_intervention_sparse_basis import random_rotation,sparse_rotation
from dual_joint_function_canonical import HERE
from train_three_count_language import PAD,ThreeCountAttention,examples


FIT=("original","qk1_constant","qk2_constant")
ALL=(*FIT,"both_constant")
THRESHOLDS=(.6,.7,.8,.9)
TRUTH=np.zeros((4,8))
for family in range(4):
    TRUTH[family,2*family]=-1/np.sqrt(2)
    TRUTH[family,2*family+1]=1/np.sqrt(2)


def tables(model):
    embedding=model.embed.weight.detach().numpy()
    query=embedding[8:10];key=embedding[:8]
    width=model.qk_width
    output={name:[] for name in ALL}
    for head in range(model.heads):
        def projected(name,inputs):
            weight=getattr(model,name).weight.detach().numpy()
            sl=slice(head*width,(head+1)*width)
            return rms(inputs@weight[sl].T)
        score1=projected("q1",query)@projected("k1",key).T/width
        score2=projected("q2",query)@projected("k2",key).T/width
        value=key@model.v.weight.detach().numpy()[head]
        read=model.read.weight.detach().numpy()[:,head]
        for name,score in (("original",score1*score2),
                           ("qk1_constant",score2),
                           ("qk2_constant",score1),
                           ("both_constant",np.ones_like(score1))):
            output[name].append(score[:,:,None]*value[None,:,None]*
                                read[None,None,:])
    gate=float(np.max(np.abs(sum(output["original"])-sum(head_tables(model)))))
    assert gate<1e-10,gate
    return {name:flatten(sum(value)) for name,value in output.items()},gate


def similarity(matrix,dictionary):
    response=matrix@dictionary.T
    response/=np.maximum(np.linalg.norm(response,axis=0,keepdims=True),1e-30)
    return np.abs(response.T@response)


def count_coactivity(tokens,dictionary):
    keys=tokens[:,:-1]
    counts=np.stack([np.bincount(row[row!=PAD],minlength=8) for row in keys])
    activity=counts@dictionary.T
    return np.abs(np.corrcoef(activity,rowvar=False))


def evaluate_groups(original,dictionary,groups):
    summaries=[]
    for group in groups:
        projection=dictionary[group].T@dictionary[group]
        coverage=np.sum((TRUTH@projection)**2,axis=1)
        component=original@projection
        effects=np.sum((component@TRUTH.T)**2,axis=0)
        purity=effects/max(effects.sum(),1e-30)
        summaries.append(dict(indices=group,coverage=coverage.tolist(),
            effect_purity=purity.tolist(),effect_energy=effects.tolist()))
    assignment={};complete=False
    if len(groups)>=3:
        score=np.array([[row["coverage"][bit]*row["effect_purity"][bit]
                         for row in summaries] for bit in range(3)])
        bit_ids,group_ids=so.linear_sum_assignment(-score)
        assignment={"ABC"[bit]:int(group) for bit,group in zip(bit_ids,group_ids)}
        complete=all(summaries[assignment[bit]]["coverage"][i]>.9 and
                     summaries[assignment[bit]]["effect_purity"][i]>.9
                     for i,bit in enumerate("ABC"))
    return dict(groups=groups,n_groups=len(groups),group_summaries=summaries,
                matched_groups=assignment,complete_A_B_C=complete)


def run(n_seeds=10,n_unlabeled=8192,n_test=10000,
        fit_conditions=FIT):
    torch.set_num_threads(1)
    rows=[]
    for seed in range(n_seeds):
        model=ThreeCountAttention().double()
        model.load_state_dict(torch.load(HERE/f"three_count_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        model.eval()
        matrices,gate=tables(model)
        gate_tokens,_,_=examples(256,88000+seed,False)
        direct=model(gate_tokens).detach().numpy()
        bias=model.read.bias.detach().numpy()
        direct_gate=float(np.max(np.abs(direct-logits(
            [unflatten(matrices["original"])],gate_tokens.numpy(),bias))))
        assert direct_gate<1e-10,direct_gate
        stack=np.vstack([matrices[name]/np.linalg.norm(matrices[name])
                         for name in fit_conditions])
        _,singular,right=np.linalg.svd(stack,full_matrices=False)
        dictionary=right[:6]
        rotation,sparse,opt=sparse_rotation(dictionary,seed+19000)
        assert np.max(np.abs(dictionary.T@dictionary-
                             sparse.T@sparse))<1e-10
        unlabeled,_,_=examples(n_unlabeled,91000+seed,True)
        variants={"svd":dictionary,"sparse":sparse}
        scores={}
        for variant,d in variants.items():
            fitted=matrices["original"]@d.T@d
            recon={name:float(np.sum((matrix-matrix@d.T@d)**2)/
                              np.sum(matrix**2))
                   for name,matrix in matrices.items()}
            signatures={"ordinary_function":similarity(matrices["original"],d),
                "causal_function":similarity(stack,d),
                "sentence_coactivity":count_coactivity(unlabeled.numpy(),d)}
            graphs={}
            for method,sim in signatures.items():
                graphs[method]={}
                for threshold in THRESHOLDS:
                    groups=connected_groups(sim,threshold)
                    graphs[method][str(threshold)]=evaluate_groups(
                        matrices["original"],d,groups)
            scores[variant]=dict(key_l1=float(np.abs(d).sum()),
                reconstruction=recon,graphs=graphs,
                dictionary=d.tolist(),optimization=opt if variant=="sparse" else None)
            ordinary=graphs["ordinary_function"]["0.8"]
            heldout=evaluate_groups(matrices["both_constant"],d,
                                     ordinary["groups"])
            scores[variant]["ordinary_groups_on_double_edit"]=dict(
                relative_table_mse=recon["both_constant"],
                group_effects=heldout,
                matched_bit_purity={bit:heldout["group_summaries"][
                    ordinary["matched_groups"][bit]]["effect_purity"][i]
                    for i,bit in enumerate("ABC")
                    if bit in ordinary["matched_groups"]})
            # Post-fit task interventions on groups chosen without labels.
            task_tests={}
            for rate in (.35,.75,1.2):
                test_tokens,test_targets,attrs=examples(
                    n_test,93000+seed+int(rate*1000),False,rate)
                tokens=test_tokens.numpy();targets=test_targets.numpy()
                bias=model.read.bias.detach().numpy()
                reference=logits([unflatten(fitted)],tokens,bias)
                rare=attrs["heldout"]
                assert rare.any()
                base_pred=reference>0
                task_deletions={}
                for method in signatures:
                    chosen=graphs[method]["0.8"]
                    deletions={}
                    for bit,group_index in chosen["matched_groups"].items():
                        indices=chosen["groups"][group_index]
                        component=matrices["original"]@d[indices].T@d[indices]
                        edited=logits([unflatten(fitted-component)],tokens,bias)
                        pred=edited>0
                        order=np.stack((np.where(attrs["rare"],2,0),
                                        np.where(attrs["rare"],0,1),
                                        np.where(attrs["rare"],1,2)),axis=1)
                        semantic={name:float(np.mean(pred[np.arange(len(pred)),order[:,i]]==
                                  attrs["bits"][:,i])) for i,name in enumerate("ABC")}
                        deletions[bit]=dict(semantic_accuracy=semantic,
                            rare_heldout_joint_accuracy=float(np.mean(np.all(
                                pred[rare]==targets[rare].astype(bool),axis=1))),
                            relative_logit_effect=float(np.sum((edited-reference)**2)/
                                                        np.sum(reference**2)))
                    task_deletions[method]=deletions
                task_tests[str(rate)]=dict(
                    n_rare_heldout=int(rare.sum()),
                    reduced_joint_accuracy=float(np.mean(np.all(
                        base_pred==targets.astype(bool),axis=1))),
                    reduced_rare_heldout_joint_accuracy=float(np.mean(np.all(
                        base_pred[rare]==targets[rare].astype(bool),axis=1))),
                    threshold_0_8_task_deletions=task_deletions)
            scores[variant]["shifted_count_task_tests"]=task_tests
        rng=np.random.default_rng(99000+seed)
        random_complete={"ordinary_function":[],"causal_function":[],
                         "sentence_coactivity":[]}
        for _ in range(32):
            d=random_rotation(6,rng)@dictionary
            signatures={"ordinary_function":similarity(matrices["original"],d),
                "causal_function":similarity(stack,d),
                "sentence_coactivity":count_coactivity(unlabeled.numpy(),d)}
            for method,sim in signatures.items():
                groups=connected_groups(sim,.8)
                random_complete[method].append(evaluate_groups(
                    matrices["original"],d,groups)["complete_A_B_C"])
        rows.append(dict(seed=seed,exact_original_gate=gate,
            direct_forward_gate=direct_gate,
            fitted_singular_values=singular.tolist(),
            random_0_8_complete_rates={method:float(np.mean(values))
                for method,values in random_complete.items()},
            variants=scores))
        print(seed,scores["sparse"]["graphs"]["causal_function"]["0.8"]
              ["complete_A_B_C"],flush=True)
    output=dict(n_seeds=n_seeds,fit_conditions=list(fit_conditions),
        heldout_condition=("both_constant" if "both_constant" not in
                           fit_conditions else None),rank=6,
        thresholds=list(THRESHOLDS),n_unlabeled=n_unlabeled,
        n_test_per_rate=n_test,test_rates=[.35,.75,1.2],
        unlabeled_rule="training-distribution sentences excluding rare whole-combination holdout; labels discarded",
        grouping="connected components of absolute cosine of query/output functional signatures or absolute Pearson sentence-coordinate coactivity",
        postfit_success=">.9 key-contrast subspace coverage and >.9 original-function contrast-effect purity for distinct A/B/C groups",
        random_control="32 independent orthogonal rotations per seed for all three grouping signatures at threshold .8",
        rows=rows)
    name=("three_count_causal_grouping_all_four.json" if
          "both_constant" in fit_conditions else "three_count_causal_grouping.json")
    (HERE/name).write_text(
        json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--include-double",action="store_true")
    args=parser.parse_args()
    run(fit_conditions=ALL if args.include_double else FIT)
