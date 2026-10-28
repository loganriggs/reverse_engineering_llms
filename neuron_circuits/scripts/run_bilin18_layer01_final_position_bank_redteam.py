#!/usr/bin/env python3
"""Pair-clustered and label-permutation red-team of final-position banks."""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_layer01_final_position_bank_redteam.json")
OUTPUT=TENSOR_ROOT/"layer01_final_position_bank_redteam.json"
REPORT=TENSOR_ROOT/"LAYER01_FINAL_POSITION_BANK_REDTEAM.md"


def jaccard(left, right):
    a,b=set(left),set(right)
    return len(a&b)/len(a|b)


def pair_rows(records):
    grouped=defaultdict(list)
    for row in records: grouped[(row["pair"],row["family"])].append(row)
    result=[]
    for (pair,family),rows in grouped.items():
        names=list(rows[0]["family_effects"])
        effects={name:float(np.median([row["family_effects"][name] for row in rows])) for name in names}
        correct=effects[family]; wrong=float(np.median([value for name,value in effects.items() if name!=family]))
        result.append({"pair":pair,"family":family,"effects":effects,"advantage":correct-wrong,"top1":correct==max(effects.values())})
    return result


def bootstrap_interval(values, samples, rng):
    values=np.asarray(values,dtype=np.float64); n=len(values)
    draws=np.median(values[rng.integers(0,n,size=(samples,n))],axis=1)
    return [float(value) for value in np.quantile(draws,[.025,.5,.975])]


def permutation_pvalue(rows, samples, rng):
    labels=np.asarray([row["family"] for row in rows],dtype=object)
    # The mean retains graded evidence under size-preserving permutations.  A
    # median becomes a low-power two-valued statistic for balanced two-family
    # controls and can assign a large p-value even to a perfect mapping.
    observed=float(np.mean([row["advantage"] for row in rows]))
    null=[]
    for _ in range(samples):
        assigned=rng.permutation(labels)
        advantages=[]
        for row,label in zip(rows,assigned):
            chosen=row["effects"][label]
            alternatives=[value for name,value in row["effects"].items() if name!=label]
            advantages.append(chosen-float(np.median(alternatives)))
        null.append(float(np.mean(advantages)))
    return observed,(1+sum(value>=observed for value in null))/(len(null)+1),[float(value) for value in np.quantile(null,[.01,.5,.99])]


def overlap_test(layer):
    banks=layer["bank_indices"]; names=list(banks)
    matrix=np.asarray([[jaccard(banks[left]["final"],banks[right]["sequence"]) for right in names] for left in names])
    observed=float(np.median(np.diag(matrix)) - np.median(matrix[~np.eye(len(names),dtype=bool)]))
    permutations=list(itertools.permutations(range(len(names))))
    null=[]
    for permutation in permutations:
        selected=np.asarray([matrix[index,permutation[index]] for index in range(len(names))])
        mask=np.ones_like(matrix,dtype=bool)
        for index,target in enumerate(permutation): mask[index,target]=False
        null.append(float(np.median(selected)-np.median(matrix[mask])))
    p=(1+sum(value>=observed for value in null))/(len(null)+1)
    return {"same_median":float(np.median(np.diag(matrix))),"wrong_median":float(np.median(matrix[~np.eye(len(names),dtype=bool)])),"advantage":observed,"exact_p":p,"matrix":matrix.tolist(),"families":names}


def main():
    config=json.loads(CONFIG.read_text()); primary=json.loads(Path(config["primary_output"]).read_text()); single=json.loads(Path(config["single_context_output"]).read_text())
    rng=np.random.default_rng(config["seed"]); results={}
    for layer in primary["config"]["layers"]:
        layer_result={"overlap":overlap_test(primary["layers"][str(layer)])}
        for corpus in primary["layers"][str(layer)]["summaries"]:
            rows=pair_rows([row for row in primary["layers"][str(layer)]["records"] if row["corpus"]==corpus])
            permutation_stat,pvalue,null=permutation_pvalue(rows,config["permutation_samples"],rng)
            single_adv=single["layers"][str(layer)]["summaries"][corpus]["advantage"]
            median_advantage=float(np.median([row["advantage"] for row in rows]))
            layer_result[corpus]={
                "pair_count":len(rows),"median_advantage":median_advantage,
                "permutation_mean_statistic":permutation_stat,
                "bootstrap_ci":bootstrap_interval([row["advantage"] for row in rows],config["bootstrap_samples"],rng),
                "permutation_p":pvalue,"permutation_null_q01_median_q99":null,
                "pair_win_rate":float(np.mean([row["advantage"]>0 for row in rows])),
                "pair_top1_rate":float(np.mean([row["top1"] for row in rows])),
                "single_context_advantage":single_adv,"multicontext_improvement":median_advantage-single_adv,
                "family_advantages":{family:float(np.median([row["advantage"] for row in rows if row["family"]==family])) for family in sorted({row["family"] for row in rows})},
            }
        results[str(layer)]=layer_result
    cells=[results[str(layer)][corpus] for layer in primary["config"]["layers"] for corpus in primary["layers"][str(layer)]["summaries"]]
    failed_single=[(layer,corpus) for layer in primary["config"]["layers"] for corpus in primary["layers"][str(layer)]["summaries"] if single["layers"][str(layer)]["summaries"][corpus]["advantage"]<.03]
    hypotheses={
        "R0_pair_clustered_advantage_is_positive":{"supported":all(row["bootstrap_ci"][0]>0 for row in cells)},
        "R1_family_mapping_beats_permutation":{"supported":all(row["permutation_p"]<.05 for row in cells)},
        "R2_correct_family_is_usually_top1":{"supported":all(row["pair_top1_rate"]>=.65 for row in cells)},
        "R3_two_context_discovery_rescues_single_context_failure":{"supported":all(results[str(layer)][corpus]["multicontext_improvement"]>=.03 for layer,corpus in failed_single)},
        "R4_position_scopes_share_family_specific_axes":{"supported":all(results[str(layer)]["overlap"]["advantage"]>0 and results[str(layer)]["overlap"]["exact_p"]<.05 for layer in primary["config"]["layers"])},
    }
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results},indent=2))
    lines=["# Red-team of final-position semantic-bank failure","","The stronger two-FineWeb-context discovery banks are evaluated with lexical pairs—not contexts—as the replication unit. Five thousand pair bootstraps and five thousand size-preserving family-label permutations test whether weak medians hide reliable mapping. Exact permutations of the five family labels test whether final-position and distributed-sequence banks share family-specific native axes.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","| Layer | Corpus | Median advantage (95% pair CI) | Pair win / top-1 | Label p | Single→multi improvement |","|---:|---|---:|---:|---:|---:|"]
    for layer in primary["config"]["layers"]:
        for corpus in primary["layers"][str(layer)]["summaries"]:
            row=results[str(layer)][corpus]; ci=row["bootstrap_ci"]
            lines.append(f"| {layer} | {corpus} | {row['median_advantage']:.3f} ({ci[0]:.3f}, {ci[2]:.3f}) | {row['pair_win_rate']:.2f} / {row['pair_top1_rate']:.2f} | {row['permutation_p']:.4f} | {row['multicontext_improvement']:+.3f} |")
    lines += ["","| Layer | Same-scope-family overlap | Wrong-family overlap | Advantage | Exact mapping p |","|---:|---:|---:|---:|---:|"]
    for layer in primary["config"]["layers"]:
        row=results[str(layer)]["overlap"]; lines.append(f"| {layer} | {row['same_median']:.3f} | {row['wrong_median']:.3f} | {row['advantage']:.3f} | {row['exact_p']:.4f} |")
    lines += ["","A negative result here is specifically about a reusable **family-labelled native-neuron bank at the final position**. It does not negate per-example native sparsity, distributed-prefix family structure, subspace-level reuse, or an attention-level relational operator. The high per-example oracle remains the positive control."]
    REPORT.write_text("\n".join(lines)+"\n"); print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__=="__main__": main()
