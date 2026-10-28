#!/usr/bin/env python3
"""Pair-clustered red-team of high-data final-write subspace specificity."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_layer01_final_write_subspace_redteam.json")
OUTPUT=TENSOR_ROOT/"layer01_final_write_subspace_redteam.json"
REPORT=TENSOR_ROOT/"LAYER01_FINAL_WRITE_SUBSPACE_REDTEAM.md"


def aggregate(records,rank):
    groups=defaultdict(list)
    for row in records: groups[(row["pair"],row["family"])].append(row)
    output=[]
    for (pair,family),rows in groups.items():
        names=list(rows[0]["ranks"][str(rank)]["family_effects"])
        effects={name:float(np.median([row["ranks"][str(rank)]["family_effects"][name] for row in rows])) for name in names}; correct=effects[family]; wrong=float(np.median([value for name,value in effects.items() if name!=family]))
        output.append({"pair":pair,"family":family,"effects":effects,"advantage":correct-wrong,"top1":correct==max(effects.values())})
    return output


def bootstrap(values,samples,rng):
    values=np.asarray(values); draws=np.median(values[rng.integers(0,len(values),size=(samples,len(values)))],axis=1)
    return [float(value) for value in np.quantile(draws,[.025,.5,.975])]


def permutation(rows,samples,rng):
    labels=np.asarray([row["family"] for row in rows],dtype=object); observed=float(np.mean([row["advantage"] for row in rows])); null=[]
    for _ in range(samples):
        advantages=[]
        for row,label in zip(rows,rng.permutation(labels)):
            advantages.append(row["effects"][label]-float(np.median([value for name,value in row["effects"].items() if name!=label])))
        null.append(float(np.mean(advantages)))
    return (1+sum(value>=observed for value in null))/(samples+1)


def main():
    config=json.loads(CONFIG.read_text()); source=json.loads(Path(config["source_output"]).read_text()); rng=np.random.default_rng(config["seed"]); results={}
    for layer,layer_row in source["layers"].items():
        results[layer]={}
        for corpus in layer_row["summaries"]:
            rows=aggregate([row for row in layer_row["records"] if row["corpus"]==corpus],config["rank"]); results[layer][corpus]={"pair_count":len(rows),"median_advantage":float(np.median([row["advantage"] for row in rows])),"bootstrap_ci":bootstrap([row["advantage"] for row in rows],config["bootstrap_samples"],rng),"permutation_p":permutation(rows,config["permutation_samples"],rng),"pair_win_rate":float(np.mean([row["advantage"]>0 for row in rows])),"pair_top1_rate":float(np.mean([row["top1"] for row in rows])),"family_advantages":{family:float(np.median([row["advantage"] for row in rows if row["family"]==family])) for family in sorted({row["family"] for row in rows})}}
    cells=[row for layer in results.values() for row in layer.values()]
    hypotheses={"T0_pair_advantage_is_positive":{"supported":all(row["bootstrap_ci"][0]>0 for row in cells)},"T1_family_labels_beat_permutation":{"supported":all(row["permutation_p"]<.05 for row in cells)},"T2_correct_family_is_usually_top1":{"supported":all(row["pair_top1_rate"]>=.65 for row in cells)}}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results},indent=2)); lines=["# Pair-clustered red-team of rank-24 final-write subspaces","","The eight-context discovery bases remain frozen. Evaluation contexts are aggregated within each of the seventeen prospective lexical pairs before 5,000 pair bootstraps and 5,000 size-preserving family-label permutations.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","| Layer | Corpus | Median advantage (95% CI) | Win / top-1 | Label p |","|---:|---|---:|---:|---:|"]
    for layer,layer_rows in results.items():
        for corpus,row in layer_rows.items():
            ci=row["bootstrap_ci"]; lines.append(f"| {layer} | {corpus} | {row['median_advantage']:.3f} ({ci[0]:.3f}, {ci[2]:.3f}) | {row['pair_win_rate']:.2f} / {row['pair_top1_rate']:.2f} | {row['permutation_p']:.4f} |")
    lines += ["","A significant cell would support partial family structure inside an otherwise low-fidelity query-write span. Universal failure is not required to reject a universal reusable-feature claim; cell-specific positives must be retained as such."]
    REPORT.write_text("\n".join(lines)+"\n"); print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__=="__main__": main()
