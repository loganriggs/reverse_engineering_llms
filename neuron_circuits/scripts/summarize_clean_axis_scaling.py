#!/usr/bin/env python3
"""Aggregate clean-axis scaling runs using width-normalized sparse budgets."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np


def interval(values):
    if not values:
        return {"mean": None, "sd": None, "ci95_low": None, "ci95_high": None, "count": 0}
    mean = statistics.mean(values); sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci95_low": mean-half, "ci95_high": mean+half, "count": len(values)}


def recovery(row, method):
    if method == "raw": return row["raw_confirmation"]["kl_recovery"]
    if method == "rotation": return row["rotation_confirmation"]["kl_recovery"]["mean"]
    if method == "pca": return row["pca_confirmation"]["kl_recovery"]
    raise ValueError(method)


def threshold(rows, method, target):
    return next((row["rank"] for row in rows if recovery(row, method) >= target), None)


def run_metrics(run):
    width = run["config"]["neurons"]
    output = {}
    for family, analysis in run["analysis"]["families"].items():
        output[family] = {}
        depth = len(analysis["layers"])
        for position, layer in (("first", "1"), ("last", str(depth))):
            rows = analysis["layers"][layer]
            low, high = math.ceil(width/64), width/4
            sparse = [row for row in rows if low <= row["rank"] <= high]
            numerator = sum(recovery(row,"raw")-recovery(row,"rotation") for row in sparse)
            denominator = sum(recovery(row,"pca")-recovery(row,"rotation") for row in sparse)
            metrics = {
                "clean_axis_index": numerator / denominator if abs(denominator) > 1e-9 else None,
                "raw_axis_advantage": statistics.mean(recovery(row,"raw")-recovery(row,"rotation") for row in sparse),
                "subspace_advantage": statistics.mean(recovery(row,"pca")-recovery(row,"raw") for row in sparse),
            }
            for target in (.8,.9):
                raw_rank, pca_rank = threshold(rows,"raw",target), threshold(rows,"pca",target)
                metrics[f"efficiency_{int(target*100)}"] = pca_rank/raw_rank if raw_rank and pca_rank else None
                metrics[f"raw_rank_{int(target*100)}"] = raw_rank
                metrics[f"pca_rank_{int(target*100)}"] = pca_rank
            output[family][position] = metrics
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("input",type=Path); args=parser.parse_args()
    manifest=json.loads((args.input/"manifest.json").read_text(encoding="utf-8"))
    unique={row["slug"]:row for row in manifest["runs"]}
    grouped=defaultdict(list); accuracy=defaultdict(list); convergence=defaultdict(list)
    for slug, descriptor in unique.items():
        for path in sorted((args.input/slug).glob("seed_*/results.json")):
            run=json.loads(path.read_text(encoding="utf-8")); grouped[slug].append(run_metrics(run))
            accuracy[slug].append(run["evaluations"]["confirmation"]["accuracy"]); convergence[slug].append(run["training"]["converged"])
    summary={"base":manifest["base"],"grid":manifest["grid"],"conditions":{}}
    metric_names=("clean_axis_index","raw_axis_advantage","subspace_advantage","efficiency_80","efficiency_90")
    for slug,runs in grouped.items():
        descriptor=unique[slug]
        condition={"config":{key:descriptor[key] for key in ("classes","operations","depth","neurons")},"seeds":len(runs),"converged":sum(convergence[slug]),"accuracy":interval(accuracy[slug]),"families":{}}
        for family in ("operation","class"):
            condition["families"][family]={}
            for position in ("first","last"):
                condition["families"][family][position]={metric:interval([run[family][position][metric] for run in runs if run[family][position][metric] is not None]) for metric in metric_names}
            condition["families"][family]["last_minus_first"] = {
                metric: interval([
                    run[family]["last"][metric] - run[family]["first"][metric]
                    for run in runs
                    if run[family]["last"][metric] is not None and run[family]["first"][metric] is not None
                ])
                for metric in metric_names
            }
        summary["conditions"][slug]=condition
    (args.input/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    base=manifest["base"]
    fig,axes=plt.subplots(2,4,figsize=(16,7),sharey="row")
    for column,(factor,values) in enumerate(manifest["grid"].items()):
        for row,(family,color) in enumerate((("operation","#0072b2"),("class","#d55e00"))):
            axis=axes[row,column]; means=[]; lows=[]; highs=[]
            for value in values:
                config=base|{factor:value}; slug=f"c{config['classes']}_o{config['operations']}_d{config['depth']}_w{config['neurons']}"
                metric=summary["conditions"][slug]["families"][family]["last"]["clean_axis_index"]
                means.append(metric["mean"]); lows.append(metric["ci95_low"]); highs.append(metric["ci95_high"])
            axis.plot(values,means,marker="o",color=color); axis.fill_between(values,lows,highs,color=color,alpha=.18)
            axis.axhline(0,color="#777",linewidth=.8); axis.axhline(1,color="#777",linewidth=.8,linestyle=":")
            axis.set_title(factor); axis.grid(alpha=.2)
            if column==0: axis.set_ylabel(f"{family}\nClean Axis Index")
            axis.set_xlabel(factor)
    fig.suptitle("Clean-neuron axis alignment under scaling (final layer, mean ± 95% CI)",fontweight="bold")
    fig.tight_layout(); fig.savefig(args.input/"clean_axis_scaling.png",dpi=200); plt.close(fig)


if __name__=="__main__": main()
