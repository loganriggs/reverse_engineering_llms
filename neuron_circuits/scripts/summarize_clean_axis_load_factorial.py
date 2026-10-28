#!/usr/bin/env python3
"""Aggregate the class-count by width follow-up and fit the sealed interaction model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from summarize_clean_axis_scaling import interval, run_metrics


ROOT=Path("artifacts/clean_axis_scaling")


def coefficients(rows):
    log_c=np.log([row["classes"] for row in rows]); log_w=np.log([row["neurons"] for row in rows])
    log_c=log_c-np.log(30); log_w=log_w-np.log(128)
    design=np.column_stack((np.ones(len(rows)),log_c,log_w,log_c*log_w))
    return np.linalg.lstsq(design,np.array([row["value"] for row in rows]),rcond=None)[0]


def main():
    manifest=json.loads((ROOT/"load_factorial_manifest.json").read_text(encoding="utf-8"))
    records=[]; cells={}
    for classes in manifest["classes"]:
        for neurons in manifest["neurons"]:
            slug=f"c{classes}_o4_d2_w{neurons}"; cells[slug]={"classes":classes,"neurons":neurons,"runs":[]}
            for path in sorted((ROOT/slug).glob("seed_*/results.json")):
                run=json.loads(path.read_text(encoding="utf-8")); metrics=run_metrics(run)
                seed=run["config"]["seed"]; cells[slug]["runs"].append({"seed":seed,"metrics":metrics,"accuracy":run["evaluations"]["confirmation"]["accuracy"],"converged":run["training"]["converged"]})
                for family in ("operation","class"):
                    for metric in ("clean_axis_index","raw_axis_advantage","efficiency_80"):
                        value=metrics[family]["last"][metric]
                        if value is not None: records.append({"classes":classes,"neurons":neurons,"seed":seed,"family":family,"metric":metric,"value":value})
    output={"cells":{},"models":{}}
    for slug,cell in cells.items():
        output["cells"][slug]={"classes":cell["classes"],"neurons":cell["neurons"],"seeds":len(cell["runs"]),"converged":sum(row["converged"] for row in cell["runs"]),"accuracy":interval([row["accuracy"] for row in cell["runs"]]),"families":{}}
        for family in ("operation","class"):
            output["cells"][slug]["families"][family]={metric:interval([row["metrics"][family]["last"][metric] for row in cell["runs"] if row["metrics"][family]["last"][metric] is not None]) for metric in ("clean_axis_index","raw_axis_advantage","subspace_advantage","efficiency_80","efficiency_90")}
    rng=np.random.default_rng(20261017)
    for family in ("operation","class"):
        output["models"][family]={}
        for metric in ("clean_axis_index","raw_axis_advantage","efficiency_80"):
            rows=[row for row in records if row["family"]==family and row["metric"]==metric]
            estimate=coefficients(rows); boot=[]
            for _ in range(5000):
                sampled=rng.integers(0,8,size=8); selected=[]
                for seed in sampled: selected.extend(row for row in rows if row["seed"]==seed)
                boot.append(coefficients(selected))
            boot=np.array(boot); names=("intercept","log_classes","log_width","interaction")
            output["models"][family][metric]={name:{"estimate":float(estimate[i]),"ci95_low":float(np.quantile(boot[:,i],.025)),"ci95_high":float(np.quantile(boot[:,i],.975))} for i,name in enumerate(names)}
    (ROOT/"load_factorial_summary.json").write_text(json.dumps(output,indent=2),encoding="utf-8")

    fig,axes=plt.subplots(2,2,figsize=(10,8));
    for row,family in enumerate(("operation","class")):
        for column,(metric,title) in enumerate((("clean_axis_index","Clean Axis Index"),("raw_axis_advantage","Raw − rotation KL recovery"))):
            matrix=np.array([[output["cells"][f"c{classes}_o4_d2_w{neurons}"]["families"][family][metric]["mean"] for neurons in manifest["neurons"]] for classes in manifest["classes"]])
            axis=axes[row,column]; image=axis.imshow(matrix,aspect="auto",origin="lower",cmap="viridis")
            for i in range(3):
                for j in range(3): axis.text(j,i,f"{matrix[i,j]:.3f}",ha="center",va="center",color="white" if matrix[i,j]>.55*matrix.max() else "black")
            axis.set_xticks(range(3),manifest["neurons"]); axis.set_yticks(range(3),manifest["classes"]); axis.set_xlabel("neurons / layer"); axis.set_ylabel("classes"); axis.set_title(f"{family}: {title}"); fig.colorbar(image,ax=axis,fraction=.046)
    fig.suptitle("Class complexity × neuron width",fontweight="bold"); fig.tight_layout(); fig.savefig(ROOT/"load_factorial.png",dpi=200); plt.close(fig)


if __name__=="__main__": main()
