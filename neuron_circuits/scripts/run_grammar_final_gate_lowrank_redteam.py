#!/usr/bin/env python3
"""Red-team the apparent low-dimensional dense code in final bilinear gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_address_router import collect, prototypes, apply_prototypes, score, select_panel
from run_grammar_final_gate_code_dimension import gate, pca, project, eligible_mask


CONFIG = Path("configs/grammar_final_gate_lowrank_redteam.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_final_gate_lowrank_redteam.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_FINAL_GATE_LOWRANK_REDTEAM.md")


def analyze(spec, config, device):
    root=Path(spec["root"]); manifest=load_manifest(root/"manifest.npz")
    model=load_model(root/f"rho_{config['rho']:g}/depth_4/checkpoint.pt",device)
    facts=np.asarray(manifest["train_facts"]);labels,selected,changed=labels_for_rho(manifest,"train",config["rho"])
    eligible=eligible_mask(model,manifest,facts,labels,config["evaluation_templates"],device)
    panel=select_panel(labels,selected,changed,eligible,facts,config,config["seed"]+spec["seed"])
    discovery={}
    for flag,status in ((False,"unselected"),(True,"selected")):
        idx=panel[flag]["discovery"]
        data,_,_=collect(model,encode_prompts(manifest,facts[idx],config["discovery_template"]),device)
        values=gate(model,data["premlp"]);mean,basis=pca(values)
        discovery[status]={"idx":idx,"gate":values,"write":data["write"],"mean":mean,"basis":basis}
    pooled_gate=torch.cat([discovery["unselected"]["gate"],discovery["selected"]["gate"]])
    pooled_mean,pooled_basis=pca(pooled_gate);down=model.blocks[3].down.weight.detach().cpu()
    output={}
    for flag,status in ((False,"unselected"),(True,"selected")):
        other="selected" if status=="unselected" else "unselected";eval_idx=panel[flag]["evaluation"]
        train_idx=discovery[status]["idx"]
        table=prototypes(discovery[status]["write"],facts[train_idx,2],labels[train_idx],True)
        importance=discovery[status]["gate"].var(0)*down.square().sum(0);coordinate_order=importance.argsort(descending=True)
        output[status]={}
        for template in config["evaluation_templates"]:
            data,_,_=collect(model,encode_prompts(manifest,facts[eval_idx],template),device)
            native=gate(model,data["premlp"]);oracle=apply_prototypes(table,facts[eval_idx,2],labels[eval_idx],True)
            methods={"native":score(native@down.T,data["write"],data["after"],data["zero_logits"],data["logits"],labels[eval_idx],oracle,model,config["minimum_fisher_effect_norm"])}
            for rank in config["ranks"]:
                own=discovery[status]; opposite=discovery[other]
                own_state=own["mean"].expand_as(native) if rank==0 else project(native,own["mean"],own["basis"][:rank])
                pooled_state=pooled_mean.expand_as(native) if rank==0 else project(native,pooled_mean,pooled_basis[:rank])
                opposite_state=own["mean"].expand_as(native) if rank==0 else project(native,own["mean"],opposite["basis"][:rank])
                coordinate=own["mean"].expand_as(native).clone()
                if rank: coordinate[:,coordinate_order[:rank]]=native[:,coordinate_order[:rank]]
                candidates={f"own_pca_{rank}":own_state,f"pooled_pca_{rank}":pooled_state,
                            f"opposite_basis_{rank}":opposite_state,f"coordinate_{rank}":coordinate}
                for name,candidate in candidates.items():
                    methods[name]=score(candidate@down.T,data["write"],data["after"],data["zero_logits"],data["logits"],labels[eval_idx],oracle,model,config["minimum_fisher_effect_norm"])
            bottom=project(native,discovery[status]["mean"],discovery[status]["basis"][-8:])
            methods["bottom_8"]=score(bottom@down.T,data["write"],data["after"],data["zero_logits"],data["logits"],labels[eval_idx],oracle,model,config["minimum_fisher_effect_norm"])
            generator=torch.Generator().manual_seed(config["seed"]+spec["seed"]*100+template+int(flag))
            random=[]
            for _ in range(config["random_subspaces"]):
                basis=torch.linalg.qr(torch.randn(192,8,generator=generator)).Q.T
                candidate=project(native,discovery[status]["mean"],basis)
                row=score(candidate@down.T,data["write"],data["after"],data["zero_logits"],data["logits"],labels[eval_idx],oracle,model,config["minimum_fisher_effect_norm"])
                random.append(row["median_effect_fidelity"])
            methods["random_8"]={"median_fidelity_mean":float(np.mean(random)),"median_fidelity_p95":float(np.quantile(random,.95)),"replicates":len(random)}
            output[status][str(template)]=methods
    return {"seed":spec["seed"],"results":output}


def main():
    config=json.loads(CONFIG.read_text());device="cuda" if torch.cuda.is_available() else "cpu"
    runs=[analyze(spec,config,device) for spec in config["models"]];first,held=map(str,config["evaluation_templates"])
    def val(run,status,template,method,key="median_effect_fidelity"):return run["results"][status][template][method][key]
    m0=all(val(run,status,template,"native")>=.9999 for run in runs for status in ("selected","unselected") for template in (first,held))
    m1=all(val(run,status,held,"own_pca_0")<.5 for run in runs for status in ("selected","unselected"))
    m2=all(val(run,"selected",held,"own_pca_8")>=.85 for run in runs)
    m3=all(val(run,status,held,"own_pca_8")>=run["results"][status][held]["random_8"]["median_fidelity_p95"]+.20 for run in runs for status in ("selected","unselected"))
    m4=all(abs(val(run,status,held,"own_pca_8")-val(run,status,held,"opposite_basis_8"))<=.10 for run in runs for status in ("selected","unselected"))
    m5=all(val(run,status,held,"own_pca_8")-val(run,status,held,"coordinate_8")>=.30 for run in runs for status in ("selected","unselected"))
    m6=all(val(run,"selected",held,"own_pca_8","write_r2")>=.70 for run in runs)
    m7=all(val(run,status,held,"bottom_8")<.30 for run in runs for status in ("selected","unselected"))
    m8=all(abs(val(run,"selected",first,"own_pca_8")-val(run,"selected",held,"own_pca_8"))<=.10 for run in runs)
    m9=all(row["effectful_examples"]>=64 for run in runs for status in ("selected","unselected") for template in (first,held) for name,row in run["results"][status][template].items() if name!="random_8")
    hypotheses={name:{"supported":bool(value)} for name,value in zip(config["hypotheses"],(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9))}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"runs":runs},indent=2,allow_nan=False))
    lines=["# Low-rank final-gate red team","","All candidate post-product gates are passed through the original down projection and exact suffix. Rank zero is the discovery mean. Opposite-status bases use the target-status mean, so only orientation transfers.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Seed | Status | mean | PCA r1/r2/r4/r8/r16 | opposite8 | coord8 | bottom8 | random8 mean/p95 |","|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        for status in ("unselected","selected"):
            random=run["results"][status][held]["random_8"]
            ranks="/".join(f"{val(run,status,held,f'own_pca_{r}'):.3f}" for r in (1,2,4,8,16))
            lines.append(f"| {run['seed']} | {status} | {val(run,status,held,'own_pca_0'):.3f} | {ranks} | {val(run,status,held,'opposite_basis_8'):.3f} | {val(run,status,held,'coordinate_8'):.3f} | {val(run,status,held,'bottom_8'):.3f} | {random['median_fidelity_mean']:.3f}/{random['median_fidelity_p95']:.3f} |")
    lines += ["","A low-dimensional dense gate code is evidence against treating the 192 native channels as 192 independent computational primitives. It does not make the PCA directions semantic or identify the upstream fact lookup."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",OUTPUT,REPORT,flush=True)


if __name__=="__main__":main()
