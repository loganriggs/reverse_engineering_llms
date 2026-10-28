#!/usr/bin/env python3
"""Distinguish semantic OV modes from generic compression of each head operator."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_lexical_generalization import token_pairs
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_semantic_construction_head_programs import suffix, trace_to_site
from run_bilin18_semantic_payload_cross_basis import target_matrices
from run_bilin18_semantic_payload_svd import approximate_current, ov_modes
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_semantic_payload_family_specificity.json")
OUTPUT = TENSOR_ROOT / "bilin18_semantic_payload_family_specificity.json"
REPORT = TENSOR_ROOT / "BILIN18_SEMANTIC_PAYLOAD_FAMILY_SPECIFICITY.md"
FIGURE = TENSOR_ROOT / "bilin18_semantic_payload_family_specificity.png"


def metric(target, value):
    target, value = target.double(), value.double(); energy = target.square().sum().clamp_min(1e-30)
    return {"projection_share": float((target * value).sum() / energy), "energy_fidelity": float(1 - (target - value).square().sum() / energy)}


@torch.inference_mode()
def collect(model, raw, records, contrast, layer, program, rank, modes, matrices, projection, coefficients, batch_size):
    current_coefficient, bus_coefficient = coefficients
    state_targets, state_values, output_targets, output_values, native_targets = [], [], [], [], []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]; size = len(batch); original = np.stack([raw[row[0], :128].copy() for row in batch]); changed = original.copy()
        for index, row in enumerate(batch): changed[index, row[3]] = row[5]
        ids = torch.as_tensor(np.concatenate((original, changed)), device="cuda"); trace, native_logits = trace_to_site(model, ids, layer); details = trace["details"]
        direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)
        base_state = trace["output"][:size]; carried = trace["mixed"][size:,127] - trace["mixed"][:size,127]; mlp = trace["mlp"][size:,127] - trace["mlp"][:size,127]; no_attention = base_state.clone(); no_attention[:,127] = base_state[:,127] + carried + mlp
        p0 = details["pattern"][:size,program,127]; p1 = details["pattern"][size:,program,127]; current0 = details["current_value"][:size,:,program].permute(0,2,1,3); bus0 = details["value_bus"][:size,:,program].permute(0,2,1,3); bus1 = details["value_bus"][size:,:,program].permute(0,2,1,3); base_value = current_coefficient*current0 + bus_coefficient*bus0; delta_bus = bus_coefficient*(bus1-bus0)
        def write(pattern,value): return torch.einsum("bhs,bhsd,hdk->bk",pattern,value,projection)
        base_write=write(p0,base_value); route=write(p1,base_value)-base_write; bus=write(p1,delta_bus)
        normalized=F.rms_norm(trace["mixed"],(model.config.n_embd,)); delta=normalized[size:]-normalized[:size]
        exact_current=torch.einsum("bsd,hdk->bhsk",delta,matrices); exact=route+torch.einsum("bhs,bhsd->bd",p1,exact_current)+bus
        approx_current=approximate_current(delta,modes,program,rank,random=False); approx=route+torch.einsum("bhs,bhsd->bd",p1,approx_current)+bus
        states=[no_attention]; full=no_attention.clone(); full[:,127]+=exact; states.append(full); compressed=no_attention.clone(); compressed[:,127]+=approx; states.append(compressed)
        logits=suffix(model,layer,torch.cat(states),torch.cat([trace["x0"][:size]]*3),torch.cat([trace["bus"][:size]]*3)).reshape(3,size,-1); base_score=logits[0]@contrast
        output_targets.append((direction*((logits[1]@contrast)-base_score)).cpu()); output_values.append((direction*((logits[2]@contrast)-base_score)).cpu()); native_targets.append((direction*((native_logits[size:]@contrast)-(native_logits[:size]@contrast))).cpu()); state_targets.append(exact.cpu()); state_values.append(approx.cpu())
    state_target=torch.cat(state_targets).flatten(); state_value=torch.cat(state_values).flatten(); output_target=torch.cat(output_targets); output_value=torch.cat(output_values); native_target=torch.cat(native_targets)
    return {"state":metric(state_target,state_value),"output":metric(output_target,output_value),"branch_vs_native":metric(native_target,output_target),"branch_energy":float(output_target.double().square().mean())}


def main():
    config=json.loads(CONFIG.read_text()); cross_config=json.loads(Path(config["source_config"]).read_text()); svd_config=json.loads(Path(cross_config["source_config"]).read_text()); operand_config=json.loads(Path(svd_config["source_config"]).read_text()); head_red_config=json.loads(Path(operand_config["source_config"]).read_text()); head_config=json.loads(Path(head_red_config["source_config"]).read_text()); payload=json.loads(Path(config["payload_results"]).read_text()); heads=json.loads(Path(config["head_results"]).read_text())
    construction_config=json.loads(Path(head_config["source_config"]).read_text()); carrier_config=json.loads(Path(construction_config["source_config"]).read_text()); base=json.loads(Path(carrier_config["source_config"]).read_text()); family_source=json.loads(Path(carrier_config["family_config"]).read_text())
    tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True); corpus=np.load(base["corpus"],mmap_mode="r"); counts=np.bincount(np.asarray(corpus).reshape(-1),minlength=tokenizer.vocab_size); flags=np.stack([vocabulary_features(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)]); boundary_flags=flags[:,FEATURES.index("newline")].astype(bool)|flags[:,FEATURES.index("period")].astype(bool)
    family_pairs={"spelling":token_pairs(tokenizer,base["spelling_pairs"])}; family_pairs.update({name:[tuple(pair) for pair in pairs] for name,pairs in family_source["families"].items()}); families=list(head_config["sites"])
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval(); program_specs={}
    for family in families:
        layer=head_config["sites"][family]; program=heads["summary"][family]["head_order"][:heads["summary"][family]["discovery_k90"]]; rank=payload["summary"][family]["k90"]; modes,_=ov_modes(model,layer,program,config["seed"]); matrices,projection,current,bus=target_matrices(model,layer,program); program_specs[family]=(layer,program,rank,modes,matrices,projection,current,bus)
    results=[]
    for edit_index,edit_family in enumerate(families):
        pairs=family_pairs[edit_family]; midpoint=len(pairs)//2; folds=((pairs[:midpoint],pairs[midpoint:]),(pairs[midpoint:],pairs[:midpoint]))
        for fold,(targets,cues) in enumerate(folds):
            contrast=make_contrasts(model.lm_head.weight.shape[0],targets,0,head_config["seed"]+100*edit_index+fold,"cuda")[:,0]
            for split,(begin,end) in enumerate(base["splits"]):
                raw=np.asarray(corpus[begin:end,:128],dtype=np.int64); records=select_records(raw,boundary_flags,cues,counts,127,head_config["examples_per_label_per_split"],head_config["seed"]+1000*edit_index+10*fold+split)
                for program_family,spec in program_specs.items():
                    layer,program,rank,modes,matrices,projection,current,bus=spec; row=collect(model,raw,records,contrast,layer,program,rank,modes,matrices,projection,(current,bus),head_config["batch_size"]); row.update({"program_family":program_family,"edit_family":edit_family,"fold":fold,"split":split}); results.append(row)
                print(f"family specificity edit={edit_family} fold={fold} split={split}",flush=True)
    matrices_out={metric_name:{program:{edit:float(np.mean([row[metric_name]["energy_fidelity"] for row in results if row["program_family"]==program and row["edit_family"]==edit])) for edit in families} for program in families} for metric_name in ("state","output")}
    branch={program:{edit:float(np.mean([row["branch_vs_native"]["projection_share"] for row in results if row["program_family"]==program and row["edit_family"]==edit])) for edit in families} for program in families}
    state_margin={p:matrices_out["state"][p][p]-max(matrices_out["state"][p][e] for e in families if e!=p) for p in families}; output_margin={p:matrices_out["output"][p][p]-max(matrices_out["output"][p][e] for e in families if e!=p) for p in families}; branch_margin={p:branch[p][p]-max(branch[p][e] for e in families if e!=p) for p in families}; non_target={p:float(np.median([matrices_out["state"][p][e] for e in families if e!=p])) for p in families}
    state_flat=[matrices_out["state"][p][e] for p in families for e in families]; output_flat=[matrices_out["output"][p][e] for p in families for e in families]; correlation=float(np.corrcoef(state_flat,output_flat)[0,1])
    hypotheses={
      "G0_semantic_edits_prefer_their_own_payload_modes":{"supported":sum(v>=.20 for v in state_margin.values())>=3,"observed":state_margin},
      "G1_semantic_output_reconstruction_is_family_specific":{"supported":sum(v>=.20 for v in output_margin.values())>=3,"observed":output_margin},
      "G2_payload_modes_are_generic_operator_compressors":{"supported":sum(v>=.80 for v in non_target.values())>=3,"observed":non_target},
      "G3_head_invocation_is_family_selective":{"supported":sum(v>=.10 for v in branch_margin.values())>=3,"observed":branch_margin},
      "G4_state_and_output_specificity_agree":{"supported":correlation>=.60,"observed":correlation},
    }
    for row in hypotheses.values():row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"state_fidelity":matrices_out["state"],"output_fidelity":matrices_out["output"],"branch_projection":branch,"results":results},indent=2,allow_nan=False))
    lines=["# Semantic-family specificity of weight-derived payload modes","","Each selected head program and its previously frozen k90 OV basis are held fixed while the cue edit family is varied among spelling, sentiment, gender, and tense. State fidelity tests whether the basis generically compresses that head's attention change; output fidelity tests the declared edit family's semantic contrast; branch projection measures how much the fixed program contributes to the complete native semantic edit.","","## Locked decisions",""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{row['observed']}`" for name,row in hypotheses.items())
    for metric_name,title in (("state","State fidelity"),("output","Semantic-output fidelity")):
        lines += ["",f"## {title}","","| Program \\ edit | "+" | ".join(families)+" |","|---|"+"---:|"*len(families)]
        for program in families: lines.append("| "+program+" | "+" | ".join(f"{matrices_out[metric_name][program][edit]:.3f}" for edit in families)+" |")
    lines += ["","## Branch projection onto complete native semantic effect","","| Program \\ edit | "+" | ".join(families)+" |","|---|"+"---:|"*len(families)]
    for program in families:lines.append("| "+program+" | "+" | ".join(f"{branch[program][edit]:.3f}" for edit in families)+" |")
    lines += ["","## Interpretation boundary","","If a head's top OV modes compress every edit equally, they are generic operator coordinates even if one supervised invocation is semantic. Diagonal state specificity instead means that the target edit preferentially occupies those weight modes. Branch projection is a separate gate: an accurately compressed head can remain behaviorally irrelevant for a given edit."]
    REPORT.write_text("\n".join(lines)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for axis,(metric_name,title) in zip(axes,(("state","State fidelity"),("output","Semantic-output fidelity"))):
        array=np.asarray([[matrices_out[metric_name][p][e] for e in families] for p in families]); image=axis.imshow(array,vmin=min(0,float(array.min())),vmax=1,cmap="viridis"); axis.set(xticks=np.arange(4),xticklabels=families,yticks=np.arange(4),yticklabels=families,xlabel="Edit family",ylabel="Frozen head program",title=title)
        for i in range(4):
            for j in range(4):axis.text(j,i,f"{array[i,j]:.2f}",ha="center",va="center",color="white" if array[i,j]<.5 else "black")
    fig.savefig(FIGURE,dpi=180);plt.close(fig);print(json.dumps({"hypotheses":hypotheses,"state":matrices_out["state"],"output":matrices_out["output"]},indent=2));print(REPORT)


if __name__=="__main__":main()
