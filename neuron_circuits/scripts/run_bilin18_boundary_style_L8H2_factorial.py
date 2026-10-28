#!/usr/bin/env python3
"""Factor the native L8H2 cue edge into QK/current-value/persistent-bus roles."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_factorial import FACTORS, WORLDS, shapley, summarize_scalar, summarize_vector
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_boundary_style_L8H2_factorial.json")
OUTPUT = TENSOR_ROOT / "bilin18_boundary_style_L8H2_factorial.json"
REPORT = TENSOR_ROOT / "BILIN18_BOUNDARY_STYLE_L8H2_FACTORIAL.md"
FIGURE = TENSOR_ROOT / "bilin18_boundary_style_L8H2_factorial.png"


@torch.inference_mode()
def collect(model, raw, records, contrasts, config):
    layer, head, destination = 8, 2, 127
    dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads
    projection = model.blocks[layer].attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)[head]
    local_targets, output_targets, random_targets = [], [], []
    local_parts_all = {name: [] for name in FACTORS}; output_parts_all = {name: [] for name in FACTORS}; random_parts_all = {name: [] for name in FACTORS}
    local_error = output_error = 0.0
    for start in range(0, len(records), config["batch_size"]):
        batch = records[start:start + config["batch_size"]]; batch_size = len(batch); paired = torch.arange(batch_size, device="cuda")
        original = np.stack([raw[row[0], :128].copy() for row in batch]); swapped = original.copy()
        for index, row in enumerate(batch): swapped[index, row[3]] = row[5]
        cue = torch.as_tensor([row[3] for row in batch], device="cuda"); boundary = torch.as_tensor([row[1] for row in batch], device="cuda")
        ids = torch.as_tensor(np.concatenate([original, swapped]), device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for block_index, block in enumerate(model.blocks):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            if block_index < layer:
                pre = mixed + details["write"]; state = F.rms_norm(pre, (dimension,))
                x = pre + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
                bus = details["value_bus"]; continue
            factors = {
                "qk1": details["score1"][torch.cat([paired, paired + batch_size]), head,
                                                torch.cat([boundary, boundary]), torch.cat([cue, cue])].float().reshape(2, batch_size),
                "qk2": details["score2"][torch.cat([paired, paired + batch_size]), head,
                                                torch.cat([boundary, boundary]), torch.cat([cue, cue])].float().reshape(2, batch_size),
                "current_value": details["current_value"][torch.cat([paired, paired + batch_size]), torch.cat([cue, cue]), head].float().reshape(2, batch_size, head_dim),
                "persistent_bus": details["value_bus"][torch.cat([paired, paired + batch_size]), torch.cat([cue, cue]), head].float().reshape(2, batch_size, head_dim),
            }
            writes = {}
            for world in WORLDS:
                value = ((1 - block.attn.lamb.float()) * factors["current_value"][world[2]]
                         + block.attn.lamb.float() * factors["persistent_bus"][world[3]])
                writes[world] = (factors["qk1"][world[0]] * factors["qk2"][world[1]])[:, None] * value @ projection
            local_target = writes[(1,1,1,1)] - writes[(0,0,0,0)]; local_parts = shapley(writes)
            local_error = max(local_error, float((sum(local_parts.values()) - local_target).norm() / local_target.norm().clamp_min(1e-30)))
            original_attention = details["write"][:batch_size].float(); original_edge = writes[(0,0,0,0)]
            attention = []
            for world in WORLDS:
                modified = original_attention.clone(); modified[paired, boundary] += writes[world] - original_edge; attention.append(modified)
            attention = torch.cat(attention); repeats = len(WORLDS)
            mixed0 = mixed[:batch_size].float().repeat(repeats, 1, 1); x00 = x0[:batch_size].float().repeat(repeats, 1, 1)
            bus0 = details["value_bus"][:batch_size].float().repeat(repeats, 1, 1, 1)
            state = F.rms_norm(mixed0 + attention, (dimension,)); suffix_x = mixed0 + attention + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias; suffix_bus = bus0
            for suffix in model.blocks[layer + 1:]: suffix_x, suffix_bus = suffix(suffix_x, suffix_bus, x00)
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(suffix_x, (dimension,))[:, destination]).float() / 30)
            scores = (logits @ contrasts).reshape(repeats, batch_size, -1); score_worlds = {world: scores[index] for index, world in enumerate(WORLDS)}
            output_target = score_worlds[(1,1,1,1)] - score_worlds[(0,0,0,0)]; output_parts = shapley(score_worlds)
            output_error = max(output_error, float((sum(output_parts.values()) - output_target).norm() / output_target.norm().clamp_min(1e-30)))
            direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)
            local_targets.append(local_target.cpu()); output_targets.append((direction * output_target[:,0]).cpu()); random_targets.append((direction[:,None] * output_target[:,1:]).cpu())
            for name in FACTORS:
                local_parts_all[name].append(local_parts[name].cpu()); output_parts_all[name].append((direction * output_parts[name][:,0]).cpu()); random_parts_all[name].append((direction[:,None] * output_parts[name][:,1:]).cpu())
            break
    local_target = torch.cat(local_targets); output_target = torch.cat(output_targets); random_target = torch.cat(random_targets)
    local_parts = {name: torch.cat(value) for name, value in local_parts_all.items()}; output_parts = {name: torch.cat(value) for name, value in output_parts_all.items()}; random_parts = {name: torch.cat(value) for name, value in random_parts_all.items()}
    for pieces in (local_parts, output_parts, random_parts):
        pieces["qk_combined"] = pieces["qk1"] + pieces["qk2"]; pieces["value_combined"] = pieces["current_value"] + pieces["persistent_bus"]
    total_mean = float(output_target.mean()); random_means = random_target.mean(0)
    summary = {"examples": len(output_target), "local_recomposition_error": local_error, "output_recomposition_error": output_error,
               "lambda": float(model.blocks[layer].attn.lamb),
               "total": {"win": float((output_target > 0).float().mean()), "mean": total_mean,
                         "random_percentile": float((random_means < total_mean).float().mean())},
               "local": summarize_vector(local_target, local_parts), "output": summarize_scalar(output_target, output_parts)}
    for name in ("qk_combined", "value_combined"):
        random_mean = random_parts[name].mean(0); summary["output"][name]["random_percentile"] = float((random_mean < output_parts[name].mean()).float().mean())
    return summary


def main():
    config = json.loads(CONFIG.read_text()); base = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    pairs = [(tokenizer.encode(us, add_special_tokens=False)[0], tokenizer.encode(uk, add_special_tokens=False)[0]) for us, uk in base["spelling_pairs"]]
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results = []
    for fold in range(config["folds"]):
        targets = set(range(fold * config["target_pairs_per_fold"], (fold + 1) * config["target_pairs_per_fold"])); target_pairs = [p for i,p in enumerate(pairs) if i in targets]; cues = [p for i,p in enumerate(pairs) if i not in targets]
        contrasts = make_contrasts(model.lm_head.weight.shape[0], target_pairs, config["random_contrasts"], config["seed"] + fold, "cuda")
        for split, (begin,end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[begin:end,:128], dtype=np.int64); records = select_records(raw, boundary_flags, cues, counts, 127, config["examples_per_label_per_split"], config["seed"] + fold*10 + split)
            print("L8H2 factorial", fold, split, len(records), flush=True); results.append({"fold":fold,"split":split,**collect(model,raw,records,contrasts,config)})
    p0 = max(max(r["local_recomposition_error"],r["output_recomposition_error"]) for r in results)<=1e-5
    p1 = all(r["total"]["win"]>=.75 for r in results); p2=all(r["output"]["value_combined"]["mean"]>r["output"]["qk_combined"]["mean"] for r in results)
    p3=all(r["output"]["persistent_bus"]["mean"]*r["output"]["current_value"]["mean"]<0 and min(abs(r["output"]["persistent_bus"]["mean"]),abs(r["output"]["current_value"]["mean"]))>=abs(r["total"]["mean"]) for r in results)
    p4=all(r["output"]["qk_combined"]["mean"]<0 for r in results)
    p5cells=sum(r["total"]["random_percentile"]>=.95 and r["output"]["value_combined"]["random_percentile"]>=.95 for r in results); p5=p5cells>=5
    hypotheses={"P0_factorial_recomposes":{"supported":p0},"P1_hybrid_edge_preserves_native_transport":{"supported":p1},"P2_value_change_dominates_QK_change":{"supported":p2},"P3_value_is_a_cancellation_code":{"supported":p3},"P4_QK_change_opposes_payload":{"supported":p4},"P5_declared_contrast_beats_random":{"supported":p5,"cells":p5cells},"P6_lexical_difference_payload_is_supported":{"supported":all((p0,p1,p2,p3,p4,p5))}}
    for row in hypotheses.values(): row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results},indent=2,allow_nan=False))
    lines=["# Tensor-factor ontology of the L8H2 spelling-cue edge","","The validated upstream edge is factored into QK1, QK2, the contextual Layer-8 value, and the persistent Layer-0 value bus. Every one of the sixteen hybrid edges is inserted into the original context and propagated through all downstream nonlinearities. Vector-valued Shapley values exactly allocate the finite endpoint difference.","","## Locked results",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Fold/split | total win/mean | QK mean | current-V mean | bus mean | combined-V mean |","|---|---:|---:|---:|---:|---:|"]
    for r in results: lines.append(f"| {r['fold']}/{r['split']} | {r['total']['win']:.3f}/{r['total']['mean']:+.4g} | {r['output']['qk_combined']['mean']:+.4g} | {r['output']['current_value']['mean']:+.4g} | {r['output']['persistent_bus']['mean']:+.4g} | {r['output']['value_combined']['mean']:+.4g} |")
    lines += ["","## Interpretation boundary","","Layer 8 uses `mixed_value = -3 * current_value + 4 * persistent_bus`. Large opposite Shapley terms therefore indicate a native difference code, not two independent semantic features. The persistent bus supplies a stable lexical reference while the current value supplies its contextualized counterpart. The smaller residual passed by their cancellation is the edge payload relevant to the spelling intervention."]
    REPORT.write_text("\n".join(lines)+"\n")
    x=np.arange(len(results)); labels=[f"f{r['fold']}s{r['split']}" for r in results]; fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained")
    for offset,name,label in ((-.3,"qk_combined","QK"),(-.1,"current_value","current V"),(.1,"persistent_bus","persistent bus"),(.3,"value_combined","combined V")): ax.bar(x+offset,[r["output"][name]["mean"] for r in results],.19,label=label)
    ax.axhline(0,color="black",linewidth=.8); ax.set(xticks=x,xticklabels=labels,ylabel="Directed spelling-contrast shift",title="What does L8H2 transport from the cue?"); ax.legend(); ax.grid(alpha=.2); fig.savefig(FIGURE,dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses":hypotheses,"results":results},indent=2)); print(REPORT)


if __name__=="__main__": main()
