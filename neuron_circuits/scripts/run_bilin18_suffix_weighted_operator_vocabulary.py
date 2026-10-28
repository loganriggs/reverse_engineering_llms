#!/usr/bin/env python3
"""Learn local MLP operator modes under actual suffix-output sensitivity."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native
from run_bilin18_dynamic_mlp_jacobian_spectrum import exact_incoming_adjoints
from run_bilin18_early_operator_tensor_vocabulary import operator_modes, subspace_overlap
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, mlp_vjp
from run_bilin18_operator_vocabulary_causal_composition import reconstruct_state, select_sites
from run_bilin18_output_metric_whitening_redteam import residual_adjoints
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_suffix_weighted_operator_vocabulary.json")
OUTPUT = TENSOR_ROOT / "bilin18_suffix_weighted_operator_vocabulary.json"
REPORT = TENSOR_ROOT / "BILIN18_SUFFIX_WEIGHTED_OPERATOR_VOCABULARY.md"
FIGURE = TENSOR_ROOT / "bilin18_suffix_weighted_operator_vocabulary.png"
BASES = TENSOR_ROOT / "bilin18_suffix_weighted_operator_vocabulary.pt"


def incoming_trace(model, trace, position, axes, blocks):
    current = residual_adjoints(trace[-1]["output"], position, axes)
    result = {}
    for layer in range(len(model.blocks) - 1, min(blocks) - 1, -1):
        if layer in blocks:
            result[layer] = current[:, 0, position].detach()
        block = model.blocks[layer]
        updated = current + mlp_vjp(block, trace[layer]["pre"].float(), current)
        previous = trace[layer - 1]["output"].float(); x0 = trace[layer]["x0"].float()
        mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0
        routed = attention_vjp(block, mixed, trace[layer - 1]["value_bus"], updated)
        current = block.lambdas[0].float() * (updated + routed)
    return result


def task_probe_vectors(block, incoming, replicates, seed, normalize):
    incoming = incoming.repeat_interleave(replicates, dim=0)
    if normalize:
        incoming = incoming / incoming.norm(dim=1, keepdim=True).clamp_min(1e-30)
    dimension = incoming.shape[1]
    generator = torch.Generator(device=incoming.device).manual_seed(seed)
    q = torch.randn(len(incoming), dimension, generator=generator, device=incoming.device) / np.sqrt(dimension)
    down = block.mlp.Down.weight.float(); left = block.mlp.Left.weight.float(); right = block.mlp.Right.weight.float()
    output = incoming @ down
    return (output * (q @ left.T)) @ right + (output * (q @ right.T)) @ left


def reconstruct_with_modes(state, basis, modes, rank):
    mean = basis["mean"].to(state.device); vectors = basis["state_vectors"].to(state.device); rms = basis["state_rms"].to(state.device)
    whitened = ((state - mean) @ vectors) / rms
    selected = modes[:, :rank]
    recovered = (whitened @ selected) @ selected.T
    return mean + (recovered * rms) @ vectors.T


def response(block, pre, incoming, normalized_state):
    x = pre.float(); dimension = x.numel(); rms = x.square().mean().sqrt()
    left_weight = block.mlp.Left.weight.float(); right_weight = block.mlp.Right.weight.float(); down = block.mlp.Down.weight.float()
    left = left_weight @ normalized_state; right = right_weight @ normalized_state
    channel = incoming @ down
    core = (channel * right[None]) @ left_weight + (channel * left[None]) @ right_weight
    coupling = (core * x[None]).sum(1, keepdim=True)
    return core / rms - x[None] * coupling / (dimension * rms.pow(3))


def median_cosine(target, prediction):
    value = F.cosine_similarity(target, prediction, dim=1)
    return float(value.median())


def evaluate_context(block, pre, incoming, state_basis, task_modes, ranks):
    state = F.rms_norm(pre.float(), (pre.numel(),))
    exact = response(block, pre, incoming, state)
    result = {}
    for rank in ranks:
        for name, modes in task_modes.items():
            reconstructed = reconstruct_with_modes(state, state_basis, modes, rank)
            result[f"{name}_{rank}"] = median_cosine(exact, response(block, pre, incoming, reconstructed))
        for family in ("first", "pca"):
            reconstructed = reconstruct_state(state, state_basis, rank, family)
            result[f"frobenius_{rank}" if family == "first" else f"pca_{rank}"] = median_cosine(exact, response(block, pre, incoming, reconstructed))
    return result


def summarize(records, block):
    rows = [row for row in records if row["block"] == block]
    return {key: float(np.median([row["metrics"][key] for row in rows])) for key in rows[0]["metrics"]}


def main(config_path=CONFIG):
    requested = json.loads(Path(config_path).read_text())
    if "base_config" in requested:
        config = json.loads(Path(requested["base_config"]).read_text())
        config.update({key: value for key, value in requested.items() if key != "base_config"})
    else:
        config = requested
    artifact_stem = config.get("artifact_stem", "bilin18_suffix_weighted_operator_vocabulary")
    output_path = TENSOR_ROOT / f"{artifact_stem}.json"
    report_path = TENSOR_ROOT / f"{artifact_stem.upper()}.md"
    figure_path = TENSOR_ROOT / f"{artifact_stem}.png"
    bases_path = TENSOR_ROOT / f"{artifact_stem}.pt"
    core = torch.load(config["core_path"], map_location="cpu", weights_only=False)
    fisher = torch.load(config["fisher_axes_path"], map_location="cpu", weights_only=False)
    output_axes = core["output"][:, :config["rank"]].float().cuda() @ fisher["discovery"]["residual_dual"].float().cuda()
    state_bases = torch.load(config["state_bases"], map_location="cpu", weights_only=True)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r")

    incoming_rows = {block: [] for block in config["blocks"]}
    discovery = corpus[config["discovery_start"]:config["discovery_start"] + config["rows_per_split"]]
    for row_index, raw in enumerate(discovery):
        ids = torch.as_tensor(np.asarray(raw[:128], dtype=np.int64), device="cuda")[None]
        trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        for position in config["positions"]:
            values = incoming_trace(model, trace, position, output_axes, config["blocks"])
            for block in config["blocks"]: incoming_rows[block].append(values[block])
        print(f"suffix basis discovery {row_index + 1}/{len(discovery)}", flush=True)

    task_bases, stability, raw_unit = {}, {}, {}
    for block in config["blocks"]:
        incoming = torch.cat(incoming_rows[block])
        state = state_bases[str(block)]; vectors = state["state_vectors"].cuda(); rms = state["state_rms"].cuda()
        task_bases[str(block)] = {}
        for family, normalize in (("raw", False), ("unit", True)):
            first = task_probe_vectors(model.blocks[block], incoming, config["q_replicates"], config["seed"] + 10 * block, normalize)
            second = task_probe_vectors(model.blocks[block], incoming, config["q_replicates"], config["seed"] + 10 * block + 1, normalize)
            singular_a, modes_a = operator_modes(vectors, rms, first); singular_b, modes_b = operator_modes(vectors, rms, second)
            task_bases[str(block)][family] = modes_a.cpu()
            stability[f"{block}_{family}"] = subspace_overlap(modes_a[:, :64], modes_b[:, :64])
            task_bases[str(block)][f"{family}_second"] = modes_b.cpu()
            task_bases[str(block)][f"{family}_singular"] = singular_a.cpu()
        raw_unit[str(block)] = subspace_overlap(task_bases[str(block)]["raw"][:, :64].cuda(), task_bases[str(block)]["unit"][:, :64].cuda())
    torch.save(task_bases, bases_path)

    residual_records = []
    evaluation = corpus[config["evaluation_start"]:config["evaluation_start"] + config["rows_per_split"]]
    for row_index, raw in enumerate(evaluation):
        ids = torch.as_tensor(np.asarray(raw[:128], dtype=np.int64), device="cuda")[None]
        trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        for position in config["positions"]:
            values = incoming_trace(model, trace, position, output_axes, config["blocks"])
            for block in config["blocks"]:
                modes = {name: value.cuda() for name, value in task_bases[str(block)].items() if name in ("raw", "unit")}
                metrics = evaluate_context(model.blocks[block], trace[block]["pre"][0, position], values[block], state_bases[str(block)], modes, config["ranks"])
                residual_records.append({"row": row_index, "position": position, "block": block, "metrics": metrics})
        print(f"suffix basis evaluation {row_index + 1}/{len(evaluation)}", flush=True)

    causal = json.loads(Path(config["source_config"]).read_text()); old = json.loads(Path(config["source_results"]).read_text()); source = json.loads(Path(causal["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); positive = flags.sum(0).astype(np.float32); negative = len(flags)-positive
    contrasts = torch.as_tensor(flags/positive.clip(1)[None]-(1-flags)/negative.clip(1)[None],device="cuda",dtype=torch.float32)
    heldout_start = source["start_row"] + source["discovery_rows"]; semantic_records=[]
    for site_index,row in enumerate(select_sites(causal,old,config["sites_per_feature"])):
        feature=FEATURES.index(row["feature"]); contrast=contrasts[:,feature]; position=row["recipient_position"]
        raw=np.asarray(corpus[heldout_start+row["recipient_row"],:source["sequence_length"]],dtype=np.int64); ids=torch.as_tensor(raw,device="cuda")[None]
        trace=forward_layers_native(model,ids,len(model.blocks)-1); incoming=exact_incoming_adjoints(model,trace,position,contrast,first_block=min(config["blocks"]))
        for block in config["blocks"]:
            modes={name:value.cuda() for name,value in task_bases[str(block)].items() if name in ("raw","unit")}
            metrics=evaluate_context(model.blocks[block],trace[block]["pre"][0,position],incoming[block][:,0,position],state_bases[str(block)],modes,config["ranks"])
            semantic_records.append({"site":site_index,"feature":row["feature"],"block":block,"metrics":metrics})
    residual_summary={str(block):summarize(residual_records,block) for block in config["blocks"]}; semantic_summary={str(block):summarize(semantic_records,block) for block in config["blocks"]}
    hypotheses={
        "T0_independent_q_sketches_recover_task_subspace":{"supported":all(stability[f"{block}_unit"]>=.80 for block in config["blocks"])},
        "T1_task_rank64_reconstructs_residual_objectives":{"supported":all(residual_summary[str(block)]["unit_64"]>=.90 for block in config["blocks"])},
        "T2_task_rank64_beats_frobenius":{"supported":all(residual_summary[str(block)]["unit_64"]-residual_summary[str(block)]["frobenius_64"]>=.05 for block in config["blocks"])},
        "T3_task_rank64_beats_state_pca":{"supported":all(residual_summary[str(block)]["unit_64"]-residual_summary[str(block)]["pca_64"]>=.05 for block in config["blocks"])},
        "T4_task_rank64_transfers_to_semantics":{"supported":sum(semantic_summary[str(block)]["unit_64"]>semantic_summary[str(block)]["frobenius_64"] for block in config["blocks"])>=4},
        "T5_task_rank256_is_high_fidelity":{"supported":all(residual_summary[str(block)]["unit_256"]>=.95 and semantic_summary[str(block)]["unit_256"]>=.95 for block in config["blocks"])},
        "T6_adjoint_magnitude_changes_the_basis":{"supported":any(raw_unit[str(block)]<.90 for block in config["blocks"])},
    }
    output_path.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"stability":stability,"raw_unit_overlap":raw_unit,"residual_summary":residual_summary,"semantic_summary":semantic_summary,"residual_records":residual_records,"semantic_records":semantic_records},indent=2,allow_nan=False))
    lines=["# Suffix-weighted early operator vocabulary","","The exact bilinear operator tensor is weighted by real incoming adjoints obtained by transporting the stable residual-output subspace through the complete native suffix. Random tangent probes provide the second tensor leg. Bases are fit on disjoint FineWeb and evaluated on new residual objectives plus held-out human semantic objectives. Frobenius-operator modes and state PCA use the same state covariance and rank budgets.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["",f"![Suffix-weighted vocabulary]({figure_path.name})","","| Block | Unit stability | Raw/unit overlap | Residual unit64 / Frobenius / PCA | Residual unit256 | Semantic unit64 / Frobenius / PCA | Semantic unit256 |","|---:|---:|---:|---:|---:|---:|---:|"]
    for block in config["blocks"]:
        r=residual_summary[str(block)];s=semantic_summary[str(block)];lines.append(f"| {block} | {stability[f'{block}_unit']:.3f} | {raw_unit[str(block)]:.3f} | {r['unit_64']:.3f}/{r['frobenius_64']:.3f}/{r['pca_64']:.3f} | {r['unit_256']:.3f} | {s['unit_64']:.3f}/{s['frobenius_64']:.3f}/{s['pca_64']:.3f} | {s['unit_256']:.3f} |")
    lines += ["","## Guardrails","","This is a declared residual-output metric, not a universal operator ontology. The suffix-weighted basis uses native forward states and exact suffix adjoints on discovery data but no semantic labels. Transfer to the semantic panel tests reuse beyond its fitting objectives. Unit normalization asks for recurring directions; the raw variant allows high-gain contexts to dominate. Local response reconstruction is not yet a simultaneous multi-block causal replacement."]
    report_path.write_text("\n".join(lines)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout="constrained");blocks=config["blocks"]
    for name in ("unit_64","raw_64","frobenius_64","pca_64","unit_256"):
        axes[0].plot(blocks,[residual_summary[str(block)][name] for block in blocks],marker="o",label=name)
        axes[1].plot(blocks,[semantic_summary[str(block)][name] for block in blocks],marker="o",label=name)
    axes[0].set(xlabel="Block",ylabel="Median response cosine",title="Held-out residual-output objectives");axes[1].set(xlabel="Block",ylabel="Median response cosine",title="Held-out semantic objectives")
    for axis in axes:axis.grid(alpha=.2);axis.legend()
    fig.savefig(figure_path,dpi=180);plt.close(fig)
    print(json.dumps({"hypotheses":hypotheses,"stability":stability,"raw_unit_overlap":raw_unit,"residual_summary":residual_summary,"semantic_summary":semantic_summary},indent=2));print(report_path)


if __name__=="__main__":main(Path(sys.argv[1]) if len(sys.argv)>1 else CONFIG)
