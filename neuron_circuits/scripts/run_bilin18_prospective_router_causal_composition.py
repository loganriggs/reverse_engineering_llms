#!/usr/bin/env python3
"""Causally compose prospective residual-plus-attention operator states."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, selected_delta
from run_bilin12_layer01_frozen_taylor_behavior import logits_for_deltas
from run_bilin18_dynamic_mlp_jacobian_spectrum import exact_incoming_adjoints
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import semantic_gradient
from run_bilin18_layer1_suffix_bidegree_pilot import metrics
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, final_adjoint, mlp_vjp
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_operator_vocabulary_causal_composition import (
    approximate_mlp_vjp,
    select_sites,
    summarize_program,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_prospective_router_causal_composition.json")
OUTPUT = TENSOR_ROOT / "bilin18_prospective_router_causal_composition.json"
REPORT = TENSOR_ROOT / "BILIN18_PROSPECTIVE_ROUTER_CAUSAL_COMPOSITION.md"
FIGURE = TENSOR_ROOT / "bilin18_prospective_router_causal_composition.png"


def reconstruct(vector, mean, basis, rank):
    centered = vector - mean.to(vector.device)
    selected = basis[:, :rank].to(vector.device)
    return mean.to(vector.device) + (centered @ selected) @ selected.T


def routed_state(model, trace, layer, position, projections, rank):
    block = model.blocks[layer]
    previous = trace[layer]["x0"] if layer == 0 else trace[layer - 1]["output"]
    mixed = block.lambdas[0].float() * previous.float() + block.lambdas[1].float() * trace[layer]["x0"].float()
    attention = trace[layer]["pre"].float() - mixed
    row = projections[str(layer)]
    mixed_reconstructed = reconstruct(
        mixed[0, position], row["mixed_mean"], row["mixed_vectors"], rank,
    )
    attention_reconstructed = reconstruct(
        attention[0, position], row["attention_mean"], row["attention_vectors"], rank,
    )
    return F.rms_norm(mixed_reconstructed + attention_reconstructed, (mixed.shape[-1],))


def program(model, trace, position, contrast, source_layer, projections, config):
    specs = [("exact", None)] + [(f"direct_{rank}", rank) for rank in config["ranks"]]
    current = final_adjoint(model, trace[-1]["output"], position, contrast).repeat(len(specs), 1, 1, 1)
    for layer in range(len(model.blocks) - 1, source_layer, -1):
        block = model.blocks[layer]
        responses = mlp_vjp(block, trace[layer]["pre"].float(), current)
        if layer in config["approximated_blocks"]:
            pre = trace[layer]["pre"][0, position].float()
            for row, (_, rank) in enumerate(specs):
                if rank is None: continue
                state = routed_state(model, trace, layer, position, projections, rank)
                responses[row, 0, position] = approximate_mlp_vjp(
                    block, pre, current[row, 0, position], state,
                )
        updated = current + responses
        previous = trace[layer - 1]["output"].float()
        mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * trace[layer]["x0"].float()
        routed = attention_vjp(block, mixed, trace[layer - 1]["value_bus"], updated)
        current = block.lambdas[0].float() * (updated + routed)
    down = model.blocks[source_layer].mlp.Down.weight.float()
    return {name: current[row, 0, position] @ down for row, (name, _) in enumerate(specs)}


def main():
    config = json.loads(CONFIG.read_text())
    causal = json.loads(Path(config["source_config"]).read_text())
    old = json.loads(Path(config["source_results"]).read_text())
    source = json.loads(Path(causal["source_config"]).read_text())
    projections = torch.load(config["projections"], map_location="cpu", weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive
    contrasts = torch.as_tensor(
        flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None],
        device="cuda", dtype=torch.float32,
    )
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r")
    heldout_start = source["start_row"] + source["discovery_rows"]
    sites = select_sites(causal, old, config["sites_per_feature"])
    records, local = [], []
    maximum_state_error = 0.0
    for site_index, row in enumerate(sites):
        feature = FEATURES.index(row["feature"]); contrast = contrasts[:, feature]
        position = row["recipient_position"]
        donor_raw = np.asarray(corpus[heldout_start + row["donor_row"], :source["sequence_length"]], dtype=np.int64)
        recipient_raw = np.asarray(corpus[heldout_start + row["recipient_row"], :source["sequence_length"]], dtype=np.int64)
        donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]
        recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        donor_trace = forward_layers_native(model, donor_ids, max(config["source_layers"]))
        trace = forward_layers_native(model, recipient_ids, len(model.blocks) - 1)
        incoming = exact_incoming_adjoints(
            model, trace, position, contrast, first_block=min(config["approximated_blocks"]),
        )
        for block_index in config["approximated_blocks"]:
            pre = trace[block_index]["pre"][0, position].float()
            native_state = F.rms_norm(pre, (pre.numel(),))
            previous = trace[block_index]["x0"] if block_index == 0 else trace[block_index - 1]["output"]
            block = model.blocks[block_index]
            mixed = block.lambdas[0].float() * previous.float() + block.lambdas[1].float() * trace[block_index]["x0"].float()
            exact_attention = trace[block_index]["pre"].float() - mixed
            exact_composed = F.rms_norm(mixed[0, position] + exact_attention[0, position], (pre.numel(),))
            maximum_state_error = max(maximum_state_error, float((native_state - exact_composed).abs().max()))
            native = mlp_vjp(block, trace[block_index]["pre"].float(), incoming[block_index])[0, 0, position]
            methods = {}
            for rank in config["ranks"]:
                state = routed_state(model, trace, block_index, position, projections, rank)
                predicted = approximate_mlp_vjp(
                    block, pre, incoming[block_index][0, 0, position], state,
                )
                methods[f"direct_{rank}"] = float(F.cosine_similarity(predicted, native, dim=0))
            local.append({"site": site_index, "feature": row["feature"], "block": block_index, "cosines": methods})
        for source_layer in config["source_layers"]:
            exact = semantic_gradient(model, trace, source_layer, position, contrast)
            predictions = program(model, trace, position, contrast, source_layer, projections, config)
            predictions["direct_exact"] = exact
            delta = product_at(model, donor_trace, source_layer, row["donor_position"]) - product_at(model, trace, source_layer, position)
            patches = {"base": torch.zeros_like(delta)}
            for name, prediction in predictions.items():
                support = (prediction * delta).topk(config["support_rank"]).indices
                patches[name] = selected_delta(delta, support)
            logits = logits_for_deltas(model, source_layer, trace[source_layer], patches, position)
            base_score = class_score(logits["base"], contrast)
            finite = {
                name: class_score(value, contrast) - base_score
                for name, value in logits.items() if name != "base"
            }
            records.append({
                "site": site_index, "feature": row["feature"], "source_layer": source_layer,
                "metrics": {name: metrics(value, exact, delta, config["support_rank"]) for name, value in predictions.items()},
                "finite_shifts": finite,
            })
        print(f"prospective causal {site_index + 1}/{len(sites)} {row['feature']}", flush=True)
    local_summary = {}
    for block in config["approximated_blocks"]:
        rows = [row for row in local if row["block"] == block]
        local_summary[str(block)] = {
            name: float(np.median([row["cosines"][name] for row in rows]))
            for name in rows[0]["cosines"]
        }
    methods = list(records[0]["metrics"])
    program_summary = {
        str(layer): {
            method: summarize_program(records, layer, method) for method in methods
        }
        for layer in config["source_layers"]
    }
    hypotheses = {
        "C0_direct_formula_matches_native_state": {"supported": maximum_state_error <= 1e-5},
        "C1_direct64_reconstructs_semantic_local_response": {"supported": all(local_summary[str(block)]["direct_64"] >= .80 for block in config["approximated_blocks"])},
        "C2_direct64_composes_through_suffix": {"supported": all(program_summary[str(layer)]["direct_64"]["gradient_cosine_median"] >= .60 for layer in config["source_layers"])},
        "C3_direct256_composes_strongly": {"supported": all(program_summary[str(layer)]["direct_256"]["gradient_cosine_median"] >= .80 for layer in config["source_layers"])},
        "C4_direct64_is_finitely_directional": {"supported": all(program_summary[str(layer)]["direct_64"]["finite_directional_fraction"] >= .90 for layer in config["source_layers"])},
        "C5_composition_improves_for_later_sources": {"supported": program_summary["3"]["direct_64"]["gradient_cosine_median"] - program_summary["0"]["direct_64"]["gradient_cosine_median"] >= .15},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({
        "config": config, "maximum_state_error": maximum_state_error, "hypotheses": hypotheses,
        "local_summary": local_summary, "program_summary": program_summary,
        "local_records": local, "records": records,
    }, indent=2, allow_nan=False))
    lines = [
        "# Causal composition of the prospective operator router", "",
        "Discovery-frozen PCA bases compress each Block 1--5 mixed residual and attention write separately. At each held-out semantic site, the components are reconstructed, literally added, RMS-normalized, and used as the state conditioning the bilinear MLP Jacobian. No regression to the target operator or intervention effect is fitted. Every attention derivative, other position, and later block remains exact.", "",
        f"Maximum exact residual-plus-attention state reconstruction error: `{maximum_state_error:.2e}`.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Prospective causal composition](bilin18_prospective_router_causal_composition.png)", "",
              "| Block | Direct64 local cosine | Direct256 local cosine |", "|---:|---:|---:|"]
    for block in config["approximated_blocks"]:
        row = local_summary[str(block)]
        lines.append(f"| {block} | {row['direct_64']:.3f} | {row['direct_256']:.3f} |")
    lines += ["", "| Source layer | Direct64 gradient cosine | Direct256 gradient cosine | Direct64 finite direction |",
              "|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        row = program_summary[str(layer)]
        lines.append(f"| {layer} | {row['direct_64']['gradient_cosine_median']:.3f} | {row['direct_256']['gradient_cosine_median']:.3f} | {row['direct_64']['finite_directional_fraction']:.3f} |")
    lines += ["", "## Guardrails", "",
              "This validates a compact interface between attention and each MLP, not a semantic interpretation of its coordinates. The PCA bases are data-derived and dense. Only Blocks 1--5 are compressed; the remaining suffix is exact. Finite validation tests source-channel selection induced by the approximate gradient, not a forward replacement of every compressed state."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    blocks = config["approximated_blocks"]
    for method in ("direct_64", "direct_256"):
        axes[0].plot(blocks, [local_summary[str(block)][method] for block in blocks], marker="o", label=method)
    layers = config["source_layers"]
    for method in ("direct_64", "direct_256"):
        axes[1].plot(layers, [program_summary[str(layer)][method]["gradient_cosine_median"] for layer in layers], marker="o", label=method)
    axes[0].set(xlabel="Approximated block", ylabel="Median response cosine", title="Local semantic adjoints")
    axes[1].set(xlabel="Source layer", ylabel="Median gradient cosine", title="Composition through Blocks 1-5")
    for axis in axes: axis.grid(alpha=.2); axis.legend()
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "local_summary": local_summary, "program_summary": program_summary}, indent=2))
    print(REPORT)


if __name__ == "__main__": main()
