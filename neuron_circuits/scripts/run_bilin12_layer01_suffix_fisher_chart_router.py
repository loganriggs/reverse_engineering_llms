#!/usr/bin/env python3
"""Prospective chart routing with the exact suffix JVP and Fisher metric."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_adaptive_chart_router import (
    chart_program,
    cosine,
    midpoint_product_jvp,
)
from run_bilin12_layer01_dynamic_alignment_replication import (
    evaluate_deltas,
    final_product_delta,
    forward_layers_native,
    median,
    pair_cluster_interval,
    selected_delta,
)
from run_bilin18_layer01_native_rotation_tournament import (
    make_rotation,
    selected_rotated,
)
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin12_layer01_suffix_fisher_chart_router.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_suffix_fisher_chart_router.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_SUFFIX_FISHER_CHART_ROUTER.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_suffix_fisher_chart_router.png"


@torch.inference_mode()
def trace_from_x0(model, x0, last_layer):
    dimension = model.config.n_embd
    x, value_bus = x0, None
    records = []
    for block in model.blocks[:last_layer + 1]:
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
        records.append({"x0": x0, "value_bus": value_bus, "output": x})
    return records


def repeat_first(tensor, count):
    return tensor.repeat(count, *([1] * (tensor.ndim - 1)))


def suffix_logits(model, layer, x0, value_bus, output, position):
    dimension = model.config.n_embd
    x, bus = output, value_bus
    for block in model.blocks[layer + 1:]:
        x, bus = block(x, bus, x0)
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, position]).float()
    return 30 * torch.tanh(raw / 30)


def suffix_jvp_batch(model, layer, record, write_tangents, position):
    count = len(write_tangents)
    x0 = repeat_first(record["x0"], count)
    bus = repeat_first(record["value_bus"], count)
    primal = repeat_first(record["output"], count)
    tangent = torch.zeros_like(primal)
    tangent[:, position] = torch.stack(write_tangents)
    function = lambda value: suffix_logits(model, layer, x0, bus, value, position)
    with torch.enable_grad():
        logits, derivative = torch.func.jvp(function, (primal,), (tangent,))
    return logits.float(), derivative.float(), primal, tangent, function


def main():
    red = json.loads(CONFIG.read_text())
    router = json.loads(Path(red["source_config"]).read_text())
    parent = json.loads(Path(router["source_config"]).read_text())
    config = json.loads(Path(parent["source_config"]).read_text())
    pair_config = json.loads(Path(config["evaluation_pair_config"]).read_text())
    seeds = parent["rotation_seeds"]
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in config["evaluation"].items()
    }
    layers = config["layers"]
    edit_position, position = config["edit_position"], config["prediction_position"]
    rank = red["rank"]
    rotations = {layer: {
        seed: make_rotation(
            model.blocks[layer].mlp.Down.weight.shape[1], config["rotation_rounds"], seed + layer, "cuda",
        ) for seed in seeds
    } for layer in layers}
    records = {str(layer): [] for layer in layers}

    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        rng = np.random.default_rng(config["window_seed"] + 1000 + corpus_index)
        for family, pairs in pair_config["new_families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(rows, source_token, edit_position, position, config["evaluation_contexts_per_pair"], rng)
                for context_index, (row_index, start) in enumerate(windows):
                    raw = np.asarray(rows[row_index, start:start + config["sequence_length"]], dtype=np.int64).copy()
                    changed = raw.copy(); changed[edit_position] = target_token
                    base_ids = torch.as_tensor(raw, device="cuda")[None]
                    changed_ids = torch.as_tensor(changed, device="cuda")[None]
                    with torch.no_grad():
                        base = forward_layers_native(model, base_ids, max(layers))
                        counter = forward_layers_native(model, changed_ids, max(layers))
                    # The midpoint x0 is shared across target layers.
                    midpoint_trace = None
                    for layer in layers:
                        decoder = model.blocks[layer].mlp.Down.weight.float()
                        full = final_product_delta(model, layer, base[layer], counter[layer], position)
                        tangent, point, _ = midpoint_product_jvp(model, base_ids, changed_ids, layer, position, router["path_point"])
                        if midpoint_trace is None:
                            with torch.no_grad(): midpoint_trace = trace_from_x0(model, point, max(layers))

                        names = ["native"] + [str(seed) for seed in seeds]
                        actual_patches = {}
                        tangent_patches = {}
                        local_scores = {}
                        patch, local, indices = chart_program(full, tangent, decoder, rank)
                        actual_patches["native"] = patch; local_scores["native"] = local
                        tangent_patches["native"] = selected_delta(tangent, indices)
                        for seed in seeds:
                            name = str(seed); permutation = rotations[layer][seed]
                            patch, local, indices = chart_program(full, tangent, decoder, rank, permutation)
                            actual_patches[name] = patch; local_scores[name] = local
                            tangent_patches[name] = selected_rotated(tangent, indices, permutation)

                        write_tangents = [F.linear(tangent, decoder)] + [F.linear(tangent_patches[name], decoder) for name in names]
                        logits, derivatives, primal, output_tangent, function = suffix_jvp_batch(
                            model, layer, midpoint_trace[layer], write_tangents, position,
                        )
                        reference_logits = logits[0]
                        target_effect = derivatives[0]
                        predicted_fidelity = {
                            name: fisher_fidelity(reference_logits, target_effect, derivatives[index + 1])[0]
                            for index, name in enumerate(names)
                        }
                        rotation_names = [str(seed) for seed in seeds]
                        fisher_choice = max(rotation_names, key=lambda name: predicted_fidelity[name])
                        local_choice = max(rotation_names, key=lambda name: local_scores[name])

                        deltas = {f"chart_{name}": patch for name, patch in actual_patches.items()}
                        with torch.no_grad(): effects = evaluate_deltas(model, layer, base[layer], deltas, full, position)
                        actual = {name: effects[f"chart_{name}"] for name in names}
                        oracle_choice = max(rotation_names, key=lambda name: actual[name])

                        # Validate only the full tangent row to keep the numerical
                        # control independent of which chart was selected.
                        epsilon = red["finite_difference_epsilon"]
                        with torch.no_grad():
                            positive = function(primal + epsilon * output_tangent)[0]
                            negative = function(primal - epsilon * output_tangent)[0]
                        finite = (positive - negative) / (2 * epsilon)
                        suffix_cosine = cosine(target_effect, finite)
                        suffix_error = float((target_effect - finite).norm() / finite.norm().clamp_min(1e-30))
                        records[str(layer)].append({
                            "corpus": corpus, "family": family, "pair": f"{source_token}->{target_token}", "context": context_index,
                            "fisher_choice": fisher_choice, "local_choice": local_choice, "oracle_choice": oracle_choice,
                            "fisher_oracle_agreement": fisher_choice == oracle_choice,
                            "fisher_local_agreement": fisher_choice == local_choice,
                            "suffix_jvp_cosine": suffix_cosine, "suffix_jvp_relative_error": suffix_error,
                            "predicted_fidelity": predicted_fidelity,
                            "actual": {
                                "native": actual["native"],
                                "fixed_rotation": actual[rotation_names[0]],
                                "local_router": actual[local_choice],
                                "fisher_router": actual[fisher_choice],
                                "oracle": actual[oracle_choice],
                            },
                        })
                print("suffix Fisher router", corpus, family, source_token, target_token, flush=True)

    summaries = {str(layer): {} for layer in layers}
    for layer in layers:
        for corpus_index, corpus in enumerate(config["evaluation"]):
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            methods = {name: median([row["actual"][name] for row in selected]) for name in selected[0]["actual"]}
            advantage, interval = pair_cluster_interval(
                selected, lambda row: row["actual"]["fisher_router"] - row["actual"]["native"],
                red["bootstrap_samples"], red["bootstrap_seed"] + layer * 100 + corpus_index,
            )
            summaries[str(layer)][corpus] = {
                "methods": methods,
                "fisher_minus_native_pair_median": advantage,
                "fisher_minus_native_pair_ci": interval,
                "fisher_oracle_agreement": float(np.mean([row["fisher_oracle_agreement"] for row in selected])),
                "fisher_local_disagreement": float(np.mean([not row["fisher_local_agreement"] for row in selected])),
                "suffix_jvp_minimum_cosine": min(row["suffix_jvp_cosine"] for row in selected),
                "suffix_jvp_maximum_relative_error": max(row["suffix_jvp_relative_error"] for row in selected),
            }

    def cell(layer, corpus): return summaries[str(layer)][corpus]
    numerical = all(cell(l, c)["suffix_jvp_minimum_cosine"] >= 0.999 and cell(l, c)["suffix_jvp_maximum_relative_error"] <= 0.02 for l in layers for c in config["evaluation"])
    recovers = all(cell(l, c)["methods"]["oracle"] - cell(l, c)["methods"]["fisher_router"] <= 0.05 and cell(l, c)["fisher_oracle_agreement"] >= 0.30 for l in layers for c in config["evaluation"])
    beats_local = all(cell(l, c)["methods"]["fisher_router"] - cell(l, c)["methods"]["local_router"] >= 0.05 for l in layers for c in config["evaluation"])
    beats_native = all(cell(l, c)["fisher_minus_native_pair_median"] >= 0.03 and cell(l, c)["fisher_minus_native_pair_ci"][0] > 0 for l in layers for c in config["evaluation"])
    fractions = {}; closes = True
    for layer in layers:
        fractions[str(layer)] = {}
        for corpus in config["evaluation"]:
            methods = cell(layer, corpus)["methods"]
            fraction = (methods["fisher_router"] - methods["fixed_rotation"]) / max(methods["oracle"] - methods["fixed_rotation"], 1e-30)
            fractions[str(layer)][corpus] = fraction; closes = closes and fraction >= .5
    changes = all(cell(l, c)["fisher_local_disagreement"] >= .5 for l in layers for c in config["evaluation"])
    ood = all(abs((cell(l, "fineweb")["methods"]["fisher_router"] - cell(l, "fineweb")["methods"]["native"]) - (cell(l, "tinystories")["methods"]["fisher_router"] - cell(l, "tinystories")["methods"]["native"])) <= .1 for l in layers)
    hypotheses = {
        "F0_batched_suffix_jvp_is_numerically_correct": {"supported": numerical},
        "F1_suffix_fisher_router_recovers_the_midpoint_chart_oracle": {"supported": recovers},
        "F2_suffix_fisher_router_beats_local_router": {"supported": beats_local},
        "F3_suffix_fisher_adaptive_dictionary_beats_native": {"supported": beats_native},
        "F4_suffix_fisher_router_closes_half_the_chart_oracle_gap": {"supported": closes, "fractions": fractions},
        "F5_output_conditioning_changes_chart_choice": {"supported": changes},
        "F6_suffix_router_transfers_ood": {"supported": ood},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Prospective suffix-Fisher adaptive chart router", "", "For each native or rotated rank-256 midpoint program, the selected product tangent is written through the native Down matrix and propagated through the exact remaining tensor-network JVP at the midpoint state. Chart choice maximizes predicted softmax-Fisher fidelity to the full tangent. The router never sees the finite endpoint logits used for evaluation.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Suffix-Fisher router](bilin12_layer01_suffix_fisher_chart_router.png)", "", "| Layer | Corpus | Native / fixed / local router / suffix-Fisher router / oracle | Fisher−native pair median (95% CI) | Fisher↔oracle agreement | Fisher/local disagreement | Suffix JVP min cosine / max error |", "|---:|---|---:|---:|---:|---:|---:|"]
    for layer in layers:
        for corpus in config["evaluation"]:
            row = cell(layer, corpus); m = row["methods"]; ci = row["fisher_minus_native_pair_ci"]
            lines.append(f"| {layer} | {corpus} | {m['native']:.3f}/{m['fixed_rotation']:.3f}/{m['local_router']:.3f}/{m['fisher_router']:.3f}/{m['oracle']:.3f} | {row['fisher_minus_native_pair_median']:+.3f} ({ci[0]:+.3f}, {ci[1]:+.3f}) | {row['fisher_oracle_agreement']:.3f} | {row['fisher_local_disagreement']:.3f} | {row['suffix_jvp_minimum_cosine']:.6f}/{row['suffix_jvp_maximum_relative_error']:.4f} |")
    lines += ["", "This is a stronger tensor explanation than local write fidelity because it contracts the chart through all downstream weights and the output metric. Failure would mean the per-example best-of-seven finite chart depends on curvature or finite interactions not captured by one midpoint tangent, rather than merely on choosing the correct local norm."]
    REPORT.write_text("\n".join(lines) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained"); method_names = ["native", "fixed_rotation", "local_router", "fisher_router", "oracle"]
    for axis, layer in zip(axes, layers):
        x = np.arange(len(method_names)); width = .35
        for offset, corpus in zip((-.175, .175), config["evaluation"]): axis.bar(x + offset, [cell(layer, corpus)["methods"][name] for name in method_names], width, label=corpus)
        axis.set_xticks(x); axis.set_xticklabels(method_names, rotation=25, ha="right"); axis.set_ylim(0, 1.02); axis.set_title(f"Layer {layer}"); axis.set_ylabel("median Fisher fidelity"); axis.grid(axis="y", alpha=.25); axis.legend()
    figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
