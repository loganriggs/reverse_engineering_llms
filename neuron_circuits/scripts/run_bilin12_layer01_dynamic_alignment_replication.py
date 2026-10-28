#!/usr/bin/env python3
"""Independent-checkpoint replication of Layer-0/1 dynamic neuron alignment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin18_layer01_native_rotation_tournament import (
    make_rotation,
    rotate,
    selected_native,
    selected_rotated,
)
from run_bilin18_layer8_router_causal_validation import suffix_from_layer
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin12_layer01_dynamic_alignment_replication.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_dynamic_alignment_replication.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_DYNAMIC_ALIGNMENT_REPLICATION.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_dynamic_alignment_replication.png"
METHODS = ("native_activation", "native_local", "rotated_local")


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def first_threshold(curve, ranks, threshold):
    for rank, value in zip(ranks, curve):
        if value >= threshold:
            return int(rank)
    return None


def pair_cluster_interval(records, value_function, samples, seed):
    """Bootstrap the median after collapsing correlated contexts within pairs."""
    pair_names = sorted({row["pair"] for row in records})
    pair_values = np.asarray([
        median([value_function(row) for row in records if row["pair"] == pair])
        for pair in pair_names
    ], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draws[index] = np.median(generator.choice(pair_values, len(pair_values), replace=True))
    return float(np.median(pair_values)), [float(value) for value in np.quantile(draws, [0.025, 0.975])]


@torch.inference_mode()
def final_product_delta(model, layer, base, changed, position):
    block = model.blocks[layer]
    dimension = model.config.n_embd
    base_state = F.rms_norm(base["pre"][0, position].float(), (dimension,))
    changed_state = F.rms_norm(changed["pre"][0, position].float(), (dimension,))
    base_product = block.mlp.Left(base_state) * block.mlp.Right(base_state)
    changed_product = block.mlp.Left(changed_state) * block.mlp.Right(changed_state)
    return changed_product.float() - base_product.float()


@torch.inference_mode()
def evaluate_deltas(model, layer, base, deltas, full, position):
    """Patch only the final-position MLP product and recompute the exact suffix."""
    block = model.blocks[layer]
    decoder = block.mlp.Down.weight.float()
    names = list(deltas)
    outputs = [base["output"]]
    x0s = [base["x0"]]
    buses = [base["value_bus"]]
    for delta in list(deltas.values()) + [full]:
        output = base["output"].clone()
        output[:, position] += F.linear(delta, decoder).to(output.dtype)
        outputs.append(output)
        x0s.append(base["x0"])
        buses.append(base["value_bus"])
    logits = suffix_from_layer(
        model, layer, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), position,
    )
    base_logits = logits[0]
    target = logits[-1] - base_logits
    return {
        name: fisher_fidelity(base_logits, target, logits[index + 1] - base_logits)[0]
        for index, name in enumerate(names)
    }


def selected_delta(full, indices):
    return selected_native(full, indices.to(full.device))


@torch.inference_mode()
def forward_layers_native(model, ids, last_layer):
    """Trace checkpoint-native blocks without assuming one- or two-QK attention."""
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x, value_bus = x0, None
    records = []
    for block in model.blocks[:last_layer + 1]:
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        mlp_write = block.mlp.Down(product) + block.mlp.Down_bias
        x = pre + mlp_write
        records.append({
            "x0": x0,
            "value_bus": value_bus,
            "pre": pre,
            "mlp_write": mlp_write,
            "output": x,
        })
    return records


@torch.inference_mode()
def main(config_path=CONFIG, output_path=OUTPUT, report_path=REPORT, figure_path=FIGURE):
    config = json.loads(Path(config_path).read_text())
    discovery_spec = json.loads(Path(config["discovery_pair_config"]).read_text())
    evaluation_spec = json.loads(Path(config["evaluation_pair_config"]).read_text())
    reference = json.loads(Path(config["reference_results"]).read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in config["evaluation"].items()
    }
    families = list(discovery_spec["families"])
    ranks = config["ranks"]
    position = config["prediction_position"]
    edit_position = config["edit_position"]

    rotations = {}
    rotated_decoders = {}
    for layer in config["layers"]:
        decoder = model.blocks[layer].mlp.Down.weight.float()
        rotations[layer] = make_rotation(
            decoder.shape[1], config["rotation_rounds"], config["rotation_seed"] + layer, "cuda",
        )
        rotated_decoders[layer] = rotate(decoder, rotations[layer])

    # Discover frozen semantic banks only on the old FineWeb lexical pairs.
    discovery_rows = arrays[config["discovery_corpus"]]
    discovery_scores = {
        layer: {family: [] for family in families} for layer in config["layers"]
    }
    discovery_global = {layer: [] for layer in config["layers"]}
    rng = np.random.default_rng(config["window_seed"])
    for family, pairs in discovery_spec["families"].items():
        for source_token, target_token in pairs:
            windows = select_windows(
                discovery_rows, source_token, edit_position, position,
                config["discovery_contexts_per_pair"], rng,
            )
            for row_index, start in windows:
                raw = np.asarray(
                    discovery_rows[row_index, start:start + config["sequence_length"]], dtype=np.int64,
                ).copy()
                changed = raw.copy(); changed[edit_position] = target_token
                base = forward_layers_native(model, torch.as_tensor(raw, device="cuda")[None], last_layer=max(config["layers"]))
                counter = forward_layers_native(model, torch.as_tensor(changed, device="cuda")[None], last_layer=max(config["layers"]))
                for layer in config["layers"]:
                    delta = final_product_delta(model, layer, base[layer], counter[layer], position)
                    decoder = model.blocks[layer].mlp.Down.weight.float()
                    write = F.linear(delta, decoder)
                    score = (delta * (write @ decoder)).abs().cpu()
                    discovery_scores[layer][family].append(score)
                    discovery_global[layer].append(score)
            print("discovery", family, source_token, target_token, flush=True)

    banks = {}
    for layer in config["layers"]:
        banks[layer] = {
            family: torch.stack(rows).mean(0).topk(config["bank_rank"]).indices.cuda()
            for family, rows in discovery_scores[layer].items()
        }
        banks[layer]["global"] = torch.stack(discovery_global[layer]).mean(0).topk(config["bank_rank"]).indices.cuda()

    # Evaluate entirely new lexical pairs on both in- and out-of-distribution text.
    results = {str(layer): [] for layer in config["layers"]}
    maximum_rotation_error = 0.0
    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        rng = np.random.default_rng(config["window_seed"] + 1000 + corpus_index)
        for family, pairs in evaluation_spec["new_families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows, source_token, edit_position, position,
                    config["evaluation_contexts_per_pair"], rng,
                )
                for context_index, (row_index, start) in enumerate(windows):
                    raw = np.asarray(rows[row_index, start:start + config["sequence_length"]], dtype=np.int64).copy()
                    changed = raw.copy(); changed[edit_position] = target_token
                    base = forward_layers_native(model, torch.as_tensor(raw, device="cuda")[None], last_layer=max(config["layers"]))
                    counter = forward_layers_native(model, torch.as_tensor(changed, device="cuda")[None], last_layer=max(config["layers"]))
                    for layer in config["layers"]:
                        decoder = model.blocks[layer].mlp.Down.weight.float()
                        full = final_product_delta(model, layer, base[layer], counter[layer], position)
                        target_write = F.linear(full, decoder)
                        native_score = (full * (target_write @ decoder)).abs()
                        rotated_full = rotate(full, rotations[layer])
                        rotated_score = (rotated_full * (target_write @ rotated_decoders[layer])).abs()
                        restored = selected_rotated(
                            full, torch.arange(len(full), device="cuda"), rotations[layer],
                        )
                        maximum_rotation_error = max(
                            maximum_rotation_error,
                            float((restored - full).norm() / full.norm().clamp_min(1e-30)),
                        )
                        deltas = {}
                        for rank in ranks:
                            deltas[f"native_activation_{rank}"] = selected_delta(full, full.abs().topk(rank).indices)
                            deltas[f"native_local_{rank}"] = selected_delta(full, native_score.topk(rank).indices)
                            deltas[f"rotated_local_{rank}"] = selected_rotated(
                                full, rotated_score.topk(rank).indices, rotations[layer],
                            )
                        for bank_family in families:
                            deltas[f"bank_{bank_family}"] = selected_delta(full, banks[layer][bank_family])
                        deltas["bank_global"] = selected_delta(full, banks[layer]["global"])
                        effects = evaluate_deltas(model, layer, base[layer], deltas, full, position)
                        wrong = median([
                            effects[f"bank_{other}"] for other in families if other != family
                        ])
                        results[str(layer)].append({
                            "corpus": corpus,
                            "family": family,
                            "pair": f"{source_token}->{target_token}",
                            "context": context_index,
                            "fidelity": effects,
                            "correct_bank": effects[f"bank_{family}"],
                            "wrong_bank_median": wrong,
                            "global_bank": effects["bank_global"],
                        })
                print("evaluation", corpus, family, source_token, target_token, flush=True)

    summaries = {}
    comparison_rank = config["comparison_rank"]
    for layer in config["layers"]:
        summaries[str(layer)] = {}
        for corpus_index, corpus in enumerate(config["evaluation"]):
            selected = [row for row in results[str(layer)] if row["corpus"] == corpus]
            curves = {
                method: [median([row["fidelity"][f"{method}_{rank}"] for row in selected]) for rank in ranks]
                for method in METHODS
            }
            native_advantage, native_ci = pair_cluster_interval(
                selected,
                lambda row: row["fidelity"][f"native_local_{comparison_rank}"] - row["fidelity"][f"rotated_local_{comparison_rank}"],
                config["bootstrap_samples"], config["bootstrap_seed"] + layer * 100 + corpus_index,
            )
            bank_advantage, bank_ci = pair_cluster_interval(
                selected, lambda row: row["correct_bank"] - row["wrong_bank_median"],
                config["bootstrap_samples"], config["bootstrap_seed"] + 10000 + layer * 100 + corpus_index,
            )
            summaries[str(layer)][corpus] = {
                "curves": curves,
                "s50": {method: first_threshold(curves[method], ranks, 0.50) for method in METHODS},
                "s90": {method: first_threshold(curves[method], ranks, 0.90) for method in METHODS},
                "native_advantage_pair_median": native_advantage,
                "native_advantage_pair_ci": native_ci,
                "correct_bank": median([row["correct_bank"] for row in selected]),
                "wrong_bank": median([row["wrong_bank_median"] for row in selected]),
                "global_bank": median([row["global_bank"] for row in selected]),
                "bank_advantage_pair_median": bank_advantage,
                "bank_advantage_pair_ci": bank_ci,
            }

    def cell(layer, corpus):
        return summaries[str(layer)][corpus]

    comparison_index = ranks.index(comparison_rank)
    numerical = all(
        cell(layer, corpus)["curves"][method][-1] >= 0.95
        for layer in config["layers"] for corpus in config["evaluation"] for method in METHODS
    ) and maximum_rotation_error <= 1e-5
    moderate = all(
        cell(layer, corpus)["s50"]["native_local"] is not None
        and cell(layer, corpus)["s50"]["native_local"] <= 256
        for layer in config["layers"] for corpus in config["evaluation"]
    )
    privileged = all(
        cell(layer, corpus)["curves"]["native_local"][comparison_index]
        - cell(layer, corpus)["curves"]["rotated_local"][comparison_index] >= 0.05
        and cell(layer, corpus)["native_advantage_pair_ci"][0] > 0
        for layer in config["layers"] for corpus in config["evaluation"]
    )
    broad = all(
        cell(layer, corpus)["s90"]["native_local"] is not None
        and cell(layer, corpus)["s90"]["native_local"] >= 1024
        for layer in config["layers"] for corpus in config["evaluation"]
    )
    banks_insufficient = all(
        cell(layer, corpus)["correct_bank"] < 0.35
        and cell(layer, corpus)["curves"]["native_local"][comparison_index] - cell(layer, corpus)["correct_bank"] >= 0.25
        for layer in config["layers"] for corpus in config["evaluation"]
    )
    bank_identity = all(
        cell(layer, corpus)["bank_advantage_pair_median"] >= 0.03
        and cell(layer, corpus)["bank_advantage_pair_ci"][0] > 0
        for layer in config["layers"] for corpus in config["evaluation"]
    )
    ood = all(
        abs(
            (cell(layer, "fineweb")["curves"]["native_local"][comparison_index] - cell(layer, "fineweb")["curves"]["rotated_local"][comparison_index])
            - (cell(layer, "tinystories")["curves"]["native_local"][comparison_index] - cell(layer, "tinystories")["curves"]["rotated_local"][comparison_index])
        ) <= 0.10 for layer in config["layers"]
    )
    activation = all(
        cell(layer, corpus)["curves"]["native_activation"][comparison_index] >= 0.50
        and cell(layer, corpus)["curves"]["native_activation"][comparison_index]
        >= cell(layer, corpus)["curves"]["native_local"][comparison_index] - 0.15
        for layer in config["layers"] for corpus in config["evaluation"]
    )
    hypotheses = {
        "R0_numerical_controls_reconstruct": {"supported": numerical, "maximum_rotation_error": maximum_rotation_error},
        "R1_dynamic_native_execution_is_moderately_sparse": {"supported": moderate},
        "R2_native_basis_is_privileged": {"supported": privileged},
        "R3_high_fidelity_native_execution_is_broad": {"supported": broad},
        "R4_fixed_semantic_banks_are_not_sufficient": {"supported": banks_insufficient},
        "R5_fixed_semantic_identity_is_detectable": {"supported": bank_identity},
        "R6_dynamic_alignment_transfers_ood": {"supported": ood},
        "R7_qualitative_ontology_replicates": {"supported": moderate and privileged and broad and banks_insufficient},
        "R8_activation_ranking_is_state_informative": {"supported": activation},
    }

    payload = {
        "config": config,
        "model_config": {
            "n_layer": model.config.n_layer,
            "n_embd": model.config.n_embd,
            "n_head": model.config.n_head,
            "mlp_channels": model.blocks[0].mlp.Down.weight.shape[1],
        },
        "hypotheses": hypotheses,
        "maximum_rotation_error": maximum_rotation_error,
        "summaries": summaries,
        "records": results,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Independent-checkpoint replication of Layer-0/1 dynamic neuron alignment", "",
        f"The unchanged final-query MLP assay is repeated on `{config['model_repo']}` at commit `{config['snapshot_commit']}` (12 layers, width 768, 3,072 bilinear channels). Frozen semantic banks are learned from the old FineWeb lexical pairs; native/rotated frontiers and banks are evaluated on seventeen new lexical pairs in FineWeb and TinyStories.", "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "", "![Dynamic alignment replication](bilin12_layer01_dynamic_alignment_replication.png)", "",
        "| Layer | Corpus | Native activation S50/S90 | Native local S50/S90 | Rotated local S50/S90 | Rank-256 native/rotated (pair CI) | Correct/wrong/global bank (bank CI) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for layer in config["layers"]:
        for corpus in config["evaluation"]:
            row = cell(layer, corpus)
            fmt = lambda method: f"{row['s50'][method]}/{row['s90'][method]}"
            n = row["curves"]["native_local"][comparison_index]
            r = row["curves"]["rotated_local"][comparison_index]
            nci = row["native_advantage_pair_ci"]
            bci = row["bank_advantage_pair_ci"]
            lines.append(
                f"| {layer} | {corpus} | {fmt('native_activation')} | {fmt('native_local')} | {fmt('rotated_local')} | "
                f"{n:.3f}/{r:.3f} ({nci[0]:+.3f}, {nci[1]:+.3f}) | "
                f"{row['correct_bank']:.3f}/{row['wrong_bank']:.3f}/{row['global_bank']:.3f} ({bci[0]:+.3f}, {bci[1]:+.3f}) |"
            )
    lines += [
        "", f"Maximum inverse-rotation relative error: **{maximum_rotation_error:.2e}**.", "",
        "The native frontier tests execution alignment: coordinates are selected separately for each realized edit. The frozen banks test semantic identity alignment: the same discovered channel set must serve unseen lexical pairs. Passing the first while failing the second is the conditional-native-program ontology, not classical one-concept/one-neuron monosemanticity.", "",
        "## Original 18-layer reference", "",
        "| Layer | Corpus | Native local S50/S90 | Rotated local S50/S90 | Rank-256 advantage |", "|---:|---|---:|---:|---:|",
    ]
    reference_ranks = reference["config"]["ranks"]
    reference_index = reference_ranks.index(256)
    for layer in config["layers"]:
        for corpus in config["evaluation"]:
            row = reference["layers"][str(layer)]["summaries"][corpus]
            advantage = row["curves"]["native_local"][reference_index] - row["curves"]["rotated_local"][reference_index]
            lines.append(f"| {layer} | {corpus} | {row['s50']['native_local']}/{row['s90']['native_local']} | {row['s50']['rotated_local']}/{row['s90']['rotated_local']} | {advantage:+.3f} |")
    Path(report_path).write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = {"native_activation": "tab:blue", "native_local": "tab:green", "rotated_local": "tab:red"}
    for axis, (layer, corpus) in zip(axes.flat, [(l, c) for l in config["layers"] for c in config["evaluation"]]):
        row = cell(layer, corpus)
        for method in METHODS:
            axis.plot(ranks, row["curves"][method], marker="o", label=method, color=colors[method])
        axis.axhline(0.9, color="grey", linewidth=1, linestyle=":")
        axis.set_xscale("log", base=2); axis.set_ylim(-0.1, 1.05)
        axis.set_title(f"Layer {layer} / {corpus}"); axis.set_xlabel("selected coordinates"); axis.set_ylabel("Fisher fidelity")
        axis.grid(alpha=0.25); axis.legend(fontsize=7)
    figure.savefig(figure_path, dpi=180); plt.close(figure)
    print(json.dumps(hypotheses, indent=2)); print(report_path)


if __name__ == "__main__":
    main()
