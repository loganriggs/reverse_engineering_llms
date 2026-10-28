#!/usr/bin/env python3
"""Compare native-neuron and compositional explanations across MLP Layers 0--3."""

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
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, selected_delta
from run_bilin12_layer01_frozen_taylor_behavior import logits_for_deltas
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import semantic_gradient
from run_bilin18_layer1_suffix_bidegree_pilot import metrics
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, final_adjoint, mlp_vjp
from run_bilin18_layer1_suffix_qk_role_pilot import ROLES, attention_role_vjps
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layers0_3_compositional_ontology.json")


def specifications(source_layer: int, block_count: int, near_widths: list[int]):
    suffix = list(range(source_layer + 1, block_count))
    result = {
        "mlp_only": {"layers": set(), "roles": set()},
        "queries": {"layers": set(suffix), "roles": {"q1", "q2"}},
        "keys": {"layers": set(suffix), "roles": {"k1", "k2"}},
        "query_key": {"layers": set(suffix), "roles": {"q1", "k1", "q2", "k2"}},
        "value": {"layers": set(suffix), "roles": {"value"}},
        "all_full": {"layers": set(suffix), "roles": set(ROLES)},
    }
    for width in near_widths:
        result[f"near_{width}"] = {
            "layers": set(suffix[:width]), "roles": set(ROLES),
        }
    return result


def compositional_program(model, trace, source_layer, position, contrast, specs):
    """Compose exact local adjoints while enabling selected attention roles."""
    names = list(specs)
    current = final_adjoint(model, trace[-1]["output"], position, contrast).repeat(
        len(names), 1, 1, 1,
    )
    maximum_local_error = 0.0
    for layer in range(len(model.blocks) - 1, source_layer, -1):
        block = model.blocks[layer]
        updated = current + mlp_vjp(block, trace[layer]["pre"].float(), current)
        enabled = [
            row for row, name in enumerate(names)
            if layer in specs[name]["layers"] and specs[name]["roles"]
        ]
        if enabled:
            incoming = updated[enabled]
            previous = trace[layer - 1]["output"].float()
            x0 = trace[layer]["x0"].float()
            mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0
            bus = trace[layer - 1]["value_bus"]
            pieces = attention_role_vjps(block, mixed, bus, incoming)
            native = attention_vjp(block, mixed, bus, incoming)
            recomposed = sum(pieces.values())
            error = float((recomposed - native).norm() / native.norm().clamp_min(1e-30))
            maximum_local_error = max(maximum_local_error, error)
            for local, row in enumerate(enabled):
                roles = specs[names[row]]["roles"]
                updated[row] += sum(
                    (pieces[role][local] for role in roles),
                    torch.zeros_like(updated[row]),
                )
        current = block.lambdas[0].float() * updated
    down = model.blocks[source_layer].mlp.Down.weight.float()
    return {
        name: current[row, 0, position] @ down
        for row, name in enumerate(names)
    }, maximum_local_error


def select_sites(causal, old, sites_per_feature):
    selected = []
    for feature in causal["features"]:
        rows = [row for row in old["records"] if row["feature"] == feature]
        extreme = [row for row in rows if row["selection"] == "extreme"]
        uniform = [row for row in rows if row["selection"] == "uniform"]
        first = sites_per_feature // 2
        selected.extend(extreme[:first])
        selected.extend(uniform[:sites_per_feature - first])
    return selected


def positive_mass_statistics(contribution):
    """Concentration of locally helpful channels, without harmful-channel cancellation."""
    ordered = contribution.clamp_min(0).sort(descending=True).values
    total = ordered.sum()
    if float(total) <= 0:
        return {"positive_k90": None, "positive_count": 0, "positive_mass": 0.0}
    k90 = int(torch.searchsorted(ordered.cumsum(0), .9 * total).item()) + 1
    return {
        "positive_k90": k90,
        "positive_count": int((ordered > 0).sum()),
        "positive_mass": float(total),
    }


def median_metrics(records, layer, method):
    rows = [row for row in records if row["layer"] == layer]
    values = [row["program_metrics"][method] for row in rows]
    shifts = np.asarray([row["finite_shifts"][method] for row in rows])
    return {
        "gradient_cosine_median": float(np.median([value["gradient_cosine"] for value in values])),
        "gradient_cosine_q10": float(np.quantile([value["gradient_cosine"] for value in values], .1)),
        "gradient_relative_error_median": float(np.median([value["gradient_relative_error"] for value in values])),
        "contribution_cosine_median": float(np.median([value["contribution_cosine"] for value in values])),
        "true_leverage_ratio_median": float(np.median([value["true_leverage_ratio"] for value in values])),
        "finite_directional_fraction": float(np.mean(shifts > 0)),
        "finite_mean_shift": float(shifts.mean()),
    }


def first_rank_at_fraction(rank_summaries, ranks, fraction=.9):
    target = rank_summaries[str(ranks[-1])]["finite_mean_shift"]
    if target <= 0:
        return None
    for rank in ranks:
        if rank_summaries[str(rank)]["finite_mean_shift"] >= fraction * target:
            return rank
    return None


def main(config_path=None):
    config = json.loads(Path(config_path or CONFIG).read_text())
    artifact_stem = config.get("artifact_stem", "bilin18_layers0_3_compositional_ontology")
    output = TENSOR_ROOT / f"{artifact_stem}.json"
    report = TENSOR_ROOT / f"{artifact_stem.upper()}.md"
    figure = TENSOR_ROOT / f"{artifact_stem}.png"
    causal = json.loads(Path(config["source_config"]).read_text())
    old = json.loads(Path(config["source_results"]).read_text())
    source = json.loads(Path(causal["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([
        vocabulary_features(tokenizer.decode([index]))
        for index in range(tokenizer.vocab_size)
    ])
    positive = flags.sum(0).astype(np.float32)
    negative = len(flags) - positive
    contrasts = torch.as_tensor(
        flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None],
        device="cuda", dtype=torch.float32,
    )
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    corpus = np.load(source["corpus"], mmap_mode="r")
    heldout_start = source["start_row"] + source["discovery_rows"]
    sites = select_sites(causal, old, config["sites_per_feature"])
    records = []
    maximum_local_error = 0.0
    ranks = config["coalition_ranks"]

    # Each donor/recipient pair is traced once, then inspected at four boundaries.
    for site_index, row in enumerate(sites):
        feature_index = FEATURES.index(row["feature"])
        contrast = contrasts[:, feature_index]
        donor_raw = np.asarray(
            corpus[heldout_start + row["donor_row"], :source["sequence_length"]],
            dtype=np.int64,
        )
        recipient_raw = np.asarray(
            corpus[heldout_start + row["recipient_row"], :source["sequence_length"]],
            dtype=np.int64,
        )
        donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]
        recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        with torch.no_grad():
            donor_trace = forward_layers_native(model, donor_ids, max(config["source_layers"]))
            recipient_trace = forward_layers_native(model, recipient_ids, len(model.blocks) - 1)

        for source_layer in config["source_layers"]:
            position = row["recipient_position"]
            delta = (
                product_at(model, donor_trace, source_layer, row["donor_position"])
                - product_at(model, recipient_trace, source_layer, position)
            )
            exact = semantic_gradient(model, recipient_trace, source_layer, position, contrast)
            specs = specifications(source_layer, len(model.blocks), config["near_widths"])
            predictions, error = compositional_program(
                model, recipient_trace, source_layer, position, contrast, specs,
            )
            maximum_local_error = max(maximum_local_error, error)
            predictions["exact"] = exact

            patch_deltas = {"base": torch.zeros_like(delta), "full_product": delta}
            for method, prediction in predictions.items():
                support = (prediction * delta).topk(config["support_rank"]).indices
                patch_deltas[method] = selected_delta(delta, support)
            exact_contribution = exact * delta
            concentration = positive_mass_statistics(exact_contribution)
            for rank in ranks:
                support = exact_contribution.topk(rank).indices
                patch_deltas[f"rank_{rank}"] = selected_delta(delta, support)

            with torch.no_grad():
                logits = logits_for_deltas(
                    model, source_layer, recipient_trace[source_layer], patch_deltas, position,
                )
            base_score = class_score(logits["base"], contrast)
            finite = {
                name: class_score(value, contrast) - base_score
                for name, value in logits.items() if name != "base"
            }
            records.append({
                "layer": source_layer,
                "feature": row["feature"],
                "selection": row["selection"],
                "pair": row["pair"],
                "program_metrics": {
                    name: metrics(prediction, exact, delta, config["support_rank"])
                    for name, prediction in predictions.items()
                },
                "positive_mass": concentration,
                "finite_shifts": finite,
            })
            print(
                f"site {site_index + 1}/{len(sites)} layer {source_layer} "
                f"{row['feature']} {row['selection']}", flush=True,
            )

    summaries = {}
    for layer in config["source_layers"]:
        layer_rows = [row for row in records if row["layer"] == layer]
        program_names = list(layer_rows[0]["program_metrics"])
        programs = {
            name: median_metrics(records, layer, name) for name in program_names
        }
        rank_summaries = {}
        for rank in ranks:
            shifts = np.asarray([row["finite_shifts"][f"rank_{rank}"] for row in layer_rows])
            rank_summaries[str(rank)] = {
                "finite_directional_fraction": float(np.mean(shifts > 0)),
                "finite_mean_shift": float(shifts.mean()),
                "finite_median_shift": float(np.median(shifts)),
            }
        full = np.asarray([row["finite_shifts"]["full_product"] for row in layer_rows])
        summaries[str(layer)] = {
            "programs": programs,
            "ranks": rank_summaries,
            "full_product": {
                "finite_directional_fraction": float(np.mean(full > 0)),
                "finite_mean_shift": float(full.mean()),
            },
            "first_rank_at_90pct_of_rank1024": first_rank_at_fraction(rank_summaries, ranks),
            "positive_mass_k90_median": float(np.median([
                row["positive_mass"]["positive_k90"] for row in layer_rows
                if row["positive_mass"]["positive_k90"] is not None
            ])),
            "positive_mass_k90_q10": float(np.quantile([
                row["positive_mass"]["positive_k90"] for row in layer_rows
                if row["positive_mass"]["positive_k90"] is not None
            ], .1)),
            "positive_mass_k90_q90": float(np.quantile([
                row["positive_mass"]["positive_k90"] for row in layer_rows
                if row["positive_mass"]["positive_k90"] is not None
            ], .9)),
        }

    layer_values = [summaries[str(layer)] for layer in config["source_layers"]]
    rank90 = [value["first_rank_at_90pct_of_rank1024"] for value in layer_values]
    hypotheses = {
        "D0_exact_composition_reconstructs_each_layer": {"supported": all(
            value["programs"]["all_full"]["gradient_cosine_median"] >= .999
            and value["programs"]["all_full"]["gradient_relative_error_median"] <= .02
            for value in layer_values
        )},
        "D1_local_role_factorization_is_exact": {"supported": maximum_local_error <= 1e-5},
        "D2_eight_nearby_attention_blocks_are_sufficient": {"supported": all(
            value["programs"]["near_8"]["gradient_cosine_median"] >= .90
            for value in layer_values
        )},
        "D3_qk_is_a_stable_same_position_grammar": {"supported": all(
            value["programs"]["query_key"]["gradient_cosine_median"] >= .85
            for value in layer_values
        )},
        "D4_qk_beats_value": {"supported": all(
            value["programs"]["query_key"]["gradient_cosine_median"]
            - value["programs"]["value"]["gradient_cosine_median"] >= .10
            for value in layer_values
        )},
        "D5_rank64_is_directionally_sufficient": {"supported": all(
            value["ranks"]["64"]["finite_directional_fraction"] >= .90
            for value in layer_values
        )},
        "D6_rank64_recovers_most_rank1024_effect": {"supported": all(
            value["ranks"]["64"]["finite_mean_shift"]
            >= .80 * value["ranks"]["1024"]["finite_mean_shift"]
            for value in layer_values
        )},
        "D7_single_neurons_are_not_sufficient": {"supported": all(
            value["ranks"]["1"]["finite_mean_shift"]
            < .25 * value["ranks"]["1024"]["finite_mean_shift"]
            for value in layer_values
        )},
        "D8_depth_does_not_make_native_coalitions_less_compact": {"supported": (
            all(rank is not None for rank in rank90)
            and all(right <= left for left, right in zip(rank90, rank90[1:]))
        )},
    }

    output.write_text(json.dumps({
        "config": config,
        "maximum_local_recomposition_error": maximum_local_error,
        "hypotheses": hypotheses,
        "summaries": summaries,
        "records": records,
    }, indent=2, allow_nan=False))

    methods = ["mlp_only", "queries", "keys", "query_key", "value", "near_4", "near_8", "all_full"]
    first_layer, last_layer = min(config["source_layers"]), max(config["source_layers"])
    lines = [
        f"# Layers {first_layer}–{last_layer} compositional ontology",
        "",
        f"The same {len(sites)} held-out semantic donor–recipient pairs are inspected at each of {len(config['source_layers'])} bilinear MLP product boundaries. Exact downstream gradients are composed block by block. This avoids explicitly expanding the network into a high-order polynomial: complexity is represented as a sequence of local, state-conditioned operators.",
        "",
        f"Maximum local Q/K/value recomposition error: `{maximum_local_error:.2e}`.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", f"![Depth comparison]({figure.name})", "", "## Compositional programs", "", "| Source layer | MLP only | Queries | Keys | QK | Value | Near 4 | Near 8 | Exact |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        programs = summaries[str(layer)]["programs"]
        lines.append("| " + str(layer) + " | " + " | ".join(
            f"{programs[name]['gradient_cosine_median']:.3f}" for name in methods
        ) + " |")
    lines += ["", "## Native-neuron finite interventions", "", "| Source layer | Top 1 | Top 4 | Top 16 | Top 64 | Top 256 | Top 1024 | Positive-mass K90 median (Q10–Q90) |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        value = summaries[str(layer)]
        shifts = [value["ranks"][str(rank)]["finite_mean_shift"] for rank in ranks]
        lines.append("| " + str(layer) + " | " + " | ".join(f"{shift:+.4f}" for shift in shifts) + f" | {value['positive_mass_k90_median']:.0f} ({value['positive_mass_k90_q10']:.0f}–{value['positive_mass_k90_q90']:.0f}) |")
    lines += [
        "", "## Interpretation guardrail", "",
        "The native-channel curves use an example-specific activation-times-exact-gradient oracle. They measure how many native channels are sufficient for a known edit at that state; they are not a prospective monosemanticity score. The role programs are likewise local linear explanations of downstream sensitivity, then checked with finite channel interventions.",
    ]
    report.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    x = np.arange(len(config["source_layers"]))
    for method in ("mlp_only", "query_key", "value", "near_4", "near_8"):
        axes[0].plot(x, [summaries[str(layer)]["programs"][method]["gradient_cosine_median"] for layer in config["source_layers"]], marker="o", label=method)
    axes[0].set_xticks(x, config["source_layers"]); axes[0].set(xlabel="Source MLP layer", ylabel="Median gradient cosine", title="Compositional description across depth"); axes[0].axhline(.9, color="gray", linestyle=":"); axes[0].grid(alpha=.2); axes[0].legend()
    for rank in ranks:
        axes[1].plot(x, [summaries[str(layer)]["ranks"][str(rank)]["finite_mean_shift"] for layer in config["source_layers"]], marker="o", label=f"top {rank}")
    axes[1].set_xticks(x, config["source_layers"]); axes[1].set(xlabel="Source MLP layer", ylabel="Mean finite semantic-score shift", title="Example-specific native coalition curve"); axes[1].grid(alpha=.2); axes[1].legend(ncol=2)
    fig.savefig(figure, dpi=180); plt.close(fig)
    print(json.dumps({"maximum_local_recomposition_error": maximum_local_error, "hypotheses": hypotheses, "summaries": summaries}, indent=2))
    print(report)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
