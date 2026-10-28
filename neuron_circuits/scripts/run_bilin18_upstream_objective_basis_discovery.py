#!/usr/bin/env python3
"""Discover objective coordinates from upstream tensor programs, then freeze them."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native
from run_bilin12_router_jacobian_sparse_basis_redteam import greedy_overlap, varimax
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer17_tangent_output_semantics import evaluate_individual, select_individual, vocabulary_table
from run_bilin18_output_basis_upstream_compactness import (
    aggregate, batched_source_gradients, effective_support, top_fraction,
)
from run_bilin18_output_objective_sparse_discovery import fit_sparse_bases
from run_bilin18_layer1_supervised_coalition_causality import product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_upstream_objective_basis_discovery.json")
OUTPUT = TENSOR_ROOT / "bilin18_upstream_objective_basis_discovery.json"
REPORT = TENSOR_ROOT / "BILIN18_UPSTREAM_OBJECTIVE_BASIS_DISCOVERY.md"
FIGURE = TENSOR_ROOT / "bilin18_upstream_objective_basis_discovery.png"


@torch.inference_mode(False)
def collect_programs(model, rows, positions, layers, contrasts):
    sites, gradient_observations, contribution_observations = [], [], []
    for row_index, raw_row in enumerate(rows):
        raw = np.asarray(raw_row[:128], dtype=np.int64); ids = torch.as_tensor(raw, device="cuda")[None]
        with torch.no_grad(): trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        for position in positions:
            gradients = batched_source_gradients(model, trace, position, contrasts, layers)
            for layer in layers:
                gradient = gradients[layer].detach().cpu(); product = product_at(model, trace, layer, position).detach().cpu(); contribution = gradient * product[None]
                # Each objective and cell receives equal RMS before fitting, so
                # high-gain contexts do not determine the coordinate system.
                normalized_gradient = gradient / gradient.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
                normalized_contribution = contribution / contribution.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
                gradient_observations.append(normalized_gradient.T)
                contribution_observations.append(normalized_contribution.T)
                sites.append({"row": row_index, "position": position, "layer": layer, "gradient": gradient, "product": product})
        print(f"program basis rows {row_index + 1}/{len(rows)}", flush=True)
    return sites, torch.cat(gradient_observations), torch.cat(contribution_observations)


def fit_varimax(observations):
    centered = observations - observations.mean(0)
    scale = centered.std(0).clamp_min(1e-6)
    return varimax(centered / scale)[0]


def global_kurtosis(observations, bases):
    """Coordinate sparsity across all context-channel observations."""
    centered = observations - observations.mean(0)
    z = centered / centered.std(0).clamp_min(1e-6)
    result = {}
    for name, q in bases.items():
        rotated = z @ q
        standardized = rotated / rotated.std(0).clamp_min(1e-6)
        excess = standardized.pow(4).mean(0) - 3
        result[name] = {
            "mean_excess_kurtosis": float(excess.mean()),
            "mean_absolute_excess_kurtosis": float(excess.abs().mean()),
        }
    return result


def evaluate_sites(sites, bases, rank):
    records = []
    for site in sites:
        metrics = {}
        for name, q in bases.items():
            gradient = q.T @ site["gradient"]; contribution = gradient * site["product"][None]
            metrics[name] = {"gradient_top64_energy": float(top_fraction(gradient, rank, squared=True).median()), "gradient_effective_support": float(effective_support(gradient).median()), "contribution_top64_mass": float(top_fraction(contribution, rank).median()), "contribution_effective_support": float(effective_support(contribution).median())}
        records.append({"row": site["row"], "position": site["position"], "layer": site["layer"], "metrics": metrics})
    return records


def semantic_evaluation(scores, labels, names, bases, minimum):
    train = np.arange(len(scores)) % 2 == 0; test = ~train; result = {}
    for name, q in bases.items():
        selection = select_individual((scores[train] @ q).numpy(), labels[train], names, minimum)
        result[name] = evaluate_individual((scores[test] @ q).numpy(), labels[test], selection)["median_auc"]
    return result


def event_score_matrices(sites, bases):
    """Per-site axis scores for the rare-event ontology audit."""
    result = {family: {name: [] for name in bases} for family in ("gradient", "contribution")}
    for site in sites:
        for name, q in bases.items():
            gradient = q.T @ site["gradient"]
            contribution = gradient * site["product"][None]
            for family, values in (("gradient", gradient), ("contribution", contribution)):
                rms = values.square().mean(-1).sqrt().clamp_min(1e-12)
                # Large values mean an axis is concentrated into an unusually
                # extreme native channel on this particular example.
                result[family][name].append(values.abs().amax(-1) / rms)
    return {family: {name: torch.stack(values) for name, values in rows.items()} for family, rows in result.items()}


def event_labels(sites, raw_rows, vocabulary_flags, positions, layers):
    arrays, names = [], []
    current = np.asarray([raw_rows[site["row"], site["position"]] for site in sites], dtype=np.int64)
    next_ids = np.asarray([raw_rows[site["row"], site["position"] + 1] for site in sites], dtype=np.int64)
    for prefix, ids in (("current", current), ("next", next_ids)):
        arrays.append(vocabulary_flags[ids]); names.extend([f"{prefix}:{name}" for name in FEATURES])
    for position in positions:
        arrays.append(np.asarray([site["position"] == position for site in sites], dtype=np.uint8)[:, None]); names.append(f"position:{position}")
    for layer in layers:
        arrays.append(np.asarray([site["layer"] == layer for site in sites], dtype=np.uint8)[:, None]); names.append(f"layer:{layer}")
    return np.concatenate(arrays, axis=1), names


def evaluate_event_semantics(discovery_scores, evaluation_scores, discovery_labels, evaluation_labels, names, minimum):
    results, selections = {}, {}
    for family in discovery_scores:
        results[family], selections[family] = {}, {}
        for basis in discovery_scores[family]:
            selected = select_individual(discovery_scores[family][basis].numpy(), discovery_labels, names, minimum)
            evaluated = evaluate_individual(evaluation_scores[family][basis].numpy(), evaluation_labels, selected)
            results[family][basis] = evaluated; selections[family][basis] = selected
    return results, selections


def main():
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    core = torch.load(config["core_path"], map_location="cpu", weights_only=False); axes = core["output"][:, :config["rank"]].float().cuda()
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    vocabulary_scores = (model.lm_head.weight.float()[:tokenizer.vocab_size] @ axes).detach().cpu()
    contrasts = torch.zeros(model.lm_head.weight.shape[0], config["rank"], device="cuda"); contrasts[:tokenizer.vocab_size] = vocabulary_scores.cuda()
    corpus = np.load(config["corpus"], mmap_mode="r")
    discovery_rows = corpus[config["discovery_start"]:config["discovery_start"] + config["rows_per_split"]]
    evaluation_rows = corpus[config["evaluation_start"]:config["evaluation_start"] + config["rows_per_split"]]
    print("collect discovery programs", flush=True); discovery_sites, gradient_obs, contribution_obs = collect_programs(model, discovery_rows, config["positions"], config["source_layers"], contrasts)
    gradient_q = fit_varimax(gradient_obs); contribution_q = fit_varimax(contribution_obs)
    midpoint = config["rows_per_split"] // 2; cells_per_row = len(config["positions"]) * len(config["source_layers"]); channels = model.blocks[0].mlp.Left.out_features
    observations_per_row = cells_per_row * channels
    gradient_halves = [fit_varimax(gradient_obs[start * observations_per_row:stop * observations_per_row]) for start, stop in ((0, midpoint), (midpoint, config["rows_per_split"]))]
    contribution_halves = [fit_varimax(contribution_obs[start * observations_per_row:stop * observations_per_row]) for start, stop in ((0, midpoint), (midpoint, config["rows_per_split"]))]
    stability = {"gradient": greedy_overlap(*gradient_halves), "contribution": greedy_overlap(*contribution_halves)}
    generator = torch.Generator().manual_seed(config["seed"])
    output_sparse, _ = fit_sparse_bases(vocabulary_scores[np.arange(tokenizer.vocab_size) % 2 == 0], generator, {**config, "ica_starts": 5, "ica_iterations": 1000})
    output_q = output_sparse["varimax"]
    overlap_with_output = {"gradient": greedy_overlap(gradient_q, output_q), "contribution": greedy_overlap(contribution_q, output_q)}
    random_qs = [torch.linalg.qr(torch.randn(config["rank"], config["rank"], generator=generator)).Q for _ in range(config["random_rotations"])]
    bases = {"native": torch.eye(config["rank"]), "output_varimax": output_q, "upstream_gradient": gradient_q, "upstream_contribution": contribution_q, **{f"random_{i}": q for i, q in enumerate(random_qs)}}
    discovery_records = evaluate_sites(discovery_sites, bases, config["support_rank"])
    discovery_summaries = {str(layer): {basis: aggregate(discovery_records, layer, basis) for basis in bases} for layer in config["source_layers"]}
    discovery_global = {
        "gradient": global_kurtosis(gradient_obs, bases),
        "contribution": global_kurtosis(contribution_obs, bases),
    }
    print("collect evaluation programs", flush=True); evaluation_sites, evaluation_gradient_obs, evaluation_contribution_obs = collect_programs(model, evaluation_rows, config["positions"], config["source_layers"], contrasts)
    records = evaluate_sites(evaluation_sites, bases, config["support_rank"])
    summaries = {str(layer): {basis: aggregate(records, layer, basis) for basis in bases} for layer in config["source_layers"]}
    random_summary = {}
    for layer in config["source_layers"]:
        random_summary[str(layer)] = {}
        for metric in summaries[str(layer)]["native"]:
            values = [summaries[str(layer)][f"random_{i}"][metric] for i in range(config["random_rotations"])]
            random_summary[str(layer)][metric] = {"median": float(np.median(values)), "p95": float(np.quantile(values, .95))}
    _, vocabulary_labels, vocabulary_names = vocabulary_table(tokenizer)
    semantic = semantic_evaluation(vocabulary_scores, vocabulary_labels, vocabulary_names, bases, config["minimum_positive"])
    semantic_random_p95 = float(np.quantile([semantic[f"random_{i}"] for i in range(config["random_rotations"])], .95))
    evaluation_global = {
        "gradient": global_kurtosis(evaluation_gradient_obs, bases),
        "contribution": global_kurtosis(evaluation_contribution_obs, bases),
    }
    vocabulary_flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    discovery_event_labels, event_label_names = event_labels(discovery_sites, np.asarray(discovery_rows), vocabulary_flags, config["positions"], config["source_layers"])
    evaluation_event_labels, _ = event_labels(evaluation_sites, np.asarray(evaluation_rows), vocabulary_flags, config["positions"], config["source_layers"])
    discovery_event_scores = event_score_matrices(discovery_sites, bases)
    evaluation_event_scores = event_score_matrices(evaluation_sites, bases)
    event_semantics, event_selections = evaluate_event_semantics(discovery_event_scores, evaluation_event_scores, discovery_event_labels, evaluation_event_labels, event_label_names, config["minimum_positive"])
    event_random_p95 = {family: float(np.quantile([event_semantics[family][f"random_{i}"]["median_auc"] for i in range(config["random_rotations"])], .95)) for family in event_semantics}
    def beats_everywhere(basis, metrics):
        return all(summaries[str(layer)][basis][metric] >= random_summary[str(layer)][metric]["p95"] + .02 for layer in config["source_layers"] for metric in metrics)
    hypotheses = {
        "U0_gradient_basis_transfers": {"supported": beats_everywhere("upstream_gradient", ("gradient_top64_energy",))},
        "U1_contribution_basis_transfers": {"supported": beats_everywhere("upstream_contribution", ("contribution_top64_mass",))},
        "U2_upstream_coordinates_are_reproducible": {"supported": min(stability.values()) >= .80},
        "U3_one_basis_serves_both_upstream_metrics": {"supported": any(beats_everywhere(basis, ("gradient_top64_energy", "contribution_top64_mass")) for basis in ("upstream_gradient", "upstream_contribution"))},
        "U4_upstream_basis_is_distinct_from_output_basis": {"supported": max(overlap_with_output.values()) < .80},
        "U5_upstream_sparsity_implies_semantics": {"supported": max(semantic["upstream_gradient"], semantic["upstream_contribution"]) >= semantic_random_p95 + .03},
    }
    result = {"config": config, "hypotheses": hypotheses, "split_half_stability": stability, "overlap_with_output_varimax": overlap_with_output, "semantic_auc": semantic, "semantic_random_p95": semantic_random_p95, "discovery_summaries": discovery_summaries, "evaluation_summaries": summaries, "discovery_global_kurtosis": discovery_global, "evaluation_global_kurtosis": evaluation_global, "event_semantics": event_semantics, "event_semantic_random_p95": event_random_p95, "event_selections": {family: {basis: event_selections[family][basis] for basis in ("upstream_gradient", "upstream_contribution")} for family in event_selections}, "random_summary": random_summary, "records": records}
    torch.save({"gradient": gradient_q, "contribution": contribution_q, "output_varimax": output_q}, TENSOR_ROOT / "bilin18_upstream_objective_bases.pt")
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ["# Label-free discovery of upstream objective coordinates", "", "Two orthogonal bases are fitted without semantic labels on discovery FineWeb: one maximizes sparsity of exact Layer-0--3 gradient programs, and one maximizes sparsity of native activation-weighted contributions. Every cell and objective is RMS-normalized before fitting. Bases are frozen before disjoint FineWeb evaluation and compared with the output-varimax basis and 32 matched rotations of the identical rank-16 subspace.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Split-half coordinate overlap: gradient **{stability['gradient']:.3f}**, contribution **{stability['contribution']:.3f}**.", f"Overlap with output-varimax: gradient **{overlap_with_output['gradient']:.3f}**, contribution **{overlap_with_output['contribution']:.3f}**.", f"Vocabulary semantic AUROC: gradient **{semantic['upstream_gradient']:.3f}**, contribution **{semantic['upstream_contribution']:.3f}**, random p95 **{semantic_random_p95:.3f}**.", f"Fitted-axis global excess kurtosis, discovery/evaluation: gradient **{discovery_global['gradient']['upstream_gradient']['mean_excess_kurtosis']:.3f}/{evaluation_global['gradient']['upstream_gradient']['mean_excess_kurtosis']:.3f}**; contribution **{discovery_global['contribution']['upstream_contribution']['mean_excess_kurtosis']:.3f}/{evaluation_global['contribution']['upstream_contribution']['mean_excess_kurtosis']:.3f}**.", "", "![Upstream basis discovery](bilin18_upstream_objective_basis_discovery.png)", "", "| Layer | Basis | Discovery gradient/contribution | Evaluation gradient/contribution |", "|---:|---|---:|---:|"]
    for layer in config["source_layers"]:
        for basis in ("native", "output_varimax", "upstream_gradient", "upstream_contribution"):
            row = summaries[str(layer)][basis]; discovery_row = discovery_summaries[str(layer)][basis]; lines.append(f"| {layer} | {basis} | {discovery_row['gradient_top64_energy']:.3f}/{discovery_row['contribution_top64_mass']:.3f} | {row['gradient_top64_energy']:.3f}/{row['contribution_top64_mass']:.3f} |")
        lines.append(f"| {layer} | random p95 | — | {random_summary[str(layer)]['gradient_top64_energy']['p95']:.3f}/{random_summary[str(layer)]['contribution_top64_mass']['p95']:.3f} |")
    lines += ["", "## Post-hoc rare-event audit", "", "Each axis receives a per-site score equal to its largest native-channel magnitude divided by its RMS channel magnitude. Simple current-token, next-token, absolute-position, and source-layer predicates are selected on discovery rows only and frozen for evaluation.", "", "| Event family | Gradient-basis AUROC | Contribution-basis AUROC | Random p95 |", "|---|---:|---:|---:|"]
    for family in ("gradient", "contribution"):
        lines.append(f"| {family} | {event_semantics[family]['upstream_gradient']['median_auc']:.3f} | {event_semantics[family]['upstream_contribution']['median_auc']:.3f} | {event_random_p95[family]:.3f} |")
    lines += ["", "### Selected gradient-axis event labels", "", "| Axis | Gradient fit | Contribution fit |", "|---:|---|---|"]
    for axis in range(config["rank"]):
        left = event_selections["gradient"]["upstream_gradient"][axis]; right = event_selections["contribution"]["upstream_contribution"][axis]
        lines.append(f"| {axis} | `{left['label']}` ({event_semantics['gradient']['upstream_gradient']['mode_aucs'][axis]:.2f}) | `{right['label']}` ({event_semantics['contribution']['upstream_contribution']['mode_aucs'][axis]:.2f}) |")
    lines += ["", "## Guardrails", "", "This is an unsupervised coordinate search inside a pre-existing output subspace, not discovery of the subspace itself. Transfer and split-half stability distinguish reusable axes from a discovery-set sparsity fit. Semantic labels are evaluated only after all rotations are frozen. Global kurtosis can reflect rare context-channel events while per-example channel support remains broad; these are different definitions of sparsity."]
    REPORT.write_text("\n".join(lines) + "\n")
    x = np.arange(len(config["source_layers"])); fig, plots = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for basis in ("native", "output_varimax", "upstream_gradient", "upstream_contribution"):
        plots[0].plot(x, [summaries[str(layer)][basis]["gradient_top64_energy"] for layer in config["source_layers"]], marker="o", label=basis)
        plots[1].plot(x, [summaries[str(layer)][basis]["contribution_top64_mass"] for layer in config["source_layers"]], marker="o", label=basis)
    plots[0].plot(x, [random_summary[str(layer)]["gradient_top64_energy"]["p95"] for layer in config["source_layers"]], "k--", label="random p95")
    plots[1].plot(x, [random_summary[str(layer)]["contribution_top64_mass"]["p95"] for layer in config["source_layers"]], "k--", label="random p95")
    for plot in plots: plot.set_xticks(x, config["source_layers"]); plot.set_xlabel("Source layer"); plot.grid(alpha=.2); plot.legend()
    plots[0].set(ylabel="Top-64 squared-gradient fraction", title="Held-out gradient compactness")
    plots[1].set(ylabel="Top-64 contribution mass", title="Held-out native-contribution compactness")
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "split_half_stability": stability, "overlap_with_output_varimax": overlap_with_output, "semantic_auc": {key: semantic[key] for key in ("native", "output_varimax", "upstream_gradient", "upstream_contribution")}, "semantic_random_p95": semantic_random_p95, "discovery_global_kurtosis": {family: {basis: values[basis] for basis in ("native", "upstream_gradient", "upstream_contribution")} for family, values in discovery_global.items()}, "evaluation_global_kurtosis": {family: {basis: values[basis] for basis in ("native", "upstream_gradient", "upstream_contribution")} for family, values in evaluation_global.items()}, "event_semantics": {family: {basis: event_semantics[family][basis]["median_auc"] for basis in ("upstream_gradient", "upstream_contribution")} for family in event_semantics}, "event_random_p95": event_random_p95, "summaries": {layer: {basis: summaries[layer][basis] for basis in ("native", "output_varimax", "upstream_gradient", "upstream_contribution")} for layer in summaries}, "random_summary": random_summary}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
