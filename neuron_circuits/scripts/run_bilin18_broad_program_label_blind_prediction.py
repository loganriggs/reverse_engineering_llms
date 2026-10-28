#!/usr/bin/env python3
"""Can broad semantic circuits be found without semantic output labels?

Three discovery rankings are compared at each already-frozen construction site:

1. static random-function scale from the weights alone;
2. native head-write change on the paired inputs, without answer labels;
3. full-vocabulary logit RMS caused by that write, without answer labels.

The selected top-three heads are then evaluated against the held-out semantic
contrast and every equally sized within-layer head subset.  The script also
asks whether the semantic programs are enriched in the weight-motif graph that
was frozen before the broad task battery existed.
"""

from __future__ import annotations

from collections import defaultdict
import itertools
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from redteam_bilin18_broad_regime_construction import reconstruct_selected
from run_bilin18_broad_regime_construction import (
    head_deltas, head_program_batch, length_batches, metric, repeat_batch, suffix, trace,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_unsupervised_weight_circuit_atlas import random_function_signatures


CONFIG = Path("configs/bilin18_broad_program_label_blind_prediction.json")
OUTPUT = TENSOR_ROOT / "bilin18_broad_program_label_blind_prediction.json"
REPORT = TENSOR_ROOT / "BILIN18_BROAD_PROGRAM_LABEL_BLIND_PREDICTION.md"
FIGURE = TENSOR_ROOT / "bilin18_broad_program_label_blind_prediction.png"


def top_subset(scores, size):
    return sorted(np.argsort(-np.asarray(scores, dtype=np.float64))[:size].tolist())


def percentile(values, observed):
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values <= observed))


def motif_statistics(program_nodes, components, layers, heads, subset_size, replicates, seed):
    """Matched null preserves the number of selected heads at every layer."""
    membership = {}
    for component_index, component in enumerate(components):
        for node in component:
            membership[node] = component_index

    def score(nodes):
        nodes = list(nodes)
        member_count = sum(node in membership for node in nodes)
        closure = sum(
            membership.get(a, -1) >= 0 and membership.get(a) == membership.get(b)
            for index, a in enumerate(nodes) for b in nodes[index + 1:]
        )
        return member_count, closure

    observed_members, observed_closure = score(program_nodes)
    rng = np.random.default_rng(seed)
    null_members = np.empty(replicates, dtype=np.int16)
    null_closure = np.empty(replicates, dtype=np.int16)
    for replicate in range(replicates):
        nodes = []
        for layer in layers:
            chosen = rng.choice(heads, size=subset_size, replace=False)
            nodes.extend(layer * heads + int(head) for head in chosen)
        null_members[replicate], null_closure[replicate] = score(nodes)
    return {
        "selected_heads": len(program_nodes),
        "motif_members": int(observed_members),
        "motif_member_fraction": float(observed_members / max(len(program_nodes), 1)),
        "matched_member_mean": float(null_members.mean()),
        "member_one_sided_p": float((1 + np.sum(null_members >= observed_members)) / (replicates + 1)),
        "component_pairs": int(observed_closure),
        "matched_component_pair_mean": float(null_closure.mean()),
        "component_pair_one_sided_p": float((1 + np.sum(null_closure >= observed_closure)) / (replicates + 1)),
    }


@torch.inference_mode()
def input_conditioned_scores_batch(model, pairs, layer):
    """State and output magnitudes for each head, without semantic contrasts."""
    size = len(pairs)
    ids = torch.as_tensor(
        [pair["base_ids"] for pair in pairs] + [pair["source_ids"] for pair in pairs],
        device=model.embedding.weight.device,
    )
    x0, traces, _ = trace(model, ids)
    row = traces[layer]
    delta = head_deltas(model, layer, row["details"], size).float()
    state_square = delta.square().mean(-1)

    base_state = row["output"][:size]
    carried = row["mixed"][size:, -1] - row["mixed"][:size, -1]
    mlp = row["mlp"][size:, -1] - row["mlp"][:size, -1]
    no_attention = base_state.clone()
    no_attention[:, -1] = base_state[:, -1] + carried + mlp
    states = [no_attention]
    for head in range(model.config.n_head):
        state = no_attention.clone()
        state[:, -1] += delta[:, head].to(state.dtype)
        states.append(state)
    logits = suffix(
        model, layer, torch.cat(states), repeat_batch(x0[:size], len(states)),
        repeat_batch(row["bus"][:size], len(states)),
    ).reshape(len(states), size, -1)
    output_square = (logits[1:] - logits[0:1]).float().square().mean(-1).T
    return state_square.cpu(), output_square.cpu(), float((delta.sum(1) - (
        row["attention"][size:, -1] - row["attention"][:size, -1]
    )).abs().max())


def aggregate_label_blind_scores(model, pairs, layer, batch_size):
    state, output = [], []
    maximum_error = 0.0
    for batch in length_batches(pairs, batch_size):
        state_square, output_square, error = input_conditioned_scores_batch(model, batch, layer)
        state.append(state_square); output.append(output_square)
        maximum_error = max(maximum_error, error)
    return {
        "state_norm": torch.cat(state).mean(0).sqrt().numpy(),
        "output_rms": torch.cat(output).mean(0).sqrt().numpy(),
    }, maximum_error


def evaluate_subsets(model, tokenizer, pairs, layer, named_subsets, chunk_size, batch_size):
    """Evaluate named and all equal-size subsets on held-out semantic effects."""
    size = len(next(iter(named_subsets.values())))
    subsets = list(itertools.combinations(range(model.config.n_head), size))
    unique = {tuple(sorted(value)) for value in named_subsets.values()}
    unique.update(subsets)
    names = {subset: "subset_" + "_".join(map(str, subset)) for subset in sorted(unique)}
    effects = defaultdict(list); targets = []
    for batch in length_batches(pairs, batch_size):
        batch_target = None
        ordered = list(names)
        for start in range(0, len(ordered), chunk_size):
            chunk = ordered[start:start + chunk_size]
            programs = {names[subset]: list(subset) for subset in chunk}
            programs["all"] = list(range(model.config.n_head))
            target, values, _ = head_program_batch(model, tokenizer, batch, layer, programs)
            if batch_target is None:
                batch_target = target
            for subset in chunk:
                effects[names[subset]].append(values[names[subset]])
        targets.append(batch_target)
    target = torch.cat(targets)
    subset_metrics = {
        subset: metric(target, torch.cat(effects[name]))
        for subset, name in names.items()
    }
    all_fidelity = [subset_metrics[subset]["energy_fidelity"] for subset in subsets]
    return {
        name: {
            "heads": list(sorted(heads)),
            **subset_metrics[tuple(sorted(heads))],
            "subset_percentile": percentile(all_fidelity, subset_metrics[tuple(sorted(heads))]["energy_fidelity"]),
        }
        for name, heads in named_subsets.items()
    }, {
        "subsets": len(subsets),
        "median": float(np.median(all_fidelity)),
        "p90": float(np.quantile(all_fidelity, .9)),
        "best": float(np.max(all_fidelity)),
    }


def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source"]).read_text())
    redteam = json.loads(Path(config["redteam_source"]).read_text())
    motif = json.loads(Path(config["motif_source"]).read_text())
    atlas = json.loads(Path(config["motif_atlas_source"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    selected = reconstruct_selected(tokenizer, source)
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)

    # Reproduce the already-frozen random-function representation solely to
    # recover per-head scale, which the atlas saved only as global quantiles.
    signature_a, scale_a = random_function_signatures(
        model, atlas["config"]["signature_probes"], atlas["config"]["seed_a"],
        atlas["config"]["relative_distances"],
    )
    del signature_a
    signature_b, scale_b = random_function_signatures(
        model, atlas["config"]["signature_probes"], atlas["config"]["seed_b"],
        atlas["config"]["relative_distances"],
    )
    del signature_b
    static_scale = np.sqrt(scale_a * scale_b).reshape(model.config.n_layer, model.config.n_head)

    material = [family for family, row in redteam["results"].items() if row["material_attention"]]
    results = {}; maximum_error = 0.0
    for family in material:
        layer = int(source["head_summary"][family]["site"])
        discovery = selected[family]["discovery"]
        heldout = selected[family]["heldout"]
        scores, error = aggregate_label_blind_scores(
            model, discovery, layer, source["config"]["batch_size"],
        )
        maximum_error = max(maximum_error, error)
        named = {
            "static_weight_scale": top_subset(static_scale[layer], config["subset_size"]),
            "input_state_norm": top_subset(scores["state_norm"], config["subset_size"]),
            "label_blind_output_rms": top_subset(scores["output_rms"], config["subset_size"]),
            "semantic_causal": sorted(source["head_summary"][family]["head_order"][:config["subset_size"]]),
        }
        evaluated, null = evaluate_subsets(
            model, tokenizer, heldout, layer, named, config["subset_chunk_size"],
            source["config"]["batch_size"],
        )
        results[family] = {
            "layer": layer,
            "discovery_pairs": len(discovery),
            "heldout_pairs": len(heldout),
            "rankings": {
                "static_weight_scale": np.argsort(-static_scale[layer]).tolist(),
                "input_state_norm": np.argsort(-scores["state_norm"]).tolist(),
                "label_blind_output_rms": np.argsort(-scores["output_rms"]).tolist(),
                "semantic_causal": source["head_summary"][family]["head_order"],
            },
            "scores": {
                "static_weight_scale": static_scale[layer].tolist(),
                "input_state_norm": scores["state_norm"].tolist(),
                "label_blind_output_rms": scores["output_rms"].tolist(),
            },
            "heldout": evaluated,
            "subset_null": null,
        }
        print(f"label-blind {family} L{layer} " + " ".join(
            f"{name}={row['energy_fidelity']:.3f}/{row['subset_percentile']:.3f}"
            for name, row in evaluated.items()
        ), flush=True)

    components = [
        [int(member["layer"]) * model.config.n_head + int(member["head"]) for member in component]
        for component in motif["bus_robust"]["components"]
    ]
    program_nodes = []
    program_layers = []
    for family in material:
        layer = results[family]["layer"]
        program_layers.append(layer)
        program_nodes.extend(
            layer * model.config.n_head + head
            for head in source["head_summary"][family]["head_order"][:config["subset_size"]]
        )
    motif_stats = motif_statistics(
        program_nodes, components, program_layers, model.config.n_head,
        config["subset_size"], config["motif_null_replicates"], config["seed"],
    )

    predictors = ("static_weight_scale", "input_state_norm", "label_blind_output_rms", "semantic_causal")
    means = {
        name: {
            "energy_fidelity": float(np.mean([row["heldout"][name]["energy_fidelity"] for row in results.values()])),
            "subset_percentile": float(np.mean([row["heldout"][name]["subset_percentile"] for row in results.values()])),
        }
        for name in predictors
    }
    state_pass = [family for family, row in results.items()
                  if row["heldout"]["input_state_norm"]["energy_fidelity"] >= config["minimum_energy_fidelity"]
                  and row["heldout"]["input_state_norm"]["subset_percentile"] >= config["minimum_subset_percentile"]]
    output_pass = [family for family, row in results.items()
                   if row["heldout"]["label_blind_output_rms"]["energy_fidelity"] >= config["minimum_energy_fidelity"]
                   and row["heldout"]["label_blind_output_rms"]["subset_percentile"] >= config["minimum_subset_percentile"]]
    static_above_median = [family for family, row in results.items()
                           if row["heldout"]["static_weight_scale"]["subset_percentile"] >= .5]
    best_blind = max(means["input_state_norm"]["energy_fidelity"], means["label_blind_output_rms"]["energy_fidelity"])
    hypotheses = {
        "P0_exact_recomposition": {"supported": maximum_error <= 1e-3, "maximum_head_sum_abs": maximum_error},
        "P1_frozen_weight_motifs_are_enriched": {"supported": min(motif_stats["member_one_sided_p"], motif_stats["component_pair_one_sided_p"]) <= .05, **motif_stats},
        "P2_static_weight_scale_predicts_programs": {"supported": len(static_above_median) >= 4 and means["static_weight_scale"]["subset_percentile"] >= .65, "families_above_median": static_above_median, "mean": means["static_weight_scale"]},
        "P3_input_conditioned_state_scale_predicts_programs": {"supported": len(state_pass) >= 4, "passing_families": state_pass, "mean": means["input_state_norm"]},
        "P4_label_blind_output_scale_predicts_programs": {"supported": len(output_pass) >= 4, "passing_families": output_pass, "mean": means["label_blind_output_rms"]},
        "P5_output_scale_improves_on_state_scale": {"supported": means["label_blind_output_rms"]["energy_fidelity"] - means["input_state_norm"]["energy_fidelity"] >= .05, "mean_fidelity_difference": means["label_blind_output_rms"]["energy_fidelity"] - means["input_state_norm"]["energy_fidelity"]},
        "P6_semantic_selection_adds_information": {"supported": means["semantic_causal"]["energy_fidelity"] - best_blind >= .05, "mean_fidelity_advantage": means["semantic_causal"]["energy_fidelity"] - best_blind},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {"config": config, "hypotheses": hypotheses, "motif_enrichment": motif_stats, "means": means, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    labels = {
        "static_weight_scale": "static weights", "input_state_norm": "input state",
        "label_blind_output_rms": "unlabeled output", "semantic_causal": "semantic causal",
    }
    lines = ["# Can broad circuits be found without task labels?", "",
             "The construction layer and semantic program were already frozen. This prospective red team asks whether the selected heads could instead have been found from an older label-blind weight atlas, static head scale, or the paired input computation with all answer labels hidden.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Family | Site | Static heads / fidelity / pct | State heads / fidelity / pct | Unlabeled-output heads / fidelity / pct | Semantic heads / fidelity / pct |", "|---|---:|---|---|---|---|"]
    for family, row in results.items():
        cells = []
        for name in predictors:
            value = row["heldout"][name]
            heads = ",".join(f"H{x}" for x in value["heads"])
            cells.append(f"{heads} / {value['energy_fidelity']:.3f} / {value['subset_percentile']:.3f}")
        lines.append(f"| {family} | L{row['layer']} | " + " | ".join(cells) + " |")
    lines += ["", "## Meaning", "",
              "Static weight scale asks which heads are generically large. Input-state scale sees which heads respond to the counterfactual pair but not what answer is desired. Unlabeled-output scale additionally sees which changed head writes survive the suffix into any vocabulary logits, but still receives no semantic contrast. Semantic-causal selection is the task-informed ceiling, not an unsupervised method.", "",
              "A failed static predictor does not falsify weight-based operator discovery: invocation is input-conditioned. A successful unlabeled predictor finds behaviorally active heads, but semantic labeling and signed causal validation remain separate problems."]
    REPORT.write_text("\n".join(lines) + "\n")

    families = list(results)
    x = np.arange(len(families)); width = .19
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for index, name in enumerate(predictors):
        axes[0].bar(x + (index - 1.5) * width, [results[f]["heldout"][name]["energy_fidelity"] for f in families], width, label=labels[name])
        axes[1].bar(x + (index - 1.5) * width, [results[f]["heldout"][name]["subset_percentile"] for f in families], width)
    axes[0].axhline(config["minimum_energy_fidelity"], color="black", ls="--", lw=1)
    axes[1].axhline(config["minimum_subset_percentile"], color="black", ls="--", lw=1)
    axes[0].set_ylabel("held-out energy fidelity"); axes[0].legend(ncol=2)
    axes[1].set_ylabel("percentile among 84 subsets"); axes[1].set_xticks(x, families, rotation=25, ha="right")
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
