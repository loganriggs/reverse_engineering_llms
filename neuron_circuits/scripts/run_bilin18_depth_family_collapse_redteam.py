#!/usr/bin/env python3
"""Test whether Layer-17 semantic families collapse or multiplex in one bottleneck."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_layer3_family_interaction_modes import evaluate_writes, output_basis
from run_bilin18_layer3_frozen_channel_transfer import channel_state, recover_configs
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_depth_family_collapse_redteam.json")
OUTPUT = TENSOR_ROOT / "depth_family_collapse_redteam.json"
REPORT = TENSOR_ROOT / "DEPTH_FAMILY_COLLAPSE_REDTEAM.md"
FIGURE = TENSOR_ROOT / "depth_family_collapse_redteam.png"


def overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.T @ right).square().sum() / min(left.shape[1], right.shape[1]))


def pair_balanced(records: list[dict], getter) -> float:
    groups = defaultdict(list)
    for record in records:
        groups[f"{record['source_id']}->{record['target_id']}"].append(float(getter(record)))
    return median([median(values) for values in groups.values()])


def projection_energy(write: torch.Tensor, basis: torch.Tensor) -> float:
    return float((write @ basis).square().sum() / write.square().sum().clamp_min(1e-30))


def rotation_invariant_metric(block, basis: torch.Tensor, rank: int) -> torch.Tensor:
    metric = torch.zeros(
        basis.shape[0], basis.shape[0], device=basis.device, dtype=torch.float32,
    )
    for axis in range(basis.shape[1]):
        coefficients = block.mlp.Down.weight.float().T @ basis[:, axis]
        matrix = block.mlp.Left.weight.float().T @ (
            coefficients[:, None] * block.mlp.Right.weight.float()
        )
        matrix = 0.5 * (matrix + matrix.T)
        metric += matrix @ matrix
    values, vectors = torch.linalg.eigh(metric)
    return vectors[:, values.argsort(descending=True)[:rank]]


def leave_one_pair_out(
    pair_writes: dict[str, dict[str, torch.Tensor]], families: list[str], rank: int,
) -> dict:
    gaps = []
    rows = []
    inclusive = {
        family: output_basis(list(pair_writes[family].values()), rank)
        for family in families
    }
    for family in families:
        for pair, write in pair_writes[family].items():
            training = [value for other, value in pair_writes[family].items() if other != pair]
            correct = output_basis(training, rank)
            correct_energy = projection_energy(write, correct)
            wrong = {
                other: projection_energy(write, inclusive[other])
                for other in families if other != family
            }
            best_wrong = max(wrong.values())
            gaps.append(correct_energy - best_wrong)
            rows.append({
                "family": family, "pair": pair, "correct": correct_energy,
                "best_wrong": best_wrong, "gap": correct_energy - best_wrong,
            })
    return {"median_gap": median(gaps), "rows": rows}


def coordinate_accuracy(
    pair_writes: dict[str, dict[str, torch.Tensor]], global_basis: torch.Tensor,
    families: list[str], rng: np.random.Generator, replicates: int,
) -> dict:
    vectors, labels = [], []
    for family in families:
        for write in pair_writes[family].values():
            vector = (write @ global_basis).flatten().cpu().numpy()
            vector /= max(np.linalg.norm(vector), 1e-30)
            vectors.append(vector)
            labels.append(family)
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    def score(candidate_labels: np.ndarray) -> float:
        predictions = []
        for index in range(len(vectors)):
            centroids = {}
            for family in families:
                selected = vectors[(candidate_labels == family) & (np.arange(len(vectors)) != index)]
                centroid = selected.mean(0)
                centroid /= max(np.linalg.norm(centroid), 1e-30)
                centroids[family] = centroid
            predictions.append(max(families, key=lambda family: vectors[index] @ centroids[family]))
        return float(np.mean(np.asarray(predictions) == candidate_labels))

    observed = score(labels)
    nulls = [score(rng.permutation(labels)) for _ in range(replicates)]
    permutation_p = (1 + sum(value >= observed for value in nulls)) / (replicates + 1)
    return {
        "coordinate_accuracy": observed,
        "permutation_p": permutation_p,
        "null_mean": float(np.mean(nulls)),
        "null_95": float(np.quantile(nulls, .95)),
    }


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    family_config = json.loads(Path(config["source_config"]).read_text())
    lopo_config = json.loads(Path(family_config["source_config"]).read_text())
    frozen_config = json.loads(Path(lopo_config["source_config"]).read_text())
    fixed, parent = recover_configs(frozen_config)
    native = json.loads(Path(config["native_results"]).read_text())
    depth_arrays = np.load(config["depth_arrays"])
    unseen = json.loads(Path(config["unseen_results"]).read_text())
    semantic_config = json.loads(Path(config["semantic_config"]).read_text())
    families = list(parent["families"])
    layers = config["layers"]
    rng = np.random.default_rng(config["seed"])
    discovery = [
        record for record in native["semantic_records"]
        if record["corpus"] == frozen_config["discovery_corpus"]
        and record["context_index"] == frozen_config["discovery_context_index"]
    ]
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")
        for corpus, spec in fixed["evaluation"].items()
    }
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()

    pair_writes = {
        layer: {family: {} for family in families} for layer in layers
    }
    for record in discovery:
        spec = fixed["evaluation"][record["corpus"]]
        row = arrays[record["corpus"]][spec["start"] + record["row"]]
        raw = np.asarray(
            row[record["source_start"]:record["source_start"] + fixed["sequence_length"]],
            dtype=np.int64,
        ).copy()
        changed = raw.copy()
        changed[fixed["edit_position"]] = int(record["pair"].split("->")[1])
        base = forward_layers(model, torch.as_tensor(raw, device="cuda")[None], last_layer=max(layers))
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None], last_layer=max(layers),
        )
        for layer in layers:
            delta = counterfactual[layer]["mlp_write"].float() - base[layer]["mlp_write"].float()
            pair_writes[layer][record["family"]][record["pair"]] = delta[
                0, fixed["edit_position"]:fixed["causal_position"] + 1
            ]
        print("discovery", record["family"], record["pair"], flush=True)

    layer_rows = {}
    for layer in layers:
        block = model.blocks[layer]
        family_bases = {
            family: torch.from_numpy(depth_arrays[f"layer{layer}_{family}_basis"]).cuda()
            for family in families
        }
        output_family_overlap = [
            overlap(family_bases[left], family_bases[right])
            for left, right in combinations(families, 2)
        ]
        input_bases = {
            family: rotation_invariant_metric(
                block, family_bases[family], config["input_rank"],
            )
            for family in families
        }
        input_family_overlap = [
            overlap(input_bases[left], input_bases[right])
            for left, right in combinations(families, 2)
        ]
        _, global_vectors = output_map_eigensystem(
            block.mlp.Down.weight.float(), block.mlp.Left.weight.float(),
            block.mlp.Right.weight.float(),
        )
        loo = leave_one_pair_out(pair_writes[layer], families, 4)
        coordinates = coordinate_accuracy(
            pair_writes[layer], global_vectors[:, :4], families, rng,
            config["permutation_replicates"],
        )
        layer_rows[str(layer)] = {
            "output_family_overlap": output_family_overlap,
            "median_output_family_overlap": median(output_family_overlap),
            "input_family_overlap": input_family_overlap,
            "median_input_family_overlap": median(input_family_overlap),
            "leave_one_pair_out": loo,
            **coordinates,
        }
        print("layer summary", layer, flush=True)

    # Finite Layer-17 unseen-word substitutions test whether family labels matter causally.
    corpus_arrays = {
        corpus: np.asarray(
            np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]],
            dtype=np.int64,
        )
        for corpus, spec in semantic_config["corpora"].items()
        if corpus in ("fineweb", "tinystories")
    }
    counts = defaultdict(int)
    panel = []
    for record in unseen["records"]:
        key = (record["corpus"], record["family"], record["source_id"], record["target_id"])
        if counts[key] < config["unseen_contexts_per_pair"]:
            panel.append(record)
            counts[key] += 1
    late_bases = {
        family: torch.from_numpy(depth_arrays[f"layer17_{family}_basis"]).cuda()
        for family in families
    }
    causal_records = []
    for record in panel:
        raw = corpus_arrays[record["corpus"]][
            record["row"],
            record["source_start"]:record["source_start"] + 64,
        ].copy()
        changed = raw.copy()
        changed[47] = record["target_id"]
        base = forward_layers(model, torch.as_tensor(raw, device="cuda")[None], last_layer=17)[17]
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None], last_layer=17,
        )[17]
        state = channel_state(model, 17, base, counterfactual, 63)
        target = state["target_write"].float()
        family = record["family"]
        correct_basis = late_bases[family]
        writes = {
            "full_mlp": target,
            "family_exact": (target @ correct_basis) @ correct_basis.T,
        }
        wrong_family = []
        for wrong in families:
            if wrong == family:
                continue
            name = f"wrong_family:{wrong}"
            basis = late_bases[wrong]
            writes[name] = (target @ basis) @ basis.T
            wrong_family.append(name)
        effects = evaluate_writes(model, 17, base, counterfactual, writes, 63)
        causal_records.append({
            **{key: record[key] for key in ("corpus", "family", "source_id", "target_id")},
            "family_full_mlp_fidelity": effects["family_exact"]["full_mlp_fidelity"],
            "best_wrong_family_fidelity": max(
                effects[name]["family_patch_fidelity"] for name in wrong_family
            ),
        })
        print("late causal", record["corpus"], family, flush=True)

    causal_summary = {}
    for corpus in ("fineweb", "tinystories"):
        selected = [record for record in causal_records if record["corpus"] == corpus]
        family_summary = {}
        for family in families:
            family_selected = [record for record in selected if record["family"] == family]
            family_summary[family] = {
                "pairs": len({(r["source_id"], r["target_id"]) for r in family_selected}),
                "best_wrong_family_fidelity": pair_balanced(
                    family_selected, lambda r: r["best_wrong_family_fidelity"],
                ),
                "family_full_mlp_fidelity": pair_balanced(
                    family_selected, lambda r: r["family_full_mlp_fidelity"],
                ),
            }
        causal_summary[corpus] = {
            "pairs": len({(r["source_id"], r["target_id"]) for r in selected}),
            "best_wrong_family_fidelity": pair_balanced(
                selected, lambda r: r["best_wrong_family_fidelity"],
            ),
            "family_full_mlp_fidelity": pair_balanced(
                selected, lambda r: r["family_full_mlp_fidelity"],
            ),
            "families": family_summary,
        }

    hypotheses = {
        "C1_late_family_output_spaces_collapse": {
            "supported": layer_rows["17"]["median_output_family_overlap"] >= .50
            and all(
                layer_rows["17"]["median_output_family_overlap"]
                - layer_rows[str(layer)]["median_output_family_overlap"] >= .40
                for layer in layers if layer != 17
            ),
        },
        "C2_late_family_input_operators_also_collapse": {
            "supported": layer_rows["17"]["median_input_family_overlap"] >= .50,
        },
        "C3_leave_one_pair_out_output_specificity_disappears_late": {
            "supported": layer_rows["17"]["leave_one_pair_out"]["median_gap"] <= .10
            and any(
                layer_rows[str(layer)]["leave_one_pair_out"]["median_gap"] >= .10
                for layer in layers if layer != 17
            ),
        },
        "C4_shared_late_coordinates_retain_family_information": {
            "supported": layer_rows["17"]["coordinate_accuracy"] >= .50
            and layer_rows["17"]["permutation_p"] <= .05,
        },
        "C5_wrong_late_family_spaces_are_causally_substitutable": {
            "supported": all(
                causal_summary[corpus]["best_wrong_family_fidelity"] >= .80
                for corpus in causal_summary
            ),
        },
        "C6_late_shared_patch_is_behaviorally_substantial": {
            "supported": all(
                causal_summary[corpus]["family_full_mlp_fidelity"] >= .70
                for corpus in causal_summary
            ),
        },
        "C7_input_separation_survives_output_collapse": {
            "supported": layer_rows["17"]["median_output_family_overlap"]
            - layer_rows["17"]["median_input_family_overlap"] >= .30,
        },
        "C8_late_global_coordinates_are_more_family_specific_than_early": {
            "supported": all(
                layer_rows["17"]["coordinate_accuracy"]
                - layer_rows[str(layer)]["coordinate_accuracy"] >= .10
                for layer in layers if layer != 17
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "layers": layer_rows,
        "causal_summary": causal_summary,
        "causal_records": causal_records,
    }, indent=2))

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(layers))
    width = .36
    axes[0].bar(x - width / 2, [layer_rows[str(layer)]["median_output_family_overlap"] for layer in layers], width, label="output space")
    axes[0].bar(x + width / 2, [layer_rows[str(layer)]["median_input_family_overlap"] for layer in layers], width, label="input metric")
    axes[0].set_xticks(x, layers)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Median cross-family overlap")
    axes[0].set_title("Do family operators collapse?")
    axes[0].legend()
    axes[1].bar(x, [layer_rows[str(layer)]["leave_one_pair_out"]["median_gap"] for layer in layers])
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x, layers)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Correct minus best-wrong projection")
    axes[1].set_title("Leave-one-pair-out specificity")
    axes[2].bar(x, [layer_rows[str(layer)]["coordinate_accuracy"] for layer in layers])
    axes[2].plot(x, [layer_rows[str(layer)]["null_95"] for layer in layers], "r--o", label="permutation 95th")
    axes[2].set_xticks(x, layers)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("LOPO family accuracy")
    axes[2].set_title("Family information inside global top-4")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    lines = [
        "# Red-team of semantic-family collapse at Layer 17",
        "",
        "The depth-conditioned result makes five semantic family spaces nearly coincide with the same late global top-four output bottleneck. This follow-up tests whether family distinctions survive in rotation-invariant input operators, leave-one-pair-out write specificity, or coordinate trajectories inside that shared bottleneck. Finite unseen-word interventions ask whether a wrong-family late subspace is causally substitutable.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Family collapse red-team](depth_family_collapse_redteam.png)",
        "",
        "| Layer | Output family overlap | Input family overlap | LOPO projection gap | Coordinate accuracy / null 95th | Permutation p |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in layers:
        row = layer_rows[str(layer)]
        lines.append(
            f"| {layer} | {row['median_output_family_overlap']:.3f} | "
            f"{row['median_input_family_overlap']:.3f} | "
            f"{row['leave_one_pair_out']['median_gap']:.3f} | "
            f"{row['coordinate_accuracy']:.3f} / {row['null_95']:.3f} | "
            f"{row['permutation_p']:.4f} |"
        )
    lines += [
        "",
        "| Unseen corpus | Pairs | Best wrong-family fidelity | Correct-family full-MLP fidelity |",
        "|---|---:|---:|---:|",
    ]
    for corpus, row in causal_summary.items():
        lines.append(
            f"| {corpus} | {row['pairs']} | {row['best_wrong_family_fidelity']:.3f} | "
            f"{row['family_full_mlp_fidelity']:.3f} |"
        )
    lines += [
        "",
        "| Unseen corpus / family | Pairs | Best wrong-family fidelity | Correct-family full-MLP fidelity |",
        "|---|---:|---:|---:|",
    ]
    for corpus, corpus_row in causal_summary.items():
        for family, row in corpus_row["families"].items():
            lines.append(
                f"| {corpus} / {family} | {row['pairs']} | "
                f"{row['best_wrong_family_fidelity']:.3f} | "
                f"{row['family_full_mlp_fidelity']:.3f} |"
            )
    lines += [
        "",
        "Output-subspace identity, input-operator identity, and coordinate multiplexing are distinct possibilities. High family classification inside a shared output space would support multiplexing; low classification plus substitutable wrong-family patches would support a generic bottleneck. Neither outcome says the upstream model lacks semantic distinctions—it localizes where those distinctions cease to correspond to separate native subspaces.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
