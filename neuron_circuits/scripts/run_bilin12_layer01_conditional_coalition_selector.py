#!/usr/bin/env python3
"""Decompose the information needed to select Layer-0/1 native coalitions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_adaptive_chart_router import midpoint_product_jvp
from run_bilin12_layer01_dynamic_alignment_replication import (
    evaluate_deltas,
    final_product_delta,
    forward_layers_native,
    median,
    pair_cluster_interval,
    selected_delta,
)
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_conditional_coalition_selector.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_conditional_coalition_selector.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_CONDITIONAL_COALITION_SELECTOR.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_conditional_coalition_selector.png"

LEARNED = (
    "context", "edit", "additive", "additive_family",
    "interaction", "interaction_family", "interaction_shuffled",
)
ALL_METHODS = LEARNED + (
    "family_bank", "global_bank", "base_activation",
    "base_jvp", "midpoint_jvp", "endpoint_oracle",
)


def normalize_rows(values):
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def make_projection(input_dimension, output_dimension, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(input_dimension, output_dimension, generator=generator) / input_dimension ** .5


def feature_tables(context, source, edit, family, family_count, projections):
    context = normalize_rows(context.float())
    source = normalize_rows(source.float())
    edit = normalize_rows(edit.float())
    edit_pair = torch.cat((source, edit), -1)
    additive = torch.cat((context, edit_pair), -1)
    one_hot = F.one_hot(family.long(), family_count).float()
    interaction = (context @ projections[0]) * (edit_pair @ projections[1])
    return {
        "context": context,
        "edit": edit_pair,
        "additive": additive,
        "additive_family": torch.cat((additive, one_hot), -1),
        "interaction": torch.cat((additive, interaction), -1),
        "interaction_family": torch.cat((additive, interaction, one_hot), -1),
        "interaction_shuffled": torch.cat((additive, interaction), -1),
    }


def fit_dual_ridge(features, targets, relative_penalty):
    x_mean = features.mean(0, keepdim=True)
    x_scale = features.std(0, unbiased=False, keepdim=True).clamp_min(1e-4)
    z = (features - x_mean) / x_scale
    y_mean = targets.mean(0, keepdim=True)
    centered = targets - y_mean
    gram = z @ z.T
    penalty = relative_penalty * float(torch.trace(gram)) / max(len(gram), 1)
    dual = torch.linalg.solve(
        gram + penalty * torch.eye(len(gram), dtype=gram.dtype), centered,
    )
    return {"x_mean": x_mean, "x_scale": x_scale, "z": z, "y_mean": y_mean, "dual": dual}


def ridge_predict(fitted, features):
    z = (features - fitted["x_mean"]) / fitted["x_scale"]
    return fitted["y_mean"] + (z @ fitted["z"].T) @ fitted["dual"]


def cosine_rows(left, right):
    return (normalize_rows(left) * normalize_rows(right)).sum(-1)


def local_scores(vector, decoder):
    write = F.linear(vector, decoder)
    return (vector * (write @ decoder)).abs()


def jaccard(left, right):
    a, b = set(int(value) for value in left), set(int(value) for value in right)
    return len(a & b) / max(len(a | b), 1)


def calibration_pair_set(rows, families, count):
    selected = set()
    for family in families:
        pairs = sorted({row["pair"] for row in rows if row["family"] == family})
        selected.update(pairs[-count:])
    return selected


def shuffled_pair_indices(rows, seed):
    pairs = sorted({row["pair"] for row in rows})
    rng = np.random.default_rng(seed)
    shifted = pairs.copy()
    while True:
        rng.shuffle(shifted)
        if all(left != right for left, right in zip(pairs, shifted)):
            break
    mapping = dict(zip(pairs, shifted))
    by_pair = {pair: [index for index, row in enumerate(rows) if row["pair"] == pair] for pair in pairs}
    counters = {pair: 0 for pair in pairs}
    indices = []
    for row in rows:
        target_pair = mapping[row["pair"]]
        candidates = by_pair[target_pair]
        indices.append(candidates[counters[target_pair] % len(candidates)])
        counters[target_pair] += 1
    return indices, mapping


def collect_discovery(model, rows, pairs, config, source_config, layers, family_to_index):
    records = []
    rng = np.random.default_rng(config["window_seed"])
    edit_position, position = source_config["edit_position"], source_config["prediction_position"]
    for family, family_pairs in pairs.items():
        for source_token, target_token in family_pairs:
            windows = select_windows(
                rows, source_token, edit_position, position,
                config["discovery_contexts_per_pair"], rng,
            )
            for context_index, (row_index, start) in enumerate(windows):
                raw = np.asarray(rows[row_index, start:start + source_config["sequence_length"]], dtype=np.int64).copy()
                changed = raw.copy(); changed[edit_position] = target_token
                base_ids = torch.as_tensor(raw, device="cuda")[None]
                changed_ids = torch.as_tensor(changed, device="cuda")[None]
                with torch.no_grad():
                    base = forward_layers_native(model, base_ids, max(layers))
                    dimension = model.config.n_embd
                    base_x0 = F.rms_norm(model.embedding(base_ids), (dimension,)).float()
                    changed_x0 = F.rms_norm(model.embedding(changed_ids), (dimension,)).float()
                row = {
                    "family": family,
                    "family_index": family_to_index[family],
                    "pair": f"{source_token}->{target_token}",
                    "context_index": context_index,
                    "source": base_x0[0, edit_position].cpu(),
                    "edit": (changed_x0 - base_x0)[0, edit_position].cpu(),
                    "layers": {},
                }
                for layer in layers:
                    tangent, _, _ = midpoint_product_jvp(
                        model, base_ids, changed_ids, layer, position, .5,
                    )
                    context = F.rms_norm(
                        base[layer]["pre"][0, position].float(), (dimension,),
                    )
                    row["layers"][str(layer)] = {
                        "context": context.cpu(),
                        "target": normalize_rows(tangent[None]).cpu()[0],
                    }
                records.append(row)
            print("selector discovery", family, source_token, target_token, flush=True)
    return records


def tensors(records, layer):
    return (
        torch.stack([row["layers"][str(layer)]["context"] for row in records]),
        torch.stack([row["source"] for row in records]),
        torch.stack([row["edit"] for row in records]),
        torch.tensor([row["family_index"] for row in records]),
        torch.stack([row["layers"][str(layer)]["target"] for row in records]),
    )


def main():
    red = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(red["source_config"]).read_text())
    discovery_spec = json.loads(Path(source_config["discovery_pair_config"]).read_text())
    evaluation_spec = json.loads(Path(source_config["evaluation_pair_config"]).read_text())
    families = list(discovery_spec["families"])
    family_to_index = {family: index for index, family in enumerate(families)}
    discovery_pairs = {
        f"{source}->{target}" for pairs in discovery_spec["families"].values()
        for source, target in pairs
    }
    evaluation_pairs = {
        f"{source}->{target}" for pairs in evaluation_spec["new_families"].values()
        for source, target in pairs
    }

    model = load_snapshot_model(Path(source_config["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in source_config["evaluation"].items()
    }
    layers = source_config["layers"]
    discovery = collect_discovery(
        model, arrays[source_config["discovery_corpus"]], discovery_spec["families"],
        red, source_config, layers, family_to_index,
    )

    projections = {}
    fitted = {}
    calibration = {}
    family_banks = {}
    global_banks = {}
    shuffle_indices, shuffle_mapping = shuffled_pair_indices(discovery, red["shuffle_seed"])
    calibration_pairs = calibration_pair_set(
        discovery, families, red["calibration_pairs_per_family"],
    )
    training_indices = [index for index, row in enumerate(discovery) if row["pair"] not in calibration_pairs]
    calibration_indices = [index for index, row in enumerate(discovery) if row["pair"] in calibration_pairs]

    for layer in layers:
        context, source, edit, family, target = tensors(discovery, layer)
        projections[layer] = (
            make_projection(context.shape[1], red["interaction_features"], red["feature_seed"] + layer * 2),
            make_projection(source.shape[1] + edit.shape[1], red["interaction_features"], red["feature_seed"] + layer * 2 + 1),
        )
        feature = feature_tables(
            context, source, edit, family, len(families), projections[layer],
        )
        fitted[layer] = {}
        calibration[str(layer)] = {}
        for method in LEARNED[:-1]:
            scores = {}
            for penalty in red["ridge_relative_penalties"]:
                model_fit = fit_dual_ridge(
                    feature[method][training_indices], target[training_indices], penalty,
                )
                predicted = ridge_predict(model_fit, feature[method][calibration_indices])
                scores[str(penalty)] = float(cosine_rows(predicted, target[calibration_indices]).median())
            selected_penalty = max(red["ridge_relative_penalties"], key=lambda value: scores[str(value)])
            fitted[layer][method] = fit_dual_ridge(feature[method], target, selected_penalty)
            calibration[str(layer)][method] = {
                "selected_penalty": selected_penalty,
                "median_cosine_by_penalty": scores,
            }
        interaction_penalty = calibration[str(layer)]["interaction"]["selected_penalty"]
        fitted[layer]["interaction_shuffled"] = fit_dual_ridge(
            feature["interaction_shuffled"], target[shuffle_indices], interaction_penalty,
        )
        decoder = model.blocks[layer].mlp.Down.weight.float().cpu()
        discovery_scores = torch.stack([local_scores(row, decoder) for row in target])
        global_banks[layer] = discovery_scores.mean(0).topk(red["rank"]).indices.cuda()
        family_banks[layer] = {
            name: discovery_scores[family == family_to_index[name]].mean(0).topk(red["rank"]).indices.cuda()
            for name in families
        }
        print("selector fitted layer", layer, flush=True)

    edit_position, position = source_config["edit_position"], source_config["prediction_position"]
    records = {str(layer): [] for layer in layers}
    full_patch_fidelities = {str(layer): {corpus: [] for corpus in arrays} for layer in layers}
    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        rng = np.random.default_rng(red["window_seed"] + 1000 + corpus_index)
        for family, family_pairs in evaluation_spec["new_families"].items():
            for source_token, target_token in family_pairs:
                windows = select_windows(
                    rows, source_token, edit_position, position,
                    red["evaluation_contexts_per_pair"], rng,
                )
                for context_index, (row_index, start) in enumerate(windows):
                    raw = np.asarray(rows[row_index, start:start + source_config["sequence_length"]], dtype=np.int64).copy()
                    changed = raw.copy(); changed[edit_position] = target_token
                    base_ids = torch.as_tensor(raw, device="cuda")[None]
                    changed_ids = torch.as_tensor(changed, device="cuda")[None]
                    with torch.no_grad():
                        base = forward_layers_native(model, base_ids, max(layers))
                        counter = forward_layers_native(model, changed_ids, max(layers))
                        dimension = model.config.n_embd
                        base_x0 = F.rms_norm(model.embedding(base_ids), (dimension,)).float()
                        changed_x0 = F.rms_norm(model.embedding(changed_ids), (dimension,)).float()
                    source_row = base_x0[0, edit_position].cpu()[None]
                    edit_row = (changed_x0 - base_x0)[0, edit_position].cpu()[None]
                    family_row = torch.tensor([family_to_index[family]])
                    for layer in layers:
                        decoder_gpu = model.blocks[layer].mlp.Down.weight.float()
                        decoder_cpu = decoder_gpu.cpu()
                        full = final_product_delta(model, layer, base[layer], counter[layer], position)
                        midpoint, _, _ = midpoint_product_jvp(
                            model, base_ids, changed_ids, layer, position, .5,
                        )
                        base_jvp, _, _ = midpoint_product_jvp(
                            model, base_ids, changed_ids, layer, position, 0.0,
                        )
                        state = F.rms_norm(base[layer]["pre"][0, position].float(), (dimension,))
                        base_product = (
                            model.blocks[layer].mlp.Left(state)
                            * model.blocks[layer].mlp.Right(state)
                        )
                        feature = feature_tables(
                            state.cpu()[None], source_row, edit_row, family_row,
                            len(families), projections[layer],
                        )
                        predicted = {
                            method: ridge_predict(fitted[layer][method], feature[method])[0].cuda()
                            for method in LEARNED
                        }
                        vectors = {
                            **predicted,
                            "base_activation": base_product,
                            "base_jvp": base_jvp,
                            "midpoint_jvp": midpoint,
                            "endpoint_oracle": full,
                        }
                        endpoint_indices = local_scores(full, decoder_gpu).topk(red["rank"]).indices
                        indices = {
                            method: local_scores(vector, decoder_gpu).topk(red["rank"]).indices
                            for method, vector in vectors.items()
                        }
                        indices["family_bank"] = family_banks[layer][family]
                        indices["global_bank"] = global_banks[layer]
                        deltas = {
                            method: selected_delta(full, indices[method]) for method in ALL_METHODS
                        }
                        deltas["full_patch"] = full
                        with torch.no_grad():
                            effects = evaluate_deltas(model, layer, base[layer], deltas, full, position)
                        full_patch_fidelities[str(layer)][corpus].append(effects["full_patch"])
                        target_unit = normalize_rows(midpoint.cpu()[None])[0]
                        row_cosines = {
                            method: float(cosine_rows(predicted[method].cpu()[None], target_unit[None])[0])
                            for method in LEARNED
                        }
                        records[str(layer)].append({
                            "corpus": corpus,
                            "family": family,
                            "pair": f"{source_token}->{target_token}",
                            "context": context_index,
                            "fidelity": {method: effects[method] for method in ALL_METHODS},
                            "support_jaccard": {
                                method: jaccard(indices[method].cpu().tolist(), endpoint_indices.cpu().tolist())
                                for method in ALL_METHODS
                            },
                            "midpoint_cosine": row_cosines,
                        })
                print("selector evaluation", corpus, family, source_token, target_token, flush=True)

    summaries = {str(layer): {} for layer in layers}
    for layer in layers:
        for corpus_index, corpus in enumerate(arrays):
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            summary = {
                "fidelity": {method: median([row["fidelity"][method] for row in selected]) for method in ALL_METHODS},
                "support_jaccard": {method: median([row["support_jaccard"][method] for row in selected]) for method in ALL_METHODS},
                "midpoint_cosine": {method: median([row["midpoint_cosine"][method] for row in selected]) for method in LEARNED},
                "full_patch_fidelity": median(full_patch_fidelities[str(layer)][corpus]),
            }
            comparisons = (
                ("interaction_minus_additive", lambda row: row["fidelity"]["interaction"] - row["fidelity"]["additive"], 0),
                ("family_gain", lambda row: row["fidelity"]["interaction_family"] - row["fidelity"]["interaction"], 10000),
                ("interaction_minus_one_sided", lambda row: row["fidelity"]["interaction"] - max(row["fidelity"]["context"], row["fidelity"]["edit"]), 20000),
                ("midpoint_minus_interaction", lambda row: row["fidelity"]["midpoint_jvp"] - row["fidelity"]["interaction"], 30000),
                ("interaction_minus_shuffled", lambda row: row["fidelity"]["interaction"] - row["fidelity"]["interaction_shuffled"], 40000),
            )
            for name, function, offset in comparisons:
                value, interval = pair_cluster_interval(
                    selected, function, red["bootstrap_samples"],
                    red["bootstrap_seed"] + offset + layer * 100 + corpus_index,
                )
                summary[f"{name}_pair_median"] = value
                summary[f"{name}_pair_ci"] = interval
            summaries[str(layer)][corpus] = summary

    cell = lambda layer, corpus: summaries[str(layer)][corpus]
    disjoint = not bool(discovery_pairs & evaluation_pairs)
    shuffle_valid = all(left != right for left, right in shuffle_mapping.items())
    numerical = disjoint and shuffle_valid and all(
        cell(layer, corpus)["full_patch_fidelity"] >= .95
        for layer in layers for corpus in arrays
    )
    hypotheses = {
        "C0_pair_splits_and_numerics_are_valid": {
            "supported": numerical,
            "pair_sets_disjoint": disjoint,
            "shuffle_has_no_fixed_pair": shuffle_valid,
        },
        "C1_context_edit_interaction_beats_additive_selection": {"supported": all(
            cell(layer, corpus)["interaction_minus_additive_pair_median"] >= .05
            and cell(layer, corpus)["interaction_minus_additive_pair_ci"][0] > 0
            for layer in layers for corpus in arrays
        )},
        "C2_human_family_labels_improve_the_interaction_selector": {"supported": all(
            cell(layer, corpus)["family_gain_pair_median"] >= .03
            for layer in layers for corpus in arrays
        )},
        "C3_conditional_selection_beats_one_sided_selection": {"supported": all(
            cell(layer, corpus)["interaction_minus_one_sided_pair_median"] >= .05
            and cell(layer, corpus)["interaction_minus_one_sided_pair_ci"][0] > 0
            for layer in layers for corpus in arrays
        )},
        "C4_conditional_selection_is_useful_on_held_out_pairs": {"supported": all(
            cell(layer, corpus)["fidelity"]["interaction"] >= .50
            and cell(layer, corpus)["support_jaccard"]["interaction"] >= .15
            for layer in layers for corpus in arrays
        )},
        "C5_exact_tensor_contraction_retains_a_material_advantage": {"supported": all(
            cell(layer, corpus)["midpoint_minus_interaction_pair_median"] >= .10
            for layer in layers for corpus in arrays
        )},
        "C6_fixed_family_banks_remain_weak": {"supported": all(
            cell(layer, corpus)["fidelity"]["family_bank"] < .35
            for layer in layers for corpus in arrays
        )},
        "C7_interaction_success_is_not_a_shuffled_target_artifact": {"supported": all(
            cell(layer, corpus)["interaction_minus_shuffled_pair_median"] >= .10
            for layer in layers for corpus in arrays
        )},
        "C8_conditional_selector_transfers_out_of_distribution": {"supported": all(
            abs(cell(layer, "fineweb")["fidelity"]["interaction"] - cell(layer, "tinystories")["fidelity"]["interaction"]) <= .10
            for layer in layers
        )},
        "C9_exact_midpoint_reference_replicates": {"supported": all(
            cell(layer, corpus)["fidelity"]["midpoint_jvp"] >= .70
            for layer in layers for corpus in arrays
        )},
        "C10_endpoint_oracle_is_high_fidelity": {"supported": all(
            cell(layer, corpus)["fidelity"]["endpoint_oracle"] >= .80
            for layer in layers for corpus in arrays
        )},
    }
    OUTPUT.write_text(json.dumps({
        "config": red,
        "hypotheses": hypotheses,
        "calibration_pairs": sorted(calibration_pairs),
        "calibration": calibration,
        "shuffle_mapping": shuffle_mapping,
        "summaries": summaries,
        "records": records,
    }, indent=2, allow_nan=False))

    lines = [
        "# Conditional native-coalition selector decomposition", "",
        "Ridge selectors are trained only on the old FineWeb lexical pairs and evaluated on seventeen disjoint pairs in FineWeb and TinyStories. Their target is the normalized midpoint product tangent. Context-only, edit-only, additive, context×edit random-feature, and family-label variants all choose 256 native channels; only the selected channels' exact finite endpoint deltas are then patched through the complete suffix. Exact base/midpoint JVPs and endpoint-informed selection are mechanistic references, not learned models.",
        "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "", "![Conditional coalition selector](bilin12_layer01_conditional_coalition_selector.png)", "",
        "| Layer | Corpus | Context / edit / additive / interaction / +family fidelity | Family/global bank | Shuffled / midpoint / endpoint | Interaction support / cosine | Interaction−additive pair median (95% CI) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for layer in layers:
        for corpus in arrays:
            row = cell(layer, corpus); f = row["fidelity"]; ci = row["interaction_minus_additive_pair_ci"]
            lines.append(
                f"| {layer} | {corpus} | {f['context']:.3f} / {f['edit']:.3f} / {f['additive']:.3f} / {f['interaction']:.3f} / {f['interaction_family']:.3f} | "
                f"{f['family_bank']:.3f} / {f['global_bank']:.3f} | {f['interaction_shuffled']:.3f} / {f['midpoint_jvp']:.3f} / {f['endpoint_oracle']:.3f} | "
                f"{row['support_jaccard']['interaction']:.3f} / {row['midpoint_cosine']['interaction']:.3f} | "
                f"{row['interaction_minus_additive_pair_median']:+.3f} ({ci[0]:+.3f}, {ci[1]:+.3f}) |"
            )
    lines += ["", "## Selected ridge penalties", "", "| Layer | " + " | ".join(LEARNED[:-1]) + " |", "|---:|" + "---:|" * (len(LEARNED) - 1)]
    for layer in layers:
        lines.append(f"| {layer} | " + " | ".join(str(calibration[str(layer)][method]["selected_penalty"]) for method in LEARNED[:-1]) + " |")
    lines += [
        "",
        "Selection fidelity is the causal metric: predicted vectors are used only to choose native coordinates, after which the true finite deltas on those coordinates are inserted. Midpoint cosine measures predictor geometry but cannot by itself establish a useful circuit. The learned predictors are restricted to the span supported by discovery examples; failure does not imply that no more expressive weight-aware compiler exists—the exact tensor JVP is such a compiler.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    labels = [f"L{layer}\n{corpus}" for layer in layers for corpus in arrays]
    x = np.arange(len(labels)); shown = ("context", "edit", "additive", "interaction", "interaction_family", "midpoint_jvp")
    width = .13
    for index, method in enumerate(shown):
        offset = (index - (len(shown) - 1) / 2) * width
        axes[0].bar(x + offset, [cell(layer, corpus)["fidelity"][method] for layer in layers for corpus in arrays], width, label=method)
        axes[1].bar(x + offset, [cell(layer, corpus)["support_jaccard"][method] for layer in layers for corpus in arrays], width, label=method)
    axes[0].set_xticks(x, labels); axes[0].set_ylim(0, 1); axes[0].set_ylabel("finite causal fidelity"); axes[0].set_title("Held-out coalition selection")
    axes[1].set_xticks(x, labels); axes[1].set_ylim(0, 1); axes[1].set_ylabel("endpoint top-256 Jaccard"); axes[1].set_title("Native support recovery")
    for axis in axes: axis.legend(fontsize=7)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
