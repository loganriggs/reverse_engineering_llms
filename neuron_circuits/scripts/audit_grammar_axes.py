#!/usr/bin/env python3
"""Audit native-axis privilege, semantic subspaces, and fact addressing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.grammar_model import GrammarTransformer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--panel-size", type=int, default=2048)
    parser.add_argument("--rotations", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-underfit", action="store_true",
                        help="Run exploratory audits even when the preregistered fidelity gate fails.")
    return parser.parse_args()


def load_model(path: Path, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GrammarTransformer(
        VOCAB.size, checkpoint["width"], checkpoint["heads"], checkpoint["depth"],
        mlp_kind=checkpoint["mlp_kind"],
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


@torch.inference_mode()
def collect(model, prompts: np.ndarray, device: str, batch_size: int = 2048):
    layers = [[] for _ in model.blocks]
    for start in range(0, len(prompts), batch_size):
        tokens = torch.from_numpy(prompts[start:start + batch_size].astype(np.int64)).to(device)
        _, activations = model(tokens, return_activations=True)
        for layer, activation in enumerate(activations): layers[layer].append(activation.cpu())
    return [torch.cat(parts).float() for parts in layers]


def features_for(manifest: dict, facts: np.ndarray, current_answers: np.ndarray):
    digits = np.asarray(manifest["digits"])
    subject, obj, relation = facts.T
    sd, od = digits[subject], digits[obj]
    structured = np.asarray(manifest["permutations"])[
        relation,
        # Stored structured labels are easier and less error-prone to recover
        # by matching rows in the caller, so this placeholder is overwritten.
        np.zeros(len(facts), dtype=int),
    ]
    branch = np.zeros(len(facts), dtype=np.int64)
    branch[relation == 4] = sd[relation == 4, 2] >= 4
    branch[relation == 5] = od[relation == 5, 2] >= 4
    branch[relation == 6] = sd[relation == 6, 0] >= 4
    branch[relation == 7] = od[relation == 7, 0] >= 4
    semantic = {"relation": relation.astype(np.int64), "branch": branch}
    for position in range(3):
        semantic[f"subject_digit_{position}"] = sd[:, position].astype(np.int64)
        semantic[f"object_digit_{position}"] = od[:, position].astype(np.int64)
        semantic[f"relation&subject_digit_{position}"] = relation * 8 + sd[:, position]
        semantic[f"relation&object_digit_{position}"] = relation * 8 + od[:, position]
    semantic["relation&branch"] = relation * 2 + branch
    output = {"current_answer": current_answers.astype(np.int64),
              "relation&current_answer": relation * 8 + current_answers}
    address = {"subject": subject, "object": obj,
               "fact": np.arange(len(facts), dtype=np.int64)}
    return semantic, output, address


def with_structured_feature(features: dict[str, np.ndarray], facts: np.ndarray, structured: np.ndarray):
    result = dict(features)
    result["structured_answer"] = structured.astype(np.int64)
    result["relation&structured_answer"] = facts[:, 2] * 8 + structured
    return result


def predict_categories(train: torch.Tensor, values: np.ndarray) -> torch.Tensor:
    _, inverse = np.unique(values, return_inverse=True)
    index = torch.from_numpy(inverse).long()
    count = int(index.max()) + 1
    sums = torch.zeros(count, train.shape[1]).index_add_(0, index, train)
    counts = torch.bincount(index, minlength=count).clamp_min(1).to(train.dtype)
    return (sums / counts[:, None])[index]


def best_feature_r2(train: torch.Tensor, selection: torch.Tensor, test: torch.Tensor, features: dict[str, np.ndarray]):
    selection_variance = ((selection - selection.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    test_variance = ((test - test.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    best_selection = torch.full((train.shape[1],), -torch.inf)
    best_test = torch.full_like(best_selection, -torch.inf)
    best_name = torch.zeros(train.shape[1], dtype=torch.long)
    names = list(features)
    for feature_index, name in enumerate(names):
        prediction = predict_categories(train, features[name])
        selection_r2 = 1 - ((selection - prediction) ** 2).sum(0) / selection_variance
        test_r2 = 1 - ((test - prediction) ** 2).sum(0) / test_variance
        update = selection_r2 > best_selection
        best_selection[update] = selection_r2[update]
        best_test[update] = test_r2[update]
        best_name[update] = feature_index
    return best_test.numpy(), [names[index] for index in best_name.tolist()]


def single_feature_r2(train: torch.Tensor, test: torch.Tensor, values: np.ndarray):
    prediction = predict_categories(train, values)
    variance = ((test - test.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    return (1 - ((test - prediction) ** 2).sum(0) / variance).numpy()


def summarize(values: np.ndarray) -> dict:
    positive = np.clip(values, 0, None)
    count = max(1, round(.1 * len(values)))
    return {
        "median": float(np.median(values)), "mean": float(np.mean(values)),
        "top_decile_mean": float(np.mean(np.sort(values)[-count:])),
        "maximum": float(np.max(values)), "fraction_above_0.8": float(np.mean(values >= .8)),
        "top_decile_positive_mass_fraction": float(np.sort(positive)[-count:].sum() / max(positive.sum(), 1e-12)),
    }


def subgroup_audit(discovery, selection, transfer, semantic, output, address, rotations, keep):
    train = discovery[keep]; select = selection[keep]; test = transfer[keep]
    semantic_subset = {key: value[keep] for key, value in semantic.items()}
    output_subset = {key: value[keep] for key, value in output.items()}
    result = {}
    native_r2, names = best_feature_r2(train, select, test, semantic_subset)
    native = summarize(native_r2); native["best_features"] = names
    native["heldout_r2_by_channel"] = native_r2.tolist()
    native["current_answer_r2"] = summarize(single_feature_r2(train, test, output_subset["current_answer"]))
    # Exact-fact prediction across templates is a stability score, not by
    # itself evidence of an address code: a deterministic semantic state can
    # also be stable.  Residualize each axis by its selected semantic feature
    # before measuring the fact-specific remainder.
    fact_ids = np.arange(len(train))
    native["exact_fact_template_stability_r2"] = summarize(single_feature_r2(train, test, fact_ids))
    semantic_prediction = torch.empty_like(train)
    for feature_name in set(names):
        columns = torch.tensor([name == feature_name for name in names], dtype=torch.bool)
        semantic_prediction[:, columns] = predict_categories(train, semantic_subset[feature_name])[:, columns]
    residual_train = train - semantic_prediction
    residual_test = test - semantic_prediction
    native["residual_fact_template_stability_r2"] = summarize(
        single_feature_r2(residual_train, residual_test, fact_ids)
    )
    result["native"] = native
    rotated_summaries = []
    for rotation in rotations:
        rotated_r2, _ = best_feature_r2(train @ rotation, select @ rotation, test @ rotation, semantic_subset)
        rotated_summaries.append(summarize(rotated_r2))
    result["rotations"] = rotated_summaries
    for metric in ("median", "top_decile_mean", "maximum", "fraction_above_0.8", "top_decile_positive_mass_fraction"):
        null = np.asarray([row[metric] for row in rotated_summaries])
        result.setdefault("axis_privilege", {})[metric] = {
            "native_minus_rotation_median": float(native[metric] - np.median(null)),
            "rotation_median": float(np.median(null)),
            "rotation_min": float(np.min(null)), "rotation_max": float(np.max(null)),
            "empirical_upper_tail_p": float((1 + np.sum(null >= native[metric])) / (len(null) + 1)),
        }
    return result


def main():
    args = parse_args()
    manifest = load_manifest(args.root / "manifest.npz")
    runs = json.loads((args.root / "pilot_results.json").read_text())["runs"]
    for run in runs:
        destination = args.root / f"rho_{run['rho']:g}" / f"depth_{run['depth']}"
        output_path = destination / "axis_audit.json"
        if output_path.exists() and not args.force:
            print(f"SKIP {destination}", flush=True); continue
        if (not args.include_underfit and
                (run["best"]["fixed_accuracy"] < .97 or run["best"]["transfer_accuracy"] < .95)):
            print(f"UNDERFIT {destination}", flush=True); continue
        labels, selected, changed = labels_for_rho(manifest, "train", run["rho"])
        facts = np.asarray(manifest["train_facts"])
        with np.load(destination / "predictions.npz") as stored:
            correct = np.ones(len(facts), dtype=bool)
            for template in (0, 6): correct &= stored[f"train_template{template}_prediction"] == labels
        eligible = np.flatnonzero(correct)
        rng = np.random.default_rng(41000 + run["depth"] + round(100 * run["rho"]))
        panel = rng.choice(eligible, min(args.panel_size, len(eligible)), replace=False)
        panel_facts, panel_labels = facts[panel], labels[panel]
        semantic, output, address = features_for(manifest, panel_facts, panel_labels)
        semantic = with_structured_feature(semantic, panel_facts, manifest["train_structured"][panel])
        model = load_model(destination / "checkpoint.pt", args.device)
        activations = {
            template: collect(model, encode_prompts(manifest, panel_facts, template), args.device)
            for template in (0, 1, 6)
        }
        layers = []
        for layer in range(model.layers_count):
            rotations = []
            for rotation_index in range(args.rotations):
                generator = torch.Generator().manual_seed(51000 + layer * 100 + rotation_index + run["depth"])
                rotations.append(torch.linalg.qr(torch.randn(model.channels, model.channels, generator=generator)).Q)
            groups = {"all": np.ones(len(panel), dtype=bool)}
            if np.any(~selected[panel]): groups["structured"] = ~selected[panel]
            if np.any(changed[panel]): groups["changed"] = changed[panel]
            layer_result = {"layer": layer, "groups": {}}
            for name, keep in groups.items():
                if keep.sum() < 64: continue
                layer_result["groups"][name] = subgroup_audit(
                    activations[0][layer], activations[1][layer], activations[6][layer],
                    semantic, output, address, rotations, keep,
                )
            layers.append(layer_result)
        payload = {"rho": run["rho"], "depth": run["depth"], "mlp_kind": run["mlp_kind"],
                   "panel_size": len(panel), "panel_indices": panel.tolist(), "rotations": args.rotations,
                   "fidelity_gate_passed": bool(run["best"]["fixed_accuracy"] >= .97 and run["best"]["transfer_accuracy"] >= .95),
                   "layers": layers}
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        print(f"DONE {destination}", flush=True)


if __name__ == "__main__":
    main()
