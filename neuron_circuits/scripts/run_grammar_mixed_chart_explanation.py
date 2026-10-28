#!/usr/bin/env python3
"""Can simple semantic, output, or address variables explain mixed-model charts?"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_generalization_chart import collect, eligible, root_for
from run_grammar_mixed_natural_metric_redteam import matrix_sqrt


CONFIG = Path("configs/grammar_mixed_chart_explanation.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_mixed_chart_explanation.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_MIXED_CHART_EXPLANATION.md")


def one_hot(values, count):
    output = np.zeros((len(values), count), dtype=np.float32)
    output[np.arange(len(values)), values.astype(int)] = 1
    return output


def feature_sets(manifest, facts, current, competitors):
    digits = np.asarray(manifest["digits"]); subject, obj, relation = facts.T
    sd, od = digits[subject], digits[obj]
    # Evaluation facts are in a different table, so compute their structured
    # answer directly from the manifest's saved relation values/permutations.
    from neuron_circuits.grammar_data import relation_values
    values = relation_values(np.asarray(manifest["traits"])[subject], np.asarray(manifest["traits"])[obj], relation)
    structured = np.asarray(manifest["permutations"])[relation, values]
    branch = np.zeros(len(facts), dtype=np.int64)
    branch[relation == 4] = sd[relation == 4, 2] >= 4
    branch[relation == 5] = od[relation == 5, 2] >= 4
    branch[relation == 6] = sd[relation == 6, 0] >= 4
    branch[relation == 7] = od[relation == 7, 0] >= 4
    semantic = np.concatenate((one_hot(relation, 8), one_hot(structured, 8),
                               one_hot(relation * 8 + structured, 64),
                               one_hot(relation * 2 + branch, 16)), 1)
    output = np.concatenate((one_hot(current, 8), one_hot(competitors, 8),
                             one_hot(current * 8 + competitors, 64)), 1)
    token_parts = [one_hot(relation, 8)]
    for column in range(3):
        token_parts += [one_hot(sd[:, column], 8), one_hot(relation * 8 + sd[:, column], 64),
                        one_hot(od[:, column], 8), one_hot(relation * 8 + od[:, column], 64)]
    token = np.concatenate(token_parts, 1)
    entity = np.concatenate((one_hot(subject, 512), one_hot(obj, 512), one_hot(relation, 8)), 1)
    address = np.concatenate((token, entity), 1)
    return {"semantic": semantic, "output": output, "address": address,
            "semantic_output": np.concatenate((semantic, output), 1),
            "combined": np.concatenate((semantic, output, address), 1)}


def margin_operator(row, layer, positions, targets, covariance_sqrt):
    jacobian = row["jacobians"][layer][positions]
    logits = row["logits"][positions]
    operator = torch.einsum("boc,cd->bod", jacobian, covariance_sqrt)
    rows = torch.arange(len(positions)); wrong = logits.clone(); wrong[rows, targets] = -torch.inf
    competitors = wrong.argmax(1)
    return operator[rows, targets] - operator[rows, competitors], competitors.numpy()


def contextual_r2(actual, predicted, train_mean):
    denominator = (actual - train_mean).square().sum().clamp_min(1e-30)
    return 1 - float((actual - predicted).square().sum() / denominator)


def total_fidelity(actual, predicted):
    return 1 - float((actual - predicted).square().sum() / actual.square().sum().clamp_min(1e-30))


def ridge_fit(train_x, train_y, test_x, multiplier):
    x_mean = train_x.mean(0, keepdim=True); y_mean = train_y.mean(0, keepdim=True)
    x = train_x - x_mean; y = train_y - y_mean
    kernel = x @ x.T; scale = torch.diagonal(kernel).mean().clamp_min(1e-12)
    coefficient = torch.linalg.solve(kernel + multiplier * scale * torch.eye(len(kernel)), y)
    prediction = (test_x - x_mean) @ x.T @ coefficient + y_mean
    return prediction, y_mean


def choose_fit(train_x, train_y, validation_x, validation_y, test_x, test_y, multipliers):
    candidates = []
    for multiplier in multipliers:
        prediction, mean = ridge_fit(train_x, train_y, validation_x, multiplier)
        candidates.append((contextual_r2(validation_y, prediction, mean), multiplier))
    validation_r2, multiplier = max(candidates)
    validation_prediction, mean = ridge_fit(train_x, train_y, validation_x, multiplier)
    test_prediction, _ = ridge_fit(train_x, train_y, test_x, multiplier)
    return {"multiplier": multiplier, "validation_r2": validation_r2,
            "validation_total_fidelity": total_fidelity(validation_y, validation_prediction),
            "heldout_r2": contextual_r2(test_y, test_prediction, mean),
            "heldout_total_fidelity": total_fidelity(test_y, test_prediction)}


def analyze_seed(seed, config, device):
    root = root_for(seed); manifest = load_manifest(root / "manifest.npz")
    model = load_model(root / "rho_0.5" / "depth_4" / "checkpoint.pt", device)
    facts, labels, selected, _, keep = eligible(
        model, manifest, "train", config["rho"],
        [config["discovery_template"], config["validation_template"]], device,
    )
    rng = np.random.default_rng(config["seed"] + seed); discovery, validation = [], []
    for relation in range(8):
        for answer in range(8):
            for status in (False, True):
                indices = np.flatnonzero(keep & (facts[:, 2] == relation) & (labels == answer) & (selected == status))
                rng.shuffle(indices); first = config["discovery_per_relation_answer_status"]
                second = first + config["validation_per_relation_answer_status"]
                if len(indices) < second: raise RuntimeError((seed, relation, answer, status, len(indices)))
                discovery.extend(indices[:first]); validation.extend(indices[first:second])
    discovery, validation = np.asarray(discovery), np.asarray(validation)
    held_facts, held_labels, held_selected, _, held_keep = eligible(
        model, manifest, "heldout_pair", config["rho"], [config["evaluation_template"]], device,
    )
    heldout = np.flatnonzero(held_keep & ~held_selected)
    discovery_row = collect(model, manifest, "train", facts, discovery, config["discovery_template"], config, device)
    validation_row = collect(model, manifest, "train", facts, validation, config["validation_template"], config, device)
    heldout_row = collect(model, manifest, "heldout_pair", held_facts, heldout, config["evaluation_template"], config, device)
    layers = []
    for layer in range(4):
        activation = torch.cat((discovery_row["activations"][layer], validation_row["activations"][layer]))
        centered = activation - activation.mean(0)
        covariance_sqrt = matrix_sqrt(centered.T @ centered / len(centered), config["covariance_ridge_fraction"])
        targets = {
            "discovery": torch.from_numpy(labels[discovery].astype(np.int64)),
            "validation": torch.from_numpy(labels[validation].astype(np.int64)),
            "heldout": torch.from_numpy(held_labels[heldout].astype(np.int64)),
        }
        rows = {"discovery": discovery_row, "validation": validation_row, "heldout": heldout_row}
        operators, competitors = {}, {}
        for name, row in rows.items():
            positions = torch.arange(len(row["indices"]))
            operators[name], competitors[name] = margin_operator(row, layer, positions, targets[name], covariance_sqrt)
        features = {
            "discovery": feature_sets(manifest, facts[discovery], labels[discovery], competitors["discovery"]),
            "validation": feature_sets(manifest, facts[validation], labels[validation], competitors["validation"]),
            "heldout": feature_sets(manifest, held_facts[heldout], held_labels[heldout], competitors["heldout"]),
        }
        result = {}
        train_mask = ~selected[discovery]; validation_mask = ~selected[validation]
        train_mask_t = torch.from_numpy(train_mask); validation_mask_t = torch.from_numpy(validation_mask)
        for feature_name in features["discovery"]:
            result[feature_name] = choose_fit(
                torch.from_numpy(features["discovery"][feature_name][train_mask]).float(), operators["discovery"][train_mask_t],
                torch.from_numpy(features["validation"][feature_name][validation_mask]).float(), operators["validation"][validation_mask_t],
                torch.from_numpy(features["heldout"][feature_name]).float(), operators["heldout"], config["ridge_multipliers"],
            )
        # The null keeps feature covariance fixed and breaks its alignment with
        # the causal operators. Lambda selection is repeated inside each null.
        nulls = []
        real_x = features["discovery"]["combined"][train_mask]
        real_val = features["validation"]["combined"][validation_mask]
        real_test = features["heldout"]["combined"]
        for repeat in range(config["random_null_replicates"]):
            permutation = rng.permutation(len(real_x))
            row = choose_fit(torch.from_numpy(real_x).float(), operators["discovery"][train_mask_t][torch.from_numpy(permutation)],
                             torch.from_numpy(real_val).float(), operators["validation"][validation_mask_t],
                             torch.from_numpy(real_test).float(), operators["heldout"], config["ridge_multipliers"])
            nulls.append(row["heldout_r2"])
        result["combined_null_p95"] = float(np.quantile(nulls, .95)); layers.append(result)
    return {"seed": seed, "heldout_sites": len(heldout), "layers": layers}


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = []
    for seed in config["seeds"]:
        print("audit seed", seed, flush=True); seeds.append(analyze_seed(seed, config, device))
    def metric(seed, feature, key): return seed["layers"][0][feature][key]
    p0 = all(metric(seed, "semantic", "heldout_r2") >= .3 for seed in seeds)
    p1 = all(metric(seed, "address", "heldout_r2") >= metric(seed, "semantic", "heldout_r2") + .1 for seed in seeds)
    p2 = all(metric(seed, "combined", "heldout_r2") >= .5 for seed in seeds)
    p3 = all(metric(seed, "combined", "heldout_r2") >= seed["layers"][0]["combined_null_p95"] + .1 for seed in seeds)
    p4 = all(metric(seed, "combined", "heldout_r2") >= metric(seed, "combined", "validation_r2") - .2 for seed in seeds)
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], (p0, p1, p2, p3, p4))}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "seeds": seeds}, indent=2, allow_nan=False))
    lines = ["# Simple explanations of mixed-model causal charts", "",
             "Bounded ridge models are fit on unselected training facts in one template, selected on disjoint unselected facts in a held-out template, and evaluated on correctly generalized held-out pairs. The target is the activation-covariance-weighted correct-versus-rival margin operator. Semantic, output-role, token/address, and combined feature vocabularies are fixed before fitting.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "The transfer-closeness condition is vacuous at early layers where both validation and held-out R2 are near zero; it is substantive only where predictive R2 is high."]
    lines += ["", "| Seed | Layer | Semantic R2 | Output R2 | Address R2 | Combined R2 | Combined validation R2 | Null p95 |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in seeds:
        for layer, row in enumerate(seed["layers"]):
            lines.append(f"| {seed['seed']} | {layer} | {row['semantic']['heldout_r2']:.3f} | {row['output']['heldout_r2']:.3f} | "
                         f"{row['address']['heldout_r2']:.3f} | {row['combined']['heldout_r2']:.3f} | "
                         f"{row['combined']['validation_r2']:.3f} | {row['combined_null_p95']:.3f} |")
    lines += ["", "A failed bounded explanation does not prove irreducibility: the model may use a compact nonlinear or tensor program outside this feature class. It does falsify the claim that the observed chart family is already explained by these simple additive semantic/address variables."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
