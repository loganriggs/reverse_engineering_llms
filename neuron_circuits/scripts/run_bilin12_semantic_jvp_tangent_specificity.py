#!/usr/bin/env python3
"""Red-team whether JVP-selected source support is target-edit specific."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import logits_from_attention_writes, trace
from run_bilin12_boundary_carrier_depth_audit import fisher_terms
from run_bilin12_final_destination_source_program import source_write_contributions
from run_bilin12_semantic_source_jvp_predictor import cosine, differentiable_source_trace
from run_bilin12_semantic_source_program_transfer import jaccard
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_semantic_jvp_tangent_specificity.json")
OUTPUT = TENSOR_ROOT / "bilin12_semantic_jvp_tangent_specificity.json"
REPORT = TENSOR_ROOT / "BILIN12_SEMANTIC_JVP_TANGENT_SPECIFICITY.md"
FIGURE = TENSOR_ROOT / "bilin12_semantic_jvp_tangent_specificity.png"


def normalized_token(model, token: int):
    ids = torch.tensor([[token]], device="cuda")
    return F.rms_norm(model.embedding(ids), (model.config.n_embd,))[0, 0]


def tangent_trace(model, base_x0, tangent, layers, destination):
    function = lambda value: differentiable_source_trace(model, value, layers, destination)
    _, linear = torch.func.jvp(function, (base_x0,), (tangent,))
    return tuple(value.float() for value in linear)


def finite_metric(base_logits, target_delta, prediction_delta):
    target, _, _, error = fisher_terms(base_logits, target_delta, prediction_delta)
    return float(target.sum()), float(error.sum()), float(1 - error.sum() / target.sum().clamp_min(1e-30))


def finish(rows):
    target = sum(row[0] for row in rows)
    error = sum(row[1] for row in rows)
    values = [row[2] for row in rows]
    return {
        "examples": len(rows),
        "fisher_fidelity": 1 - error / max(target, 1e-30),
        "mean_individual_fidelity": float(np.mean(values)),
        "q05_individual_fidelity": float(np.quantile(values, 0.05)),
        "minimum_individual_fidelity": float(np.min(values)),
    }


def main():
    red = json.loads(CONFIG.read_text())
    config = json.loads(Path(red["source_config"]).read_text())
    prior = json.loads(Path(red["source_output"]).read_text())
    semantic = json.loads(Path(config["semantic_source"]).read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    pairs = [(family, source, target) for family, rows in semantic["families"].items() for source, target in rows]
    pair_index = {(source, target): index for index, (_, source, target) in enumerate(pairs)}
    token_vectors = {token: normalized_token(model, token) for _, source, target in pairs for token in (source, target)}
    sources = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"] : spec["start"] + spec["rows"]]
        for corpus, spec in config["evaluation"].items()
    }
    generator = torch.Generator(device="cuda").manual_seed(red["seed"])
    metric_rows = defaultdict(list)
    records = []

    for prior_record in prior["records"]:
        corpus = prior_record["corpus"]
        row_index = prior_record["row"]
        start = prior_record["start"]
        source_token, target_token = (int(value) for value in prior_record["pair"].split("->"))
        index = pair_index[(source_token, target_token)]
        _, shuffled_source, shuffled_target = pairs[(index + 1) % len(pairs)]
        wrong_target = pairs[(index + 7) % len(pairs)][2]
        if wrong_target == target_token:
            wrong_target = pairs[(index + 8) % len(pairs)][2]

        raw = np.asarray(sources[corpus][row_index, start : start + config["sequence_length"]], dtype=np.int64).copy()
        changed = raw.copy()
        changed[config["edit_position"]] = target_token
        base_ids = torch.as_tensor(raw, device="cuda")[None]
        changed_ids = torch.as_tensor(changed, device="cuda")[None]
        with torch.no_grad():
            base_rows, _, _ = trace(model, base_ids)
            changed_rows, _, _ = trace(model, changed_ids)
            base_x0 = F.rms_norm(model.embedding(base_ids), (model.config.n_embd,))

        correct = torch.zeros_like(base_x0)
        correct[:, config["edit_position"]] = token_vectors[target_token] - base_x0[:, config["edit_position"]]
        wrong = torch.zeros_like(base_x0)
        wrong[:, config["edit_position"]] = token_vectors[wrong_target] - base_x0[:, config["edit_position"]]
        shuffled = torch.zeros_like(base_x0)
        shuffled[:, config["edit_position"]] = token_vectors[shuffled_target] - token_vectors[shuffled_source]

        with torch.enable_grad():
            correct_linear = tangent_trace(model, base_x0, correct, config["layers"], config["prediction_position"])
            wrong_linear = tangent_trace(model, base_x0, wrong, config["layers"], config["prediction_position"])
            shuffled_linear = tangent_trace(model, base_x0, shuffled, config["layers"], config["prediction_position"])
            random_linears = []
            for _ in range(red["random_probes"]):
                random = torch.zeros_like(base_x0)
                vector = torch.randn((model.config.n_embd,), generator=generator, device="cuda")
                vector = vector / vector.norm().clamp_min(1e-30) * correct[:, config["edit_position"]].norm()
                random[:, config["edit_position"]] = vector
                random_linears.append(tangent_trace(model, base_x0, random, config["layers"], config["prediction_position"]))

        layer_records = {}
        for layer_index, layer in enumerate(config["layers"]):
            base = base_rows[layer]
            changed_row = changed_rows[layer]
            block = model.blocks[layer]
            exact_base = source_write_contributions(block, base["details"], config["prediction_position"]).float()
            exact_changed = source_write_contributions(block, changed_row["details"], config["prediction_position"]).float()
            exact_delta = exact_changed - exact_base
            random_score = torch.stack([rows[layer_index].square().sum(-1) for rows in random_linears]).mean(0).sqrt()
            predicted = {
                "correct": correct_linear[layer_index],
                "wrong_target": wrong_linear[layer_index],
                "shuffled_pair": shuffled_linear[layer_index],
            }
            scores = {
                **{name: value.norm(dim=-1) for name, value in predicted.items()},
                "random_susceptibility": random_score,
                "base_source_norm": exact_base.norm(dim=-1),
            }
            orders = {name: score.argsort(-1, descending=True) for name, score in scores.items()}

            full_write = base["details"]["write"].clone()
            full_write[:, config["prediction_position"]] = changed_row["details"]["write"][:, config["prediction_position"]]
            base_logits, target_logits = logits_from_attention_writes(model, layer, base, torch.stack((base["details"]["write"], full_write)))
            target_delta = target_logits - base_logits
            worlds = []
            labels = []
            for method, order in orders.items():
                for budget in red["budgets"]:
                    mask = torch.zeros(1, config["sequence_length"], device="cuda")
                    mask.scatter_(1, order[:, :budget], 1)
                    approximation = (mask[:, :, None] * exact_delta).sum(1)
                    write = base["details"]["write"].clone()
                    write[:, config["prediction_position"]] += approximation.to(write.dtype)
                    worlds.append(write)
                    labels.append(f"{method}/{budget}")
            linear_write = base["details"]["write"].clone()
            linear_write[:, config["prediction_position"]] += correct_linear[layer_index].sum(1).to(linear_write.dtype)
            worlds.append(linear_write)
            labels.append("correct_linear/all")
            logits = logits_from_attention_writes(model, layer, base, torch.stack(worlds))
            per_metrics = {}
            for world_index, label in enumerate(labels):
                row = finite_metric(base_logits, target_delta, logits[world_index] - base_logits)
                metric_rows[label].append(row)
                per_metrics[label] = row[2]

            correct_top = orders["correct"][0, :16].cpu().tolist()
            wrong_top = orders["wrong_target"][0, :16].cpu().tolist()
            shuffled_top = orders["shuffled_pair"][0, :16].cpu().tolist()
            sign_same = bool(torch.equal(
                correct_linear[layer_index].norm(dim=-1).argsort(-1, descending=True),
                (-correct_linear[layer_index]).norm(dim=-1).argsort(-1, descending=True),
            ))
            layer_records[str(layer)] = {
                "metrics": per_metrics,
                "correct_wrong_top16_jaccard": jaccard(correct_top, wrong_top),
                "correct_shuffled_top16_jaccard": jaccard(correct_top, shuffled_top),
                "correct_wrong_tensor_cosine": float(cosine(correct_linear[layer_index], wrong_linear[layer_index])[0]),
                "sign_reversed_ranking_identical": sign_same,
            }
        records.append({
            "corpus": corpus,
            "family": prior_record["family"],
            "pair": prior_record["pair"],
            "wrong_target": wrong_target,
            "shuffled_pair": f"{shuffled_source}->{shuffled_target}",
            "layers": layer_records,
        })
        print(corpus, prior_record["family"], prior_record["pair"], flush=True)

    summary = {label: finish(rows) for label, rows in metric_rows.items()}
    layer_values = [layer for record in records for layer in record["layers"].values()]
    overlaps = np.asarray([row["correct_wrong_top16_jaccard"] for row in layer_values])
    tensor_cosines = np.asarray([row["correct_wrong_tensor_cosine"] for row in layer_values])
    correct16 = summary["correct/16"]["fisher_fidelity"]
    wrong16 = summary["wrong_target/16"]["fisher_fidelity"]
    shuffled16 = summary["shuffled_pair/16"]["fisher_fidelity"]
    random16 = summary["random_susceptibility/16"]["fisher_fidelity"]
    base16 = summary["base_source_norm/16"]["fisher_fidelity"]
    hypotheses = {
        "T1_correct_tangent_beats_wrong_target": {"supported": bool(correct16 - wrong16 >= 0.05), "advantage": correct16 - wrong16},
        "T2_correct_tangent_beats_shuffled_pair_delta": {"supported": bool(correct16 - shuffled16 >= 0.05), "advantage": correct16 - shuffled16},
        "T3_correct_tangent_beats_generic_random_susceptibility": {"supported": bool(correct16 - random16 >= 0.05), "advantage": correct16 - random16},
        "T4_program_support_is_target_specific": {"supported": bool(np.median(overlaps) <= 0.75), "median_jaccard": float(np.median(overlaps))},
        "T5_generic_jacobian_susceptibility_beats_static_source_magnitude": {"supported": bool(random16 - base16 >= 0.05), "advantage": random16 - base16},
        "T6_correct_jvp_support_remains_robust": {"supported": bool(summary["correct/32"]["fisher_fidelity"] >= 0.90 and summary["correct/32"]["q05_individual_fidelity"] >= 0.75), "aggregate": summary["correct/32"]["fisher_fidelity"], "q05": summary["correct/32"]["q05_individual_fidelity"]},
        "T7_selection_is_much_better_than_linear_reconstruction": {"supported": bool(summary["correct/32"]["fisher_fidelity"] - summary["correct_linear/all"]["fisher_fidelity"] >= 0.25), "gap": summary["correct/32"]["fisher_fidelity"] - summary["correct_linear/all"]["fisher_fidelity"]},
        "T8_norm_selection_is_sign_blind": {"supported": all(row["sign_reversed_ranking_identical"] for row in layer_values)},
        "T9_wrong_tangents_do_not_define_semantic_direction": {"supported": bool(np.median(tensor_cosines) < 0.80), "median_tensor_cosine": float(np.median(tensor_cosines))},
        "T10_support_is_instead_a_generic_contextual_susceptibility_map": {"supported": bool(correct16 - wrong16 <= 0.02 and np.median(overlaps) >= 0.70), "fidelity_gap": correct16 - wrong16, "median_jaccard": float(np.median(overlaps))},
    }
    OUTPUT.write_text(json.dumps({"config": red, "source_config": config, "hypotheses": hypotheses, "summary": summary, "records": records}, indent=2, allow_nan=False))

    budgets = red["budgets"]
    methods = ["correct", "wrong_target", "shuffled_pair", "random_susceptibility", "base_source_norm"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), layout="constrained")
    for method in methods:
        axes[0].plot(budgets, [summary[f"{method}/{budget}"]["fisher_fidelity"] for budget in budgets], marker="o", label=method)
    axes[0].set_xscale("log", base=2)
    axes[0].set_title("True finite edges selected by tangent controls")
    axes[0].set_xlabel("source budget")
    axes[0].set_ylabel("Fisher fidelity")
    axes[1].hist(overlaps, bins=14)
    axes[1].set_title("Correct/wrong-target top-16 Jaccard")
    axes[2].hist(tensor_cosines, bins=14)
    axes[2].set_title("Correct/wrong JVP tensor cosine")
    for axis in axes:
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    lines = [
        "# Tangent-specificity red team for semantic source programs",
        "",
        "The previously successful base-point JVP ranking is compared with an unrelated replacement target, a shuffled lexical-pair delta, a four-random-probe Jacobian susceptibility estimate, and static base-source magnitude. Every method only selects positions; the selected positions are evaluated with the same true finite source deltas. This isolates support prediction from direction prediction.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Tangent specificity](bilin12_semantic_jvp_tangent_specificity.png)",
        "",
        "| Budget | Correct | Wrong target | Shuffled pair | Random susceptibility | Base source |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in budgets:
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            budget,
            summary[f"correct/{budget}"]["fisher_fidelity"],
            summary[f"wrong_target/{budget}"]["fisher_fidelity"],
            summary[f"shuffled_pair/{budget}"]["fisher_fidelity"],
            summary[f"random_susceptibility/{budget}"]["fisher_fidelity"],
            summary[f"base_source_norm/{budget}"]["fisher_fidelity"],
        ))
    lines += [
        "",
        f"Correct/wrong-target median top-16 Jaccard: **{np.median(overlaps):.3f}**. Correct/wrong-target median source-tensor cosine: **{np.median(tensor_cosines):.3f}**. All-source correct-JVP reconstruction fidelity: **{summary['correct_linear/all']['fisher_fidelity']:.3f}**.",
        "",
        "Because source norms are invariant under tangent sign, support selection cannot by itself specify whether an edit adds or removes a semantic property. A target-specific support result and a target-specific direction result are therefore distinct claims.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
