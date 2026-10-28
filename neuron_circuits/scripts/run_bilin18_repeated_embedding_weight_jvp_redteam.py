#!/usr/bin/env python3
"""Prospective position and donor red team for repeated-embedding JVP paths."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer17_tangent_bundle_quotient import build_role_tensor
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_repeated_embedding_weight_jvp import evaluate


CONFIG = Path("configs/bilin18_repeated_embedding_weight_jvp_redteam.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_repeated_embedding_weight_jvp_redteam.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_REPEATED_EMBEDDING_WEIGHT_JVP_REDTEAM.md")
FIGURE = Path("artifacts/bilin18_tensor_decomposition/bilin18_repeated_embedding_weight_jvp_redteam.png")


def cells(results):
    for corpus, by_position in results.items():
        for position_group, by_shift in by_position.items():
            for shift, row in by_shift.items():
                yield corpus, position_group, shift, row


def locked_hypotheses(results):
    rows = list(cells(results)); metrics = ("operator", "fisher")
    group_names = ["inject_0_2", "inject_3_5", "inject_6_8", "inject_9_11", "inject_12_14", "inject_15_17"]
    largest_68 = [max(group_names, key=lambda name: row["paths"]["operator"][name]["projection_share"]) == "inject_6_8" for _, _, _, row in rows]
    donor_stable = True
    for corpus, by_position in results.items():
        for _, by_shift in by_position.items():
            for metric in metrics:
                values = [row["paths"][metric]["injection_all"]["projection_share"] for row in by_shift.values()]
                donor_stable &= max(values) - min(values) <= .15
    position_stable = True
    for corpus, by_position in results.items():
        for shift in next(iter(by_position.values())):
            for metric in metrics:
                values = [by_position[group][shift]["paths"][metric]["injection_all"]["projection_share"] for group in by_position]
                position_stable &= max(values) - min(values) <= .15
    return {
        "D0_path_recomposition_survives_controls": max(value for _, _, _, row in rows for value in row["recomposition_relative_error"].values()) <= 1e-5,
        "D1_injection_dominance_survives_controls": all(row["paths"][metric]["injection_all"]["projection_share"] >= .75 for _, _, _, row in rows for metric in metrics),
        "D2_final_group_remains_minor": all(abs(row["paths"][metric]["inject_15_17"]["projection_share"]) <= .15 for _, _, _, row in rows for metric in metrics),
        "D3_middle_groups_are_jointly_material": all(row["paths"][metric]["inject_6_8"]["projection_share"] + row["paths"][metric]["inject_9_11"]["projection_share"] >= .30 for _, _, _, row in rows for metric in metrics),
        "D4_layer6_8_is_usually_largest_operator_group": np.mean(largest_68) >= .80,
        "D5_donor_pairing_does_not_control_injection_share": donor_stable,
        "D6_position_choice_does_not_control_injection_share": position_stable,
        "D7_small_edit_linearity_survives_controls": all(row["finite_fidelity"]["0.01"][metric] >= .99 for _, _, _, row in rows for metric in metrics),
        "D8_full_edit_nonlinearity_survives_controls": all(any(row["finite_fidelity"]["1.0"][metric] <= row["finite_fidelity"]["0.01"][metric] - .10 for metric in metrics) for _, _, _, row in rows),
        "D9_operator_and_fisher_agree_on_injection_dominance": all(abs(row["paths"]["operator"]["injection_all"]["projection_share"] - row["paths"]["fisher"]["injection_all"]["projection_share"]) <= .15 for _, _, _, row in rows),
        "D10_initial_plus_bus_remains_minor": all(row["paths"][metric]["initial_plus_bus"]["projection_share"] <= .25 for _, _, _, row in rows for metric in metrics),
    }


def plot_results(results):
    labels = []; operator = []; fisher = []
    for corpus, position_group, shift, row in cells(results):
        labels.append(f"{corpus[:2]}-{position_group[0]}-{shift}")
        operator.append(row["paths"]["operator"]["injection_all"]["projection_share"])
        fisher.append(row["paths"]["fisher"]["injection_all"]["projection_share"])
    x = np.arange(len(labels)); figure, axis = plt.subplots(figsize=(12, 4.4))
    axis.plot(x, operator, marker="o", label="operator"); axis.plot(x, fisher, marker="s", label="Fisher")
    axis.axhline(.75, color="black", linestyle="--", linewidth=.8); axis.set_ylim(.65, 1.05)
    axis.set_xticks(x, labels, rotation=60, ha="right"); axis.set_ylabel("repeated-injection projection share")
    axis.grid(alpha=.2); axis.legend(frameon=False); figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)


def main():
    config = json.loads(CONFIG.read_text())
    core = torch.load(config["core_path"], map_location="cpu", weights_only=False)
    context_axes = core["context"][:, :config["rank"]].float().cuda()
    tangent_axes = core["tangent"][:, :config["rank"]].float().cuda()
    output_axes = core["output"][:, :config["rank"]].float().cuda()
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    tensor, *_ = build_role_tensor(model.blocks[17], context_axes, tangent_axes, output_axes)
    flat = tensor.reshape(-1, config["rank"]).double(); gram = (flat.T @ flat).cuda()
    results = {}
    for corpus, spec in config["evaluation"].items():
        results[corpus] = {}
        for position_group, positions in config["position_groups"].items():
            results[corpus][position_group] = {}
            for shift in config["donor_shifts"]:
                print("evaluate", corpus, position_group, "shift", shift, flush=True)
                cell_config = {**config, "positions": positions}
                results[corpus][position_group][str(shift)] = evaluate(
                    model, spec, cell_config, context_axes, gram, donor_shift=shift)
    hypotheses = {name: bool(value) for name, value in locked_hypotheses(results).items()}
    payload = {"config": config, "hypotheses": hypotheses, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False)); plot_results(results)

    lines = ["# Repeated-embedding weight-JVP red team", "",
             "This disjoint replication crosses three corpora, original versus unseen positions, and all three nontrivial within-batch donor permutations.", "", "![Position and donor red team](bilin18_repeated_embedding_weight_jvp_redteam.png)", "", "## Locked hypotheses", ""]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += ["", "## Cell summaries", "", "| Corpus | positions | shift | injection operator/Fisher | middle 6-11 operator/Fisher | final 15-17 operator/Fisher | full-edit operator/Fisher |", "|---|---|---:|---:|---:|---:|---:|"]
    for corpus, position_group, shift, row in cells(results):
        op = row["paths"]["operator"]; fi = row["paths"]["fisher"]
        middle_op = op["inject_6_8"]["projection_share"] + op["inject_9_11"]["projection_share"]
        middle_fi = fi["inject_6_8"]["projection_share"] + fi["inject_9_11"]["projection_share"]
        full = row["finite_fidelity"]["1.0"]
        lines.append(f"| {corpus} | {position_group} | {shift} | {op['injection_all']['projection_share']:.3f}/{fi['injection_all']['projection_share']:.3f} | {middle_op:.3f}/{middle_fi:.3f} | {op['inject_15_17']['projection_share']:.3f}/{fi['inject_15_17']['projection_share']:.3f} | {full['operator']:.3f}/{full['fisher']:.3f} |")
    lines += ["", "## Interpretation boundary", "",
              "Passing these controls would establish a robust local architectural path ordering for ordinary token directions. It would not make the individual three-layer shares semantic modules, nor would it repair the known failure of one native-state JVP to explain a complete token substitution."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses}, indent=2), flush=True)


if __name__ == "__main__":
    main()
