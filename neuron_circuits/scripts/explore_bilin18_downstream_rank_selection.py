#!/usr/bin/env python3
"""Rank inherited CP corrections by first-order downstream task sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate_bilin18_causal_cp import load_cp
from evaluate_bilin18_decomposition_metrics import behavioral_evaluation, causal_evaluation, json_ready
from explore_bilin18_blend_rank_curve import SelectiveBlend
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_tensor_decomposition import ROOT
from train_bilin18_causal_cp import (
    distribution_kl,
    fisher_effect_loss,
    prefix_to_layer1,
    suffix_from_layer1,
    teacher_outputs,
)
from validate_bilin18_value_tangent_blend import CONFIG as EVAL_CONFIG, load_rows


CONFIG = Path("configs/bilin18_downstream_rank_selection.json")
OUTPUT = ROOT / "downstream_rank_selection_results.json"
REPORT = ROOT / "DOWNSTREAM_RANK_SELECTION_REPORT.md"
FIGURE = ROOT / "downstream_rank_selection.png"


class GatedCorrection(nn.Module):
    """Language CP plus per-ancestral-term gates on the direct correction."""

    def __init__(self, language, direct, blend_weight):
        super().__init__()
        self.language = language
        self.direct = direct
        self.blend_weight = blend_weight
        self.gates = nn.Parameter(torch.zeros(language.rank, device=language.gain.device))

    @staticmethod
    def features(model, x):
        left, right, output, gain = model.factors()
        feature = F.linear(x, left) * F.linear(x, right) * gain
        return feature, output

    def forward(self, x):
        language_feature, language_output = self.features(self.language, x)
        direct_feature, direct_output = self.features(self.direct, x)
        base = F.linear(language_feature, language_output) + self.language.bias
        correction = (F.linear(direct_feature * self.gates, direct_output)
                      - F.linear(language_feature * self.gates, language_output))
        return base + self.blend_weight * correction


def changed_input(normalized, spec, generator):
    if spec["kind"] == "gaussian":
        direction = torch.randn(
            normalized.shape, generator=generator, device=normalized.device,
            dtype=torch.float32)
        direction = direction / direction.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        return normalized + spec["scale"] * direction.to(normalized.dtype)
    if spec["kind"] == "natural_mix":
        return normalized + spec["scale"] * (normalized.roll(1, 1) - normalized)
    raise ValueError(spec["kind"])


def student_outputs(model, student, x0, value_bus, pre, base_input, changed):
    base_write = student(base_input.float()).view_as(base_input).to(pre.dtype)
    changed_write = student(changed.float()).view_as(changed).to(pre.dtype)
    base = suffix_from_layer1(model, x0, value_bus, pre + base_write)
    altered = suffix_from_layer1(model, x0, value_bus, pre + changed_write)
    return base, altered


def normalized_gradient(loss, gates, retain_graph):
    gradient = torch.autograd.grad(loss, gates, retain_graph=retain_graph)[0].detach()
    return gradient / gradient.abs().mean().clamp_min(1e-12)


def downstream_ranking(model, language, direct, rows, config):
    student = GatedCorrection(language, direct, config["blend_lambda"]).cuda().eval()
    for candidate in (language, direct):
        candidate.requires_grad_(False)
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    component_gradients = {name: torch.zeros(2048, device="cuda")
                           for name in ("base_kl", "changed_kl", "effect")}
    spec = config["selection_rows"]
    records = []
    for offset, index in enumerate(range(spec["start"], spec["start"] + spec["rows"])):
        raw = np.asarray(rows[index:index + 1, :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        x0, value_bus, pre, normalized = prefix_to_layer1(model, ids)
        for condition, intervention in config["selection_interventions"].items():
            changed = changed_input(normalized, intervention, generator)
            teacher_base, teacher_changed = teacher_outputs(
                model, x0, value_bus, pre, normalized, changed)
            student_base, student_changed = student_outputs(
                model, student, x0, value_bus, pre, normalized, changed)
            losses = {
                "base_kl": distribution_kl(teacher_base, student_base),
                "changed_kl": distribution_kl(teacher_changed, student_changed),
                "effect": fisher_effect_loss(
                    teacher_base, teacher_changed, student_base, student_changed),
            }
            names = tuple(losses)
            for position, name in enumerate(names):
                component_gradients[name] += normalized_gradient(
                    losses[name], student.gates, position < len(names) - 1)
            records.append({"row": index, "condition": condition,
                            **{name: float(value.detach()) for name, value in losses.items()}})
        print("downstream ranking row", offset + 1, "/", spec["rows"], flush=True)
    # A positive score means that turning on the inherited direct correction is
    # predicted to lower the equally weighted, per-objective-normalized loss.
    score = -sum(component_gradients.values()) / len(records)
    ranking = score.argsort(descending=True)
    return ranking.cpu(), score.cpu(), {
        name: (-gradient / len(records)).cpu() for name, gradient in component_gradients.items()
    }, records


def improvement_fraction(base, selected, full, higher_is_better):
    numerator = selected - base if higher_is_better else base - selected
    denominator = full - base if higher_is_better else base - full
    return float(numerator / denominator) if abs(denominator) > 1e-12 else float("nan")


def main():
    selection_config = json.loads(CONFIG.read_text())
    eval_config = json.loads(EVAL_CONFIG.read_text())
    eval_config["bootstrap_resamples"] = 1000
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.requires_grad_(False)
    language = load_cp(ROOT / "cp2048_causal_r2.pt", "cuda")
    direct = load_cp(ROOT / "cp_r2048.pt", "cuda")
    selection_rows = np.load(DEFAULT_CORPUS, mmap_mode="r")
    ranking, scores, component_scores, selection_records = downstream_ranking(
        model, language, direct, selection_rows, selection_config)

    candidates = {}
    for count in selection_config["selected_counts"]:
        candidates[f"top_{count}"] = (
            SelectiveBlend(language, direct, ranking[:count].cuda(),
                           selection_config["blend_lambda"]).eval(), False)
    generator = torch.Generator().manual_seed(selection_config["seed"] + 1)
    random_indices = torch.randperm(2048, generator=generator)[:512].cuda()
    candidates["random_512"] = (
        SelectiveBlend(language, direct, random_indices,
                       selection_config["blend_lambda"]).eval(), False)

    corpora = load_rows(eval_config)
    result = {
        "status": "confirmatory_downstream_ranking",
        "selection_config": selection_config,
        "selection_records": selection_records,
        "ranking": ranking.tolist(),
        "scores": scores.tolist(),
        "component_scores": {name: values.tolist() for name, values in component_scores.items()},
        "behavior": {}, "causal": {},
    }
    for corpus, rows in corpora.items():
        print("downstream curve behavior", corpus, flush=True)
        result["behavior"][corpus] = behavioral_evaluation(
            model, candidates, rows, eval_config, "cuda")
        print("downstream curve causal", corpus, flush=True)
        result["causal"][corpus] = causal_evaluation(
            model, candidates, rows[:eval_config["causal_rows"]],
            eval_config["interventions"], eval_config, "cuda")

    kl = lambda corpus, name: result["behavior"][corpus]["models"][name]["mean_kl"]["mean"]
    causal = lambda corpus, name, condition: result["causal"][corpus][name][condition]["fisher_effect_fidelity"]
    base, selected, full, random = "top_0", "top_512", "top_2048", "random_512"
    behavior_fraction = improvement_fraction(
        kl("fineweb", base), kl("fineweb", selected), kl("fineweb", full), False)
    causal_fractions = {condition: improvement_fraction(
        causal("fineweb", base, condition), causal("fineweb", selected, condition),
        causal("fineweb", full, condition), True)
        for condition in eval_config["interventions"]}
    transfer_wins = {corpus: (
        kl(corpus, selected) < kl(corpus, base)
        and all(causal(corpus, selected, condition) > causal(corpus, base, condition)
                for condition in eval_config["interventions"]))
        for corpus in corpora}
    mean_kl = lambda name: float(np.mean([kl(corpus, name) for corpus in corpora]))
    mean_causal = lambda name: float(np.mean([
        causal(corpus, name, condition) for corpus in corpora
        for condition in eval_config["interventions"]]))
    result["hypotheses"] = {
        "H1_sparse_behavior": {"supported": behavior_fraction >= 0.70,
                                "recovered_fraction": behavior_fraction},
        "H2_sparse_causal": {"supported": all(value >= 0.70 for value in causal_fractions.values())
                              and all(causal("fineweb", selected, condition)
                                      >= causal("fineweb", base, condition)
                                      for condition in eval_config["interventions"]),
                             "recovered_fractions": causal_fractions},
        "H3_transfer": {"supported": sum(transfer_wins.values()) >= 2,
                        "corpus_wins": transfer_wins},
        "H4_ranking_control": {"supported": mean_kl(selected) < mean_kl(random)
                                and mean_causal(selected) > mean_causal(random),
                               "ranked_mean_kl": mean_kl(selected),
                               "random_mean_kl": mean_kl(random),
                               "ranked_mean_causal": mean_causal(selected),
                               "random_mean_causal": mean_causal(random)},
    }
    OUTPUT.write_text(json.dumps(json_ready(result), indent=2, allow_nan=False))

    lines = ["# Downstream-ranked CP correction curve", "", "## Hypotheses", ""]
    for name, row in result["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Downstream rank selection](downstream_rank_selection.png)", "",
              "| Terms | Rank upper bound | FineWeb KL | FW Gaussian | FW natural |",
              "|---:|---:|---:|---:|---:|"]
    for count in selection_config["selected_counts"]:
        name = f"top_{count}"
        lines.append(f"| {count} | {2048 + count} | {kl('fineweb', name):.4f} "
                     f"| {causal('fineweb', name, 'gaussian_rms_0.10'):.3f} "
                     f"| {causal('fineweb', name, 'natural_mix_0.25'):.3f} |")
    lines += ["", "The ranking is the negative first-order gradient at the language model, averaged after separately normalizing base-KL, intervened-KL, and Fisher-effect gradients. Selection rows are disjoint from all evaluation rows.", "",
              "This is still an inherited-factor-basis test. A failure means no small subset was found by this local ranking; it is not a proof that no sparse basis exists.", "",
              "Raw results: `downstream_rank_selection_results.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    xs = np.asarray(selection_config["selected_counts"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for corpus in corpora:
        axes[0].plot(xs, [kl(corpus, f"top_{count}") for count in xs], marker="o", label=corpus)
        axes[1].plot(xs, [causal(corpus, f"top_{count}", "gaussian_rms_0.10") for count in xs], marker="o")
        axes[2].plot(xs, [causal(corpus, f"top_{count}", "natural_mix_0.25") for count in xs], marker="o")
    axes[0].set_ylabel("KL"); axes[0].legend()
    axes[1].set_ylabel("Gaussian causal fidelity")
    axes[2].set_ylabel("Natural causal fidelity")
    for axis in axes:
        axis.set_xlabel("Selected correction terms")
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
