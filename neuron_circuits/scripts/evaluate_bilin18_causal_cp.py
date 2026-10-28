#!/usr/bin/env python3
"""Evaluate causally distilled CP-2048 variants against the frozen baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from audit_bilin18_mlp_discriminators import SNAPSHOT, TOKENIZER
from evaluate_bilin18_decomposition_metrics import (
    behavioral_evaluation,
    causal_evaluation,
    corpus_rows,
    cp_term_cosine,
    encode_cp,
    factor_stability_summary,
    json_ready,
    linear_cka,
    semantic_categories,
)
from refine_bilin18_tensor_decomposition import load_model
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_tensor_decomposition import ROOT, natural_metrics, tensor_probe_metrics
from train_bilin18_sparse_codes import SPLITS
from transformers import AutoTokenizer


CONFIG = Path("configs/bilin18_causal_distillation.json")
CONFIG_R2 = Path("configs/bilin18_causal_distillation_r2.json")
SCORE_CONFIG = Path("configs/bilin18_decomposition_evaluation.json")
DATA = Path("artifacts/bilin18_sparse_codes")
OUTPUT = ROOT / "causal_distillation_results.json"
REPORT = ROOT / "CAUSAL_DISTILLATION_REPORT.md"
FIGURE = ROOT / "causal_distillation_summary.png"
DISPLAY = {
    "activation_baseline": "Activation CP baseline",
    "cp2048_kl": "KL distilled",
    "cp2048_causal": "Causally distilled",
    "cp2048_causal_tensor": "Causal + tensor anchor",
}


def load_cp(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = load_model("cp_r2048")
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("r1", "r2"), default="r1")
    return parser.parse_args()


def load_candidates(device, profile="r1"):
    suffix = "" if profile == "r1" else "_r2"
    return {
        "activation_baseline": load_cp(ROOT / "cp_r2048_activation.pt", device),
        "cp2048_kl": load_cp(ROOT / f"cp2048_kl{suffix}.pt", device),
        "cp2048_causal": load_cp(ROOT / f"cp2048_causal{suffix}.pt", device),
        "cp2048_causal_tensor": load_cp(ROOT / f"cp2048_causal_tensor{suffix}.pt", device),
    }


def tensor_and_write_metrics(candidates, device):
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    left = state["transformer.h.1.mlp.Left.weight"].float()
    right = state["transformer.h.1.mlp.Right.weight"].float()
    down = state["transformer.h.1.mlp.Down.weight"].float()
    bias = state["transformer.h.1.mlp.Down_bias"].float()
    del state
    inputs = np.load(DATA / "layer1_inputs_large.npz")
    products = np.load(DATA / "natural_large.npz")
    result = {}
    for index, (name, model) in enumerate(candidates.items()):
        print("tensor/write", name, flush=True)
        row = {"tensor": tensor_probe_metrics(
            model, left, right, down, 88000 + index, batches=16), "natural": {}}
        for split in SPLITS:
            x = torch.from_numpy(inputs[f"{split}__input1"].astype(np.float32))
            h = torch.from_numpy(products[f"{split}__h1"].astype(np.float32))
            row["natural"][split] = natural_metrics(model, x, h, down, bias)
        result[name] = row
    return result


def stability_metrics(candidates, device, tokens=2048):
    archive = np.load(DATA / "layer1_inputs_large.npz")
    values = torch.from_numpy(archive["fineweb_test__input1"][:tokens].astype(np.float32))
    baseline = candidates["activation_baseline"]
    baseline_code = encode_cp(baseline, values, False).float().to(device)
    result = {}
    for name in tuple(candidates)[1:]:
        print("stability", name, flush=True)
        code = encode_cp(candidates[name], values, False).float().to(device)
        similarity = cp_term_cosine(baseline, candidates[name])
        row = factor_stability_summary(similarity)
        row["activation_linear_cka"] = linear_cka(baseline_code, code)
        result[name] = row
        del code, similarity
    return result


def semantic_metrics(candidates, train_tokens=16384, test_tokens=8192):
    inputs = np.load(DATA / "layer1_inputs_large.npz")
    products = np.load(DATA / "natural_large.npz")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    train_x = torch.from_numpy(inputs["fineweb_train__input1"][:train_tokens].astype(np.float32))
    test_x = {split: torch.from_numpy(inputs[f"{split}__input1"][:test_tokens].astype(np.float32))
              for split in SPLITS}
    train_ids = products["fineweb_train__token_ids"][:train_tokens]
    test_ids = {split: products[f"{split}__token_ids"][:test_tokens] for split in SPLITS}
    result = {}
    for name, model in candidates.items():
        print("semantics", name, flush=True)
        train_code = encode_cp(model, train_x, False)
        test_codes = {split: encode_cp(model, values, False) for split, values in test_x.items()}
        result[name] = semantic_categories(
            train_code, test_codes, train_ids, test_ids, tokenizer)
        del train_code, test_codes
    return result


def scale_and_ood_metrics(model, candidates, corpora, config, device):
    held_out = {
        "gaussian_0.075": {"kind": "gaussian", "scale": 0.075},
        "gaussian_0.150": {"kind": "gaussian", "scale": 0.150},
        "natural_mix_0.175": {"kind": "natural_mix", "scale": 0.175},
        "natural_mix_0.375": {"kind": "natural_mix", "scale": 0.375},
    }
    # One call per Gaussian scale makes its directions paired across conditions.
    fineweb = {name: {} for name in candidates}
    for condition, spec in held_out.items():
        rows = causal_evaluation(
            model, {name: (candidate, False) for name, candidate in candidates.items()},
            corpora["fineweb"][:24], {condition: spec}, config, device)
        for name in candidates:
            fineweb[name][condition] = rows[name][condition]
    primary = {
        "gaussian_0.10": {"kind": "gaussian", "scale": 0.10},
        "natural_mix_0.25": {"kind": "natural_mix", "scale": 0.25},
    }
    ood = {}
    for corpus in ("wikitext103", "tinystories"):
        ood[corpus] = causal_evaluation(
            model, {name: (candidate, False) for name, candidate in candidates.items()},
            corpora[corpus][:24], primary, config, device)
    return {"fineweb_held_out_scales": fineweb, "ood_primary_scales": ood}


def hypothesis_verdicts(result, distill_config, score_config):
    behavior = result["behavior"]
    baseline = "activation_baseline"
    kl_model = "cp2048_kl"
    causal = "cp2048_causal"
    anchored = "cp2048_causal_tensor"
    h1_fineweb = (behavior["fineweb"]["models"][kl_model]["mean_kl"]["mean"]
                  < behavior["fineweb"]["models"][baseline]["mean_kl"]["mean"])
    h1_ood = all(
        behavior[corpus]["models"][kl_model]["mean_kl"]["mean"]
        <= behavior[corpus]["models"][baseline]["mean_kl"]["mean"] + 0.005
        for corpus in ("wikitext103", "tinystories"))

    h2_deltas = {}
    for condition in score_config["interventions"]:
        h2_deltas[condition] = (
            result["causal_primary"][causal][condition]["fisher_effect_fidelity"]
            - result["causal_primary"][baseline][condition]["fisher_effect_fidelity"])
    h2 = all(value > 0 for value in h2_deltas.values())

    causal_tensor = result["tensor_and_write"][causal]["tensor"]["median_fidelity"]
    anchored_tensor = result["tensor_and_write"][anchored]["tensor"]["median_fidelity"]
    kl_costs = {corpus:
        behavior[corpus]["models"][anchored]["mean_kl"]["mean"]
        - behavior[corpus]["models"][causal]["mean_kl"]["mean"]
        for corpus in behavior}
    causal_costs = {condition:
        result["causal_primary"][causal][condition]["fisher_effect_fidelity"]
        - result["causal_primary"][anchored][condition]["fisher_effect_fidelity"]
        for condition in score_config["interventions"]}
    h3 = (anchored_tensor > causal_tensor and max(kl_costs.values()) <= 0.005
          and max(causal_costs.values()) <= 0.05)

    transfer = result["causal_transfer"]
    transfer_deltas = {}
    for condition in transfer["fineweb_held_out_scales"][baseline]:
        transfer_deltas[f"fineweb/{condition}"] = (
            transfer["fineweb_held_out_scales"][causal][condition]["fisher_effect_fidelity"]
            - transfer["fineweb_held_out_scales"][baseline][condition]["fisher_effect_fidelity"])
    for corpus, rows in transfer["ood_primary_scales"].items():
        for condition in rows[baseline]:
            transfer_deltas[f"{corpus}/{condition}"] = (
                rows[causal][condition]["fisher_effect_fidelity"]
                - rows[baseline][condition]["fisher_effect_fidelity"])
    h4 = all(value > 0 for value in transfer_deltas.values())

    gates = score_config["gates"]
    gate_rows = {}
    for name in candidates_for_hypotheses(result):
        checks = {}
        for condition, row in result["causal_primary"][name].items():
            checks[f"{condition}_fidelity"] = row["fisher_effect_fidelity"] >= gates["causal_effect_fidelity_min"]
            checks[f"{condition}_cosine"] = row["fisher_effect_cosine"] >= gates["causal_effect_cosine_min"]
        gate_rows[name] = {"checks": checks, "passes": all(checks.values())}
    h5 = any(row["passes"] for row in gate_rows.values())
    return {
        "H1_kl_behavior": {"supported": h1_fineweb and h1_ood,
                           "fineweb_improved": h1_fineweb, "ood_within_tolerance": h1_ood},
        "H2_causal_training": {"supported": h2, "fidelity_improvements": h2_deltas},
        "H3_tensor_anchor": {"supported": h3, "causal_tensor_fidelity": causal_tensor,
                             "anchored_tensor_fidelity": anchored_tensor,
                             "kl_costs": kl_costs, "causal_fidelity_costs": causal_costs},
        "H4_scale_transfer": {"supported": h4, "fidelity_improvements": transfer_deltas},
        "H5_causal_gate": {"supported": h5, "models": gate_rows},
    }


def candidates_for_hypotheses(result):
    return tuple(result["causal_primary"])


def make_figure(result):
    names = tuple(result["causal_primary"])
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(3); width = 0.19
    for index, name in enumerate(names):
        axes[0, 0].bar(x + (index - 1.5) * width,
                       [result["behavior"][corpus]["models"][name]["mean_kl"]["mean"]
                        for corpus in ("fineweb", "wikitext103", "tinystories")],
                       width, color=colors[index], label=DISPLAY[name])
    axes[0, 0].set_xticks(x, ("FineWeb", "WikiText", "TinyStories"))
    axes[0, 0].set_ylabel("Teacher → student KL")
    axes[0, 0].set_title("Behavioral fidelity")
    axes[0, 0].legend(fontsize=8)

    conditions = tuple(next(iter(result["causal_primary"].values())))
    x2 = np.arange(len(conditions))
    for index, name in enumerate(names):
        axes[0, 1].bar(x2 + (index - 1.5) * width,
                       [result["causal_primary"][name][condition]["fisher_effect_fidelity"]
                        for condition in conditions], width, color=colors[index])
    axes[0, 1].axhline(0.9, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_xticks(x2, ("Gaussian .10", "Natural mix .25"))
    axes[0, 1].set_ylabel("Softmax-Fisher effect fidelity")
    axes[0, 1].set_title("Primary causal fidelity")

    axes[1, 0].bar(np.arange(len(names)),
                   [result["tensor_and_write"][name]["tensor"]["median_fidelity"] for name in names],
                   color=colors[:len(names)])
    axes[1, 0].set_xticks(np.arange(len(names)),
                          ("Baseline", "KL", "Causal", "Causal+tensor"), rotation=12)
    axes[1, 0].set_ylabel("Isotropic tensor fidelity")
    axes[1, 0].set_title("Global weight behavior")

    transfer = result["causal_transfer"]["fineweb_held_out_scales"]
    transfer_conditions = tuple(transfer[names[0]])
    for index, name in enumerate(names):
        axes[1, 1].plot(np.arange(len(transfer_conditions)),
                        [transfer[name][condition]["fisher_effect_fidelity"]
                         for condition in transfer_conditions], marker="o",
                        color=colors[index], label=DISPLAY[name])
    axes[1, 1].axhline(0.9, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xticks(np.arange(len(transfer_conditions)),
                          ("G .075", "G .15", "N .175", "N .375"))
    axes[1, 1].set_ylabel("Effect fidelity")
    axes[1, 1].set_title("Held-out intervention magnitudes")
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)


def write_report(result):
    h = result["hypotheses"]
    title = "Causal CP distillation results" + (" — optimizer-corrected R2" if result.get("profile") == "r2" else "")
    lines = [f"# {title}", "", "## Outcome", ""]
    for name, row in h.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"![Causal distillation summary]({FIGURE.name})", "",
              "## Behavior", "",
              "| Model | FineWeb KL | WikiText KL | TinyStories KL |", "|---|---:|---:|---:|"]
    for name in result["causal_primary"]:
        lines.append(f"| {DISPLAY[name]} | " + " | ".join(
            f"{result['behavior'][corpus]['models'][name]['mean_kl']['mean']:.4f}"
            for corpus in ("fineweb", "wikitext103", "tinystories")) + " |")
    lines += ["", "## Causal fidelity", "",
              "| Model | Gaussian .10 fidelity | Gaussian cosine | Natural .25 fidelity | Natural cosine |",
              "|---|---:|---:|---:|---:|"]
    for name, rows in result["causal_primary"].items():
        gaussian, natural = rows["gaussian_rms_0.10"], rows["natural_mix_0.25"]
        lines.append(f"| {DISPLAY[name]} | {gaussian['fisher_effect_fidelity']:.3f} "
                     f"| {gaussian['fisher_effect_cosine']:.3f} "
                     f"| {natural['fisher_effect_fidelity']:.3f} "
                     f"| {natural['fisher_effect_cosine']:.3f} |")
    lines += ["", "## Global/local tradeoff", "",
              "| Model | Tensor fidelity | FineWeb write fidelity | Factor cosine to baseline | Activation CKA |",
              "|---|---:|---:|---:|---:|"]
    for name in result["causal_primary"]:
        tensor = result["tensor_and_write"][name]
        if name == "activation_baseline":
            term, cka = 1.0, 1.0
        else:
            term = result["stability"][name]["a_to_b_median_best_abs_cosine"]
            cka = result["stability"][name]["activation_linear_cka"]
        lines.append(f"| {DISPLAY[name]} | {tensor['tensor']['median_fidelity']:.3f} "
                     f"| {tensor['natural']['fineweb_test']['median_fidelity']:.3f} "
                     f"| {term:.3f} | {cka:.3f} |")
    lines += ["", "## Red-team interpretation", "",
              "- Training on a causal metric can improve that metric without proving that the learned factors are canonical or human-semantic.",
              "- Held-out magnitudes and OOD corpora test interpolation and distribution transfer, but not unrelated intervention families or other layers.",
              "- All variants share one architecture and initialization. This is an objective comparison, not independent rediscovery.",
              "- The teacher itself defines the distillation target; agreement does not explain why the teacher computes the behavior.",
              "", f"Raw results: `{OUTPUT.name}`."]
    REPORT.write_text("\n".join(lines) + "\n")


def main():
    global OUTPUT, REPORT, FIGURE
    args = parse_args()
    profile = args.profile
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = CONFIG if profile == "r1" else CONFIG_R2
    if profile == "r2":
        OUTPUT = ROOT / "causal_distillation_r2_results.json"
        REPORT = ROOT / "CAUSAL_DISTILLATION_R2_REPORT.md"
        FIGURE = ROOT / "causal_distillation_r2_summary.png"
    distill_config = json.loads(config_path.read_text())
    score_config = json.loads(SCORE_CONFIG.read_text())
    candidates = load_candidates(device, profile)
    wrapped = {name: (candidate, False) for name, candidate in candidates.items()}
    model = load_full_model(device)
    corpora = corpus_rows(score_config)
    result = {"profile": profile, "distillation_config": distill_config, "score_config": score_config,
              "behavior": {}}
    for corpus, rows in corpora.items():
        print("behavior", corpus, flush=True)
        result["behavior"][corpus] = behavioral_evaluation(
            model, wrapped, rows, score_config, device)
    result["causal_primary"] = causal_evaluation(
        model, wrapped, corpora["fineweb"][:score_config["causal_rows"]],
        score_config["interventions"], score_config, device)
    result["causal_transfer"] = scale_and_ood_metrics(
        model, candidates, corpora, score_config, device)
    result["tensor_and_write"] = tensor_and_write_metrics(candidates, device)
    result["stability"] = stability_metrics(candidates, device)
    result["semantics"] = semantic_metrics(candidates)
    result["hypotheses"] = hypothesis_verdicts(result, distill_config, score_config)
    OUTPUT.write_text(json.dumps(json_ready(result), indent=2, allow_nan=False))
    make_figure(result); write_report(result)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
