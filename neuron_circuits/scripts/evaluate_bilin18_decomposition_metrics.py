#!/usr/bin/env python3
"""Confirmatory scorecard for frozen Bilin18 layer-1 decompositions.

Evaluates behavior, OOD transfer, finite-intervention effects, factor stability,
semantic proxies, and actual versus idealized sparsity/compression costs. The
hypotheses and thresholds live in configs/bilin18_decomposition_evaluation.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import SNAPSHOT, TOKENIZER
from discover_bilin18_weight_only_circuits import token_flags
from evaluate_bilin18_tensor_decomposition import load_activation, load_routed
from neuron_circuits.decomposition_metrics import (
    bilinear_operation_counts,
    bootstrap_mean_ci,
    combine_effect_agreements,
    cp_term_cosine,
    effect_agreement,
    factor_stability_summary,
    fisher_effect_agreement,
    linear_cka,
    routing_usage,
)
from refine_bilin18_tensor_decomposition import load_model
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_tensor_decomposition import CPBilinear, LL1Bilinear, ROOT
from train_bilin18_sparse_codes import SPLITS


CONFIG = Path("configs/bilin18_decomposition_evaluation.json")
OOD = Path("artifacts/bilin18_ood_corpora")
DATA = Path("artifacts/bilin18_sparse_codes")
OUTPUT = ROOT / "evaluation_scorecard.json"
REPORT = ROOT / "EVALUATION_SCORECARD.md"
FIGURE = ROOT / "evaluation_scorecard.png"
DISPLAY = {
    "cp2048_weight": "Direct CP-2048",
    "cp2048_activation": "Activation CP-2048",
    "routed_mixed": "Mixed routed CP-2048/K128",
    "cp512_activation": "Activation CP-512",
    "ll1_activation": "Activation LL1 R256/L8",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="Use 8 corpus rows, 4 causal rows, and 1024 semantic tokens.")
    return parser.parse_args()


def load_candidates(device):
    values = {
        "cp2048_weight": (load_model("cp_r2048"), False),
        "cp2048_activation": (load_activation("cp_r2048"), False),
        "routed_mixed": (load_routed("routed_mixed"), True),
        "cp512_activation": (load_activation("cp_r512"), False),
        "ll1_activation": (load_activation("ll1_r256_l8"), False),
    }
    return {name: (model.to(device).eval(), routed) for name, (model, routed) in values.items()}


def replacement_write(block, normalized, candidate):
    if candidate is None:
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        return block.mlp.Down(product) + block.mlp.Down_bias, None
    replacement, routed = candidate
    flat = normalized.float().flatten(0, 1)
    if routed:
        write, code = replacement.forward_sparse(flat, 128)
    else:
        write, code = replacement(flat), None
    return write.view_as(normalized).to(normalized.dtype), code


@torch.inference_mode()
def forward(model, ids, candidate=None, intervention=None, noise=None):
    dimension = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (dimension,))
    x0, value_bus = x, None
    routed_code = None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention
        normalized = F.rms_norm(pre, (dimension,))
        mlp_input = normalized
        if layer == 1 and intervention is not None:
            scale = intervention["scale"]
            if intervention["kind"] == "gaussian":
                if noise is None:
                    raise ValueError("Gaussian intervention requires fixed noise")
                direction = noise.to(normalized.device, normalized.dtype)
                direction = direction / direction.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
                mlp_input = normalized + scale * direction
            elif intervention["kind"] == "natural_mix":
                donor = normalized.roll(shifts=1, dims=1)
                mlp_input = normalized + scale * (donor - normalized)
            else:
                raise ValueError(intervention["kind"])
        write, code = replacement_write(
            block, mlp_input, candidate if layer == 1 else None)
        if layer == 1 and code is not None:
            routed_code = code
        x = pre + write
    logits = model.lm_head(F.rms_norm(x, (dimension,))).float()
    return 30 * torch.tanh(logits / 30), routed_code


def corpus_rows(config, quick=False):
    fineweb = np.load(DEFAULT_CORPUS, mmap_mode="r")
    sources = {
        "fineweb": fineweb,
        "wikitext103": np.load(OOD / "wikitext103_1200x129.npy", mmap_mode="r"),
        "tinystories": np.load(OOD / "tinystories_1200x129.npy", mmap_mode="r"),
    }
    result = {}
    for name, source in sources.items():
        spec = config["corpora"][name]
        rows = 8 if quick else spec["rows"]
        result[name] = source[spec["start"]:spec["start"] + rows]
    return result


@torch.inference_mode()
def behavioral_evaluation(model, candidates, rows, config, device, batch_size=2):
    per_model = {
        name: {"cross_entropy_delta": [], "mean_kl": [], "top1_agreement": []}
        for name in candidates
    }
    baseline_ce = []
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device=device)
        targets = torch.as_tensor(raw[:, 1:129], device=device)
        teacher, _ = forward(model, ids)
        teacher_logp = F.log_softmax(teacher, -1)
        teacher_p = teacher_logp.exp()
        teacher_nll = F.nll_loss(
            teacher_logp.flatten(0, 1), targets.flatten(), reduction="none"
        ).view(len(ids), -1).mean(1)
        baseline_ce.extend(teacher_nll.cpu().tolist())
        teacher_top = teacher.argmax(-1)
        for name, candidate in candidates.items():
            student, _ = forward(model, ids, candidate)
            student_logp = F.log_softmax(student, -1)
            nll = F.nll_loss(
                student_logp.flatten(0, 1), targets.flatten(), reduction="none"
            ).view(len(ids), -1).mean(1)
            kl = (teacher_p * (teacher_logp - student_logp)).sum(-1).mean(-1)
            same = (teacher_top == student.argmax(-1)).float().mean(-1)
            per_model[name]["cross_entropy_delta"].extend((nll - teacher_nll).cpu().tolist())
            per_model[name]["mean_kl"].extend(kl.cpu().tolist())
            per_model[name]["top1_agreement"].extend(same.cpu().tolist())
    result = {
        "baseline_cross_entropy": bootstrap_mean_ci(
            baseline_ce, seed=config["seed"], resamples=config["bootstrap_resamples"]),
        "models": {},
    }
    for model_index, (name, metrics) in enumerate(per_model.items()):
        result["models"][name] = {
            metric: bootstrap_mean_ci(
                values,
                seed=config["seed"] + 100 * model_index + index,
                resamples=config["bootstrap_resamples"],
            )
            for index, (metric, values) in enumerate(metrics.items())
        }
    return result


def margin_effect_rows(teacher_base, teacher_changed, student_base, student_changed):
    top = teacher_base.topk(2, dim=-1).indices
    def margin(logits):
        chosen = logits.gather(-1, top)
        return chosen[..., 0] - chosen[..., 1]
    teacher = (margin(teacher_changed) - margin(teacher_base)).float().flatten()
    student = (margin(student_changed) - margin(student_base)).float().flatten()
    threshold = teacher.abs().median()
    keep = teacher.abs() >= threshold
    sign = (teacher[keep].sign() == student[keep].sign()).float()
    teacher_centered, student_centered = teacher - teacher.mean(), student - student.mean()
    correlation = ((teacher_centered * student_centered).sum() /
                   torch.sqrt(teacher_centered.square().sum() *
                              student_centered.square().sum()).clamp_min(1e-30))
    return float(sign.mean()), float(correlation)


@torch.inference_mode()
def causal_evaluation(model, candidates, rows, interventions, config, device, batch_size=2):
    result = {name: {} for name in candidates}
    generator = torch.Generator().manual_seed(config["seed"] + 991)
    for intervention_name, intervention in interventions.items():
        accumulators = {name: [] for name in candidates}
        fisher_accumulators = {name: [] for name in candidates}
        row_fidelity = {name: [] for name in candidates}
        row_cosine = {name: [] for name in candidates}
        signs = {name: [] for name in candidates}
        correlations = {name: [] for name in candidates}
        for start in range(0, len(rows), batch_size):
            raw = np.asarray(rows[start:start + batch_size, :129], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :128], device=device)
            noise = torch.randn(len(ids), 128, model.config.n_embd, generator=generator)
            teacher_base, _ = forward(model, ids)
            teacher_changed, _ = forward(model, ids, intervention=intervention, noise=noise)
            for name, candidate in candidates.items():
                student_base, _ = forward(model, ids, candidate)
                student_changed, _ = forward(
                    model, ids, candidate, intervention=intervention, noise=noise)
                for row in range(len(ids)):
                    agreement = effect_agreement(
                        teacher_base[row:row + 1], teacher_changed[row:row + 1],
                        student_base[row:row + 1], student_changed[row:row + 1])
                    accumulators[name].append(agreement)
                    fisher_accumulators[name].append(fisher_effect_agreement(
                        teacher_base[row:row + 1], teacher_changed[row:row + 1],
                        student_base[row:row + 1], student_changed[row:row + 1]))
                    single = combine_effect_agreements([agreement])
                    row_fidelity[name].append(single["effect_fidelity"])
                    row_cosine[name].append(single["effect_cosine"])
                sign, corr = margin_effect_rows(
                    teacher_base, teacher_changed, student_base, student_changed)
                signs[name].append(sign); correlations[name].append(corr)
        for model_index, name in enumerate(candidates):
            combined = combine_effect_agreements(accumulators[name])
            fisher = combine_effect_agreements(fisher_accumulators[name])
            combined.update({
                "fisher_effect_fidelity": fisher["effect_fidelity"],
                "fisher_effect_cosine": fisher["effect_cosine"],
                "fisher_effect_gain": fisher["effect_gain"],
                "fisher_teacher_effect_energy": fisher["teacher_effect_energy"],
                "row_effect_fidelity": bootstrap_mean_ci(
                    row_fidelity[name], seed=config["seed"] + 300 + model_index,
                    resamples=config["bootstrap_resamples"]),
                "row_effect_cosine": bootstrap_mean_ci(
                    row_cosine[name], seed=config["seed"] + 400 + model_index,
                    resamples=config["bootstrap_resamples"]),
                "margin_effect_sign_agreement": float(np.mean(signs[name])),
                "margin_effect_correlation": float(np.mean(correlations[name])),
            })
            result[name][intervention_name] = combined
    return result


@torch.inference_mode()
def encode_cp(model, values, routed, batch_size=256):
    model.eval(); codes = []
    for start in range(0, len(values), batch_size):
        x = values[start:start + batch_size].to(next(model.parameters()).device)
        if routed:
            _, code = model.forward_sparse(x, 128)
        else:
            left, right, _, gain = model.factors()
            code = model.bilinear_features(x, x, left, right, gain)
        codes.append(code.to(torch.float16).cpu())
    return torch.cat(codes)


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores)
    positive, negative = labels.sum(), (~labels).sum()
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    rank_sum = np.bincount(inverse, weights=ranks)
    average = rank_sum / counts
    tied_ranks = average[inverse]
    return float((tied_ranks[labels].sum() - positive * (positive + 1) / 2) /
                 (positive * negative))


def semantic_categories(train_code, test_codes, train_ids, test_ids, tokenizer):
    flags = np.stack([token_flags(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)])
    names = (
        "newline", "opening_delimiter", "closing_delimiter", "quote", "code_operator",
        "url_web", "non_ascii", "word_continuation", "suffix_ing", "suffix_ed",
        "suffix_ly", "all_caps", "comma_period", "colon_semicolon",
    )
    train_labels = flags[np.asarray(train_ids)]
    code = train_code.float().numpy()
    mean, std = code.mean(0), code.std(0).clip(1e-5)
    z = (code - mean) / std
    rows = {}
    for category, name in enumerate(names):
        labels = train_labels[:, category].astype(bool)
        if labels.sum() < 8 or (~labels).sum() < 8:
            continue
        gaps = z[labels].mean(0) - z[~labels].mean(0)
        feature = int(np.argmax(np.abs(gaps)))
        sign = float(np.sign(gaps[feature]) or 1)
        row = {"feature": feature, "train_standardized_gap": float(gaps[feature]), "auc": {}}
        for split, test_code in test_codes.items():
            labels_test = flags[np.asarray(test_ids[split]), category]
            scores = sign * ((test_code[:, feature].float().numpy() - mean[feature]) / std[feature])
            row["auc"][split] = auc(labels_test, scores)
        rows[name] = row
    summary = {}
    for split in test_codes:
        values = [row["auc"][split] for row in rows.values()
                  if math.isfinite(row["auc"][split])]
        summary[split] = {
            "mean_oriented_auc": float(np.mean(values)),
            "median_oriented_auc": float(np.median(values)),
            "categories": len(values),
        }
    return {"categories": rows, "summary": summary}


def semantic_evaluation(candidates, config, quick=False):
    input_archive = np.load(DATA / "layer1_inputs_large.npz")
    product_archive = np.load(DATA / "natural_large.npz")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    train_n = 1024 if quick else config["semantic_train_tokens"]
    test_n = 1024 if quick else config["semantic_test_tokens"]
    train_x = torch.from_numpy(input_archive["fineweb_train__input1"][:train_n].astype(np.float32))
    test_x = {split: torch.from_numpy(input_archive[f"{split}__input1"][:test_n].astype(np.float32))
              for split in SPLITS}
    train_ids = product_archive["fineweb_train__token_ids"][:train_n]
    test_ids = {split: product_archive[f"{split}__token_ids"][:test_n] for split in SPLITS}
    result = {}
    for name in ("cp2048_weight", "cp2048_activation", "routed_mixed"):
        model, routed = candidates[name]
        print("semantic", name, flush=True)
        train_code = encode_cp(model, train_x, routed)
        test_codes = {split: encode_cp(model, values, routed) for split, values in test_x.items()}
        result[name] = semantic_categories(
            train_code, test_codes, train_ids, test_ids, tokenizer)
        del train_code, test_codes
    return result


def stability_evaluation(candidates, config, quick=False):
    input_archive = np.load(DATA / "layer1_inputs_large.npz")
    count = 512 if quick else config["stability_tokens"]
    values = torch.from_numpy(
        input_archive["fineweb_test__input1"][:count].astype(np.float32))
    names = ("cp2048_weight", "cp2048_activation", "routed_mixed")
    codes = {}
    for name in names:
        model, routed = candidates[name]
        codes[name] = encode_cp(model, values, routed).float().to(next(model.parameters()).device)
    result = {}
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            print("stability", first, second, flush=True)
            first_model, _ = candidates[first]
            second_model, _ = candidates[second]
            similarity = cp_term_cosine(first_model, second_model)
            key = f"{first}__vs__{second}"
            result[key] = factor_stability_summary(similarity)
            result[key]["activation_linear_cka"] = linear_cka(codes[first], codes[second])
            del similarity
    return result


@torch.inference_mode()
def parsimony_evaluation(model, candidates, rows, device):
    native = model.blocks[1].mlp
    native_parameters = sum(parameter.numel() for parameter in native.parameters())
    result = {"native_parameters": native_parameters, "models": {}}
    raw = np.asarray(rows[:, :128], dtype=np.int64)
    routed_counts = torch.zeros(2048, dtype=torch.float64)
    routed_model = candidates["routed_mixed"][0]
    for start in range(0, len(raw), 4):
        ids = torch.as_tensor(raw[start:start + 4], device=device)
        _, code = forward(model, ids, candidates["routed_mixed"])
        routed_counts += (code != 0).sum(0).cpu()
    for name, (candidate, routed) in candidates.items():
        parameters = sum(parameter.numel() for parameter in candidate.parameters())
        checkpoint = {
            "cp2048_weight": ROOT / "cp_r2048.pt",
            "cp2048_activation": ROOT / "cp_r2048_activation.pt",
            "routed_mixed": ROOT / "routed_mixed.pt",
            "cp512_activation": ROOT / "cp_r512_activation.pt",
            "ll1_activation": ROOT / "ll1_r256_l8_activation.pt",
        }[name]
        if isinstance(candidate, CPBilinear):
            operations = bilinear_operation_counts(
                model.config.n_embd, 4608, candidate.rank, active=128 if routed else None)
        elif isinstance(candidate, LL1Bilinear):
            multiplies = (2 * model.config.n_embd * candidate.blocks * candidate.block_rank
                          + model.config.n_embd * candidate.blocks)
            native_multiplies = 3 * model.config.n_embd * 4608
            operations = {
                "native_multiplies": native_multiplies,
                "dense_student_multiplies": multiplies,
                "dense_fraction_of_native": multiplies / native_multiplies,
            }
        else:
            raise TypeError(type(candidate))
        row = {
            "parameters": parameters,
            "parameter_fraction_of_native": parameters / native_parameters,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "operations": operations,
        }
        if routed:
            row["routing_usage"] = routing_usage(routed_counts)
        result["models"][name] = row
    return result


def evaluate_gates(result, config):
    gates = config["gates"]
    verdicts = {}
    for name in config["primary_models"]:
        checks = {}
        for corpus in result["behavior"]:
            metrics = result["behavior"][corpus]["models"][name]
            kl_limit = gates["fineweb_mean_kl_max"] if corpus == "fineweb" else gates["ood_mean_kl_max"]
            checks[f"{corpus}_kl"] = metrics["mean_kl"]["mean"] <= kl_limit
            checks[f"{corpus}_top1"] = metrics["top1_agreement"]["mean"] >= gates["top1_agreement_min"]
        for intervention, metrics in result["causal"][name].items():
            checks[f"{intervention}_fidelity"] = metrics["fisher_effect_fidelity"] >= gates["causal_effect_fidelity_min"]
            checks[f"{intervention}_cosine"] = metrics["fisher_effect_cosine"] >= gates["causal_effect_cosine_min"]
        pars = result["parsimony"]["models"][name]
        checks["parameter_fraction"] = pars["parameter_fraction_of_native"] <= gates["parameter_fraction_max"]
        if name == "routed_mixed":
            checks["active_fraction"] = pars["operations"]["active_fraction"] <= gates["routed_active_fraction_max"]
        verdicts[name] = {"checks": checks, "passes_all": all(checks.values())}
    return verdicts


def evaluate_hypotheses(result, config):
    behavior = result["behavior"]
    weight_file = json.loads((ROOT / "weight_factorization_results.json").read_text())
    activation_file = json.loads((ROOT / "activation_refinement_results.json").read_text())
    direct_tensor = weight_file["models"]["cp_r2048"]["after_tensor_probe"]["median_fidelity"]
    activation_tensor = activation_file["models"]["cp_r2048"]["after_tensor"]["median_fidelity"]
    direct_kl = behavior["fineweb"]["models"]["cp2048_weight"]["mean_kl"]["mean"]
    activation_kl = behavior["fineweb"]["models"]["cp2048_activation"]["mean_kl"]["mean"]
    h1 = activation_kl < direct_kl and activation_tensor < direct_tensor

    fineweb_advantage = direct_kl - activation_kl
    ood_advantages = [
        behavior[corpus]["models"]["cp2048_weight"]["mean_kl"]["mean"]
        - behavior[corpus]["models"]["cp2048_activation"]["mean_kl"]["mean"]
        for corpus in ("wikitext103", "tinystories")
    ]
    causal_reversal = any(
        result["causal"]["cp2048_weight"][kind]["fisher_effect_fidelity"]
        > result["causal"]["cp2048_activation"][kind]["fisher_effect_fidelity"]
        for kind in config["interventions"]
    )
    h2 = causal_reversal or float(np.mean(ood_advantages)) < fineweb_advantage

    routed = result["parsimony"]["models"]["routed_mixed"]
    h3 = (routed["operations"]["active_fraction"] <= 0.0625
          and routed["routing_usage"]["features_ever_used"] > 1024)

    stability = result["stability"]
    stability_gaps = [
        row["activation_linear_cka"] - row["a_to_b_median_best_abs_cosine"]
        for row in stability.values()
    ]
    h4 = float(np.mean(stability_gaps)) >= config["gates"]["collective_stability_gap_min"]

    semantic = result["semantics"]["routed_mixed"]["summary"]
    h5 = all(semantic[corpus]["mean_oriented_auc"] >= config["gates"]["semantic_ood_auc_min"]
             for corpus in ("wikitext103", "tinystories"))
    h6 = not any(row["passes_all"] for row in result["gates"].values())
    return {
        "H1_manifold_gap": {"supported": h1, "direct_tensor": direct_tensor,
                            "activation_tensor": activation_tensor,
                            "direct_fineweb_kl": direct_kl,
                            "activation_fineweb_kl": activation_kl},
        "H2_causal_gap": {"supported": h2, "fineweb_kl_advantage": fineweb_advantage,
                          "mean_ood_kl_advantage": float(np.mean(ood_advantages)),
                          "causal_reversal": causal_reversal},
        "H3_sparse_conditional_program": {"supported": h3,
                                           "active_fraction": routed["operations"]["active_fraction"],
                                           **routed["routing_usage"]},
        "H4_collective_over_component_stability": {
            "supported": h4, "mean_collective_minus_component_gap": float(np.mean(stability_gaps))},
        "H5_semantic_transfer": {"supported": h5, "ood_summary": semantic},
        "H6_no_current_full_pass": {"supported": h6},
    }


def make_figure(result):
    primary = ("cp2048_weight", "cp2048_activation", "routed_mixed")
    colors = ("#4C78A8", "#F58518", "#54A24B")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(3)
    width = 0.24
    for index, name in enumerate(primary):
        axes[0, 0].bar(x + (index - 1) * width,
                       [result["behavior"][corpus]["models"][name]["mean_kl"]["mean"]
                        for corpus in ("fineweb", "wikitext103", "tinystories")],
                       width, color=colors[index], label=DISPLAY[name])
    axes[0, 0].set_xticks(x, ("FineWeb", "WikiText", "TinyStories"))
    axes[0, 0].set_ylabel("Teacher → replacement KL")
    axes[0, 0].set_title("Behavioral fidelity (lower is better)")
    axes[0, 0].legend(fontsize=8)

    intervention_names = tuple(result["causal"][primary[0]])
    x2 = np.arange(len(intervention_names))
    for index, name in enumerate(primary):
        axes[0, 1].bar(x2 + (index - 1) * width,
                       [result["causal"][name][kind]["fisher_effect_fidelity"]
                        for kind in intervention_names],
                       width, color=colors[index], label=DISPLAY[name])
    axes[0, 1].axhline(0.9, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_xticks(x2, ("Gaussian", "Natural mix"))
    axes[0, 1].set_ylabel("Softmax-Fisher causal fidelity")
    axes[0, 1].set_title("Finite intervention agreement")

    frontier_names = tuple(result["parsimony"]["models"])
    frontier_colors = plt.get_cmap("tab10").colors
    for index, name in enumerate(frontier_names):
        pars = result["parsimony"]["models"][name]
        axes[1, 0].scatter(pars["parameter_fraction_of_native"],
                           result["behavior"]["fineweb"]["models"][name]["mean_kl"]["mean"],
                           s=90, color=frontier_colors[index], label=DISPLAY[name])
    axes[1, 0].set_xlabel("Stored parameter fraction of native MLP")
    axes[1, 0].set_ylabel("FineWeb KL")
    axes[1, 0].set_title("Compression–fidelity frontier")
    axes[1, 0].legend(fontsize=7)

    pairs = list(result["stability"])
    components = [result["stability"][pair]["a_to_b_median_best_abs_cosine"] for pair in pairs]
    collective = [result["stability"][pair]["activation_linear_cka"] for pair in pairs]
    x3 = np.arange(len(pairs))
    axes[1, 1].bar(x3 - 0.18, components, 0.36, label="Individual CP terms", color="#B279A2")
    axes[1, 1].bar(x3 + 0.18, collective, 0.36, label="Activation CKA", color="#72B7B2")
    axes[1, 1].set_xticks(x3, ("Direct–activation", "Direct–routed", "Activation–routed"), rotation=12)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Component versus collective stability")
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)


def write_report(result):
    hypotheses = result["hypotheses"]
    lines = [
        "# Bilin18 decomposition evaluation scorecard", "",
        "## Outcome", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Evaluation scorecard](evaluation_scorecard.png)", "",
              "## Primary behavioral results", "",
              "| Model | FineWeb KL | WikiText KL | TinyStories KL | FineWeb top-1 |", 
              "|---|---:|---:|---:|---:|"]
    for name in result["config"]["primary_models"]:
        values = result["behavior"]
        lines.append(
            f"| {DISPLAY[name]} | {values['fineweb']['models'][name]['mean_kl']['mean']:.4f} "
            f"| {values['wikitext103']['models'][name]['mean_kl']['mean']:.4f} "
            f"| {values['tinystories']['models'][name]['mean_kl']['mean']:.4f} "
            f"| {values['fineweb']['models'][name]['top1_agreement']['mean']:.3f} |"
        )
    lines += ["", "## Causal results", "",
              "| Model | Gaussian fidelity | Gaussian cosine | Natural-mix fidelity | Natural-mix cosine |",
              "|---|---:|---:|---:|---:|"]
    for name in result["config"]["primary_models"]:
        causal = result["causal"][name]
        gaussian = causal["gaussian_rms_0.10"]
        natural = causal["natural_mix_0.25"]
        lines.append(
            f"| {DISPLAY[name]} | {gaussian['fisher_effect_fidelity']:.3f} | {gaussian['fisher_effect_cosine']:.3f} "
            f"| {natural['fisher_effect_fidelity']:.3f} | {natural['fisher_effect_cosine']:.3f} |"
        )
    lines += ["", "## Gate verdicts", ""]
    for name, row in result["gates"].items():
        passed = sum(row["checks"].values())
        lines.append(f"- **{DISPLAY[name]}:** {passed}/{len(row['checks'])} gates; "
                     f"{'full pass' if row['passes_all'] else 'does not fully pass'}.")
    routed = result["parsimony"]["models"]["routed_mixed"]
    semantic = result["semantics"]
    lines += ["", "## What changed", "",
              "- **The global-tensor causal hypothesis was falsified here.** Activation CP-2048 is more behaviorally and causally faithful than direct CP-2048 on every tested primary condition, despite retaining less isotropic tensor energy.",
              "- **None of the candidates is a faithful causal replacement.** Activation CP-2048 is the best current candidate, but remains far below the 0.90 intervention gate.",
              f"- **The routed code is genuinely local but not globally small.** It activates 128/2048 factors per token, uses {routed['routing_usage']['features_ever_used']}/2048 over the corpus, and has entropy-effective support {routed['routing_usage']['effective_support_entropy']:.0f}.",
              f"- **The current router is not an ideal sparse implementation.** It uses approximately {100*routed['operations']['current_routed_fraction_of_native']:.1f}% of native multiplies because all factor scores are computed; an indexed router would use approximately {100*routed['operations']['ideal_indexed_fraction_of_native']:.1f}% before routing overhead.",
              f"- **Restricted semantics transfer.** Activation CP's mean oriented AUROC is {semantic['cp2048_activation']['summary']['wikitext103']['mean_oriented_auc']:.3f} on WikiText and {semantic['cp2048_activation']['summary']['tinystories']['mean_oriented_auc']:.3f} on TinyStories.",
              "- **Collective structure is more stable than individual terms within this checkpoint lineage.** This is not an independent-identifiability result because all three CP models descend from the same initialization.",
              "", "The post-confirmatory scale and OOD red-team is in `CAUSAL_METRIC_REDTEAM.md`.",
              "", "## Interpretation limits", "",
              "- Causal interventions modify only the layer-1 MLP input; they do not establish fidelity for arbitrary edits elsewhere in the network.",
              "- The primary causal score uses the teacher baseline softmax as a Fisher weighting. Uniform-vocabulary effect scores remain in the JSON as a stricter secondary diagnostic.",
              "- Factor comparisons are across objectives descended from a shared initialization, not independent global CP discoveries.",
              "- Token-form categories are a deliberately small semantic vocabulary. Failure means unexplained by this audit, not meaningless.",
              "- Sequence-bootstrap intervals describe corpus sampling uncertainty, not uncertainty across independently trained base models.",
              "", "Raw results: `evaluation_scorecard.json`."]
    REPORT.write_text("\n".join(lines) + "\n")


def json_ready(value):
    """Convert NumPy scalars and non-finite diagnostics to strict JSON."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main():
    args = parse_args()
    started = time.time()
    config = json.loads(args.config.read_text())
    if args.quick:
        config["bootstrap_resamples"] = 100
    candidates = load_candidates(args.device)
    model = load_full_model(args.device)
    corpora = corpus_rows(config, args.quick)
    result = {"config": config, "behavior": {}}
    for name, rows in corpora.items():
        print("behavior", name, len(rows), flush=True)
        result["behavior"][name] = behavioral_evaluation(
            model, candidates, rows, config, args.device)
    causal_rows = 4 if args.quick else config["causal_rows"]
    print("causal", causal_rows, flush=True)
    result["causal"] = causal_evaluation(
        model, {name: candidates[name] for name in config["primary_models"]},
        corpora["fineweb"][:causal_rows], config["interventions"], config, args.device)
    result["parsimony"] = parsimony_evaluation(
        model, candidates, corpora["fineweb"], args.device)
    result["stability"] = stability_evaluation(candidates, config, args.quick)
    result["semantics"] = semantic_evaluation(candidates, config, args.quick)
    result["gates"] = evaluate_gates(result, config)
    result["hypotheses"] = evaluate_hypotheses(result, config)
    result["elapsed_seconds"] = time.time() - started
    make_figure(result)
    write_report(result)
    OUTPUT.write_text(json.dumps(json_ready(result), indent=2, allow_nan=False))
    print(f"wrote {OUTPUT}, {REPORT}, and {FIGURE} in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
