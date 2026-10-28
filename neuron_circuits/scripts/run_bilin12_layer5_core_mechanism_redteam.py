#!/usr/bin/env python3
"""Red-team the semantics and pair interaction of recurrent Layer-5 channels."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, layer_input, load_snapshot_model
from discover_bilin18_weight_only_circuits import auc
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer5_core_mechanism_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer5_core_mechanism_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER5_CORE_MECHANISM_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin12_layer5_core_mechanism_redteam.png"


def load_rows(spec: dict) -> np.ndarray:
    source = np.load(spec["path"], mmap_mode="r")
    return np.asarray(source[spec["start"] : spec["start"] + spec["rows"]])


def signed_auc(target: np.ndarray, score: np.ndarray) -> dict:
    raw = float(auc(target.astype(bool), score))
    return {
        "auc": max(raw, 1.0 - raw),
        "polarity": "positive" if raw >= 0.5 else "negative",
        "raw_auc": raw,
    }


def exact_kl(reference: torch.Tensor, changed: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(reference, -1)
    return (logp.exp() * (logp - F.log_softmax(changed, -1))).sum(-1)


def center_logits(delta: torch.Tensor) -> torch.Tensor:
    return delta - delta.mean(-1, keepdim=True)


def fisher_center(delta: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    probability = F.softmax(reference, -1)
    return delta - (probability * delta).sum(-1, keepdim=True)


def run_suffix(model, x, value_bus, x0, start_layer: int) -> torch.Tensor:
    for block in model.blocks[start_layer:]:
        x, value_bus = block(x, value_bus, x0)
    dimension = model.config.n_embd
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return 30 * torch.tanh(raw / 30)


@torch.inference_mode()
def factor_semantics(model, tokenizer, config: dict) -> dict:
    block = model.blocks[config["layer"]]
    selected = torch.as_tensor(config["core_channels"], device="cuda")
    newline_id = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline_id) != 1:
        raise RuntimeError(f"newline is not one token: {newline_id}")
    newline_id = newline_id[0]
    result = {}
    for corpus, spec in config["semantic_evaluation"].items():
        states, _ = collect_states_writes(
            model, config["layer"], spec, config["positions"], config["batch_size"]
        )
        left = states @ block.mlp.Left.weight[selected].T
        right = states @ block.mlp.Right.weight[selected].T
        product = left * right
        rows = load_rows(spec)
        ids = rows[:, np.asarray(config["positions"])].reshape(-1)
        target = ids == newline_id
        corpus_rows = {}
        for column, channel in enumerate(config["core_channels"]):
            entries = {
                "left": signed_auc(target, left[:, column].cpu().numpy()),
                "right": signed_auc(target, right[:, column].cpu().numpy()),
                "product": signed_auc(target, product[:, column].cpu().numpy()),
            }
            for name, values in (("left", left[:, column]), ("right", right[:, column]), ("product", product[:, column])):
                array = values.cpu().numpy()
                entries[name].update({
                    "newline_mean": float(array[target].mean()),
                    "ordinary_mean": float(array[~target].mean()),
                })
            entries["newline_count"] = int(target.sum())
            entries["ordinary_count"] = int((~target).sum())
            corpus_rows[str(channel)] = entries
        result[corpus] = corpus_rows
    return result


@torch.inference_mode()
def newline_substitution(model, tokenizer, config: dict) -> dict:
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline_ids) != 1:
        raise RuntimeError(f"newline is not one token: {newline_ids}")
    newline_id = newline_ids[0]
    block = model.blocks[config["layer"]]
    selected = torch.as_tensor(config["core_channels"], device="cuda")
    result = {}
    for corpus, spec in config["substitution_evaluation"].items():
        rows = load_rows(spec)
        changes = defaultdict(list)
        for position in config["substitution_positions"]:
            for start in range(0, len(rows), config["batch_size"]):
                raw = np.asarray(rows[start : start + config["batch_size"], :128], dtype=np.int64)
                ordinary = torch.as_tensor(raw, device="cuda")
                keep = ordinary[:, position] != newline_id
                if not bool(keep.any()):
                    continue
                ordinary = ordinary[keep]
                changed = ordinary.clone()
                changed[:, position] = newline_id
                original_state = layer_input(model, ordinary, config["layer"])[:, position]
                changed_state = layer_input(model, changed, config["layer"])[:, position]
                original = block.mlp.Left(original_state)[:, selected] * block.mlp.Right(original_state)[:, selected]
                replaced = block.mlp.Left(changed_state)[:, selected] * block.mlp.Right(changed_state)[:, selected]
                for column, channel in enumerate(config["core_channels"]):
                    changes[str(channel)].extend((replaced[:, column] - original[:, column]).cpu().tolist())
        result[corpus] = {
            channel: {
                "count": len(values),
                "mean_change": float(np.mean(values)),
                "median_change": float(np.median(values)),
                "fraction_suppressed": float(np.mean(np.asarray(values) < 0)),
            }
            for channel, values in changes.items()
        }
    return result


@torch.inference_mode()
def causal_geometry(model, config: dict) -> dict:
    layer = config["layer"]
    dimension = model.config.n_embd
    block = model.blocks[layer]
    core = config["core_channels"]
    matched = config["matched_channels"]
    result = {}
    for corpus, spec in config["causal_evaluation"].items():
        rows = load_rows(spec)
        store = {
            scope: defaultdict(list)
            for scope in ("all_positions", "final_position")
        }
        activation = defaultdict(list)
        for start in range(0, len(rows), config["batch_size"]):
            raw = np.asarray(rows[start : start + config["batch_size"], :128], dtype=np.int64)
            ids = torch.as_tensor(raw, device="cuda")
            x0 = F.rms_norm(model.embedding(ids), (dimension,))
            x = x0
            value_bus = None
            for prefix_block in model.blocks[:layer]:
                x, value_bus = prefix_block(x, value_bus, x0)
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
            pre = mixed + attention
            state = F.rms_norm(pre, (dimension,))
            product = block.mlp.Left(state) * block.mlp.Right(state)
            write = block.mlp.Down(product) + block.mlp.Down_bias
            native_x = pre + write
            reference = run_suffix(model, native_x, value_bus, x0, layer + 1)
            for channel in core:
                activation[str(channel)].extend(product[:, -1, channel].abs().cpu().tolist())
            variants = {
                "core_0": [core[0]], "core_1": [core[1]], "core_pair": core,
                "matched_0": [matched[0]], "matched_1": [matched[1]], "matched_pair": matched,
            }
            for scope in store:
                logits = {"reference": reference}
                for name, channels in variants.items():
                    selected = torch.as_tensor(channels, device="cuda")
                    contribution = product[:, :, selected] @ block.mlp.Down.weight[:, selected].T
                    changed_x = native_x.clone()
                    if scope == "all_positions":
                        changed_x -= contribution
                    else:
                        changed_x[:, -1] -= contribution[:, -1]
                    logits[name] = run_suffix(model, changed_x, value_bus, x0, layer + 1)
                for name in variants:
                    store[scope][f"kl/{name}"].extend(exact_kl(reference, logits[name]).cpu().tolist())
                    store[scope][f"delta/{name}"].append(center_logits(logits[name] - reference).cpu())
                additive = reference + (logits["core_0"] - reference) + (logits["core_1"] - reference)
                store[scope]["kl/synthetic_additive"].extend(exact_kl(reference, additive).cpu().tolist())
                d0 = fisher_center(logits["core_0"] - reference, reference)
                d1 = fisher_center(logits["core_1"] - reference, reference)
                probability = F.softmax(reference, -1)
                numerator = (probability * d0 * d1).sum(-1)
                denominator = ((probability * d0.square()).sum(-1) * (probability * d1.square()).sum(-1)).sqrt().clamp_min(1e-30)
                store[scope]["fisher_cosine"].extend((numerator / denominator).cpu().tolist())
        corpus_rows = {}
        for scope, values in store.items():
            arrays = {}
            for key, chunks in values.items():
                arrays[key] = torch.cat(chunks).numpy() if key.startswith("delta/") else np.asarray(chunks)
            pair = arrays["delta/core_pair"]
            additive_delta = arrays["delta/core_0"] + arrays["delta/core_1"]
            residual = pair - additive_delta
            pair_norm = np.linalg.norm(pair, axis=1)
            residual_ratio = np.linalg.norm(residual, axis=1) / np.maximum(pair_norm, 1e-30)
            exact_pair = arrays["kl/core_pair"]
            synthetic = arrays["kl/synthetic_additive"]
            scope_row = {
                "mean_exact_pair_kl": float(exact_pair.mean()),
                "mean_individual_kl_sum": float((arrays["kl/core_0"] + arrays["kl/core_1"]).mean()),
                "superadditivity_ratio": float(exact_pair.mean() / max((arrays["kl/core_0"] + arrays["kl/core_1"]).mean(), 1e-30)),
                "mean_synthetic_additive_kl": float(synthetic.mean()),
                "synthetic_to_exact_kl_ratio": float(synthetic.mean() / max(exact_pair.mean(), 1e-30)),
                "mean_absolute_synthetic_kl_error_ratio": float(np.mean(np.abs(synthetic - exact_pair)) / max(exact_pair.mean(), 1e-30)),
                "nonlinear_residual_ratio": float(residual_ratio.mean()),
                "mean_fisher_cosine": float(arrays["fisher_cosine"].mean()),
                "mean_matched_pair_kl": float(arrays["kl/matched_pair"].mean()),
                "core_to_matched_kl_ratio": float(exact_pair.mean() / max(arrays["kl/matched_pair"].mean(), 1e-30)),
            }
            if scope == "final_position":
                for column, channel in enumerate(core):
                    act = np.asarray(activation[str(channel)])
                    kl = arrays[f"kl/core_{column}"]
                    low, high = np.quantile(act, [0.25, 0.75])
                    low_kl = float(kl[act <= low].mean())
                    high_kl = float(kl[act >= high].mean())
                    scope_row[f"activation_quartile_ratio/{channel}"] = high_kl / max(low_kl, 1e-30)
            corpus_rows[scope] = scope_row
        result[corpus] = corpus_rows
    return result


def main() -> None:
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    print("factor semantics", flush=True)
    semantics = factor_semantics(model, tokenizer, config)
    print("newline substitutions", flush=True)
    substitutions = newline_substitution(model, tokenizer, config)
    print("causal geometry", flush=True)
    causal = causal_geometry(model, config)

    core = [str(channel) for channel in config["core_channels"]]
    hypotheses = {
        "M1_boundary_suppression_replicates": {"supported": all(
            semantics[corpus][channel]["product"]["auc"] >= 0.80
            and semantics[corpus][channel]["product"]["polarity"] == "negative"
            for corpus in semantics for channel in core
        )},
        "M2_product_is_more_boundary_selective_than_either_factor": {"supported": all(
            semantics[corpus][channel]["product"]["auc"]
            >= max(semantics[corpus][channel]["left"]["auc"], semantics[corpus][channel]["right"]["auc"]) + 0.03
            for corpus in semantics for channel in core
        )},
        "M3_newline_substitution_suppresses_core": {"supported": all(
            substitutions[corpus][channel]["fraction_suppressed"] >= 0.75
            for corpus in substitutions for channel in core
        )},
        "M4_pair_superadditivity_is_mostly_additive_logit_geometry": {"supported": all(
            abs(causal[corpus][scope]["synthetic_to_exact_kl_ratio"] - 1) <= 0.20
            and causal[corpus][scope]["nonlinear_residual_ratio"] <= 0.20
            for corpus in causal for scope in causal[corpus]
        )},
        "M5_core_pair_has_positive_logit_coupling": {"supported": all(
            causal[corpus][scope]["mean_fisher_cosine"] > 0
            for corpus in causal for scope in causal[corpus]
        )},
        "M6_activation_predicts_individual_causal_effect": {"supported": all(
            causal[corpus]["final_position"][f"activation_quartile_ratio/{channel}"] >= 2
            for corpus in causal for channel in core
        )},
        "M7_core_exceeds_matched_causal_effect": {"supported": all(
            causal[corpus][scope]["core_to_matched_kl_ratio"] >= 5
            for corpus in causal for scope in causal[corpus]
        )},
    }
    for row in hypotheses.values():
        row["supported"] = bool(row["supported"])
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "factor_semantics": semantics,
        "newline_substitution": substitutions,
        "causal_geometry": causal,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    corpora = list(semantics)
    x = np.arange(len(corpora))
    for channel in core:
        axes[0].plot(x, [semantics[c][channel]["product"]["auc"] for c in corpora], marker="o", label=f"ch {channel}")
        axes[1].plot(x, [substitutions[c][channel]["fraction_suppressed"] for c in corpora], marker="o", label=f"ch {channel}")
    axes[0].axhline(0.8, color="black", ls="--", lw=1); axes[0].set_title("Boundary signed AUROC")
    axes[1].axhline(0.75, color="black", ls="--", lw=1); axes[1].set_title("Suppressed by newline edit")
    for scope, marker in (("all_positions", "o"), ("final_position", "s")):
        axes[2].plot(x, [causal[c][scope]["synthetic_to_exact_kl_ratio"] for c in corpora], marker=marker, label=scope)
    axes[2].axhline(1, color="black", ls="--", lw=1); axes[2].set_title("Additive / exact pair KL")
    for axis in axes:
        axis.set_xticks(x, corpora, rotation=20); axis.grid(alpha=.25); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)

    lines = [
        "# Layer-5 core-channel mechanism red team", "",
        "Fresh held-out slices test whether channels 196/202 are boundary-suppressed bilinear predicates and whether their apparently superadditive joint KL reflects an actual nonlinear circuit or ordinary additive-logit geometry.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Mechanism red team](bilin12_layer5_core_mechanism_redteam.png)", "", "## Boundary and substitution tests", "",
              "| Corpus | Channel | Product AUROC/polarity | Better factor AUROC | Newline-edit suppression | Mean product change |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus in corpora:
        for channel in core:
            row = semantics[corpus][channel]
            sub = substitutions[corpus][channel]
            better = max(row["left"]["auc"], row["right"]["auc"])
            lines.append(f"| {corpus} | {channel} | {row['product']['auc']:.3f}/{row['product']['polarity']} | {better:.3f} | {sub['fraction_suppressed']:.3f} | {sub['mean_change']:.2f} |")
    lines += ["", "## Pair geometry", "",
              "| Corpus/scope | Exact pair KL | Sum individual KL | Additive/exact KL | Nonlinear residual | Fisher cosine | Core/matched KL |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for corpus in corpora:
        for scope, row in causal[corpus].items():
            lines.append(f"| {corpus}/{scope} | {row['mean_exact_pair_kl']:.5f} | {row['mean_individual_kl_sum']:.5f} | {row['synthetic_to_exact_kl_ratio']:.3f} | {row['nonlinear_residual_ratio']:.3f} | {row['mean_fisher_cosine']:.3f} | {row['core_to_matched_kl_ratio']:.1f} |")
    lines += ["", "The factor comparison distinguishes a genuinely bilinear conjunction from a label already present in one linear factor. Literal token substitution is stronger than observational AUROC but still changes a real token embedding and therefore is not a minimal semantic intervention. The pair decomposition is decisive for the interaction claim: KL superadditivity by itself is expected whenever two additive logit changes are positively aligned in the softmax Fisher metric."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
