#!/usr/bin/env python3
"""Audit recurrent Layer-5 backbone channels on language and exact causality."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from discover_bilin18_weight_only_circuits import CATEGORIES, auc, token_flags
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer5_core_channel_audit.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer5_core_channel_audit.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER5_CORE_CHANNEL_AUDIT.md"
FIGURE = TENSOR_ROOT / "bilin12_layer5_core_channel_audit.png"


def load_rows(spec: dict) -> np.ndarray:
    source = np.load(spec["path"], mmap_mode="r")
    return np.asarray(source[spec["start"] : spec["start"] + spec["rows"]])


def descending_rank(values: torch.Tensor, index: int) -> int:
    order = values.argsort(descending=True)
    return int(torch.nonzero(order == index, as_tuple=False)[0]) + 1


def signed_auc(target: np.ndarray, score: np.ndarray) -> dict[str, float | str]:
    value = auc(target.astype(bool), score)
    if value >= 0.5:
        return {"auc": value, "polarity": "positive"}
    return {"auc": 1 - value, "polarity": "negative"}


def token_mean_extremes(
    token_ids: np.ndarray,
    values: np.ndarray,
    tokenizer,
    minimum_count: int,
    count: int = 6,
) -> dict[str, list[dict]]:
    unique, inverse, counts = np.unique(token_ids, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=values)
    means = sums / counts
    eligible = counts >= minimum_count
    unique = unique[eligible]
    counts = counts[eligible]
    means = means[eligible]
    positive = np.argsort(means)[-count:][::-1]
    negative = np.argsort(means)[:count]
    def rows(indices):
        return [
            {"token": tokenizer.decode([int(unique[index])]), "mean": float(means[index]), "count": int(counts[index])}
            for index in indices
        ]
    return {"positive": rows(positive), "negative": rows(negative)}


@torch.inference_mode()
def run_suffix(model, x, value_bus, x0, start_layer: int):
    for block in model.blocks[start_layer:]:
        x, value_bus = block(x, value_bus, x0)
    dimension = model.config.n_embd
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return 30 * torch.tanh(raw / 30)


def exact_kl(reference: torch.Tensor, changed: torch.Tensor) -> torch.Tensor:
    reference_logprob = F.log_softmax(reference, -1)
    probability = reference_logprob.exp()
    return (probability * (reference_logprob - F.log_softmax(changed, -1))).sum(-1)


@torch.inference_mode()
def causal_evaluation(
    model,
    spec: dict,
    config: dict,
    variants: dict[str, list[int]],
) -> dict:
    rows = load_rows(spec)
    dimension = model.config.n_embd
    layer = config["layer"]
    records = {
        f"{name}_{scope}": {key: [] for key in ("kl", "ce_delta", "top1")}
        for name in variants for scope in ("all_positions", "final_position")
    }
    block = model.blocks[layer]
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start : start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        targets = torch.as_tensor(
            np.asarray(rows[start : start + len(raw), 128], dtype=np.int64),
            device="cuda",
        )
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
        reference_logprob = F.log_softmax(reference, -1)
        reference_ce = -reference_logprob.gather(1, targets[:, None]).squeeze(1)
        reference_top1 = reference.argmax(1)
        for name, channels in variants.items():
            selected = torch.as_tensor(channels, device="cuda")
            contribution = product[:, :, selected] @ block.mlp.Down.weight[:, selected].T
            for scope in ("all_positions", "final_position"):
                changed_x = native_x.clone()
                if scope == "all_positions":
                    changed_x -= contribution
                else:
                    changed_x[:, -1] -= contribution[:, -1]
                logits = run_suffix(model, changed_x, value_bus, x0, layer + 1)
                logprob = F.log_softmax(logits, -1)
                ce = -logprob.gather(1, targets[:, None]).squeeze(1)
                key = f"{name}_{scope}"
                records[key]["kl"].extend(exact_kl(reference, logits).cpu().tolist())
                records[key]["ce_delta"].extend((ce - reference_ce).cpu().tolist())
                records[key]["top1"].extend((logits.argmax(1) == reference_top1).cpu().tolist())
    return {
        name: {
            "mean_exact_kl": float(np.mean(row["kl"])),
            "median_exact_kl": float(np.median(row["kl"])),
            "mean_ce_delta": float(np.mean(row["ce_delta"])),
            "top1_agreement": float(np.mean(row["top1"])),
        }
        for name, row in records.items()
    }


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([
        token_flags(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)
    ])
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    block = model.blocks[config["layer"]]
    down = block.mlp.Down.weight.float()
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    _, output_vectors = output_map_eigensystem(down, left, right)
    leading_output = output_vectors[:, 0]

    print("collecting activation-energy discovery panel", flush=True)
    discovery_states, _ = collect_states_writes(
        model, config["layer"], config["discovery"],
        config["positions"], config["batch_size"],
    )
    activations = (discovery_states @ left.T) * (discovery_states @ right.T)
    down_energy = down.square().sum(0)
    natural_write_energy = activations.square().mean(0) * down_energy
    form_energy = 0.5 * (
        left.square().sum(1) * right.square().sum(1)
        + (left * right).sum(1).square()
    )
    tensor_energy = down_energy * form_energy

    excluded = set(config["extended_channels"])
    available = [index for index in range(len(natural_write_energy)) if index not in excluded]
    matched = {}
    used = set()
    log_energy = natural_write_energy.clamp_min(1e-30).log()
    for channel in config["extended_channels"]:
        candidates = torch.as_tensor(
            [index for index in available if index not in used], device="cuda"
        )
        distance = (log_energy[candidates] - log_energy[channel]).abs()
        selected = int(candidates[distance.argmin()])
        matched[channel] = selected
        used.add(selected)
    audited_channels = config["extended_channels"] + [matched[channel] for channel in config["extended_channels"]]

    channel_rows = {}
    for channel in audited_channels:
        output_alignment = float(
            torch.dot(down[:, channel], leading_output).square()
            / down[:, channel].square().sum().clamp_min(1e-30)
        )
        channel_rows[str(channel)] = {
            "channel": channel,
            "is_core": channel in config["core_channels"],
            "matched_to": next((key for key, value in matched.items() if value == channel), None),
            "tensor_energy_rank": descending_rank(tensor_energy, channel),
            "natural_write_energy_rank": descending_rank(natural_write_energy, channel),
            "tensor_energy_top_fraction": descending_rank(tensor_energy, channel) / len(tensor_energy),
            "natural_write_energy_top_fraction": descending_rank(natural_write_energy, channel) / len(natural_write_energy),
            "leading_output_squared_cosine": output_alignment,
            "down_norm": float(down[:, channel].norm()),
            "left_norm": float(left[channel].norm()),
            "right_norm": float(right[channel].norm()),
            "output_tokens": token_mean_extremes(
                np.arange(tokenizer.vocab_size),
                model.lm_head(down[:, channel][None]).squeeze(0)[: tokenizer.vocab_size].cpu().numpy(),
                tokenizer, 1,
            ),
        }

    semantics = {}
    for corpus, spec in config["semantic_evaluation"].items():
        print("semantic audit", corpus, flush=True)
        states, _ = collect_states_writes(
            model, config["layer"], spec, config["positions"], config["batch_size"]
        )
        selected = torch.as_tensor(audited_channels, device="cuda")
        values = (
            (states @ left[selected].T) * (states @ right[selected].T)
        ).cpu().numpy()
        rows = load_rows(spec)
        positions = np.asarray(config["positions"])
        current_ids = rows[:, positions].reshape(-1)
        next_ids = rows[:, positions + 1].reshape(-1)
        corpus_rows = {}
        for column, channel in enumerate(audited_channels):
            feature_rows = {}
            for prefix, ids in (("current", current_ids), ("next", next_ids)):
                selected_flags = flags[ids]
                for category, category_name in enumerate(CATEGORIES):
                    target = selected_flags[:, category].astype(bool)
                    if min(int(target.sum()), len(target) - int(target.sum())) < config["minimum_category_examples"]:
                        continue
                    feature_rows[f"{prefix}/{category_name}"] = signed_auc(target, values[:, column])
            corpus_rows[str(channel)] = {
                "features": feature_rows,
                "current_token_means": token_mean_extremes(
                    current_ids, values[:, column], tokenizer, config["minimum_token_count"]
                ),
                "next_token_means": token_mean_extremes(
                    next_ids, values[:, column], tokenizer, config["minimum_token_count"]
                ),
                "activation_quantiles": {
                    str(q): float(np.quantile(values[:, column], q))
                    for q in (0.001, 0.01, 0.5, 0.99, 0.999)
                },
            }
        semantics[corpus] = corpus_rows

    recurrent_features = {}
    for channel in config["core_channels"]:
        common = None
        for corpus in semantics:
            strong = {
                name for name, row in semantics[corpus][str(channel)]["features"].items()
                if row["auc"] >= 0.70
            }
            common = strong if common is None else common & strong
        recurrent_features[str(channel)] = sorted(common or set())

    variants = {
        **{f"channel_{channel}": [channel] for channel in config["core_channels"]},
        **{f"matched_{channel}": [matched[channel]] for channel in config["core_channels"]},
        "core_pair": config["core_channels"],
        "matched_pair": [matched[channel] for channel in config["core_channels"]],
        "extended": config["extended_channels"],
        "matched_extended": [matched[channel] for channel in config["extended_channels"]],
    }
    causal = {}
    for corpus, spec in config["causal_evaluation"].items():
        print("causal audit", corpus, flush=True)
        causal[corpus] = causal_evaluation(model, spec, config, variants)

    core_alignment = np.median([
        channel_rows[str(channel)]["leading_output_squared_cosine"]
        for channel in config["core_channels"]
    ])
    matched_alignment = np.median([
        channel_rows[str(matched[channel])]["leading_output_squared_cosine"]
        for channel in config["core_channels"]
    ])
    hypotheses = {
        "C1_core_channels_have_top1pct_weight_energy": {
            "supported": all(channel_rows[str(channel)]["tensor_energy_top_fraction"] <= 0.01 for channel in config["core_channels"]),
        },
        "C2_core_channels_have_top1pct_natural_write_energy": {
            "supported": all(channel_rows[str(channel)]["natural_write_energy_top_fraction"] <= 0.01 for channel in config["core_channels"]),
        },
        "C3_core_pair_is_more_causal_than_matched_pair": {
            "supported": all(
                causal[corpus]["core_pair_all_positions"]["mean_exact_kl"]
                >= 1.5 * causal[corpus]["matched_pair_all_positions"]["mean_exact_kl"]
                for corpus in causal
            ),
        },
        "C4_core_effect_is_attention_mediated": {
            "supported": all(
                causal[corpus]["core_pair_all_positions"]["mean_exact_kl"]
                >= 1.5 * causal[corpus]["core_pair_final_position"]["mean_exact_kl"]
                for corpus in causal
            ),
        },
        "C5_core_has_recurrent_simple_semantics": {
            "supported": any(recurrent_features[str(channel)] for channel in config["core_channels"]),
            "features": recurrent_features,
        },
        "C6_core_is_more_aligned_with_leading_output": {
            "supported": bool(core_alignment > matched_alignment),
            "core_median": float(core_alignment),
            "matched_median": float(matched_alignment),
        },
        "C7_core_pair_supports_observed_token_loss": {
            "supported": all(causal[corpus]["core_pair_all_positions"]["mean_ce_delta"] > 0 for corpus in causal),
        },
        "C8_core_causal_effect_exceeds_individuals": {
            "supported": all(
                causal[corpus]["core_pair_all_positions"]["mean_exact_kl"]
                > sum(causal[corpus][f"channel_{channel}_all_positions"]["mean_exact_kl"] for channel in config["core_channels"])
                for corpus in causal
            ),
        },
    }
    for row in hypotheses.values():
        row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "matched_channels": {str(key): value for key, value in matched.items()},
        "channels": channel_rows,
        "recurrent_features": recurrent_features,
        "semantics": semantics,
        "causal": causal,
    }, indent=2, allow_nan=False))

    lines = [
        "# Language and causal audit of recurrent Layer-5 backbone channels",
        "",
        "Channels 196 and 202 recur in the top-eight attribution sets of all four leading spectral addresses. Channels 197 and 456 extend the aggregate backbone. Each is compared with a unique non-backbone channel matched by natural activation-weighted write energy. Signed activation semantics use frozen token categories on FineWeb, WikiText, and TinyStories. Exact zero ablations are run either at every Layer-5 position or only the final position, with layers 6–11 recomputed.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Core channel audit](bilin12_layer5_core_channel_audit.png)",
        "",
        f"Matched controls: `{matched}`. Recurrent signed category features: `{recurrent_features}`.",
        "",
        "| Channel | Role | Tensor rank | Natural-energy rank | Leading-output cos² |",
        "|---:|---|---:|---:|---:|",
    ]
    for channel in audited_channels:
        row = channel_rows[str(channel)]
        role = "core" if channel in config["core_channels"] else ("extended" if channel in config["extended_channels"] else f"matched to {row['matched_to']}")
        lines.append(
            f"| {channel} | {role} | {row['tensor_energy_rank']} | {row['natural_write_energy_rank']} | {row['leading_output_squared_cosine']:.4f} |"
        )
    lines += [
        "",
        "| Corpus | Core pair all-pos KL / final-only KL | Matched pair KL | Core-pair CE Δ | Top-1 agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus, row in causal.items():
        all_positions = row["core_pair_all_positions"]
        final_position = row["core_pair_final_position"]
        matched_pair = row["matched_pair_all_positions"]
        lines.append(
            f"| {corpus} | {all_positions['mean_exact_kl']:.5f} / {final_position['mean_exact_kl']:.5f} | "
            f"{matched_pair['mean_exact_kl']:.5f} | {all_positions['mean_ce_delta']:.5f} | {all_positions['top1_agreement']:.3f} |"
        )
    lines += [
        "",
        "Large weight or activation rank makes a channel structurally prominent, not semantic. Recurrent held-out category selectivity supplies a descriptive neuron interpretation; exact ablation supplies necessity. A pair effect larger than the sum of individual KLs indicates downstream interaction, while whole-sequence versus final-position ablation separates attention-mediated propagation from a direct residual effect.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    corpora = list(causal)
    x = np.arange(len(corpora))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].bar(
        np.arange(len(audited_channels)),
        [channel_rows[str(channel)]["natural_write_energy_rank"] for channel in audited_channels],
    )
    axes[0].set_yscale("log")
    axes[0].invert_yaxis()
    axes[0].set_xticks(np.arange(len(audited_channels)), audited_channels, rotation=30)
    axes[0].set_title("Natural write-energy rank")
    width = 0.25
    for offset, name in enumerate(("core_pair_all_positions", "core_pair_final_position", "matched_pair_all_positions")):
        axes[1].bar(
            x + (offset - 1) * width,
            [causal[corpus][name]["mean_exact_kl"] for corpus in corpora],
            width, label=name,
        )
    axes[1].set_xticks(x, corpora)
    axes[1].set_ylabel("mean exact KL")
    axes[1].set_title("Exact downstream ablation")
    axes[1].legend(fontsize=7)
    feature_names = sorted(set().union(*[set(values) for values in recurrent_features.values()]))
    if feature_names:
        for channel in config["core_channels"]:
            axes[2].plot(
                feature_names,
                [np.mean([
                    semantics[corpus][str(channel)]["features"].get(feature, {"auc": 0.5})["auc"]
                    for corpus in semantics
                ]) for feature in feature_names],
                marker="o", label=str(channel),
            )
        axes[2].tick_params(axis="x", rotation=35)
    axes[2].axhline(0.7, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylim(0.5, 1)
    axes[2].set_title("Recurrent signed category AUROC")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
