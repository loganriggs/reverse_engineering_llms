#!/usr/bin/env python3
"""Red-team the apparent sparse Layer-17 spelling consumer by source distance."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_downstream_screen import upstream_states
from run_bilin18_boundary_style_final_attention_atlas import score_and_gradient
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_boundary_style_final_attention_position_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_boundary_style_final_attention_position_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_BOUNDARY_STYLE_FINAL_ATTENTION_POSITION_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin18_boundary_style_final_attention_position_redteam.png"

BINS = (
    "pre_boundary",
    "boundary",
    "post_far",
    "post_9_16",
    "post_5_8",
    "post_3_4",
    "post_2",
    "post_1",
    "destination",
)
POST_BINS = BINS[2:8]


def position_masks(length, cue, boundary, destination):
    del cue  # Retained in the signature to make the intervention manifest explicit.
    position = torch.arange(length, device=boundary.device)[None]
    distance = destination - position
    post = (position > boundary[:, None]) & (position < destination)
    return {
        "pre_boundary": position < boundary[:, None],
        "boundary": position == boundary[:, None],
        "post_far": post & (distance >= 17),
        "post_9_16": post & (distance >= 9) & (distance <= 16),
        "post_5_8": post & (distance >= 5) & (distance <= 8),
        "post_3_4": post & (distance >= 3) & (distance <= 4),
        "post_2": post & (distance == 2),
        "post_1": post & (distance == 1),
        "destination": (position == destination) & (boundary[:, None] != destination),
    }


def edge_group(edge_delta, masks, head, bins):
    mask = sum((masks[name] for name in bins), torch.zeros_like(masks[BINS[0]]))
    return (edge_delta[:, head] * mask[:, :, None]).sum(1)


def correlation(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.std() == 0 or second.std() == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


@torch.inference_mode()
def collect(model, raw, records, contrasts, config):
    dimension = model.config.n_embd
    heads = model.config.n_head
    head_dim = dimension // heads
    destination = 127
    declared = contrasts[:, 0]
    token_index = torch.nonzero(declared).flatten()
    token_weights = declared[token_index].double()
    output_weights = model.lm_head.weight[token_index].double()
    nodes, quadrature = np.polynomial.legendre.leggauss(config["quadrature_points"])
    nodes, quadrature = (nodes + 1) / 2, quadrature / 2

    programs = {
        "h2_after": (2, POST_BINS),
        "h2_boundary": (2, ("boundary",)),
        "h2_destination": (2, ("destination",)),
        "h2_pre_boundary": (2, ("pre_boundary",)),
        "h2_all": (2, BINS),
        "h1_after": (1, POST_BINS),
        "h6_after": (6, POST_BINS),
    }
    for name in POST_BINS:
        programs[f"h2_{name}"] = (2, (name,))

    finite = {name: {"native": [], "mediated": [], "random": []} for name in programs}
    integrated_sums = {f"h{head}_{name}": 0.0 for head in range(heads) for name in BINS}
    integrated_sums.update({"carried": 0.0, "mlp": 0.0, "attention_numeric_residual": 0.0})
    count = 0
    maximum_output_error = 0.0
    maximum_edge_error = 0.0

    for start in range(0, len(records), config["batch_size"]):
        batch = records[start : start + config["batch_size"]]
        size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch])
        swapped = original.copy()
        for index, row in enumerate(batch):
            swapped[index, row[3]] = row[5]
        cue = torch.as_tensor([row[3] for row in batch], device="cuda")
        boundary = torch.as_tensor([row[1] for row in batch], device="cuda")
        direction = torch.as_tensor(
            [1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float64
        )
        x, x0, bus = upstream_states(model, original, swapped, cue, boundary)

        for layer in range(9, 18):
            block = model.blocks[layer]
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(
                block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt
            )
            pre = mixed + details["write"]
            normalized = F.rms_norm(pre, (dimension,))
            mlp = block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
            if layer < 17:
                x = pre + mlp
                bus = details["value_bus"]
                continue

            pattern = (details["score1"] * details["score2"])[:, :, destination, :].double()
            values = details["mixed_value"].double().permute(0, 2, 1, 3)
            projection = block.attn.c_proj.weight.double().T.reshape(heads, head_dim, dimension)
            edges = torch.einsum("bhs,bhsd,hdk->bhsk", pattern, values, projection)
            edges = edges.reshape(2, size, heads, original.shape[1], dimension)
            edge_delta = edges[1] - edges[0]
            masks = position_masks(original.shape[1], cue, boundary, destination)

            components = {
                "carried": mixed[size:, destination].double() - mixed[:size, destination].double(),
                "mlp": mlp[size:, destination].double() - mlp[:size, destination].double(),
            }
            for head in range(heads):
                for name, mask in masks.items():
                    components[f"h{head}_{name}"] = (
                        edge_delta[:, head] * mask[:, :, None]
                    ).sum(1)
            attention_delta = (
                details["write"][size:, destination].double()
                - details["write"][:size, destination].double()
            )
            edge_sum = sum(
                value for name, value in components.items() if name.startswith("h")
            )
            components["attention_numeric_residual"] = attention_delta - edge_sum
            maximum_edge_error = max(
                maximum_edge_error,
                float((edge_sum - attention_delta).norm() / attention_delta.norm().clamp_min(1e-30)),
            )

            base = (
                mixed[:size, destination].double()
                + details["write"][:size, destination].double()
                + mlp[:size, destination].double()
            )
            target = (
                mixed[size:, destination].double()
                + details["write"][size:, destination].double()
                + mlp[size:, destination].double()
            )
            delta = target - base
            allocations = {
                name: torch.zeros(size, device="cuda", dtype=torch.float64)
                for name in components
            }
            for node, weight in zip(nodes, quadrature):
                _, gradient = score_and_gradient(
                    base + float(node) * delta, output_weights, token_weights
                )
                for name, value in components.items():
                    allocations[name] += float(weight) * (gradient * value).sum(-1)
            score0, _ = score_and_gradient(base, output_weights, token_weights)
            score1, _ = score_and_gradient(target, output_weights, token_weights)
            target_score = score1 - score0
            reconstructed = sum(allocations.values())
            maximum_output_error = max(
                maximum_output_error,
                float((reconstructed - target_score).norm() / target_score.norm().clamp_min(1e-30)),
            )
            for name, value in allocations.items():
                integrated_sums[name] += float((direction * value).sum())

            variants = [
                details["write"][:size].float(),
                details["write"][size:].float(),
            ]
            for head, bins in programs.values():
                modified = details["write"][size:].float().clone()
                modified[:, destination] -= edge_group(edge_delta, masks, head, bins).float()
                variants.append(modified)
            repeated = len(variants)
            attention = torch.cat(variants)
            mixed_variants = torch.cat([mixed[:size], mixed[size:]] + [mixed[size:]] * (repeated - 2))
            state = F.rms_norm(mixed_variants + attention, (dimension,))
            final = mixed_variants + attention + block.mlp.Down(
                block.mlp.Left(state) * block.mlp.Right(state)
            ) + block.mlp.Down_bias
            logits = 30 * torch.tanh(
                model.lm_head(F.rms_norm(final, (dimension,))[:, destination]).float() / 30
            )
            scores = (logits @ contrasts).reshape(repeated, size, -1)
            native = direction.float()[:, None] * (scores[1] - scores[0])
            for index, name in enumerate(programs, start=2):
                frozen = direction.float()[:, None] * (scores[index] - scores[0])
                mediated = native - frozen
                finite[name]["native"].append(native[:, 0].cpu())
                finite[name]["mediated"].append(mediated[:, 0].cpu())
                finite[name]["random"].append(mediated[:, 1:].cpu())
            count += size
            break

    integrated = {name: value / count for name, value in integrated_sums.items()}
    finite_summary = {}
    for name, values in finite.items():
        native = torch.cat(values["native"])
        mediated = torch.cat(values["mediated"])
        random_means = torch.cat(values["random"]).mean(0)
        mean = float(mediated.mean())
        finite_summary[name] = {
            "native_mean": float(native.mean()),
            "mediated_mean": mean,
            "mediation_fraction": mean / max(abs(float(native.mean())), 1e-30),
            "win": float((mediated > 0).float().mean()),
            "random_percentile": float((random_means < mean).float().mean()),
        }

    program_integrated = {}
    for name, (head, bins) in programs.items():
        program_integrated[name] = sum(integrated[f"h{head}_{item}"] for item in bins)
    correlation_names = [
        "h2_boundary",
        "h2_destination",
        *[f"h2_{name}" for name in POST_BINS],
        "h1_after",
        "h6_after",
    ]
    allocation_correlation = correlation(
        [program_integrated[name] for name in correlation_names],
        [finite_summary[name]["mediated_mean"] for name in correlation_names],
    )
    positive_post = sum(max(integrated[f"h2_{name}"], 0.0) for name in POST_BINS)
    post_shares = {
        name: max(integrated[f"h2_{name}"], 0.0) / max(positive_post, 1e-30)
        for name in POST_BINS
    }
    top_post = max(post_shares, key=post_shares.get)
    numeric_share = abs(integrated["attention_numeric_residual"]) / max(
        abs(sum(integrated.values())), 1e-30
    )
    return {
        "examples": count,
        "output_recomposition_error": maximum_output_error,
        "raw_edge_recomposition_error": maximum_edge_error,
        "numeric_residual_share": numeric_share,
        "integrated": integrated,
        "program_integrated": program_integrated,
        "finite": finite_summary,
        "allocation_finite_correlation": allocation_correlation,
        "post_shares": post_shares,
        "top_post_bin": top_post,
    }


def main():
    config = json.loads(CONFIG.read_text())
    base = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack(
        [vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]
    )
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[
        :, FEATURES.index("period")
    ].astype(bool)
    pairs = [
        (
            tokenizer.encode(first, add_special_tokens=False)[0],
            tokenizer.encode(second, add_special_tokens=False)[0],
        )
        for first, second in base["spelling_pairs"]
    ]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    results = []
    for fold in range(config["folds"]):
        target_indices = set(
            range(
                fold * config["target_pairs_per_fold"],
                (fold + 1) * config["target_pairs_per_fold"],
            )
        )
        target_pairs = [pair for index, pair in enumerate(pairs) if index in target_indices]
        cues = [pair for index, pair in enumerate(pairs) if index not in target_indices]
        contrasts = make_contrasts(
            model.lm_head.weight.shape[0],
            target_pairs,
            config["random_contrasts"],
            config["seed"] + fold,
            "cuda",
        )
        for split, (begin, end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
            records = select_records(
                raw,
                boundary_flags,
                cues,
                counts,
                127,
                config["examples_per_label_per_split"],
                config["seed"] + fold * 10 + split,
            )
            print("position redteam", fold, split, len(records), flush=True)
            results.append(
                {
                    "fold": fold,
                    "split": split,
                    **collect(model, raw, records, contrasts, config),
                }
            )

    p0 = all(
        row["output_recomposition_error"] <= 1e-5
        and row["raw_edge_recomposition_error"] <= 1e-3
        and row["numeric_residual_share"] < 0.01
        for row in results
    )
    p1 = all(
        row["finite"]["h2_after"]["mediation_fraction"] >= 0.15
        and row["finite"]["h2_after"]["random_percentile"] >= 0.95
        for row in results
    )
    p2 = all(
        row["finite"][name]["mediation_fraction"] < 0.05
        for row in results
        for name in ("h1_after", "h6_after")
    )
    p3 = all(max(row["post_shares"].values()) <= 0.75 for row in results)
    stable = sum(
        results[2 * fold]["top_post_bin"] == results[2 * fold + 1]["top_post_bin"]
        for fold in range(config["folds"])
    )
    p4 = stable >= max(1, (2 * config["folds"] + 2) // 3)
    p5 = all(row["allocation_finite_correlation"] >= 0.7 for row in results)
    hypotheses = {
        "P0_exact_and_numerically_small_residual": {"supported": p0},
        "P1_H2_after_boundary_is_finitely_causal": {"supported": p1},
        "P2_same_relation_control_heads_are_weak": {"supported": p2},
        "P3_post_boundary_support_is_not_one_distance_bin": {"supported": p3},
        "P4_top_distance_relation_replicates": {"supported": p4, "folds": stable},
        "P5_local_allocation_predicts_finite_programs": {"supported": p5},
        "P6_replicated_post_boundary_echo_ontology": {
            "supported": all((p0, p1, p2, p3, p4, p5))
        },
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    OUTPUT.write_text(
        json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2)
    )

    lines = [
        "# Positional and finite red team of the Layer-17 spelling consumer",
        "",
        "The discovery-selected L17H2 post-boundary contribution is split into fixed distance-to-destination bins. Exact line-integral allocations are compared with finite edge-group freezes through the final MLP and output. H1/H6 relation-matched controls and 31 random output contrasts test head and objective specificity.",
        "",
        "## Locked results",
        "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += [
        "",
        "| Fold/split | H2-after finite fraction | random pct. | top post bin/share | allocation→finite r |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results:
        after = row["finite"]["h2_after"]
        top = row["top_post_bin"]
        lines.append(
            f"| {row['fold']}/{row['split']} | {after['mediation_fraction']:.3f} | "
            f"{after['random_percentile']:.3f} | {top}/{row['post_shares'][top]:.3f} | "
            f"{row['allocation_finite_correlation']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "A sparse head/relation result can coexist with distributed positional support: the head is the reusable read function, while many post-boundary positions may carry repeated or transformed instances of the spelling state. Conversely, concentration in one distance bin would identify a still shorter edge family. The finite freezes establish causal contribution under this intervention; they do not establish that the source positions store only spelling information.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    labels = [f"f{row['fold']}s{row['split']}" for row in results]
    x = np.arange(len(results))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    bottom = np.zeros(len(results))
    for name in POST_BINS:
        values = np.asarray([row["post_shares"][name] for row in results])
        axes[0].bar(x, values, bottom=bottom, label=name)
        bottom += values
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Positive allocation share", title="H2 post-boundary positions")
    axes[0].legend(fontsize=7)
    axes[1].bar(x - 0.2, [row["finite"]["h2_after"]["mediation_fraction"] for row in results], width=0.2, label="H2")
    axes[1].bar(x, [row["finite"]["h1_after"]["mediation_fraction"] for row in results], width=0.2, label="H1")
    axes[1].bar(x + 0.2, [row["finite"]["h6_after"]["mediation_fraction"] for row in results], width=0.2, label="H6")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Finite mediation fraction", title="Relation-matched head controls")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print(json.dumps({"hypotheses": hypotheses, "summary": [
        {"fold": row["fold"], "split": row["split"], "top": row["top_post_bin"], "share": max(row["post_shares"].values()), "finite": row["finite"]["h2_after"]["mediation_fraction"], "correlation": row["allocation_finite_correlation"]}
        for row in results
    ]}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
