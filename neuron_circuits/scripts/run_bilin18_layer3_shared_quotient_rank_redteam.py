#!/usr/bin/env python3
"""Rank and cluster sensitivity of the Layer-3 shared/residual quotient."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer3_family_interaction_modes import evaluate_writes
from run_bilin18_layer3_frozen_channel_transfer import (
    channel_state,
    recover_configs,
    top_indices,
)
from run_bilin18_layer3_native_bank_input_forms import bank_delta, input_metric
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_shared_quotient_rank_redteam.json")
OUTPUT = TENSOR_ROOT / "layer3_shared_quotient_rank_redteam.json"
REPORT = TENSOR_ROOT / "LAYER3_SHARED_QUOTIENT_RANK_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer3_shared_quotient_rank_redteam.png"


def overlap(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.T @ second).square().sum() / min(first.shape[1], second.shape[1]))


def finite_median(values) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    return float(np.median(values[np.isfinite(values)]))


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    quotient_config = json.loads(Path(config["source_config"]).read_text())
    semantic_config = json.loads(Path(quotient_config["source_config"]).read_text())
    input_config = json.loads(Path(semantic_config["source_config"]).read_text())
    specificity_config = json.loads(Path(input_config["source_config"]).read_text())
    bank_config = json.loads(Path(specificity_config["source_config"]).read_text())
    rank_config = json.loads(Path(bank_config["source_config"]).read_text())
    form_config = json.loads(Path(rank_config["source_config"]).read_text())
    lopo_config = json.loads(Path(form_config["source_config"]).read_text())
    frozen_config = json.loads(Path(lopo_config["source_config"]).read_text())
    fixed, parent = recover_configs(frozen_config)
    family_names = list(parent["families"])
    candidate_cluster = set(config["candidate_cluster"])
    native_saved = json.loads(Path(config["native_results"]).read_text())
    layer, final_position = semantic_config["layer"], fixed["causal_position"]

    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[layer]
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")
        for corpus, spec in fixed["evaluation"].items()
    }

    def reconstruct(record):
        corpus = record["corpus"]
        spec = fixed["evaluation"][corpus]
        raw = np.asarray(
            arrays[corpus][
                spec["start"] + record["row"],
                record["source_start"]:record["source_start"] + fixed["sequence_length"],
            ], dtype=np.int64,
        ).copy()
        changed = raw.copy()
        changed[fixed["edit_position"]] = int(record["pair"].split("->")[1])
        base = forward_layers(
            model, torch.as_tensor(raw, device="cuda")[None], last_layer=layer,
        )[layer]
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer,
        )[layer]
        return base, counterfactual

    discovery = [
        record for record in native_saved["semantic_records"]
        if record["corpus"] == frozen_config["discovery_corpus"]
        and record["context_index"] == frozen_config["discovery_context_index"]
    ]
    family_scores = {family: [] for family in family_names}
    for record in discovery:
        base, counterfactual = reconstruct(record)
        family_scores[record["family"]].append(channel_state(
            model, layer, base, counterfactual, final_position,
        )["scores"].abs().cpu())

    family_cache = {}
    for family in family_names:
        indices = top_indices(
            torch.stack(family_scores[family]).mean(0), quotient_config["channel_rank"],
        ).cuda()
        metric = input_metric(block, indices, quotient_config["output_rank"])
        family_cache[family] = {
            "indices": indices,
            "output": metric["output"],
            "vectors": metric["vectors"][:, : config["maximum_family_rank"]],
        }

    family32 = {
        family: cache["vectors"][:, : config["family_rank"]]
        for family, cache in family_cache.items()
    }
    mean_projector = sum(basis @ basis.T for basis in family32.values()) / len(family32)
    values, vectors = torch.linalg.eigh(mean_projector)
    order = values.argsort(descending=True)
    values, vectors = values[order], vectors[:, order]
    eigenvalue_ratio = float(values[7] / values[8].clamp_min(1e-30))

    affinity = {}
    for rank in config["affinity_ranks"]:
        within_cluster, outside_cluster = [], []
        matrix = {}
        for index, family in enumerate(family_names):
            matrix[family] = {}
            for other in family_names:
                matrix[family][other] = overlap(
                    family_cache[family]["vectors"][:, :rank],
                    family_cache[other]["vectors"][:, :rank],
                )
            for other in family_names[index + 1:]:
                value = matrix[family][other]
                if family in candidate_cluster and other in candidate_cluster:
                    within_cluster.append(value)
                elif (family in candidate_cluster) != (other in candidate_cluster):
                    outside_cluster.append(value)
        affinity[str(rank)] = {
            "matrix": matrix,
            "within_cluster": within_cluster,
            "outside_cluster": outside_cluster,
            "within_median": finite_median(within_cluster),
            "outside_median": finite_median(outside_cluster),
        }

    split = {rank: {} for rank in config["shared_ranks"]}
    for rank in config["shared_ranks"]:
        common = vectors[:, :rank]
        for family, basis in family32.items():
            inside = basis.T @ (common @ common.T) @ basis
            local_values, local_vectors = torch.linalg.eigh(0.5 * (inside + inside.T))
            local_vectors = local_vectors[:, local_values.argsort(descending=True)]
            split[rank][family] = {
                "common": common,
                "shared_like": basis @ local_vectors[:, :rank],
                "family_residual": basis @ local_vectors[:, rank:],
            }

    query_records = [
        record for record in native_saved["semantic_records"]
        if record["context_index"] == frozen_config["evaluation_context_index"]
    ]
    causal_records = []
    for record in query_records:
        base, counterfactual = reconstruct(record)
        state = channel_state(model, layer, base, counterfactual, final_position)
        family = record["family"]
        cache = family_cache[family]
        dimension = base["pre"].shape[-1]
        xb = F.rms_norm(base["pre"], (dimension,)).float()
        xc = F.rms_norm(counterfactual["pre"], (dimension,)).float()
        delta = xc - xb

        def route(basis):
            changed = xb + (delta @ basis) @ basis.T
            predicted = bank_delta(block, xb, changed, cache["indices"])
            return (predicted @ cache["output"]) @ cache["output"].T

        union = route(family32[family])
        writes = {"full_mlp": state["target_write"], "family_exact": union}
        cross_norm = {}
        for rank in config["shared_ranks"]:
            pieces = split[rank][family]
            shared_like = route(pieces["shared_like"])
            family_residual = route(pieces["family_residual"])
            cross_term = union - shared_like - family_residual
            writes[f"shared_like:{rank}"] = shared_like
            writes[f"family_residual:{rank}"] = family_residual
            writes[f"cross_term:{rank}"] = cross_term
            writes[f"global_common:{rank}"] = route(pieces["common"])
            cross_norm[str(rank)] = float(
                cross_term.norm() / union.norm().clamp_min(1e-30)
            )
        effects = evaluate_writes(model, layer, base, counterfactual, writes, final_position)
        causal_records.append({
            "corpus": record["corpus"],
            "family": family,
            "pair": record["pair"],
            "effects": {
                name: {
                    "family_patch_fidelity": value["family_patch_fidelity"],
                    "family_patch_cosine": value["family_patch_cosine"],
                }
                for name, value in effects.items()
                if name != "full_mlp" and name != "family_exact"
            },
            "cross_norm": cross_norm,
        })
        print("rank redteam", record["corpus"], family, record["pair"], flush=True)

    summary = {}
    for corpus in fixed["evaluation"]:
        summary[corpus] = {}
        for family in family_names:
            selected = [
                row for row in causal_records
                if row["corpus"] == corpus and row["family"] == family
            ]
            summary[corpus][family] = {}
            for rank in config["shared_ranks"]:
                summary[corpus][family][str(rank)] = {
                    route: median([
                        row["effects"][f"{route}:{rank}"]["family_patch_fidelity"]
                        for row in selected
                    ])
                    for route in (
                        "shared_like", "family_residual", "cross_term", "global_common",
                    )
                } | {
                    "cross_norm_share": median([row["cross_norm"][str(rank)] for row in selected])
                }
        summary[corpus]["aggregate"] = {
            str(rank): {
                route: median([
                    row["effects"][f"{route}:{rank}"]["family_patch_fidelity"]
                    for row in causal_records if row["corpus"] == corpus
                ])
                for route in (
                    "shared_like", "family_residual", "cross_term", "global_common",
                )
            } | {
                "cross_norm_share": median([
                    row["cross_norm"][str(rank)]
                    for row in causal_records if row["corpus"] == corpus
                ])
            }
            for rank in config["shared_ranks"]
        }

    hypotheses = {
        "R1_rank8_is_not_a_unique_spectral_cut": {
            "supported": eigenvalue_ratio < 1.20,
            "eigenvalue_ratio": eigenvalue_ratio,
        },
        "R2_candidate_cluster_is_weight_stable": {
            "supported": all(
                row["within_median"] - row["outside_median"] >= 0.10
                for row in affinity.values()
            ),
            "gaps": {
                rank: row["within_median"] - row["outside_median"]
                for rank, row in affinity.items()
            },
        },
        "R3_sentiment_and_polarity_are_shared_dominant": {
            "supported": all(
                summary[corpus][family][str(rank)]["shared_like"]
                > summary[corpus][family][str(rank)]["family_residual"]
                for corpus in summary
                for family in ("sentiment", "polarity")
                for rank in (8, 12, 16)
            ),
        },
        "R4_gender_and_tense_are_residual_dominant": {
            "supported": all(
                summary[corpus][family][str(rank)]["family_residual"]
                - summary[corpus][family][str(rank)]["shared_like"] >= 0.20
                for corpus in summary
                for family in ("gender", "tense")
                for rank in config["shared_ranks"]
            ),
        },
        "R5_scale_is_a_mixed_regime": {
            "supported": all(
                any(
                    summary[corpus]["increasing_scale"][str(rank)]["shared_like"] >= 0.30
                    and summary[corpus]["increasing_scale"][str(rank)]["family_residual"] >= 0.30
                    for rank in config["shared_ranks"]
                )
                for corpus in summary
            ),
        },
        "R6_cross_term_survives_rank_choice": {
            "supported": all(
                summary[corpus]["aggregate"][str(rank)]["cross_norm_share"] >= 0.05
                for corpus in summary for rank in config["shared_ranks"]
            ),
            "values": {
                f"{corpus}/{rank}": summary[corpus]["aggregate"][str(rank)]["cross_norm_share"]
                for corpus in summary for rank in config["shared_ranks"]
            },
        },
        "R7_global_mean_basis_is_never_sufficient": {
            "supported": all(
                summary[corpus]["aggregate"][str(rank)]["global_common"] < 0.70
                for corpus in summary for rank in config["shared_ranks"]
            ),
        },
        "R8_family_regime_transfers_ood": {
            "supported": all(
                np.sign(
                    summary["fineweb"][family]["8"]["shared_like"]
                    - summary["fineweb"][family]["8"]["family_residual"]
                ) == np.sign(
                    summary["tinystories"][family]["8"]["shared_like"]
                    - summary["tinystories"][family]["8"]["family_residual"]
                )
                for family in family_names
            ),
            "differences": {
                family: {
                    corpus: summary[corpus][family]["8"]["shared_like"]
                    - summary[corpus][family]["8"]["family_residual"]
                    for corpus in ("fineweb", "tinystories")
                }
                for family in family_names
            },
        },
    }

    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "eigenvalues": values[:32].cpu().tolist(),
        "affinity": affinity,
        "summary": summary,
        "records": causal_records,
    }, indent=2))

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(family_names)))
    for family, color in zip(family_names, colors):
        axes[0].plot(
            config["shared_ranks"],
            [
                summary["tinystories"][family][str(rank)]["shared_like"]
                - summary["tinystories"][family][str(rank)]["family_residual"]
                for rank in config["shared_ranks"]
            ], marker="o", color=color, label=family,
        )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Shared cut rank")
    axes[0].set_ylabel("Shared-like minus residual causal fidelity")
    axes[0].set_title("Family regimes across quotient rank")
    axes[0].legend(fontsize=8)
    ranks = config["affinity_ranks"]
    axes[1].plot(
        ranks, [affinity[str(rank)]["within_median"] for rank in ranks],
        marker="o", label="within candidate cluster",
    )
    axes[1].plot(
        ranks, [affinity[str(rank)]["outside_median"] for rank in ranks],
        marker="o", label="cluster to gender/tense",
    )
    axes[1].set_xlabel("Family input rank")
    axes[1].set_ylabel("Normalized projector affinity")
    axes[1].set_title("Weight-only operator cluster")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    lines = [
        "# Rank and cluster red-team for the Layer-3 shared quotient",
        "",
        "The shared cut is swept from 4 to 16 dimensions while preserving each family's exact "
        "top-32 union. Separately, projector affinity is recomputed at family ranks 16, 32, and 64. "
        "The candidate sentiment/increasing-scale/polarity cluster was frozen after the first "
        "quotient geometry and is tested against gender and tense before this causal sweep.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Shared quotient rank red-team](layer3_shared_quotient_rank_redteam.png)",
        "",
        f"The rank-8/rank-9 mean-projector eigenvalue ratio is **{eigenvalue_ratio:.3f}**.",
        "",
        "| Family rank | Within-cluster affinity | Cluster-to-outside affinity | Gap |",
        "|---:|---:|---:|---:|",
    ]
    for rank in config["affinity_ranks"]:
        row = affinity[str(rank)]
        lines.append(
            f"| {rank} | {row['within_median']:.3f} | {row['outside_median']:.3f} | "
            f"{row['within_median'] - row['outside_median']:.3f} |"
        )
    lines += [
        "",
        "| Corpus / family | " + " | ".join(f"shared−residual @ {rank}" for rank in config["shared_ranks"]) + " |",
        "|---|" + "---:|" * len(config["shared_ranks"]),
    ]
    for corpus in ("fineweb", "tinystories"):
        for family in family_names:
            lines.append(
                f"| {corpus} / {family} | "
                + " | ".join(
                    f"{summary[corpus][family][str(rank)]['shared_like'] - summary[corpus][family][str(rank)]['family_residual']:.3f}"
                    for rank in config["shared_ranks"]
                ) + " |"
            )
    lines += [
        "",
        "A stable sign is a regime label, not a claim that either part is independently sufficient. "
        "The layer remains bilinear, and the separately reported cross term measures the interaction "
        "lost when shared-like and residual effects are merely added.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
