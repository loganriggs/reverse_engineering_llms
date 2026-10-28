#!/usr/bin/env python3
"""Split Layer-3 family input spaces into common and family-residual quotients."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer3_family_interaction_modes import evaluate_writes
from run_bilin18_layer3_frozen_channel_transfer import (
    channel_state,
    recover_configs,
    top_indices,
)
from run_bilin18_layer3_input_mode_semantics import collect_coordinates
from run_bilin18_layer3_native_bank_input_forms import bank_delta, input_metric
from run_bilin18_layer3_semantic_subspace_rank import sampled_auc
from run_bilin18_layer3_unseen_lexical_transfer import pole_from_label, prior_ids
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_shared_backbone_quotient.json")
OUTPUT = TENSOR_ROOT / "layer3_shared_backbone_quotient.json"
ARRAYS = TENSOR_ROOT / "layer3_shared_backbone_quotient.npz"
REPORT = TENSOR_ROOT / "LAYER3_SHARED_BACKBONE_QUOTIENT.md"
FIGURE = TENSOR_ROOT / "layer3_shared_backbone_quotient.png"


def normalized_overlap(first: torch.Tensor, second: torch.Tensor) -> float:
    rank = min(first.shape[1], second.shape[1])
    return float((first.T @ second).square().sum() / max(rank, 1))


def projector_error(original: torch.Tensor, parts: list[torch.Tensor]) -> float:
    target = original @ original.T
    reconstructed = sum(part @ part.T for part in parts)
    return float((target - reconstructed).norm() / target.norm().clamp_min(1e-30))


def local_metrics(target: torch.Tensor, value: torch.Tensor) -> dict:
    target_flat = target.flatten()
    value_flat = value.flatten()
    denominator = target_flat.square().sum().clamp_min(1e-30)
    cosine = F.cosine_similarity(target_flat, value_flat, dim=0)
    return {
        "fidelity": 1 - float((value_flat - target_flat).square().sum() / denominator),
        "cosine": float(cosine),
        "norm_share": float(value_flat.norm() / target_flat.norm().clamp_min(1e-30)),
    }


def finite_median(values) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    return float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else float("nan")


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    semantic_config = json.loads(Path(config["source_config"]).read_text())
    semantic_source = json.loads(Path(config["source_semantic_results"]).read_text())
    unseen_config = json.loads(Path(config["unseen_config"]).read_text())
    unseen_result = json.loads(Path(config["unseen_results"]).read_text())

    input_config = json.loads(Path(semantic_config["source_config"]).read_text())
    specificity_config = json.loads(Path(input_config["source_config"]).read_text())
    bank_config = json.loads(Path(specificity_config["source_config"]).read_text())
    rank_config = json.loads(Path(bank_config["source_config"]).read_text())
    form_config = json.loads(Path(rank_config["source_config"]).read_text())
    lopo_config = json.loads(Path(form_config["source_config"]).read_text())
    frozen_config = json.loads(Path(lopo_config["source_config"]).read_text())
    fixed, parent = recover_configs(frozen_config)
    families = parent["families"]
    family_names = list(families)
    native_saved = json.loads(Path(config["native_results"]).read_text())
    layer = semantic_config["layer"]
    final_position = fixed["causal_position"]

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
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

    discovery_records = [
        record for record in native_saved["semantic_records"]
        if record["corpus"] == frozen_config["discovery_corpus"]
        and record["context_index"] == frozen_config["discovery_context_index"]
    ]
    family_scores = {family: [] for family in family_names}
    for record in discovery_records:
        base, counterfactual = reconstruct(record)
        score = channel_state(
            model, layer, base, counterfactual, final_position,
        )["scores"].abs().cpu()
        family_scores[record["family"]].append(score)
        print("discovery", record["family"], record["pair"], flush=True)

    family_cache = {}
    for family in family_names:
        indices = top_indices(
            torch.stack(family_scores[family]).mean(0), config["channel_rank"],
        ).cuda()
        metric = input_metric(block, indices, config["output_rank"])
        family_cache[family] = {
            "indices": indices,
            "output": metric["output"],
            "basis": metric["vectors"][:, : config["family_rank"]],
            "metric_values": metric["values"][: config["family_rank"]],
        }

    # Weight-only Grassmann mean: eigensystem of the mean family projector.
    mean_projector = sum(
        cache["basis"] @ cache["basis"].T for cache in family_cache.values()
    ) / len(family_cache)
    common_values, common_vectors = torch.linalg.eigh(mean_projector)
    order = common_values.argsort(descending=True)
    common_values = common_values[order]
    common = common_vectors[:, order[: config["shared_rank"]]]

    for family, cache in family_cache.items():
        basis = cache["basis"]
        inside = basis.T @ (common @ common.T) @ basis
        values, vectors = torch.linalg.eigh(0.5 * (inside + inside.T))
        order = values.argsort(descending=True)
        vectors, values = vectors[:, order], values[order]
        cache["shared_like"] = basis @ vectors[:, : config["shared_rank"]]
        cache["family_residual"] = basis @ vectors[:, config["shared_rank"]:]
        cache["specific"] = basis @ vectors[:, -config["specific_rank"]:]
        cache["common_alignment_values"] = values
        print("quotient", family, flush=True)

    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    random_24 = torch.linalg.qr(torch.randn(
        model.config.n_embd,
        config["family_rank"] - config["shared_rank"],
        generator=generator,
        device="cuda",
    ), mode="reduced").Q

    pairwise_original, pairwise_residual = [], []
    for index, family in enumerate(family_names):
        for other in family_names[index + 1:]:
            pairwise_original.append(normalized_overlap(
                family_cache[family]["basis"], family_cache[other]["basis"],
            ))
            pairwise_residual.append(normalized_overlap(
                family_cache[family]["family_residual"],
                family_cache[other]["family_residual"],
            ))
    geometry = {
        "common_eigenvalues": common_values[:32].cpu().tolist(),
        "common_overlap_by_family": {
            family: normalized_overlap(common, cache["basis"])
            for family, cache in family_cache.items()
        },
        "split_projector_error": {
            family: projector_error(
                cache["basis"], [cache["shared_like"], cache["family_residual"]],
            )
            for family, cache in family_cache.items()
        },
        "pairwise_original_median": finite_median(pairwise_original),
        "pairwise_residual_median": finite_median(pairwise_residual),
        "pairwise_original": pairwise_original,
        "pairwise_residual": pairwise_residual,
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

        def route(input_basis: torch.Tensor) -> torch.Tensor:
            changed = xb + (delta @ input_basis) @ input_basis.T
            predicted = bank_delta(block, xb, changed, cache["indices"])
            return (predicted @ cache["output"]) @ cache["output"].T

        union = route(cache["basis"])
        shared_like = route(cache["shared_like"])
        family_residual = route(cache["family_residual"])
        cross_term = union - shared_like - family_residual
        additive = shared_like + family_residual
        global_common = route(common)
        random_residual = route(random_24)
        exact_bank = bank_delta(block, xb, xc, cache["indices"])
        core_exact = (exact_bank @ cache["output"]) @ cache["output"].T
        writes = {
            "full_mlp": state["target_write"],
            "family_exact": union,
            "core_exact": core_exact,
            "union32": union,
            "shared_like8": shared_like,
            "family_residual24": family_residual,
            "additive_no_cross": additive,
            "cross_term": cross_term,
            "global_common8": global_common,
            "random_24": random_residual,
        }
        wrong_names = []
        for wrong in family_names:
            if wrong == family:
                continue
            name = f"wrong_residual:{wrong}"
            writes[name] = route(family_cache[wrong]["family_residual"])
            wrong_names.append(name)
        effects = evaluate_writes(
            model, layer, base, counterfactual, writes, final_position,
        )
        strongest_wrong = max(
            wrong_names, key=lambda name: effects[name]["family_patch_fidelity"],
        )
        causal_records.append({
            "corpus": record["corpus"],
            "family": family,
            "pair": record["pair"],
            "effects": effects,
            "strongest_wrong": strongest_wrong.split(":", 1)[1],
            "strongest_wrong_fidelity": effects[strongest_wrong]["family_patch_fidelity"],
            "local": {
                name: local_metrics(union, value)
                for name, value in {
                    "union32": union,
                    "shared_like8": shared_like,
                    "family_residual24": family_residual,
                    "additive_no_cross": additive,
                    "cross_term": cross_term,
                    "global_common8": global_common,
                    "random_24": random_residual,
                }.items()
            },
            "cross_term_norm_share": float(
                cross_term.norm() / union.norm().clamp_min(1e-30)
            ),
        })
        print("causal quotient", record["corpus"], family, record["pair"], flush=True)

    route_names = (
        "union32", "shared_like8", "family_residual24", "additive_no_cross",
        "cross_term", "global_common8", "random_24", "core_exact",
    )
    causal_summary = {}
    for corpus in fixed["evaluation"]:
        selected = [row for row in causal_records if row["corpus"] == corpus]
        causal_summary[corpus] = {
            route: {
                metric: median([row["effects"][route][metric] for row in selected])
                for metric in (
                    "family_patch_fidelity", "family_patch_cosine",
                    "full_mlp_fidelity", "semantic_target_projection",
                )
            }
            for route in route_names
        }
        causal_summary[corpus]["local"] = {
            route: {
                metric: median([row["local"][route][metric] for row in selected])
                for metric in ("fidelity", "cosine", "norm_share")
            }
            for route in route_names if route in selected[0]["local"]
        }
        causal_summary[corpus]["strongest_wrong_fidelity"] = median([
            row["strongest_wrong_fidelity"] for row in selected
        ])
        causal_summary[corpus]["cross_term_norm_share"] = median([
            row["cross_term_norm_share"] for row in selected
        ])

    # Natural semantic audit of the same quotient. All probe fitting uses only
    # FineWeb occurrences of the original seed_ids; unseen labels are frozen.
    direction_parts = [common, random_24]
    slices = {
        "global_common8": slice(0, config["shared_rank"]),
        "random_24": slice(
            config["shared_rank"],
            config["shared_rank"] + random_24.shape[1],
        ),
    }
    offset = config["shared_rank"] + random_24.shape[1]
    for family in family_names:
        for name in ("shared_like", "specific", "family_residual"):
            basis = family_cache[family][name]
            direction_parts.append(basis)
            slices[f"{family}:{name}"] = slice(offset, offset + basis.shape[1])
            offset += basis.shape[1]
    directions = torch.cat(direction_parts, dim=1)
    corpora = dict(semantic_config["corpora"])
    corpora["wikitext"] = unseen_config["wikitext"]
    panels = {
        corpus: collect_coordinates(model, spec, directions, semantic_config)
        for corpus, spec in corpora.items()
    }
    seed_ids = {
        family: {
            "source": [pair[0] for pair in pairs],
            "target": [pair[1] for pair in pairs],
            "any": [token for pair in pairs for token in pair],
        }
        for family, pairs in families.items()
    }
    unseen = unseen_result["lexicon_ids"]
    labels_by_family = {
        family: semantic_source["representatives"][family]["native"]["labels"]["own"]
        for family in family_names
    }
    semantic_records = []
    route_slices = {
        family: {
            "global_common8": slices["global_common8"],
            "correct_shared8": slices[f"{family}:shared_like"],
            "correct_specific8": slices[f"{family}:specific"],
            "correct_residual24": slices[f"{family}:family_residual"],
            "random_8": slice(slices["random_24"].start, slices["random_24"].start + 8),
            "random_24": slices["random_24"],
            **{
                f"wrong_residual:{wrong}": slices[f"{wrong}:family_residual"]
                for wrong in family_names if wrong != family
            },
        }
        for family in family_names
    }
    fine = panels["fineweb"]
    for family in family_names:
        pole = pole_from_label(labels_by_family[family])
        seed_mask = np.isin(fine["token_ids"], seed_ids[family][pole])
        for route_name, selected_slice in route_slices[family].items():
            discovery = fine["coordinates"][:, selected_slice]
            mean = discovery.mean(0)
            std = discovery.std(0).clip(1e-5)
            normalized = (discovery - mean) / std
            mean_positive = normalized[seed_mask].mean(0)
            mean_negative = normalized[~seed_mask].mean(0)
            probe = mean_positive - mean_negative
            for corpus, panel in panels.items():
                values = (panel["coordinates"][:, selected_slice] - mean) / std
                for token_offset in (0, 1):
                    token_ids, valid = prior_ids(panel, corpora[corpus], token_offset)
                    labels = np.isin(token_ids[valid], unseen[family][pole])
                    scores = values[valid] @ probe
                    result = sampled_auc(
                        labels, scores,
                        config["maximum_negative_examples"],
                        config["minimum_positive_count"],
                        config["seed"],
                        f"{family}/{route_name}/{corpus}/{token_offset}",
                    )
                    semantic_records.append({
                        "family": family,
                        "route": route_name,
                        "corpus": corpus,
                        "offset": token_offset,
                        "pole": pole,
                        **result,
                    })
            print("semantic quotient", family, route_name, flush=True)

    def semantic_value(family: str, route: str, corpus: str, token_offset: int = 0) -> float:
        return next(
            row["auc"] for row in semantic_records
            if row["family"] == family and row["route"] == route
            and row["corpus"] == corpus and row["offset"] == token_offset
        )

    hypotheses = {
        "B1_common_weight_subspace_exists": {
            "supported": finite_median(geometry["common_overlap_by_family"].values()) >= 0.25,
            "overlap_by_family": geometry["common_overlap_by_family"],
        },
        "B2_family_split_preserves_the_original_space": {
            "supported": max(geometry["split_projector_error"].values()) <= 1e-5,
            "errors": geometry["split_projector_error"],
        },
        "B3_residual_spaces_are_more_family_specific": {
            "supported": geometry["pairwise_original_median"]
            - geometry["pairwise_residual_median"] >= 0.05,
            "original_median": geometry["pairwise_original_median"],
            "residual_median": geometry["pairwise_residual_median"],
        },
        "C1_union_reproduces_the_causal_family_patch": {
            "supported": max(
                abs(row["local"]["union32"]["fidelity"] - 1) for row in causal_records
            ) <= 1e-5,
            "maximum_one_minus_fidelity": max(
                abs(row["local"]["union32"]["fidelity"] - 1) for row in causal_records
            ),
        },
        "C2_shared_coordinates_are_causally_substantial": {
            "supported": all(
                causal_summary[corpus]["shared_like8"]["family_patch_fidelity"] >= 0.20
                for corpus in causal_summary
            ),
            "values": {
                corpus: causal_summary[corpus]["shared_like8"]["family_patch_fidelity"]
                for corpus in causal_summary
            },
        },
        "C3_residual_coordinates_are_causally_substantial": {
            "supported": all(
                causal_summary[corpus]["family_residual24"]["family_patch_fidelity"] >= 0.50
                for corpus in causal_summary
            ),
            "values": {
                corpus: causal_summary[corpus]["family_residual24"]["family_patch_fidelity"]
                for corpus in causal_summary
            },
        },
        "C4_shared_residual_cross_term_is_nontrivial": {
            "supported": all(
                causal_summary[corpus]["cross_term_norm_share"] >= 0.05
                for corpus in causal_summary
            ),
            "norm_shares": {
                corpus: causal_summary[corpus]["cross_term_norm_share"]
                for corpus in causal_summary
            },
        },
        "C5_residual_causality_is_family_specific": {
            "supported": all(
                causal_summary[corpus]["family_residual24"]["family_patch_fidelity"]
                - causal_summary[corpus]["strongest_wrong_fidelity"] >= 0.10
                for corpus in causal_summary
            ),
            "gaps": {
                corpus: causal_summary[corpus]["family_residual24"]["family_patch_fidelity"]
                - causal_summary[corpus]["strongest_wrong_fidelity"]
                for corpus in causal_summary
            },
        },
        "C6_global_common_basis_is_not_sufficient": {
            "supported": all(
                1 - causal_summary[corpus]["global_common8"]["family_patch_fidelity"] >= 0.30
                for corpus in causal_summary
            ),
            "union_minus_common": {
                corpus: 1 - causal_summary[corpus]["global_common8"]["family_patch_fidelity"]
                for corpus in causal_summary
            },
        },
        "S1_common_space_contains_shared_lexical_roles": {
            "supported": all(
                semantic_value(family, "global_common8", corpus) >= 0.80
                for family in ("sentiment", "polarity")
                for corpus in ("tinystories", "wikitext")
            ),
            "values": {
                f"{family}/{corpus}": semantic_value(family, "global_common8", corpus)
                for family in ("sentiment", "polarity")
                for corpus in ("tinystories", "wikitext")
            },
        },
        "S2_specific_coordinates_recover_gender_and_tense": {
            "supported": all(
                semantic_value(family, "correct_specific8", corpus) >= 0.75
                and semantic_value(family, "correct_specific8", corpus)
                - semantic_value(family, "global_common8", corpus) >= 0.05
                for family in ("gender", "tense")
                for corpus in ("tinystories", "wikitext")
            ),
            "gaps": {
                f"{family}/{corpus}": semantic_value(family, "correct_specific8", corpus)
                - semantic_value(family, "global_common8", corpus)
                for family in ("gender", "tense")
                for corpus in ("tinystories", "wikitext")
            },
        },
        "S3_scale_residual_is_not_generic_decodability": {
            "supported": all(
                semantic_value("increasing_scale", "correct_residual24", corpus)
                - semantic_value("increasing_scale", "random_24", corpus) >= 0.05
                for corpus in ("tinystories", "wikitext")
            ),
            "gaps": {
                corpus: semantic_value("increasing_scale", "correct_residual24", corpus)
                - semantic_value("increasing_scale", "random_24", corpus)
                for corpus in ("tinystories", "wikitext")
            },
        },
        "S4_shared_roles_persist_after_the_token": {
            "supported": all(
                semantic_value(family, "global_common8", corpus, 1) >= 0.60
                for family in ("sentiment", "polarity")
                for corpus in ("tinystories", "wikitext")
            ),
            "values": {
                f"{family}/{corpus}": semantic_value(family, "global_common8", corpus, 1)
                for family in ("sentiment", "polarity")
                for corpus in ("tinystories", "wikitext")
            },
        },
    }

    semantic_summary = []
    for family in family_names:
        semantic_summary.append({
            "family": family,
            **{
                f"{corpus}_{route}_0": semantic_value(family, route, corpus, 0)
                for corpus in ("tinystories", "wikitext")
                for route in (
                    "global_common8", "correct_shared8", "correct_specific8",
                    "correct_residual24", "random_24",
                )
            },
            **{
                f"{corpus}_global_common8_1": semantic_value(
                    family, "global_common8", corpus, 1,
                )
                for corpus in ("tinystories", "wikitext")
            },
        })

    result = {
        "config": config,
        "hypotheses": hypotheses,
        "geometry": geometry,
        "causal_summary": causal_summary,
        "causal_records": causal_records,
        "semantic_summary": semantic_summary,
        "semantic_records": semantic_records,
    }
    OUTPUT.write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        ARRAYS,
        common=common.cpu().numpy(),
        common_values=common_values[:64].cpu().numpy(),
        **{
            f"{family}_{name}": cache[name].cpu().numpy()
            for family, cache in family_cache.items()
            for name in ("basis", "shared_like", "family_residual", "specific")
        },
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    routes = ("shared_like8", "family_residual24", "additive_no_cross", "global_common8")
    x = np.arange(len(routes))
    width = .35
    for index, corpus in enumerate(("fineweb", "tinystories")):
        axes[0].bar(
            x + (index - .5) * width,
            [causal_summary[corpus][route]["family_patch_fidelity"] for route in routes],
            width, label=corpus,
        )
    axes[0].axhline(1, color="black", linewidth=1)
    axes[0].set_xticks(x, ("shared-like 8", "residual 24", "sum without cross", "global common 8"), rotation=20, ha="right")
    axes[0].set_ylabel("Fidelity to family top-32 causal patch")
    axes[0].set_title("Causal quotient")
    axes[0].legend()
    x = np.arange(len(family_names))
    width = .2
    for offset_index, route in enumerate((
        "global_common8", "correct_specific8", "correct_residual24", "random_24",
    )):
        axes[1].bar(
            x + (offset_index - 1.5) * width,
            [semantic_value(family, route, "wikitext") for family in family_names],
            width, label=route,
        )
    axes[1].axhline(.5, color="black", linewidth=1)
    axes[1].set_xticks(x, family_names, rotation=25, ha="right")
    axes[1].set_ylabel("WikiText unseen-word AUROC")
    axes[1].set_title("Natural roles of the quotient")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    lines = [
        "# Shared lexical-backbone quotient of Layer-3 family input spaces",
        "",
        "The top-eight eigenspace of the mean of five weight-defined family projectors is the "
        "global common candidate. Each family top-32 space is then rotated internally and split "
        "orthogonally into the eight coordinates most aligned with that common space and a residual "
        "24. Their union is exactly the original family space. Finite counterfactuals test each part, "
        "their missing bilinear cross term, wrong-family residuals, and a global common basis. "
        "Natural probes are trained only on original FineWeb seed tokens and evaluated on frozen "
        "unseen lexicons.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Shared-backbone quotient](layer3_shared_backbone_quotient.png)",
        "",
        "## Weight geometry",
        "",
        f"Median original-family overlap: **{geometry['pairwise_original_median']:.3f}**; "
        f"residual-family overlap: **{geometry['pairwise_residual_median']:.3f}**.",
        "",
        "| Family | Common overlap | Split projector error |",
        "|---|---:|---:|",
    ]
    for family in family_names:
        lines.append(
            f"| {family} | {geometry['common_overlap_by_family'][family]:.3f} | "
            f"{geometry['split_projector_error'][family]:.2e} |"
        )
    lines += [
        "",
        "## Finite causal decomposition",
        "",
        "| Corpus | Shared-like 8 | Residual 24 | Sum without cross | Cross norm share | Global common 8 | Best wrong residual |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus in causal_summary:
        row = causal_summary[corpus]
        lines.append(
            f"| {corpus} | {row['shared_like8']['family_patch_fidelity']:.3f} | "
            f"{row['family_residual24']['family_patch_fidelity']:.3f} | "
            f"{row['additive_no_cross']['family_patch_fidelity']:.3f} | "
            f"{row['cross_term_norm_share']:.3f} | "
            f"{row['global_common8']['family_patch_fidelity']:.3f} | "
            f"{row['strongest_wrong_fidelity']:.3f} |"
        )
    lines += [
        "",
        "## Unseen-word roles",
        "",
        "| Family | Tiny common/specific/residual/random | Wiki common/specific/residual/random | Common +1 Tiny/Wiki |",
        "|---|---:|---:|---:|",
    ]
    for row in semantic_summary:
        family = row["family"]
        lines.append(
            f"| {family} | "
            f"{row['tinystories_global_common8_0']:.3f}/{row['tinystories_correct_specific8_0']:.3f}/"
            f"{row['tinystories_correct_residual24_0']:.3f}/{row['tinystories_random_24_0']:.3f} | "
            f"{row['wikitext_global_common8_0']:.3f}/{row['wikitext_correct_specific8_0']:.3f}/"
            f"{row['wikitext_correct_residual24_0']:.3f}/{row['wikitext_random_24_0']:.3f} | "
            f"{row['tinystories_global_common8_1']:.3f}/{row['wikitext_global_common8_1']:.3f} |"
        )
    lines += [
        "",
        "This quotient has three distinct possible successes: shared geometry, semantic decoding, "
        "and causal sufficiency. They are reported separately. A common lexical probe does not prove "
        "that the common directions carry a family edit; a causal residual does not prove that its "
        "natural activation has a clean human label; and a cross term means neither orthogonal state "
        "component is independently sufficient in a bilinear layer.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
