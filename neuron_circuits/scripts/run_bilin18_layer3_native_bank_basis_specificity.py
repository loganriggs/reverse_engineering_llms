#!/usr/bin/env python3
"""Test whether dominant output axes of reusable native banks are family-specific."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer3_family_interaction_modes import evaluate_writes
from run_bilin18_layer3_frozen_channel_transfer import channel_state, recover_configs, top_indices
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_native_bank_basis_specificity.json")
OUTPUT = TENSOR_ROOT / "layer3_native_bank_basis_specificity.json"
REPORT = TENSOR_ROOT / "LAYER3_NATIVE_BANK_BASIS_SPECIFICITY.md"
FIGURE = TENSOR_ROOT / "layer3_native_bank_basis_specificity.png"


def metric_summary(records, route):
    return {
        metric: median([record["effects"][route][metric] for record in records])
        for metric in (
            "full_mlp_fidelity", "full_mlp_cosine", "family_patch_fidelity",
            "family_patch_cosine", "semantic_target_projection",
        )
    }


def canonical_cosine(first, second, rank):
    return float(torch.linalg.svdvals(first[:, :rank].T @ second[:, :rank]).mean())


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    bank_config = json.loads(Path(config["source_config"]).read_text())
    rank_config = json.loads(Path(bank_config["source_config"]).read_text())
    form_config = json.loads(Path(rank_config["source_config"]).read_text())
    lopo_config = json.loads(Path(form_config["source_config"]).read_text())
    frozen_config = json.loads(Path(lopo_config["source_config"]).read_text())
    native_saved = json.loads(Path(form_config["native_results"]).read_text())
    fixed, parent = recover_configs(frozen_config)
    ranks, maximum_rank = config["ranks"], max(config["ranks"])
    layer = form_config["layer"]
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[layer]
    down = block.mlp.Down.weight.float()
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")
        for corpus, spec in fixed["evaluation"].items()
    }
    specs = fixed["evaluation"]

    def reconstruct(record):
        corpus = record["corpus"]
        spec = specs[corpus]
        rows = arrays[corpus][spec["start"]:spec["start"] + spec["rows"]]
        raw = np.asarray(
            rows[record["row"], record["source_start"]:record["source_start"] + fixed["sequence_length"]],
            dtype=np.int64,
        ).copy()
        changed = raw.copy()
        changed[edit_position] = int(record["pair"].split("->")[1])
        base = forward_layers(model, torch.as_tensor(raw, device="cuda")[None], last_layer=layer)
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer,
        )
        return base[layer], counterfactual[layer]

    discovery_records = [
        record for record in native_saved["semantic_records"]
        if record["corpus"] == frozen_config["discovery_corpus"]
        and record["context_index"] == frozen_config["discovery_context_index"]
    ]
    discovery_scores, family_by_pair = {}, {}
    for record in discovery_records:
        base, counterfactual = reconstruct(record)
        state = channel_state(model, layer, base, counterfactual, final_position)
        discovery_scores[record["pair"]] = state["scores"].abs().cpu()
        family_by_pair[record["pair"]] = record["family"]
    family_names = list(parent["families"])

    def bank_basis(indices):
        u, _, _ = torch.linalg.svd(down[:, indices], full_matrices=False)
        return u[:, :maximum_rank]

    inclusive_wrong = {}
    for family in family_names:
        indices = top_indices(torch.stack([
            score for pair, score in discovery_scores.items() if family_by_pair[pair] == family
        ]).mean(0), frozen_config["rank"]).cuda()
        inclusive_wrong[family] = bank_basis(indices)
    pair_cache = {}
    for pair, family in family_by_pair.items():
        family_indices = top_indices(torch.stack([
            score for candidate_pair, score in discovery_scores.items()
            if family_by_pair[candidate_pair] == family and candidate_pair != pair
        ]).mean(0), frozen_config["rank"]).cuda()
        global_indices = top_indices(torch.stack([
            score for candidate_pair, score in discovery_scores.items()
            if candidate_pair != pair
        ]).mean(0), frozen_config["rank"]).cuda()
        family_basis = bank_basis(family_indices)
        global_basis = bank_basis(global_indices)
        pair_cache[pair] = {
            "indices": family_indices,
            "family_basis": family_basis,
            "global_basis": global_basis,
            "canonical": {
                str(rank): canonical_cosine(family_basis, global_basis, rank) for rank in ranks
            },
        }

    query_records = [
        record for record in native_saved["semantic_records"]
        if record["context_index"] == frozen_config["evaluation_context_index"]
    ]
    evaluated = []
    for record in query_records:
        base, counterfactual = reconstruct(record)
        state = channel_state(model, layer, base, counterfactual, final_position)
        pair, family = record["pair"], record["family"]
        cache = pair_cache[pair]
        selected = torch.zeros_like(state["delta_product"])
        selected[..., cache["indices"]] = state["delta_product"][..., cache["indices"]]
        native_write = F.linear(selected, down, bias=None)
        writes = {"full_mlp": state["target_write"], "family_exact": native_write}
        wrong_names = {}
        for rank in ranks:
            for label, basis in (
                ("family", cache["family_basis"]), ("global", cache["global_basis"]),
            ):
                chosen = basis[:, :rank]
                writes[f"{label}_{rank}"] = (native_write @ chosen) @ chosen.T
            wrong_names[rank] = []
            for index, wrong in enumerate(name for name in family_names if name != family):
                name = f"wrong_{rank}_{index}"
                chosen = inclusive_wrong[wrong][:, :rank]
                writes[name] = (native_write @ chosen) @ chosen.T
                wrong_names[rank].append(name)
        effects = evaluate_writes(model, layer, base, counterfactual, writes, final_position)
        for rank in ranks:
            wrong_rows = [effects[name] for name in wrong_names[rank]]
            effects[f"wrong_best_{rank}"] = dict(max(
                wrong_rows, key=lambda row: row["family_patch_fidelity"],
            ))
            effects[f"wrong_median_{rank}"] = {
                metric: float(np.median([row[metric] for row in wrong_rows]))
                for metric in wrong_rows[0]
            }
        kept = ["family_exact"] + [
            name for rank in ranks for name in (
                f"family_{rank}", f"global_{rank}", f"wrong_best_{rank}", f"wrong_median_{rank}",
            )
        ]
        evaluated.append({
            "corpus": record["corpus"],
            "family": family,
            "pair": pair,
            "canonical_cosine": cache["canonical"],
            "effects": {name: effects[name] for name in kept},
        })
        print("evaluation", record["corpus"], family, pair, flush=True)

    route_names = ["family_exact"] + [
        name for rank in ranks for name in (
            f"family_{rank}", f"global_{rank}", f"wrong_best_{rank}", f"wrong_median_{rank}",
        )
    ]
    summaries = {}
    for corpus in fixed["evaluation"]:
        records = [record for record in evaluated if record["corpus"] == corpus]
        summaries[corpus] = {name: metric_summary(records, name) for name in route_names}
        summaries[corpus]["canonical_cosine"] = {
            str(rank): median([record["canonical_cosine"][str(rank)] for record in records])
            for rank in ranks
        }
    native = lambda corpus, name: summaries[corpus][name]["family_patch_fidelity"]
    full = lambda corpus, name: summaries[corpus][name]["full_mlp_fidelity"]
    hypotheses = {
        "P1_family_basis_beats_global_at32": {
            "supported": all(native(c, "family_32") - native(c, "global_32") >= 0.05 for c in summaries),
            "gaps": {c: native(c, "family_32") - native(c, "global_32") for c in summaries},
        },
        "P2_family_basis_beats_best_wrong_at32": {
            "supported": all(native(c, "family_32") - native(c, "wrong_best_32") >= 0.05 for c in summaries),
            "gaps": {c: native(c, "family_32") - native(c, "wrong_best_32") for c in summaries},
        },
        "P3_family_rank64_nearly_recovers_native_bank": {
            "supported": all(native(c, "family_64") >= 0.90 for c in summaries),
        },
        "P4_family_rank64_preserves_full_mlp_relevance": {
            "supported": all(full(c, "family_exact") - full(c, "family_64") <= 0.05 for c in summaries),
            "losses": {c: full(c, "family_exact") - full(c, "family_64") for c in summaries},
        },
        "P5_family_and_global_axes_are_distinct": {
            "supported": all(summaries[c]["canonical_cosine"]["32"] <= 0.75 for c in summaries),
            "values": {c: summaries[c]["canonical_cosine"]["32"] for c in summaries},
        },
        "P6_basis_specificity_transfers_ood": {
            "supported": native("tinystories", "family_32") > native("tinystories", "wrong_best_32"),
            "gap": native("tinystories", "family_32") - native("tinystories", "wrong_best_32"),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "summaries": summaries,
        "records": evaluated,
    }, indent=2))

    lines = [
        "# Specificity of Layer-3 native-bank output axes",
        "",
        "The exact current write of each leave-one-pair-out family bank is projected into weight-SVD bases from that bank, a global channel bank, and every wrong-family bank. This holds the native write fixed and asks whether the compressed output axes themselves are family-specific.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Native-bank basis specificity](layer3_native_bank_basis_specificity.png)",
        "",
        "| Corpus | Rank | Family | Global | Best wrong | Median wrong | Family/global canonical cosine |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, summary in summaries.items():
        for rank in ranks:
            lines.append(
                f"| {corpus} | {rank} | {summary[f'family_{rank}']['family_patch_fidelity']:.3f} | "
                f"{summary[f'global_{rank}']['family_patch_fidelity']:.3f} | "
                f"{summary[f'wrong_best_{rank}']['family_patch_fidelity']:.3f} | "
                f"{summary[f'wrong_median_{rank}']['family_patch_fidelity']:.3f} | "
                f"{summary['canonical_cosine'][str(rank)]:.3f} |"
            )
        lines.append(
            f"| {corpus} | exact native | 1.000 | — | — | — | — |"
        )
    lines += [
        "",
        "If family, global, and wrong-bank bases perform similarly, the bank identity lives mainly in which input-side channels are activated while output compression is generic. A family-specific basis instead supplies a second reusable object: a characteristic write subspace as well as a characteristic channel address set.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        for label in ("family", "global", "wrong_best", "wrong_median"):
            axis.plot(ranks, [summaries[corpus][f"{label}_{rank}"]["family_patch_fidelity"] for rank in ranks], marker="o", label=label)
        axis.set_xscale("log", base=2)
        axis.set_xticks(ranks, ranks)
        axis.set_xlabel("weight-output rank")
        axis.set_ylabel("fidelity to exact native patch")
        axis.set_title(corpus)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
