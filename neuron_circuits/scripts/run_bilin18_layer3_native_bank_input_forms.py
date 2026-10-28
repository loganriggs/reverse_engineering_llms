#!/usr/bin/env python3
"""Analyze and causally test input-side forms of family-specific Layer-3 banks."""

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


CONFIG = Path("configs/bilin18_layer3_native_bank_input_forms.json")
OUTPUT = TENSOR_ROOT / "layer3_native_bank_input_forms.json"
REPORT = TENSOR_ROOT / "LAYER3_NATIVE_BANK_INPUT_FORMS.md"
FIGURE = TENSOR_ROOT / "layer3_native_bank_input_forms.png"


def subspace_overlap(first, second, rank):
    return float((first[:, :rank].T @ second[:, :rank]).square().sum() / rank)


@torch.inference_mode()
def input_metric(block, indices, output_rank, signs=None, permutation=None):
    down = block.mlp.Down.weight.float()[:, indices]
    left = block.mlp.Left.weight.float()[indices]
    right = block.mlp.Right.weight.float()[indices]
    if signs is not None:
        down = down * signs
    if permutation is not None:
        down = down[:, permutation]
    output, _, _ = torch.linalg.svd(down, full_matrices=False)
    output = output[:, :output_rank]
    coefficients = down.T @ output
    metric = torch.zeros(
        left.shape[1], left.shape[1], device=left.device, dtype=torch.float32,
    )
    for axis in range(output_rank):
        form = left.T @ (coefficients[:, axis, None] * right)
        form = 0.5 * (form + form.T)
        metric += form @ form
    metric = 0.5 * (metric + metric.T)
    values, vectors = torch.linalg.eigh(metric)
    order = values.argsort(descending=True)
    return {
        "output": output,
        "values": values[order].clamp_min(0),
        "vectors": vectors[:, order],
    }


@torch.inference_mode()
def bank_delta(block, base_normalized, changed_normalized, indices):
    base_product = (
        block.mlp.Left(base_normalized)[..., indices]
        * block.mlp.Right(base_normalized)[..., indices]
    )
    changed_product = (
        block.mlp.Left(changed_normalized)[..., indices]
        * block.mlp.Right(changed_normalized)[..., indices]
    )
    return F.linear(
        changed_product - base_product,
        block.mlp.Down.weight.float()[:, indices],
        bias=None,
    )


def metric_summary(records, route):
    return {
        metric: median([record["effects"][route][metric] for record in records])
        for metric in (
            "full_mlp_fidelity", "full_mlp_cosine", "family_patch_fidelity",
            "family_patch_cosine", "semantic_target_projection",
        )
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    specificity_config = json.loads(Path(config["source_config"]).read_text())
    bank_config = json.loads(Path(specificity_config["source_config"]).read_text())
    rank_config = json.loads(Path(bank_config["source_config"]).read_text())
    form_config = json.loads(Path(rank_config["source_config"]).read_text())
    lopo_config = json.loads(Path(form_config["source_config"]).read_text())
    frozen_config = json.loads(Path(lopo_config["source_config"]).read_text())
    native_saved = json.loads(Path(form_config["native_results"]).read_text())
    fixed, parent = recover_configs(frozen_config)
    layer = form_config["layer"]
    output_rank, input_ranks = config["output_rank"], config["input_ranks"]
    overlap_rank = config["overlap_rank"]
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[layer]
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

    def select(predicate):
        return top_indices(torch.stack([
            score for candidate_pair, score in discovery_scores.items() if predicate(candidate_pair)
        ]).mean(0), frozen_config["rank"]).cuda()

    inclusive = {}
    for family in family_names:
        indices = select(lambda pair, family=family: family_by_pair[pair] == family)
        inclusive[family] = {
            "indices": indices,
            "metric": input_metric(block, indices, output_rank),
        }
        print("inclusive metric", family, flush=True)

    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    pair_cache = {}
    null_rows = []
    for pair, family in family_by_pair.items():
        family_indices = select(
            lambda candidate_pair, family=family, pair=pair:
            family_by_pair[candidate_pair] == family and candidate_pair != pair
        )
        global_indices = select(lambda candidate_pair, pair=pair: candidate_pair != pair)
        observed = input_metric(block, family_indices, output_rank)
        global_metric = input_metric(block, global_indices, output_rank)
        correct_overlap = subspace_overlap(
            observed["vectors"], inclusive[family]["metric"]["vectors"], overlap_rank,
        )
        global_overlap = subspace_overlap(observed["vectors"], global_metric["vectors"], overlap_rank)
        wrong_overlaps = {
            wrong: subspace_overlap(
                observed["vectors"], inclusive[wrong]["metric"]["vectors"], overlap_rank,
            )
            for wrong in family_names if wrong != family
        }
        strongest_wrong = max(wrong_overlaps, key=wrong_overlaps.get)
        observed_energy = float(
            observed["values"][:overlap_rank].sum() / observed["values"].sum().clamp_min(1e-30)
        )
        sign_energies, pairing_energies = [], []
        for replicate in range(config["null_replicates"]):
            signs = (
                torch.randint(0, 2, (family_indices.numel(),), generator=generator, device="cuda")
                * 2 - 1
            )
            permutation = torch.randperm(family_indices.numel(), generator=generator, device="cuda")
            sign_null = input_metric(block, family_indices, output_rank, signs=signs)
            pairing_null = input_metric(block, family_indices, output_rank, permutation=permutation)
            sign_energies.append(float(
                sign_null["values"][:overlap_rank].sum()
                / sign_null["values"].sum().clamp_min(1e-30)
            ))
            pairing_energies.append(float(
                pairing_null["values"][:overlap_rank].sum()
                / pairing_null["values"].sum().clamp_min(1e-30)
            ))
        null_rows.append({
            "pair": pair,
            "family": family,
            "observed_top32_energy": observed_energy,
            "sign_null_median": float(np.median(sign_energies)),
            "pairing_null_median": float(np.median(pairing_energies)),
        })
        pair_cache[pair] = {
            "family": family,
            "indices": family_indices,
            "observed": observed,
            "global": global_metric,
            "wrong": inclusive[strongest_wrong]["metric"],
            "wrong_family": strongest_wrong,
            "overlap": {
                "correct": correct_overlap,
                "global": global_overlap,
                "best_wrong": wrong_overlaps[strongest_wrong],
            },
            "energy": observed_energy,
        }
        print("pair metric", family, pair, flush=True)

    query_records = [
        record for record in native_saved["semantic_records"]
        if record["context_index"] == frozen_config["evaluation_context_index"]
    ]
    evaluated = []
    basis_types = ("correct", "global", "wrong", "low", "random")
    for record_index, record in enumerate(query_records):
        base, counterfactual = reconstruct(record)
        state = channel_state(model, layer, base, counterfactual, final_position)
        cache = pair_cache[record["pair"]]
        dimension = base["pre"].shape[-1]
        xb = F.rms_norm(base["pre"], (dimension,)).float()
        xc = F.rms_norm(counterfactual["pre"], (dimension,)).float()
        exact_bank = bank_delta(block, xb, xc, cache["indices"])
        output = cache["observed"]["output"]
        core_exact = (exact_bank @ output) @ output.T
        random_generator = torch.Generator(device="cuda").manual_seed(config["seed"] + 1000 + record_index)
        random_basis = torch.linalg.qr(torch.randn(
            dimension, max(input_ranks), generator=random_generator, device="cuda",
        ), mode="reduced").Q
        bases = {
            "correct": cache["observed"]["vectors"],
            "global": cache["global"]["vectors"],
            "wrong": cache["wrong"]["vectors"],
            "low": cache["observed"]["vectors"].flip(1),
            "random": random_basis,
        }
        writes = {
            "full_mlp": state["target_write"],
            "family_exact": core_exact,
            "exact_bank": exact_bank,
        }
        delta = xc - xb
        for basis_type, basis in bases.items():
            for rank in input_ranks:
                chosen = basis[:, :rank]
                changed = xb + (delta @ chosen) @ chosen.T
                predicted = bank_delta(block, xb, changed, cache["indices"])
                writes[f"{basis_type}_{rank}"] = (predicted @ output) @ output.T
        effects = evaluate_writes(model, layer, base, counterfactual, writes, final_position)
        kept = ["family_exact", "exact_bank"] + [
            f"{basis_type}_{rank}" for basis_type in basis_types for rank in input_ranks
        ]
        evaluated.append({
            "corpus": record["corpus"],
            "family": record["family"],
            "pair": record["pair"],
            "wrong_family": cache["wrong_family"],
            "overlap": cache["overlap"],
            "input_energy": cache["energy"],
            "effects": {name: effects[name] for name in kept},
        })
        print("causal", record["corpus"], record["family"], record["pair"], flush=True)

    route_names = ["family_exact", "exact_bank"] + [
        f"{basis_type}_{rank}" for basis_type in basis_types for rank in input_ranks
    ]
    summaries = {}
    for corpus in fixed["evaluation"]:
        records = [record for record in evaluated if record["corpus"] == corpus]
        summaries[corpus] = {name: metric_summary(records, name) for name in route_names}
        summaries[corpus]["overlap"] = {
            key: median([record["overlap"][key] for record in records])
            for key in ("correct", "global", "best_wrong")
        }
        summaries[corpus]["input_energy"] = median([record["input_energy"] for record in records])
    observed_energy = [row["observed_top32_energy"] for row in null_rows]
    sign_energy = [row["sign_null_median"] for row in null_rows]
    pairing_energy = [row["pairing_null_median"] for row in null_rows]
    core = lambda corpus, name: summaries[corpus][name]["family_patch_fidelity"]
    hypotheses = {
        "I1_family_input_metric_is_concentrated": {
            "supported": all(value >= 0.50 for value in observed_energy),
            "minimum": min(observed_energy),
            "median": median(observed_energy),
        },
        "I2_family_input_eigenspace_is_specific": {
            "supported": all(
                summaries[c]["overlap"]["correct"] - summaries[c]["overlap"]["global"] >= 0.05
                and summaries[c]["overlap"]["correct"] - summaries[c]["overlap"]["best_wrong"] >= 0.10
                for c in summaries
            ),
            "values": {c: summaries[c]["overlap"] for c in summaries},
        },
        "I3_input_concentration_exceeds_coordination_nulls": {
            "supported": median(observed_energy) - max(median(sign_energy), median(pairing_energy)) >= 0.05,
            "observed": median(observed_energy),
            "sign_null": median(sign_energy),
            "pairing_null": median(pairing_energy),
        },
        "I4_top32_input_modes_causally_recover_core": {
            "supported": all(core(c, "correct_32") >= 0.50 for c in summaries),
        },
        "I5_top32_input_modes_beat_random": {
            "supported": all(core(c, "correct_32") - core(c, "random_32") >= 0.20 for c in summaries),
            "gaps": {c: core(c, "correct_32") - core(c, "random_32") for c in summaries},
        },
        "I6_top32_input_modes_are_family_specific": {
            "supported": all(
                core(c, "correct_32") - core(c, "global_32") >= 0.10
                and core(c, "correct_32") - core(c, "wrong_32") >= 0.10
                for c in summaries
            ),
            "global_gaps": {c: core(c, "correct_32") - core(c, "global_32") for c in summaries},
            "wrong_gaps": {c: core(c, "correct_32") - core(c, "wrong_32") for c in summaries},
        },
        "I7_top64_input_modes_nearly_recover_core": {
            "supported": all(core(c, "correct_64") >= 0.75 for c in summaries),
        },
        "I8_input_mode_effect_transfers_ood": {
            "supported": core("tinystories", "correct_32") >= 0.75 * core("fineweb", "correct_32"),
            "retention": core("tinystories", "correct_32") / max(core("fineweb", "correct_32"), 1e-30),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "summaries": summaries,
        "null_rows": null_rows,
        "records": evaluated,
    }, indent=2))

    lines = [
        "# Input-side forms of Layer-3 family operator banks",
        "",
        "For each leave-one-pair-out family bank, its top four weight-defined output axes contract the selected bilinear channels into four symmetric input forms. The rotation-invariant input metric is `sum_j A_j^2`. Its leading eigenspace is compared with correct-family, global, strongest wrong-family, sign-null, and Down/form-pairing controls. Held-out state changes are then projected into top, low, global, wrong, and random input subspaces before recomputing the exact family bank and the full model suffix.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Family-bank input forms](layer3_native_bank_input_forms.png)",
        "",
        f"Median top-32 input energy: **{median(observed_energy):.3f}**; sign-null **{median(sign_energy):.3f}**; pairing-null **{median(pairing_energy):.3f}**.",
        "",
        "| Corpus | Input basis | Rank | Core-patch fidelity | Full-MLP fidelity |",
        "|---|---|---:|---:|---:|",
    ]
    for corpus, summary in summaries.items():
        for basis_type in basis_types:
            for rank in input_ranks:
                row = summary[f"{basis_type}_{rank}"]
                lines.append(
                    f"| {corpus} | {basis_type} | {rank} | {row['family_patch_fidelity']:.3f} | "
                    f"{row['full_mlp_fidelity']:.3f} |"
                )
        lines += [
            f"| {corpus} | exact output core | — | 1.000 | {summary['family_exact']['full_mlp_fidelity']:.3f} |",
            f"| {corpus} | exact full bank | — | {summary['exact_bank']['family_patch_fidelity']:.3f} | {summary['exact_bank']['full_mlp_fidelity']:.3f} |",
            "",
            f"Correct/global/best-wrong rank-{overlap_rank} input overlap: "
            f"**{summary['overlap']['correct']:.3f} / {summary['overlap']['global']:.3f} / {summary['overlap']['best_wrong']:.3f}**.",
            "",
        ]
    lines += [
        "A concentrated, recurring metric is a weight-level description of which residual directions the bank can discriminate. Causal projection is the stronger test: it asks whether the actual held-out semantic state change travels through those directions. A failure there means the global form energy and the naturally used edit path are different explanatory objects.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        for basis_type in basis_types:
            axis.plot(input_ranks, [summaries[corpus][f"{basis_type}_{rank}"]["family_patch_fidelity"] for rank in input_ranks], marker="o", label=basis_type)
        axis.set_xscale("log", base=2)
        axis.set_xticks(input_ranks, input_ranks)
        axis.set_xlabel("input-subspace rank")
        axis.set_ylabel("fidelity to exact output-core patch")
        axis.set_title(corpus)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
