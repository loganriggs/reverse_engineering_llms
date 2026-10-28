#!/usr/bin/env python3
"""Causal-null and edit-scale red team for Layer-3 family input forms."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer3_family_interaction_modes import evaluate_writes
from run_bilin18_layer3_frozen_channel_transfer import channel_state, recover_configs, top_indices
from run_bilin18_layer3_native_bank_input_forms import bank_delta, input_metric, subspace_overlap
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_input_form_causal_nulls.json")
OUTPUT = TENSOR_ROOT / "layer3_input_form_causal_nulls.json"
REPORT = TENSOR_ROOT / "LAYER3_INPUT_FORM_CAUSAL_NULLS.md"
FIGURE = TENSOR_ROOT / "layer3_input_form_causal_nulls.png"


def factor_metric(block, indices):
    left = block.mlp.Left.weight.float()[indices]
    right = block.mlp.Right.weight.float()[indices]
    metric = left.T @ left + right.T @ right
    values, vectors = torch.linalg.eigh(0.5 * (metric + metric.T))
    return vectors[:, values.argsort(descending=True)]


def summarize(records, route):
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
    input_config = json.loads(Path(config["source_config"]).read_text())
    specificity_config = json.loads(Path(input_config["source_config"]).read_text())
    bank_config = json.loads(Path(specificity_config["source_config"]).read_text())
    rank_config = json.loads(Path(bank_config["source_config"]).read_text())
    form_config = json.loads(Path(rank_config["source_config"]).read_text())
    lopo_config = json.loads(Path(form_config["source_config"]).read_text())
    frozen_config = json.loads(Path(lopo_config["source_config"]).read_text())
    native_saved = json.loads(Path(form_config["native_results"]).read_text())
    fixed, parent = recover_configs(frozen_config)
    layer, output_rank = form_config["layer"], input_config["output_rank"]
    ranks, scales = config["ranks"], config["scales"]
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

    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    pair_cache = {}
    for pair, family in family_by_pair.items():
        indices = top_indices(torch.stack([
            score for candidate_pair, score in discovery_scores.items()
            if family_by_pair[candidate_pair] == family and candidate_pair != pair
        ]).mean(0), frozen_config["rank"]).cuda()
        observed = input_metric(block, indices, output_rank)
        sign_candidates, pairing_candidates = [], []
        for _ in range(config["null_candidates"]):
            signs = torch.randint(0, 2, (indices.numel(),), generator=generator, device="cuda") * 2 - 1
            permutation = torch.randperm(indices.numel(), generator=generator, device="cuda")
            sign_null = input_metric(block, indices, output_rank, signs=signs)
            pairing_null = input_metric(block, indices, output_rank, permutation=permutation)
            sign_candidates.append((
                subspace_overlap(observed["vectors"], sign_null["vectors"], 32), sign_null,
            ))
            pairing_candidates.append((
                subspace_overlap(observed["vectors"], pairing_null["vectors"], 32), pairing_null,
            ))
        sign_overlap, sign_null = max(sign_candidates, key=lambda row: row[0])
        pairing_overlap, pairing_null = max(pairing_candidates, key=lambda row: row[0])
        pair_cache[pair] = {
            "indices": indices,
            "output": observed["output"],
            "correct": observed["vectors"],
            "sign": sign_null["vectors"],
            "pairing": pairing_null["vectors"],
            "factor": factor_metric(block, indices),
            "null_overlap": {"sign": sign_overlap, "pairing": pairing_overlap},
        }
        print("null selection", family, pair, flush=True)

    query_records = [
        record for record in native_saved["semantic_records"]
        if record["context_index"] == frozen_config["evaluation_context_index"]
    ]
    basis_types = ("correct", "sign", "pairing", "factor", "random")
    evaluated = []
    for record_index, record in enumerate(query_records):
        base, counterfactual = reconstruct(record)
        state = channel_state(model, layer, base, counterfactual, final_position)
        cache = pair_cache[record["pair"]]
        dimension = base["pre"].shape[-1]
        xb = F.rms_norm(base["pre"], (dimension,)).float()
        xc = F.rms_norm(counterfactual["pre"], (dimension,)).float()
        delta = xc - xb
        random_generator = torch.Generator(device="cuda").manual_seed(config["seed"] + 1000 + record_index)
        random_basis = torch.linalg.qr(torch.randn(
            dimension, max(ranks), generator=random_generator, device="cuda",
        ), mode="reduced").Q
        bases = {
            "correct": cache["correct"],
            "sign": cache["sign"],
            "pairing": cache["pairing"],
            "factor": cache["factor"],
            "random": random_basis,
        }
        delta_energy = {
            f"{basis_type}_{rank}": float(
                ((delta @ basis[:, :rank]).square().sum() / delta.square().sum().clamp_min(1e-30))
            )
            for basis_type, basis in bases.items() for rank in ranks
        }
        scale_results = {}
        for scale in scales:
            changed_full = xb + scale * delta
            full_mlp = bank_delta(
                block, xb, changed_full, torch.arange(block.mlp.Left.weight.shape[0], device="cuda"),
            )
            exact_bank = bank_delta(block, xb, changed_full, cache["indices"])
            core_exact = (exact_bank @ cache["output"]) @ cache["output"].T
            writes = {
                "full_mlp": full_mlp,
                "family_exact": core_exact,
                "exact_bank": exact_bank,
            }
            for basis_type, basis in bases.items():
                for rank in ranks:
                    chosen = basis[:, :rank]
                    changed = xb + scale * (delta @ chosen) @ chosen.T
                    predicted = bank_delta(block, xb, changed, cache["indices"])
                    writes[f"{basis_type}_{rank}"] = (
                        (predicted @ cache["output"]) @ cache["output"].T
                    )
            effects = evaluate_writes(model, layer, base, counterfactual, writes, final_position)
            kept = ["family_exact", "exact_bank"] + [
                f"{basis_type}_{rank}" for basis_type in basis_types for rank in ranks
            ]
            scale_results[str(scale)] = {name: effects[name] for name in kept}
        evaluated.append({
            "corpus": record["corpus"],
            "family": record["family"],
            "pair": record["pair"],
            "null_overlap": cache["null_overlap"],
            "delta_energy": delta_energy,
            "scales": scale_results,
        })
        print("causal scales", record["corpus"], record["family"], record["pair"], flush=True)

    route_names = ["family_exact", "exact_bank"] + [
        f"{basis_type}_{rank}" for basis_type in basis_types for rank in ranks
    ]
    summaries = {}
    for corpus in fixed["evaluation"]:
        records = [record for record in evaluated if record["corpus"] == corpus]
        summaries[corpus] = {
            str(scale): {
                name: {
                    metric: median([
                        record["scales"][str(scale)][name][metric] for record in records
                    ])
                    for metric in (
                        "full_mlp_fidelity", "full_mlp_cosine", "family_patch_fidelity",
                        "family_patch_cosine", "semantic_target_projection",
                    )
                }
                for name in route_names
            }
            for scale in scales
        }
        summaries[corpus]["delta_energy"] = {
            name: median([record["delta_energy"][name] for record in records])
            for name in (f"{basis_type}_{rank}" for basis_type in basis_types for rank in ranks)
        }
        summaries[corpus]["null_overlap"] = {
            kind: median([record["null_overlap"][kind] for record in records])
            for kind in ("sign", "pairing")
        }
    core = lambda corpus, scale, route: summaries[corpus][str(scale)][route]["family_patch_fidelity"]
    hypotheses = {
        "J1_top4_beats_adversarial_sign_null": {
            "supported": all(core(c, 1.0, "correct_4") - core(c, 1.0, "sign_4") >= 0.10 for c in summaries),
            "gaps": {c: core(c, 1.0, "correct_4") - core(c, 1.0, "sign_4") for c in summaries},
        },
        "J2_top4_beats_adversarial_pairing_null": {
            "supported": all(core(c, 1.0, "correct_4") - core(c, 1.0, "pairing_4") >= 0.10 for c in summaries),
            "gaps": {c: core(c, 1.0, "correct_4") - core(c, 1.0, "pairing_4") for c in summaries},
        },
        "J3_top4_beats_unweighted_factor_span": {
            "supported": all(core(c, 1.0, "correct_4") - core(c, 1.0, "factor_4") >= 0.10 for c in summaries),
            "gaps": {c: core(c, 1.0, "correct_4") - core(c, 1.0, "factor_4") for c in summaries},
        },
        "J4_top4_survives_tangent_regime": {
            "supported": all(core(c, 0.1, "correct_4") >= 0.70 for c in summaries),
        },
        "J5_top32_is_stable_across_scale": {
            "supported": all(core(c, scale, "correct_32") >= 0.90 for c in summaries for scale in scales),
        },
        "J6_top4_is_not_just_delta_energy": {
            "supported": all(
                summaries[c]["delta_energy"]["correct_4"] <= 0.25
                and core(c, 1.0, "correct_4") >= 0.70
                for c in summaries
            ),
            "values": {
                c: {
                    "delta_energy": summaries[c]["delta_energy"]["correct_4"],
                    "core_fidelity": core(c, 1.0, "correct_4"),
                }
                for c in summaries
            },
        },
        "J7_null_redteam_transfers_ood": {
            "supported": all(
                core("tinystories", 1.0, "correct_4") > core("tinystories", 1.0, f"{control}_4")
                for control in ("sign", "pairing", "factor")
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "summaries": summaries,
        "records": evaluated,
    }, indent=2))

    lines = [
        "# Causal-null and scale red team for Layer-3 family input forms",
        "",
        "For every leave-one-pair-out bank, four sign and four Down/form-pairing null metrics are generated. The null with greatest rank-32 overlap to the observed metric is selected before causal evaluation. An unweighted Left/Right factor-span basis and random basis are additional controls. Each basis receives the same projected natural state change at scales 0.1--1.0, after which the exact bank and model suffix are recomputed.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Input-form causal nulls](layer3_input_form_causal_nulls.png)",
        "",
        "| Corpus | Scale | Correct-4 | Sign-4 | Pairing-4 | Factor-4 | Random-4 | Correct-32 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, summary in summaries.items():
        for scale in scales:
            rows = summary[str(scale)]
            lines.append(
                f"| {corpus} | {scale:.2f} | {rows['correct_4']['family_patch_fidelity']:.3f} | "
                f"{rows['sign_4']['family_patch_fidelity']:.3f} | {rows['pairing_4']['family_patch_fidelity']:.3f} | "
                f"{rows['factor_4']['family_patch_fidelity']:.3f} | {rows['random_4']['family_patch_fidelity']:.3f} | "
                f"{rows['correct_32']['family_patch_fidelity']:.3f} |"
            )
        lines += [
            "",
            f"Rank-4 state-delta energy captured (correct/sign/pairing/factor/random): "
            f"**{summary['delta_energy']['correct_4']:.3f} / {summary['delta_energy']['sign_4']:.3f} / "
            f"{summary['delta_energy']['pairing_4']:.3f} / {summary['delta_energy']['factor_4']:.3f} / "
            f"{summary['delta_energy']['random_4']:.3f}**.",
            f"Adversarial sign/pairing rank-32 overlap with observed: **{summary['null_overlap']['sign']:.3f} / {summary['null_overlap']['pairing']:.3f}**.",
            "",
        ]
    lines += [
        "The scale sweep separates a stable tangent discrimination from a full-edit curvature artifact. The adversarial nulls test whether the causal subspace follows learned Down-to-form coordination or merely the selected channels' generic input geometry.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        for basis_type in basis_types:
            axis.plot(scales, [summaries[corpus][str(scale)][f"{basis_type}_4"]["family_patch_fidelity"] for scale in scales], marker="o", label=basis_type)
        axis.set_xlabel("edit scale")
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
