#!/usr/bin/env python3
"""Test whether full-scale quadratic family computation is a sparse pair web."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer3_family_interaction_modes import evaluate_writes
from run_bilin18_layer3_frozen_channel_transfer import channel_state, recover_configs, top_indices
from run_bilin18_layer3_input_polynomial_split import forms_for_output
from run_bilin18_layer3_native_bank_input_forms import bank_delta, input_metric
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_quadratic_interaction_sparsity.json")
OUTPUT = TENSOR_ROOT / "layer3_quadratic_interaction_sparsity.json"
REPORT = TENSOR_ROOT / "LAYER3_QUADRATIC_INTERACTION_SPARSITY.md"
FIGURE = TENSOR_ROOT / "layer3_quadratic_interaction_sparsity.png"


def pair_table(forms, basis):
    reduced = torch.stack([basis.T @ form @ basis for form in forms])
    dimension = basis.shape[1]
    pairs = torch.triu_indices(dimension, dimension, device=basis.device)
    diagonal = pairs[0] == pairs[1]
    coefficients = reduced[:, pairs[0], pairs[1]]
    coefficients[:, ~diagonal] *= 2
    raw_energy = reduced.square().sum()
    diagonal_energy = torch.diagonal(reduced, dim1=1, dim2=2).square().sum()
    return {
        "reduced": reduced,
        "pairs": pairs,
        "coefficients": coefficients,
        "diagonal": diagonal,
        "offdiagonal_energy_fraction": float((raw_energy - diagonal_energy) / raw_energy.clamp_min(1e-30)),
    }


def pair_scores(z, table, selection):
    pairs = table["pairs"][:, selection]
    products = z[..., pairs[0]] * z[..., pairs[1]]
    coefficients = table["coefficients"][:, selection]
    return products @ coefficients.T


def first_rank(curve, budgets, threshold):
    for budget in budgets:
        if curve[str(budget)] >= threshold:
            return budget
    return None


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    polynomial_config = json.loads(Path(config["source_config"]).read_text())
    k90_config = json.loads(Path(polynomial_config["source_config"]).read_text())
    null_config = json.loads(Path(k90_config["source_config"]).read_text())
    input_config = json.loads(Path(null_config["source_config"]).read_text())
    specificity_config = json.loads(Path(input_config["source_config"]).read_text())
    bank_config = json.loads(Path(specificity_config["source_config"]).read_text())
    rank_config = json.loads(Path(bank_config["source_config"]).read_text())
    form_config = json.loads(Path(rank_config["source_config"]).read_text())
    lopo_config = json.loads(Path(form_config["source_config"]).read_text())
    frozen_config = json.loads(Path(lopo_config["source_config"]).read_text())
    native_saved = json.loads(Path(form_config["native_results"]).read_text())
    fixed, parent = recover_configs(frozen_config)
    layer, output_rank = form_config["layer"], input_config["output_rank"]
    budgets = config["budgets"]
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
    dimension = model.config.n_embd

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
    discovery, family_by_pair = {}, {}
    for record in discovery_records:
        base, counterfactual = reconstruct(record)
        state = channel_state(model, layer, base, counterfactual, final_position)
        xb = F.rms_norm(base["pre"], (dimension,)).float()
        xc = F.rms_norm(counterfactual["pre"], (dimension,)).float()
        discovery[record["pair"]] = {
            "score": state["scores"].abs().cpu(),
            "delta": xc - xb,
        }
        family_by_pair[record["pair"]] = record["family"]

    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    pair_cache = {}
    for pair, family in family_by_pair.items():
        other_pairs = [
            candidate_pair for candidate_pair in discovery
            if family_by_pair[candidate_pair] == family and candidate_pair != pair
        ]
        indices = top_indices(torch.stack([
            discovery[candidate_pair]["score"] for candidate_pair in other_pairs
        ]).mean(0), frozen_config["rank"]).cuda()
        metric = input_metric(block, indices, output_rank)
        input_rank = polynomial_config["input_rank_by_family"][family]
        basis = metric["vectors"][:, :input_rank]
        forms = forms_for_output(block, indices, metric["output"])
        table = pair_table(forms, basis)
        importance = torch.zeros(table["pairs"].shape[1], device="cuda")
        for candidate_pair in other_pairs:
            z = discovery[candidate_pair]["delta"] @ basis
            products = z[..., table["pairs"][0]] * z[..., table["pairs"][1]]
            importance += (products.square().sum((0, 1)) * table["coefficients"].square().sum(0))
        activation_order = importance.argsort(descending=True)
        weight_order = table["coefficients"].square().sum(0).argsort(descending=True)
        random_order = torch.randperm(table["pairs"].shape[1], generator=generator, device="cuda")
        pair_cache[pair] = {
            "family": family,
            "indices": indices,
            "output": metric["output"],
            "basis": basis,
            "table": table,
            "activation_order": activation_order,
            "weight_order": weight_order,
            "random_order": random_order,
        }
        print("pair program", family, pair, table["pairs"].shape[1], flush=True)

    query_records = [
        record for record in native_saved["semantic_records"]
        if record["context_index"] == frozen_config["evaluation_context_index"]
    ]
    evaluated = []
    for record in query_records:
        base, counterfactual = reconstruct(record)
        cache = pair_cache[record["pair"]]
        xb = F.rms_norm(base["pre"], (dimension,)).float()
        xc = F.rms_norm(counterfactual["pre"], (dimension,)).float()
        delta = xc - xb
        projected = (delta @ cache["basis"]) @ cache["basis"].T
        z = delta @ cache["basis"]
        exact_bank = bank_delta(block, xb, xb + projected, cache["indices"])
        full_core = (exact_bank @ cache["output"]) @ cache["output"].T
        all_pairs = torch.arange(cache["table"]["pairs"].shape[1], device="cuda")
        exact_quadratic = pair_scores(z, cache["table"], all_pairs) @ cache["output"].T
        diagonal_indices = torch.where(cache["table"]["diagonal"])[0]
        offdiagonal_indices = torch.where(~cache["table"]["diagonal"])[0]
        writes = {
            "full_mlp": full_core,
            "family_exact": exact_quadratic,
            "diagonal": pair_scores(z, cache["table"], diagonal_indices) @ cache["output"].T,
            "offdiagonal": pair_scores(z, cache["table"], offdiagonal_indices) @ cache["output"].T,
        }
        for label, order in (
            ("activation", cache["activation_order"]),
            ("weight", cache["weight_order"]),
            ("random", cache["random_order"]),
        ):
            for budget in budgets:
                selection = order[: min(budget, order.numel())]
                writes[f"{label}_{budget}"] = (
                    pair_scores(z, cache["table"], selection) @ cache["output"].T
                )
        effects = evaluate_writes(model, layer, base, counterfactual, writes, final_position)
        local = {
            name: 1 - float(
                (write - exact_quadratic).square().sum()
                / exact_quadratic.square().sum().clamp_min(1e-30)
            )
            for name, write in writes.items() if name not in ("full_mlp", "family_exact")
        }
        kept = ["family_exact", "diagonal", "offdiagonal"] + [
            f"{label}_{budget}" for label in ("activation", "weight", "random") for budget in budgets
        ]
        evaluated.append({
            "corpus": record["corpus"],
            "family": record["family"],
            "pair": record["pair"],
            "input_rank": cache["basis"].shape[1],
            "pair_count": cache["table"]["pairs"].shape[1],
            "offdiagonal_weight_energy": cache["table"]["offdiagonal_energy_fraction"],
            "effects": {name: effects[name] for name in kept},
            "local_fidelity": local,
        })
        print("evaluate web", record["corpus"], record["family"], record["pair"], flush=True)

    summaries, by_family = {}, {}
    route_names = ["diagonal", "offdiagonal"] + [
        f"{label}_{budget}" for label in ("activation", "weight", "random") for budget in budgets
    ]
    for corpus in fixed["evaluation"]:
        records = [record for record in evaluated if record["corpus"] == corpus]
        summaries[corpus] = {
            name: {
                "quadratic_patch_fidelity": median([
                    record["effects"][name]["family_patch_fidelity"] for record in records
                ]),
                "full_core_fidelity": median([
                    record["effects"][name]["full_mlp_fidelity"] for record in records
                ]),
                "local_fidelity": median([record["local_fidelity"][name] for record in records]),
            }
            for name in route_names
        }
        by_family[corpus] = {}
        for family in parent["families"]:
            family_records = [record for record in records if record["family"] == family]
            curve = {
                str(budget): median([
                    record["effects"][f"activation_{budget}"]["family_patch_fidelity"]
                    for record in family_records
                ])
                for budget in budgets
            }
            by_family[corpus][family] = {
                "input_rank": family_records[0]["input_rank"],
                "pair_count": family_records[0]["pair_count"],
                "offdiagonal_weight_energy": median([
                    record["offdiagonal_weight_energy"] for record in family_records
                ]),
                "activation_curve": curve,
                "causal_k90": first_rank(curve, budgets, config["threshold"]),
                "diagonal_fidelity": median([
                    record["effects"]["diagonal"]["family_patch_fidelity"] for record in family_records
                ]),
                "offdiagonal_fidelity": median([
                    record["effects"]["offdiagonal"]["family_patch_fidelity"] for record in family_records
                ]),
            }
    quad = lambda corpus, name: summaries[corpus][name]["quadratic_patch_fidelity"]
    finite = lambda value: value if value is not None else 10**9
    hypotheses = {
        "M1_quadratic_tensor_is_offdiagonal": {
            "supported": all(
                row["offdiagonal_weight_energy"] >= 0.50
                for families in by_family.values() for row in families.values()
            ),
        },
        "M2_offdiagonal_terms_are_causally_substantial": {
            "supported": all(quad(c, "offdiagonal") >= 0.50 for c in summaries),
        },
        "M3_frozen_top64_pairs_are_useful": {
            "supported": all(quad(c, "activation_64") >= 0.50 for c in summaries),
        },
        "M4_frozen_top256_pairs_are_sufficient": {
            "supported": all(quad(c, "activation_256") >= 0.80 for c in summaries),
        },
        "M5_activation_ranking_beats_weight_ranking": {
            "supported": all(quad(c, "activation_256") - quad(c, "weight_256") >= 0.10 for c in summaries),
            "gaps": {c: quad(c, "activation_256") - quad(c, "weight_256") for c in summaries},
        },
        "M6_activation_ranking_beats_random": {
            "supported": all(quad(c, "activation_256") - quad(c, "random_256") >= 0.50 for c in summaries),
            "gaps": {c: quad(c, "activation_256") - quad(c, "random_256") for c in summaries},
        },
        "M7_every_family_has_pair_k90_by4096": {
            "supported": all(
                finite(row["causal_k90"]) <= 4096
                for families in by_family.values() for row in families.values()
            ),
        },
        "M8_pair_k90_is_ood_stable": {
            "supported": all(
                max(finite(by_family["fineweb"][family]["causal_k90"]), finite(by_family["tinystories"][family]["causal_k90"]))
                <= 4 * min(finite(by_family["fineweb"][family]["causal_k90"]), finite(by_family["tinystories"][family]["causal_k90"]))
                for family in parent["families"]
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "summaries": summaries,
        "by_family": by_family,
        "records": evaluated,
    }, indent=2))

    lines = [
        "# Sparsity of signed quadratic interaction webs in Layer 3",
        "",
        "Inside each family-specific causal input basis, the four output forms become a small signed tensor `C[o,i,j]`. Symmetric coordinate pairs are ranked once using their squared contribution on the leave-one-pair-out discovery edits, then frozen for held-out FineWeb and TinyStories. Coefficient-only and random rankings, plus complete diagonal and off-diagonal terms, are controls.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Quadratic interaction sparsity](layer3_quadratic_interaction_sparsity.png)",
        "",
        "| Corpus | Family | Input rank / pair count | Off-diagonal weight energy | Diagonal / off-diagonal fidelity | Pair K90 | Activation curve (4→32768) |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for corpus, families in by_family.items():
        for family, row in families.items():
            curve = ", ".join(f"{row['activation_curve'][str(budget)]:.3f}" for budget in budgets)
            lines.append(
                f"| {corpus} | {family} | {row['input_rank']} / {row['pair_count']} | "
                f"{row['offdiagonal_weight_energy']:.3f} | {row['diagonal_fidelity']:.3f} / "
                f"{row['offdiagonal_fidelity']:.3f} | {row['causal_k90'] if row['causal_k90'] is not None else '>32768'} | {curve} |"
            )
    lines += [
        "",
        "A low pair K90 supports a sparse verb-like interaction program in the chosen causal basis. A high K90 or strong diagonal control instead indicates a broad code or mostly independent feature-strength computation. Because the basis and output core are declared first, this measures sparsity within a specific explanatory quotient rather than claiming a unique global tensor decomposition.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(parent["families"])))
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        for color, family in zip(colors, parent["families"]):
            curve = by_family[corpus][family]["activation_curve"]
            axis.plot(budgets, [curve[str(budget)] for budget in budgets], marker="o", label=family, color=color)
        axis.axhline(config["threshold"], color="black", linestyle="--", linewidth=.8)
        axis.set_xscale("log", base=2)
        axis.set_xticks(budgets, budgets, rotation=35)
        axis.set_xlabel("frozen interaction-pair budget")
        axis.set_ylabel("quadratic-patch fidelity")
        axis.set_title(corpus)
        axis.grid(alpha=.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
