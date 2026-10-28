#!/usr/bin/env python3
"""Compare family-conditioned quadratic modes with reusable native channels."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer3_frozen_channel_transfer import channel_state, recover_configs, top_indices
from run_bilin18_layer8_router_causal_validation import suffix_from_layer
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_layer8_semantic_path_decomposition import fisher_inner
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer3_family_interaction_modes.json")
OUTPUT = TENSOR_ROOT / "layer3_family_interaction_modes.json"
REPORT = TENSOR_ROOT / "LAYER3_FAMILY_INTERACTION_MODES.md"
FIGURE = TENSOR_ROOT / "layer3_family_interaction_modes.png"


def output_basis(writes, rank):
    matrix = torch.cat(writes, dim=0).float()
    _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return vh[:rank].T.contiguous()


@torch.inference_mode()
def conditioned_eigensystems(block, basis):
    systems = []
    for index in range(basis.shape[1]):
        direction = basis[:, index]
        coefficients = block.mlp.Down.weight.float().T @ direction
        matrix = block.mlp.Left.weight.float().T @ (
            coefficients[:, None] * block.mlp.Right.weight.float()
        )
        matrix = 0.5 * (matrix + matrix.T)
        values, vectors = torch.linalg.eigh(matrix)
        order = values.abs().argsort(descending=True)
        systems.append((values[order], vectors[:, order]))
    return systems


@torch.inference_mode()
def spectral_writes(base, counterfactual, basis, systems, ranks):
    dimension = base["pre"].shape[-1]
    xb = F.rms_norm(base["pre"], (dimension,)).float()
    xc = F.rms_norm(counterfactual["pre"], (dimension,)).float()
    predictions = {rank: torch.zeros_like(base["mlp_write"].float()) for rank in ranks}
    complete = torch.zeros_like(base["mlp_write"].float())
    for output_index, (values, vectors) in enumerate(systems):
        direction = basis[:, output_index]
        base_coordinates = xb @ vectors
        changed_coordinates = xc @ vectors
        terms = (changed_coordinates.square() - base_coordinates.square()) * values
        complete += terms.sum(-1, keepdim=True) * direction
        for rank in ranks:
            predictions[rank] += terms[..., :rank].sum(-1, keepdim=True) * direction
    return predictions, complete


@torch.inference_mode()
def evaluate_writes(model, layer, base, counterfactual, writes, final_position):
    names = list(writes)
    outputs = [base["output"], counterfactual["output"]]
    x0s = [base["x0"], counterfactual["x0"]]
    buses = [base["value_bus"], counterfactual["value_bus"]]
    for name in names:
        outputs.append(base["output"] + writes[name])
        x0s.append(base["x0"])
        buses.append(base["value_bus"])
    logits = suffix_from_layer(
        model, layer, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), final_position,
    )
    base_logits, counterfactual_logits = logits[0], logits[1]
    semantic = counterfactual_logits - base_logits
    semantic_energy = fisher_inner(base_logits, semantic, semantic)
    effects = {name: logits[2 + index] - base_logits for index, name in enumerate(names)}
    full = effects["full_mlp"]
    family = effects["family_exact"]
    result = {}
    for name in names:
        effect = effects[name]
        full_fidelity, full_cosine, _ = fisher_fidelity(base_logits, full, effect)
        family_fidelity, family_cosine, _ = fisher_fidelity(base_logits, family, effect)
        result[name] = {
            "full_mlp_fidelity": full_fidelity,
            "full_mlp_cosine": full_cosine,
            "family_patch_fidelity": family_fidelity,
            "family_patch_cosine": family_cosine,
            "semantic_target_projection": fisher_inner(base_logits, effect, semantic)
            / max(semantic_energy, 1e-30),
        }
    return result


def summarize(records, names):
    return {
        name: {
            metric: median([record["effects"][name][metric] for record in records])
            for metric in (
                "full_mlp_fidelity", "full_mlp_cosine", "family_patch_fidelity",
                "family_patch_cosine", "semantic_target_projection",
            )
        }
        for name in names
    } | {
        "maximum_exact_form_error": max(record["exact_form_error"] for record in records),
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    lopo_config = json.loads(Path(config["source_config"]).read_text())
    frozen_config = json.loads(Path(lopo_config["source_config"]).read_text())
    native_saved = json.loads(Path(config["native_results"]).read_text())
    fixed, parent = recover_configs(frozen_config)
    layer, output_rank = config["layer"], config["output_rank"]
    ranks = config["interaction_ranks"]
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
    discovery_writes, discovery_scores, family_by_pair = {}, {}, {}
    for record in discovery_records:
        base, counterfactual = reconstruct(record)
        state = channel_state(model, layer, base, counterfactual, final_position)
        pair = record["pair"]
        discovery_writes[pair] = [
            state["target_write"][0, edit_position:final_position + 1]
        ]
        discovery_scores[pair] = state["scores"].abs()
        family_by_pair[pair] = record["family"]
        print("discovery", record["family"], pair, flush=True)

    family_names = list(parent["families"])
    inclusive_bases = {
        family: output_basis([
            write for pair, writes in discovery_writes.items()
            if family_by_pair[pair] == family for write in writes
        ], output_rank)
        for family in family_names
    }
    pair_cache = {}
    for pair, family in family_by_pair.items():
        family_writes = [
            write for candidate_pair, writes in discovery_writes.items()
            if family_by_pair[candidate_pair] == family and candidate_pair != pair
            for write in writes
        ]
        global_writes = [
            write for candidate_pair, writes in discovery_writes.items()
            if candidate_pair != pair for write in writes
        ]
        family_basis = output_basis(family_writes, output_rank)
        pair_cache[pair] = {
            "family_basis": family_basis,
            "global_basis": output_basis(global_writes, output_rank),
            "family_systems": conditioned_eigensystems(block, family_basis),
            "native_lopo": top_indices(torch.stack([
                score for candidate_pair, score in discovery_scores.items()
                if family_by_pair[candidate_pair] == family and candidate_pair != pair
            ]).mean(0), frozen_config["rank"]),
        }
        print("forms", family, pair, flush=True)

    query_records = [
        record for record in native_saved["semantic_records"]
        if record["context_index"] == frozen_config["evaluation_context_index"]
    ]
    base_names = [
        "full_mlp", "family_exact", "global_exact", "native_lopo",
        *[f"spectral_{rank}" for rank in ranks],
        "wrong_family_median", "wrong_family_best",
    ]
    evaluated = []
    for record in query_records:
        base, counterfactual = reconstruct(record)
        state = channel_state(model, layer, base, counterfactual, final_position)
        pair, family = record["pair"], record["family"]
        cached = pair_cache[pair]
        family_basis = cached["family_basis"]
        family_exact = (state["target_write"] @ family_basis) @ family_basis.T
        global_basis = cached["global_basis"]
        global_exact = (state["target_write"] @ global_basis) @ global_basis.T
        spectral, complete = spectral_writes(
            base, counterfactual, family_basis, cached["family_systems"], ranks,
        )
        exact_error = float(
            (complete - family_exact).norm() / family_exact.norm().clamp_min(1e-30)
        )
        selected = torch.zeros_like(state["delta_product"])
        selected[..., cached["native_lopo"]] = state["delta_product"][..., cached["native_lopo"]]
        native_write = F.linear(selected, block.mlp.Down.weight.float(), bias=None)
        writes = {
            "full_mlp": state["target_write"],
            "family_exact": family_exact,
            "global_exact": global_exact,
            "native_lopo": native_write,
            **{f"spectral_{rank}": spectral[rank] for rank in ranks},
        }
        wrong_names = []
        for index, wrong_family in enumerate(name for name in family_names if name != family):
            name = f"wrong_{index}"
            wrong_basis = inclusive_bases[wrong_family]
            writes[name] = (state["target_write"] @ wrong_basis) @ wrong_basis.T
            wrong_names.append(name)
        effects = evaluate_writes(model, layer, base, counterfactual, writes, final_position)
        wrong_rows = [effects[name] for name in wrong_names]
        effects["wrong_family_median"] = {
            metric: float(np.median([row[metric] for row in wrong_rows]))
            for metric in wrong_rows[0]
        }
        effects["wrong_family_best"] = dict(max(
            wrong_rows, key=lambda row: row["full_mlp_fidelity"],
        ))
        evaluated.append({
            "corpus": record["corpus"],
            "family": family,
            "pair": pair,
            "exact_form_error": exact_error,
            "local_family_output_fidelity": 1 - float(
                (family_exact - state["target_write"]).square().sum()
                / state["target_write"].square().sum().clamp_min(1e-30)
            ),
            "effects": {name: effects[name] for name in base_names},
        })
        print("evaluation", record["corpus"], family, pair, flush=True)

    summaries = {
        corpus: summarize(
            [record for record in evaluated if record["corpus"] == corpus], base_names,
        )
        for corpus in fixed["evaluation"]
    }
    full_fidelity = lambda corpus, name: summaries[corpus][name]["full_mlp_fidelity"]
    family_fidelity = lambda corpus, name: summaries[corpus][name]["family_patch_fidelity"]
    maximum_error = max(row["exact_form_error"] for row in evaluated)
    hypotheses = {
        "Q1_family_forms_are_exact": {
            "supported": maximum_error <= 1e-4,
            "maximum_relative_error": maximum_error,
        },
        "Q2_family_output4_is_causally_substantial": {
            "supported": all(full_fidelity(c, "family_exact") >= 0.20 for c in summaries),
        },
        "Q3_rank64_interactions_recover_family_patch": {
            "supported": all(family_fidelity(c, "spectral_64") >= 0.70 for c in summaries),
        },
        "Q4_rank256_interactions_nearly_recover_family_patch": {
            "supported": all(family_fidelity(c, "spectral_256") >= 0.90 for c in summaries),
        },
        "Q5_spectral256_beats_native_family_addresses": {
            "supported": all(
                full_fidelity(c, "spectral_256") - full_fidelity(c, "native_lopo") >= 0.05
                for c in summaries
            ),
            "gaps": {
                c: full_fidelity(c, "spectral_256") - full_fidelity(c, "native_lopo")
                for c in summaries
            },
        },
        "Q6_family_output_basis_beats_global_basis": {
            "supported": all(
                full_fidelity(c, "family_exact") - full_fidelity(c, "global_exact") >= 0.03
                for c in summaries
            ),
            "gaps": {
                c: full_fidelity(c, "family_exact") - full_fidelity(c, "global_exact")
                for c in summaries
            },
        },
        "Q7_family_output_basis_is_semantically_specific": {
            "supported": all(
                full_fidelity(c, "family_exact") > full_fidelity(c, "wrong_family_best")
                for c in summaries
            ),
            "gaps": {
                c: full_fidelity(c, "family_exact") - full_fidelity(c, "wrong_family_best")
                for c in summaries
            },
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "summaries": summaries,
        "records": evaluated,
    }, indent=2))

    lines = [
        "# Family-conditioned Layer-3 interaction modes",
        "",
        "For each lexical query, a four-dimensional family output subspace is learned from the other FineWeb pairs in that family. Each output axis contracts the exact Layer-3 tensor to a symmetric input interaction matrix; its eigenmodes are truncated and evaluated in the complete suffix. Native leave-one-pair-out channels, a global output basis, and every wrong-family output basis are controls.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Family interaction modes](layer3_family_interaction_modes.png)",
        "",
        "| Corpus | Route | Full-MLP fidelity | Family-patch fidelity | Full-MLP cosine | Semantic projection |",
        "|---|---|---:|---:|---:|---:|",
    ]
    display_names = [
        "family_exact", "global_exact", "wrong_family_median", "wrong_family_best",
        "native_lopo", *[f"spectral_{rank}" for rank in ranks],
    ]
    for corpus, summary in summaries.items():
        for name in display_names:
            row = summary[name]
            lines.append(
                f"| {corpus} | {name} | {row['full_mlp_fidelity']:.3f} | "
                f"{row['family_patch_fidelity']:.3f} | {row['full_mlp_cosine']:.3f} | "
                f"{row['semantic_target_projection']:.3f} |"
            )
    lines += [
        "",
        f"Maximum exact family-form reconstruction error: `{maximum_error:.2e}`.",
        "",
        "This test separates two possible reusable objects: a fixed subset of native product addresses and a learned family output subspace whose input computation is represented by quadratic eigenmodes. Neither is a complete semantic circuit unless it also beats global and wrong-family controls and explains a substantial part of the full MLP effect.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(ranks))
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        axis.plot(x, [summaries[corpus][f"spectral_{rank}"]["family_patch_fidelity"] for rank in ranks], marker="o", label="to family patch")
        axis.plot(x, [summaries[corpus][f"spectral_{rank}"]["full_mlp_fidelity"] for rank in ranks], marker="s", label="to full MLP")
        axis.axhline(summaries[corpus]["native_lopo"]["full_mlp_fidelity"], color="grey", linestyle="--", label="native LOPO")
        axis.set_xticks(x, ranks)
        axis.set_xlabel("interaction modes per output axis")
        axis.set_ylabel("downstream fidelity")
        axis.set_title(corpus)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
