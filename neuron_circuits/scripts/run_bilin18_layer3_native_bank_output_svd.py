#!/usr/bin/env python3
"""Test whether a reusable Layer-3 native channel bank has a compact output span."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer3_family_interaction_modes import evaluate_writes, output_basis
from run_bilin18_layer3_frozen_channel_transfer import channel_state, recover_configs, top_indices
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_native_bank_output_svd.json")
OUTPUT = TENSOR_ROOT / "layer3_native_bank_output_svd.json"
REPORT = TENSOR_ROOT / "LAYER3_NATIVE_BANK_OUTPUT_SVD.md"
FIGURE = TENSOR_ROOT / "layer3_native_bank_output_svd.png"


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
    rank_config = json.loads(Path(config["source_config"]).read_text())
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
    discovery_states, family_by_pair = {}, {}
    for record in discovery_records:
        base, counterfactual = reconstruct(record)
        state = channel_state(model, layer, base, counterfactual, final_position)
        discovery_states[record["pair"]] = {
            "delta_product": state["delta_product"].cpu(),
            "scores": state["scores"].abs().cpu(),
        }
        family_by_pair[record["pair"]] = record["family"]

    pair_cache = {}
    down = block.mlp.Down.weight.float()
    for pair, family in family_by_pair.items():
        other_pairs = [
            candidate_pair for candidate_pair in discovery_states
            if family_by_pair[candidate_pair] == family and candidate_pair != pair
        ]
        indices = top_indices(torch.stack([
            discovery_states[candidate_pair]["scores"] for candidate_pair in other_pairs
        ]).mean(0), frozen_config["rank"]).cuda()
        bank = down[:, indices]
        weight_u, _, _ = torch.linalg.svd(bank, full_matrices=False)
        activation_writes = []
        for candidate_pair in other_pairs:
            delta = discovery_states[candidate_pair]["delta_product"].cuda()
            selected = delta[..., indices]
            write = F.linear(selected, bank, bias=None)
            activation_writes.append(write[0, edit_position:final_position + 1])
        activation_u = output_basis(activation_writes, maximum_rank)
        generator = torch.Generator(device="cuda").manual_seed(
            config["seed"] + sorted(family_by_pair).index(pair)
        )
        random_u = torch.linalg.qr(torch.randn(
            down.shape[0], maximum_rank, generator=generator, device="cuda",
        ), mode="reduced").Q
        pair_cache[pair] = {
            "indices": indices,
            "weight_top": weight_u[:, :maximum_rank],
            "weight_bottom": weight_u[:, -maximum_rank:],
            "activation": activation_u,
            "random": random_u,
        }

    query_records = [
        record for record in native_saved["semantic_records"]
        if record["context_index"] == frozen_config["evaluation_context_index"]
    ]
    evaluated = []
    basis_types = ("weight_top", "weight_bottom", "activation", "random")
    for record in query_records:
        base, counterfactual = reconstruct(record)
        state = channel_state(model, layer, base, counterfactual, final_position)
        cache = pair_cache[record["pair"]]
        selected = torch.zeros_like(state["delta_product"])
        selected[..., cache["indices"]] = state["delta_product"][..., cache["indices"]]
        native_write = F.linear(selected, down, bias=None)
        writes = {
            "full_mlp": state["target_write"],
            "family_exact": native_write,
        }
        for basis_type in basis_types:
            basis = cache[basis_type]
            for rank in ranks:
                chosen = basis[:, :rank]
                writes[f"{basis_type}_{rank}"] = (native_write @ chosen) @ chosen.T
        effects = evaluate_writes(model, layer, base, counterfactual, writes, final_position)
        retained = ["family_exact"] + [
            f"{basis_type}_{rank}" for basis_type in basis_types for rank in ranks
        ]
        evaluated.append({
            "corpus": record["corpus"],
            "family": record["family"],
            "pair": record["pair"],
            "effects": {name: effects[name] for name in retained},
            "local_native_fidelity": {
                f"{basis_type}_{rank}": 1 - float(
                    (writes[f"{basis_type}_{rank}"] - native_write).square().sum()
                    / native_write.square().sum().clamp_min(1e-30)
                )
                for basis_type in basis_types for rank in ranks
            },
        })
        print("evaluation", record["corpus"], record["family"], record["pair"], flush=True)

    route_names = ["family_exact"] + [
        f"{basis_type}_{rank}" for basis_type in basis_types for rank in ranks
    ]
    summaries = {}
    for corpus in fixed["evaluation"]:
        records = [record for record in evaluated if record["corpus"] == corpus]
        summaries[corpus] = {name: metric_summary(records, name) for name in route_names}
        summaries[corpus]["local_native_fidelity"] = {
            name: median([record["local_native_fidelity"][name] for record in records])
            for name in route_names if name != "family_exact"
        }
    native_fidelity = lambda corpus, name: summaries[corpus][name]["family_patch_fidelity"]
    full_fidelity = lambda corpus, name: summaries[corpus][name]["full_mlp_fidelity"]
    hypotheses = {
        "B1_weight_rank32_compresses_native_bank": {
            "supported": all(native_fidelity(c, "weight_top_32") >= 0.75 for c in summaries),
        },
        "B2_activation_rank32_nearly_recovers_native_bank": {
            "supported": all(native_fidelity(c, "activation_32") >= 0.90 for c in summaries),
        },
        "B3_activation_rank16_beats_weight_rank16": {
            "supported": all(
                native_fidelity(c, "activation_16") - native_fidelity(c, "weight_top_16") >= 0.05
                for c in summaries
            ),
            "gaps": {
                c: native_fidelity(c, "activation_16") - native_fidelity(c, "weight_top_16")
                for c in summaries
            },
        },
        "B4_activation_rank32_preserves_full_mlp_relevance": {
            "supported": all(
                full_fidelity(c, "family_exact") - full_fidelity(c, "activation_32") <= 0.05
                for c in summaries
            ),
            "losses": {
                c: full_fidelity(c, "family_exact") - full_fidelity(c, "activation_32")
                for c in summaries
            },
        },
        "B5_weight_top32_is_specific": {
            "supported": all(
                native_fidelity(c, "weight_top_32") - native_fidelity(c, "weight_bottom_32") >= 0.30
                for c in summaries
            ),
            "gaps": {
                c: native_fidelity(c, "weight_top_32") - native_fidelity(c, "weight_bottom_32")
                for c in summaries
            },
        },
        "B6_activation_code_transfers_ood": {
            "supported": native_fidelity("tinystories", "activation_32")
            >= 0.85 * native_fidelity("fineweb", "activation_32"),
            "retention": native_fidelity("tinystories", "activation_32")
            / max(native_fidelity("fineweb", "activation_32"), 1e-30),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "summaries": summaries,
        "records": evaluated,
    }, indent=2))

    lines = [
        "# Output compression of reusable Layer-3 native channel banks",
        "",
        "For each lexical query, the leave-one-pair-out family channel bank is frozen. Its current native write is projected onto four nested output bases: top and bottom left singular vectors of the bank's Down columns, an activation-weighted basis learned from the other family pairs, and a fixed random basis. Every patch is evaluated through layers 4--17.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Native-bank output SVD](layer3_native_bank_output_svd.png)",
        "",
        "| Corpus | Basis | Rank | Native-patch fidelity | Full-MLP fidelity | Local native-write fidelity |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corpus, summary in summaries.items():
        for basis_type in basis_types:
            for rank in ranks:
                name = f"{basis_type}_{rank}"
                row = summary[name]
                lines.append(
                    f"| {corpus} | {basis_type} | {rank} | {row['family_patch_fidelity']:.3f} | "
                    f"{row['full_mlp_fidelity']:.3f} | {summary['local_native_fidelity'][name]:.3f} |"
                )
        lines.append(
            f"| {corpus} | exact native | 256 channels | 1.000 | {summary['family_exact']['full_mlp_fidelity']:.3f} | 1.000 |"
        )
    lines += [
        "",
        "A compact activation-weighted basis would show that the stable native address bank writes through a reusable low-dimensional output code even though the broader family output PCA failed. A broad result instead means that reuse resides in channel identity and input-side bilinear conditions, not in a short downstream write vocabulary.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        for basis_type in basis_types:
            axis.plot(ranks, [summaries[corpus][f"{basis_type}_{rank}"]["family_patch_fidelity"] for rank in ranks], marker="o", label=basis_type)
        axis.set_xscale("log", base=2)
        axis.set_xticks(ranks, ranks)
        axis.set_xlabel("output rank")
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
