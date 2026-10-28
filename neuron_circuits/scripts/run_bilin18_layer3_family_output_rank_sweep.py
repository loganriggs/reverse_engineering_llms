#!/usr/bin/env python3
"""Sweep the size and specificity of leave-one-pair-out family output codes."""

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


CONFIG = Path("configs/bilin18_layer3_family_output_rank_sweep.json")
OUTPUT = TENSOR_ROOT / "layer3_family_output_rank_sweep.json"
REPORT = TENSOR_ROOT / "LAYER3_FAMILY_OUTPUT_RANK_SWEEP.md"
FIGURE = TENSOR_ROOT / "layer3_family_output_rank_sweep.png"


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
    form_config = json.loads(Path(config["source_config"]).read_text())
    lopo_config = json.loads(Path(form_config["source_config"]).read_text())
    frozen_config = json.loads(Path(lopo_config["source_config"]).read_text())
    native_saved = json.loads(Path(form_config["native_results"]).read_text())
    fixed, parent = recover_configs(frozen_config)
    ranks, maximum_rank = config["output_ranks"], max(config["output_ranks"])
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
    discovery_writes, discovery_scores, family_by_pair = {}, {}, {}
    for record in discovery_records:
        base, counterfactual = reconstruct(record)
        state = channel_state(model, layer, base, counterfactual, final_position)
        pair = record["pair"]
        discovery_writes[pair] = state["target_write"][0, edit_position:final_position + 1]
        discovery_scores[pair] = state["scores"].abs()
        family_by_pair[pair] = record["family"]
    family_names = list(parent["families"])
    inclusive_bases = {
        family: output_basis([
            write for pair, write in discovery_writes.items() if family_by_pair[pair] == family
        ], maximum_rank)
        for family in family_names
    }
    pair_cache = {}
    for pair, family in family_by_pair.items():
        family_writes = [
            write for candidate_pair, write in discovery_writes.items()
            if family_by_pair[candidate_pair] == family and candidate_pair != pair
        ]
        global_writes = [
            write for candidate_pair, write in discovery_writes.items()
            if candidate_pair != pair
        ]
        pair_cache[pair] = {
            "family": output_basis(family_writes, maximum_rank),
            "global": output_basis(global_writes, maximum_rank),
            "native_lopo": top_indices(torch.stack([
                score for candidate_pair, score in discovery_scores.items()
                if family_by_pair[candidate_pair] == family and candidate_pair != pair
            ]).mean(0), frozen_config["rank"]),
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
        selected[..., cache["native_lopo"]] = state["delta_product"][..., cache["native_lopo"]]
        writes = {
            "full_mlp": state["target_write"],
            "family_exact": (state["target_write"] @ cache["family"][:, :1]) @ cache["family"][:, :1].T,
            "native_lopo": F.linear(selected, block.mlp.Down.weight.float(), bias=None),
        }
        wrong_names = {}
        for rank in ranks:
            family_basis = cache["family"][:, :rank]
            global_basis = cache["global"][:, :rank]
            writes[f"family_{rank}"] = (state["target_write"] @ family_basis) @ family_basis.T
            writes[f"global_{rank}"] = (state["target_write"] @ global_basis) @ global_basis.T
            wrong_names[rank] = []
            for index, wrong_family in enumerate(name for name in family_names if name != family):
                name = f"wrong_{rank}_{index}"
                wrong_basis = inclusive_bases[wrong_family][:, :rank]
                writes[name] = (state["target_write"] @ wrong_basis) @ wrong_basis.T
                wrong_names[rank].append(name)
        effects = evaluate_writes(model, layer, base, counterfactual, writes, final_position)
        for rank in ranks:
            wrong_rows = [effects[name] for name in wrong_names[rank]]
            effects[f"wrong_best_{rank}"] = dict(max(
                wrong_rows, key=lambda row: row["full_mlp_fidelity"],
            ))
            effects[f"wrong_median_{rank}"] = {
                metric: float(np.median([row[metric] for row in wrong_rows]))
                for metric in wrong_rows[0]
            }
        retained = ["native_lopo"] + [
            name for rank in ranks for name in (
                f"family_{rank}", f"global_{rank}", f"wrong_best_{rank}", f"wrong_median_{rank}",
            )
        ]
        evaluated.append({
            "corpus": record["corpus"],
            "family": family,
            "pair": pair,
            "effects": {name: effects[name] for name in retained},
            "local_family_fidelity": {
                str(rank): 1 - float(
                    (writes[f"family_{rank}"] - state["target_write"]).square().sum()
                    / state["target_write"].square().sum().clamp_min(1e-30)
                )
                for rank in ranks
            },
        })
        print("evaluation", record["corpus"], family, pair, flush=True)

    summaries = {}
    route_names = ["native_lopo"] + [
        name for rank in ranks for name in (
            f"family_{rank}", f"global_{rank}", f"wrong_best_{rank}", f"wrong_median_{rank}",
        )
    ]
    for corpus in fixed["evaluation"]:
        records = [record for record in evaluated if record["corpus"] == corpus]
        summaries[corpus] = {name: metric_summary(records, name) for name in route_names}
        summaries[corpus]["local_family_fidelity"] = {
            str(rank): median([record["local_family_fidelity"][str(rank)] for record in records])
            for rank in ranks
        }
    fidelity = lambda corpus, name: summaries[corpus][name]["full_mlp_fidelity"]
    hypotheses = {
        "O1_family_rank32_is_substantial": {
            "supported": all(fidelity(c, "family_32") >= 0.35 for c in summaries),
        },
        "O2_family_rank32_matches_native_addresses": {
            "supported": all(fidelity(c, "family_32") >= fidelity(c, "native_lopo") for c in summaries),
            "gaps": {c: fidelity(c, "family_32") - fidelity(c, "native_lopo") for c in summaries},
        },
        "O3_family_rank16_is_semantically_specific": {
            "supported": all(fidelity(c, "family_16") > fidelity(c, "wrong_best_16") for c in summaries),
            "gaps": {c: fidelity(c, "family_16") - fidelity(c, "wrong_best_16") for c in summaries},
        },
        "O4_family_rank32_is_semantically_specific": {
            "supported": all(fidelity(c, "family_32") > fidelity(c, "wrong_best_32") for c in summaries),
            "gaps": {c: fidelity(c, "family_32") - fidelity(c, "wrong_best_32") for c in summaries},
        },
        "O5_family_rank16_beats_global": {
            "supported": all(fidelity(c, "family_16") - fidelity(c, "global_16") >= 0.03 for c in summaries),
            "gaps": {c: fidelity(c, "family_16") - fidelity(c, "global_16") for c in summaries},
        },
        "O6_family_rank32_transfers_ood": {
            "supported": fidelity("tinystories", "family_32") >= 0.75 * fidelity("fineweb", "family_32"),
            "retention": fidelity("tinystories", "family_32") / max(fidelity("fineweb", "family_32"), 1e-30),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "summaries": summaries,
        "records": evaluated,
    }, indent=2))

    lines = [
        "# Layer-3 family output-rank sweep",
        "",
        "Leave-one-pair-out family and global output subspaces are swept from rank 1 to 32. All other family bases are evaluated at each rank, and the strongest wrong family is retained as an adversarial specificity control. The native family-channel patch is unchanged.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Family output-rank sweep](layer3_family_output_rank_sweep.png)",
        "",
        "| Corpus | Rank | Family | Global | Best wrong family | Family local fidelity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, summary in summaries.items():
        for rank in ranks:
            lines.append(
                f"| {corpus} | {rank} | {summary[f'family_{rank}']['full_mlp_fidelity']:.3f} | "
                f"{summary[f'global_{rank}']['full_mlp_fidelity']:.3f} | "
                f"{summary[f'wrong_best_{rank}']['full_mlp_fidelity']:.3f} | "
                f"{summary['local_family_fidelity'][str(rank)]:.3f} |"
            )
        lines.append(
            f"| {corpus} | native-256 | {summary['native_lopo']['full_mlp_fidelity']:.3f} | — | — | — |"
        )
    lines += [
        "",
        "Increasing output rank asks whether the clean rank-4 semantic core was merely too narrow. A family basis that becomes behaviorally strong while retaining advantages over global and wrong-family bases is a reusable distributed code; loss of those advantages means the additional dimensions are generic edit transport rather than family-specific computation.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        axis.plot(ranks, [summaries[corpus][f"family_{rank}"]["full_mlp_fidelity"] for rank in ranks], marker="o", label="family")
        axis.plot(ranks, [summaries[corpus][f"global_{rank}"]["full_mlp_fidelity"] for rank in ranks], marker="s", label="global")
        axis.plot(ranks, [summaries[corpus][f"wrong_best_{rank}"]["full_mlp_fidelity"] for rank in ranks], marker="x", label="best wrong")
        axis.axhline(summaries[corpus]["native_lopo"]["full_mlp_fidelity"], color="grey", linestyle="--", label="native LOPO")
        axis.set_xscale("log", base=2)
        axis.set_xticks(ranks, ranks)
        axis.set_xlabel("output-subspace rank")
        axis.set_ylabel("fidelity to full MLP3 patch")
        axis.set_title(corpus)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
