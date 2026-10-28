#!/usr/bin/env python3
"""Localize semantic counterfactual residual carriers across Layer-8 positions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer8_router_causal_validation import exact_kl, repeat_batch, suffix_from_layer
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_layer8_semantic_path_decomposition import boundary_state, fisher_inner
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_semantic_position_paths.json")
OUTPUT = TENSOR_ROOT / "layer8_semantic_position_paths.json"
REPORT = TENSOR_ROOT / "LAYER8_SEMANTIC_POSITION_PATHS.md"
FIGURE = TENSOR_ROOT / "layer8_semantic_position_paths.png"


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


@torch.inference_mode()
def analyze_positions(model, block, base_ids, counterfactual_ids, edit_position, final_position, ranks):
    base = boundary_state(model, block, base_ids, final_position)
    counterfactual = boundary_state(model, block, counterfactual_ids, final_position)
    positions = list(range(edit_position, final_position + 1))
    variants = [base["output"], counterfactual["output"]]
    for position in positions:
        variant = base["output"].clone()
        variant[0, position] = counterfactual["output"][0, position]
        variants.append(variant)
    logits = suffix_from_layer(
        model, 8,
        repeat_batch(base["x0"], len(variants)),
        repeat_batch(base["value_bus"], len(variants)),
        torch.cat(variants), final_position,
    )
    native, target = logits[0], logits[1] - logits[0]
    target_energy = fisher_inner(native, target, target)
    single_effects = logits[2:] - logits[0]
    projections = torch.as_tensor([
        fisher_inner(native, effect, target) / max(target_energy, 1e-30)
        for effect in single_effects
    ])
    order = torch.argsort(projections, descending=True).tolist()
    cumulative_variants = []
    for rank in ranks:
        variant = base["output"].clone()
        selected = [positions[index] for index in order[:rank]]
        variant[0, selected] = counterfactual["output"][0, selected]
        cumulative_variants.append(variant)
    group_names = ("source_only", "intermediate_only", "final_only", "source_final")
    group_positions = (
        [edit_position],
        list(range(edit_position + 1, final_position)),
        [final_position],
        [edit_position, final_position],
    )
    group_variants = []
    for selected in group_positions:
        variant = base["output"].clone()
        variant[0, selected] = counterfactual["output"][0, selected]
        group_variants.append(variant)
    extra_logits = suffix_from_layer(
        model, 8,
        repeat_batch(base["x0"], len(cumulative_variants) + len(group_variants)),
        repeat_batch(base["value_bus"], len(cumulative_variants) + len(group_variants)),
        torch.cat(cumulative_variants + group_variants), final_position,
    )

    def metrics(effect_logits):
        effect = effect_logits - native
        fidelity, cosine, _ = fisher_fidelity(native, target, effect)
        return {
            "fidelity": fidelity,
            "cosine": cosine,
            "kl_to_full": exact_kl(native, effect_logits) / max(exact_kl(native, logits[1]), 1e-30),
            "target_projection": fisher_inner(native, effect, target) / max(target_energy, 1e-30),
        }

    cumulative = {
        str(rank): metrics(extra_logits[index]) | {
            "positions": [positions[item] for item in order[:rank]],
        }
        for index, rank in enumerate(ranks)
    }
    groups = {
        name: metrics(extra_logits[len(ranks) + index])
        for index, name in enumerate(group_names)
    }
    linear_sum = single_effects.sum(0)
    linear_fidelity, linear_cosine, _ = fisher_fidelity(native, target, linear_sum)
    prefix_difference = (
        counterfactual["output"][0, :edit_position] - base["output"][0, :edit_position]
    ).norm() / base["output"][0, :edit_position].norm().clamp_min(1e-30)
    return {
        "prefix_relative_difference": float(prefix_difference),
        "individual_positions": [
            {
                "position": position,
                "relative_position": position - edit_position,
                "target_projection": float(projections[index]),
                **metrics(logits[2 + index]),
            }
            for index, position in enumerate(positions)
        ],
        "ranked_positions": [positions[index] for index in order],
        "cumulative": cumulative,
        "groups": groups,
        "linear_sum_fidelity": linear_fidelity,
        "linear_sum_cosine": linear_cosine,
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    path_config = json.loads(Path(config["source_config"]).read_text())
    fixed = json.loads(Path(path_config["source_config"]).read_text())
    parent = json.loads(Path(fixed["source_config"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[8]
    results = {}
    max_prefix_difference = 0.0

    for corpus_index, (corpus, corpus_config) in enumerate(fixed["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        rng = np.random.default_rng(path_config["seed"] + corpus_index)
        records = []
        for family, pairs in parent["families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows, source_token, fixed["edit_position"], fixed["causal_position"],
                    config["contexts_per_pair"], rng,
                )
                for row_index, start in windows:
                    base_raw = np.asarray(rows[row_index, start:start + fixed["sequence_length"]], dtype=np.int64).copy()
                    counterfactual_raw = base_raw.copy()
                    counterfactual_raw[fixed["edit_position"]] = target_token
                    record = analyze_positions(
                        model, block,
                        torch.as_tensor(base_raw, device="cuda")[None],
                        torch.as_tensor(counterfactual_raw, device="cuda")[None],
                        fixed["edit_position"], fixed["causal_position"], config["ranks"],
                    )
                    record.update({"family": family, "pair": f"{source_token}->{target_token}", "row": row_index})
                    max_prefix_difference = max(max_prefix_difference, record["prefix_relative_difference"])
                    records.append(record)
                print(corpus, family, source_token, "->", target_token, flush=True)
        cumulative = {
            str(rank): {
                key: median([record["cumulative"][str(rank)][key] for record in records])
                for key in ("fidelity", "cosine", "kl_to_full", "target_projection")
            }
            for rank in config["ranks"]
        }
        groups = {
            group: {
                key: median([record["groups"][group][key] for record in records])
                for key in ("fidelity", "cosine", "kl_to_full", "target_projection")
            }
            for group in records[0]["groups"]
        }
        top_position_frequency = {
            str(relative): sum(record["ranked_positions"][0] - fixed["edit_position"] == relative for record in records) / len(records)
            for relative in range(17)
        }
        results[corpus] = {
            "cumulative": cumulative,
            "groups": groups,
            "linear_sum_fidelity": median([record["linear_sum_fidelity"] for record in records]),
            "linear_sum_cosine": median([record["linear_sum_cosine"] for record in records]),
            "top_position_frequency": top_position_frequency,
            "records": records,
        }

    hypotheses = {
        "J1_causal_prefix_is_unchanged": {"supported": max_prefix_difference <= 1e-6},
        "J2_final_position_alone_is_partial": {"supported": all(results[c]["groups"]["final_only"]["fidelity"] < 0.60 for c in results)},
        "J3_intermediate_positions_carry_signal": {"supported": all(results[c]["groups"]["intermediate_only"]["fidelity"] >= 0.30 for c in results)},
        "J4_top4_positions_form_a_compact_route": {"supported": all(results[c]["cumulative"]["4"]["fidelity"] >= 0.80 for c in results)},
        "J5_top8_positions_are_near_sufficient": {"supported": all(results[c]["cumulative"]["8"]["fidelity"] >= 0.90 for c in results)},
        "J6_position_effects_are_nearly_additive": {"supported": all(results[c]["linear_sum_fidelity"] >= 0.90 for c in results)},
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "maximum_prefix_relative_difference": max_prefix_difference,
        "results": results,
    }, indent=2))

    lines = [
        "# Position localization of Layer-8 semantic residual carriers",
        "",
        "The whole residual sequence carried nearly all of the fixed semantic-counterfactual effect. This experiment patches each of the seventeen causally affected Layer-8 positions individually, ranks them by signed Fisher projection onto the whole-residual target, evaluates exact cumulative subsets, and compares source, intermediate, and final position groups.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Semantic position paths](layer8_semantic_position_paths.png)",
        "",
        "| Corpus | Positions | Fidelity / cosine | KL/full | Target projection |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for rank in config["ranks"]:
            row = result["cumulative"][str(rank)]
            lines.append(f"| {corpus} | top {rank} | {row['fidelity']:.3f} / {row['cosine']:.3f} | {row['kl_to_full']:.3f} | {row['target_projection']:.3f} |")
    lines += [
        "",
        "| Corpus | Group | Fidelity / cosine | KL/full | Target projection |",
        "|---|---|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for group, row in result["groups"].items():
            lines.append(f"| {corpus} | {group} | {row['fidelity']:.3f} / {row['cosine']:.3f} | {row['kl_to_full']:.3f} | {row['target_projection']:.3f} |")
    lines += [
        "",
        "| Corpus | Linear sum fidelity/cosine | Most frequent individually top relative position |",
        "|---|---:|---:|",
    ]
    for corpus, result in results.items():
        top = max(result["top_position_frequency"], key=result["top_position_frequency"].get)
        lines.append(f"| {corpus} | {result['linear_sum_fidelity']:.3f} / {result['linear_sum_cosine']:.3f} | +{top} ({result['top_position_frequency'][top]:.3f}) |")
    lines += [
        "",
        f"Maximum relative residual difference before the edited token: `{max_prefix_difference:.2e}`.",
        "",
        "Ranking uses the target effect and is therefore an oracle localization, not an executable router. A compact position subset would identify where to trace next, not establish a semantic feature basis inside those positions.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ranks = config["ranks"]
    for corpus, result in results.items():
        axes[0].plot(ranks, [result["cumulative"][str(rank)]["fidelity"] for rank in ranks], marker="o", label=corpus)
        frequencies = [result["top_position_frequency"][str(relative)] for relative in range(17)]
        axes[1].plot(range(17), frequencies, marker="o", label=corpus)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("top projected positions")
    axes[0].set_ylabel("residual-target fidelity")
    axes[0].set_title("Cumulative position route")
    axes[1].set_xlabel("position relative to edit")
    axes[1].set_ylabel("frequency ranked first")
    axes[1].set_title("Where the strongest carrier sits")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
