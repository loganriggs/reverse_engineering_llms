#!/usr/bin/env python3
"""Cross-family target-derangement null for the Layer-4 residual telescope."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer4_residual_telescope import (
    COMPONENTS,
    analyze_example,
    summarize,
)
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_layer8_semantic_attention_factor_causality import vector_cosine
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer4_residual_telescope_null.json")
OUTPUT = TENSOR_ROOT / "layer4_residual_telescope_null.json"
REPORT = TENSOR_ROOT / "LAYER4_RESIDUAL_TELESCOPE_NULL.md"
FIGURE = TENSOR_ROOT / "layer4_residual_telescope_null.png"


def deranged_targets(families, seed):
    rows = [
        {"family": family, "source": source, "target": target}
        for family, pairs in families.items()
        for source, target in pairs
    ]
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        permutation = rng.permutation(len(rows))
        if all(
            rows[int(donor)]["family"] != row["family"]
            and rows[int(donor)]["target"] != row["source"]
            for row, donor in zip(rows, permutation)
        ):
            result = []
            for row, donor in zip(rows, permutation):
                donor_row = rows[int(donor)]
                result.append({
                    **row,
                    "control_target": donor_row["target"],
                    "control_target_family": donor_row["family"],
                })
            return result
    raise RuntimeError("failed to construct cross-family target derangement")


def absolute_profile(result):
    return [abs(result["components"][name]["local_projection"]) for name in COMPONENTS]


def component_share(result, name):
    profile = np.asarray(absolute_profile(result), dtype=np.float64)
    return float(profile[COMPONENTS.index(name)] / max(profile.sum(), 1e-30))


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    telescope_config = json.loads(Path(config["source_config"]).read_text())
    embedding_config = json.loads(Path(telescope_config["source_config"]).read_text())
    recurrence_config = json.loads(Path(embedding_config["source_config"]).read_text())
    value_config = json.loads(Path(recurrence_config["source_config"]).read_text())
    factor_config = json.loads(Path(value_config["source_config"]).read_text())
    frozen_config = json.loads(Path(factor_config["source_config"]).read_text())
    rank_config = json.loads(Path(frozen_config["source_config"]).read_text())
    causal_config = json.loads(Path(rank_config["source_config"]).read_text())
    atlas_config = json.loads(Path(causal_config["source_config"]).read_text())
    path_config = json.loads(Path(atlas_config["source_config"]).read_text())
    fixed = json.loads(Path(path_config["source_config"]).read_text())
    parent = json.loads(Path(fixed["source_config"]).read_text())
    frozen = json.loads(Path(telescope_config["frozen_results"]).read_text())
    semantic = json.loads(Path(config["semantic_results"]).read_text())
    controls = deranged_targets(parent["families"], config["seed"])
    target_multiset_preserved = sorted(row["target"] for row in controls) == sorted(
        row["control_target"] for row in controls
    )
    cross_family = all(row["family"] != row["control_target_family"] for row in controls)
    if not target_multiset_preserved or not cross_family:
        raise RuntimeError("invalid target derangement")

    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    layer = telescope_config["layer"]
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    edges = [
        tuple(row["edge"])
        for row in frozen["discovered_routes"][str(layer)]["magnitude"][:telescope_config["route_rank"]]
    ]
    results, records_by_corpus = {}, {}
    for corpus_index, (corpus, corpus_config) in enumerate(fixed["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        rng = np.random.default_rng(path_config["seed"] + corpus_index)
        records = []
        for control in controls:
            windows = select_windows(
                rows, control["source"], edit_position, final_position,
                atlas_config["contexts_per_pair"], rng,
            )
            row_index, start = windows[telescope_config["evaluation_context_index"]]
            base_raw = np.asarray(
                rows[row_index, start:start + fixed["sequence_length"]], dtype=np.int64,
            ).copy()
            counterfactual_raw = base_raw.copy()
            counterfactual_raw[edit_position] = control["control_target"]
            base_layers = forward_layers(
                model, torch.as_tensor(base_raw, device="cuda")[None], last_layer=layer,
            )
            counterfactual_layers = forward_layers(
                model, torch.as_tensor(counterfactual_raw, device="cuda")[None], last_layer=layer,
            )
            record = analyze_example(
                model, layer, base_layers, counterfactual_layers, edges,
                final_position, telescope_config["quadrature_points"],
            )
            record.update({
                "family": control["family"],
                "pair": f"{control['source']}->{control['target']}",
                "control_pair": f"{control['source']}->{control['control_target']}",
                "control_target_family": control["control_target_family"],
                "row": int(row_index),
                "source_start": int(start),
            })
            records.append(record)
            print(corpus, control["source"], "->", control["control_target"], flush=True)
        records_by_corpus[corpus] = records
        results[corpus] = summarize(records)

    comparisons = {}
    for corpus in results:
        semantic_result = semantic["results"][corpus]
        control_result = results[corpus]
        semantic_profile = absolute_profile(semantic_result)
        control_profile = absolute_profile(control_result)
        semantic_ratio = abs(semantic_result["groups"]["learned_writes"]) / max(
            abs(semantic_result["groups"]["embedding_all"]), 1e-30,
        )
        control_ratio = abs(control_result["groups"]["learned_writes"]) / max(
            abs(control_result["groups"]["embedding_all"]), 1e-30,
        )
        comparisons[corpus] = {
            "profile_cosine": vector_cosine(semantic_profile, control_profile),
            "semantic_mlp3_share": component_share(semantic_result, "mlp_l3"),
            "control_mlp3_share": component_share(control_result, "mlp_l3"),
            "semantic_top4_share": semantic_result["top4_absolute_local_share"],
            "control_top4_share": control_result["top4_absolute_local_share"],
            "semantic_all_downstream": semantic_result["groups"]["all_components"],
            "control_all_downstream": control_result["groups"]["all_components"],
            "semantic_learned_embedding_ratio": semantic_ratio,
            "control_learned_embedding_ratio": control_ratio,
        }

    hypotheses = {
        "Y1_control_decompositions_remain_exact": {
            "supported": all(
                results[corpus]["maximum_telescope_error"] <= 1e-4
                and results[corpus]["maximum_current_value_integration_error"] <= 1e-4
                and results[corpus]["maximum_route_integration_error"] <= 1e-4
                for corpus in results
            ),
        },
        "Y2_semantic_component_profile_is_specific": {
            "supported": all(row["profile_cosine"] <= 0.90 for row in comparisons.values()),
        },
        "Y3_semantic_mlp3_share_exceeds_control": {
            "supported": all(
                row["semantic_mlp3_share"] - row["control_mlp3_share"] >= 0.10
                for row in comparisons.values()
            ),
        },
        "Y4_semantic_concentration_exceeds_control": {
            "supported": all(
                row["semantic_top4_share"] - row["control_top4_share"] >= 0.05
                for row in comparisons.values()
            ),
        },
        "Y5_semantic_route_is_more_downstream_relevant": {
            "supported": all(
                row["semantic_all_downstream"] - row["control_all_downstream"] >= 0.02
                for row in comparisons.values()
            ),
        },
        "Y6_semantic_learned_write_advantage_is_larger": {
            "supported": all(
                row["semantic_learned_embedding_ratio"]
                >= 1.5 * row["control_learned_embedding_ratio"]
                for row in comparisons.values()
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "target_multiset_preserved": target_multiset_preserved,
        "cross_family": cross_family,
        "target_derangement": controls,
        "hypotheses": hypotheses,
        "comparisons": comparisons,
        "control_results": results,
        "records": records_by_corpus,
    }, indent=2))

    lines = [
        "# Cross-family target null for the Layer-4 residual telescope",
        "",
        "The semantic source contexts, edit positions, and complete target-token multiset are preserved, but target tokens are deranged so every replacement crosses the original semantic families. The exact residual telescope, integrated current-V attribution, frozen route, and downstream group patches are otherwise unchanged. This asks whether the clean MLP3→MLP2 hierarchy is semantic-specific or a generic token-edit transport program.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Residual telescope null](layer4_residual_telescope_null.png)",
        "",
        "| Corpus | Profile cosine | MLP3 share semantic/control | Top-4 semantic/control | All-route downstream semantic/control | Learned/embedding ratio semantic/control |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, row in comparisons.items():
        lines.append(
            f"| {corpus} | {row['profile_cosine']:.3f} | "
            f"{row['semantic_mlp3_share']:.3f} / {row['control_mlp3_share']:.3f} | "
            f"{row['semantic_top4_share']:.3f} / {row['control_top4_share']:.3f} | "
            f"{row['semantic_all_downstream']:.3f} / {row['control_all_downstream']:.3f} | "
            f"{row['semantic_learned_embedding_ratio']:.1f} / {row['control_learned_embedding_ratio']:.1f} |"
        )
    lines += [
        "",
        f"Target multiset preserved: `{target_multiset_preserved}`; every control crosses families: `{cross_family}`.",
        "",
        "A null-like component hierarchy does not erase the earlier semantic recurrence of current-V directions. It means the *location and concentration* of their construction are architectural transport properties; semantic content must be established from the directions carried within those paths, not from the path hierarchy alone.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(COMPONENTS))
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        semantic_profile = absolute_profile(semantic["results"][corpus])
        control_profile = absolute_profile(results[corpus])
        axis.plot(x, semantic_profile, marker="o", label="semantic")
        axis.plot(x, control_profile, marker="x", label="cross-family null")
        axis.set_xticks(x, [name.replace("repeat_embedding_", "emb_") for name in COMPONENTS], rotation=55, ha="right", fontsize=7)
        axis.set_title(corpus)
        axis.set_ylabel("|median local projection|")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
