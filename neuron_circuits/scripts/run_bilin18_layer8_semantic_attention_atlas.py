#!/usr/bin/env python3
"""Exact QK1/QK2/value atlas for semantic broadcast through layers 0--8."""

from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_semantic_attention_atlas.json")
OUTPUT = TENSOR_ROOT / "layer8_semantic_attention_atlas.json"
REPORT = TENSOR_ROOT / "LAYER8_SEMANTIC_ATTENTION_ATLAS.md"
FIGURE = TENSOR_ROOT / "layer8_semantic_attention_atlas.png"
COMPONENTS = ("qk1", "qk2", "value")


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


@torch.inference_mode()
def forward_layers(model, ids, last_layer=8):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x, value_bus = x0, None
    records = []
    for layer, block in enumerate(model.blocks[:last_layer + 1]):
        bus_before = value_bus
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        normalized = F.rms_norm(mixed, (dimension,))
        details = attention_details(block.attn, normalized, value_bus, model.config, model.tt)
        value_bus = details["value_bus"]
        pre = mixed + details["write"]
        mlp_input = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(mlp_input) * block.mlp.Right(mlp_input)
        mlp_write = block.mlp.Down(product) + block.mlp.Down_bias
        x = pre + mlp_write
        records.append({
            "x0": x0,
            "value_bus_before": bus_before,
            "value_bus": value_bus,
            "mixed": mixed,
            "details": details,
            "pre": pre,
            "attention_write": details["write"],
            "mlp_write": mlp_write,
            "output": x,
        })
    return records


def three_factor_shapley(a0, a1, b0, b1, v0, v1):
    da, db, dv = a1 - a0, b1 - b0, v1 - v0
    v0 = v0[:, None]
    v1 = v1[:, None]
    dv = dv[:, None]
    qk1 = da[..., None] * (
        (b0[..., None] * v0) / 3
        + (b1[..., None] * v0) / 6
        + (b0[..., None] * v1) / 6
        + (b1[..., None] * v1) / 3
    )
    qk2 = db[..., None] * (
        (a0[..., None] * v0) / 3
        + (a1[..., None] * v0) / 6
        + (a0[..., None] * v1) / 6
        + (a1[..., None] * v1) / 3
    )
    value = dv * (
        (a0 * b0)[..., None] / 3
        + (a1 * b0)[..., None] / 6
        + (a0 * b1)[..., None] / 6
        + (a1 * b1)[..., None] / 3
    )
    return {"qk1": qk1, "qk2": qk2, "value": value}


@torch.inference_mode()
def layer_edge_atlas(block, base, counterfactual, edit_position, final_position, top_edges):
    destination = slice(edit_position, final_position + 1)
    sources = slice(0, final_position + 1)
    d0, d1 = base["details"], counterfactual["details"]
    a0 = d0["score1"][0, :, destination, sources].float()
    a1 = d1["score1"][0, :, destination, sources].float()
    b0 = d0["score2"][0, :, destination, sources].float()
    b1 = d1["score2"][0, :, destination, sources].float()
    v0 = d0["mixed_value"][0, sources].float().permute(1, 0, 2)
    v1 = d1["mixed_value"][0, sources].float().permute(1, 0, 2)
    destinations = torch.arange(edit_position, final_position + 1, device=a0.device)
    source_indices = torch.arange(0, final_position + 1, device=a0.device)
    causal = source_indices[None, :] <= destinations[:, None]
    mask = causal[None, :, :, None]
    edge_delta = (a1 * b1)[..., None] * v1[:, None] - (a0 * b0)[..., None] * v0[:, None]
    edge_delta = edge_delta * mask
    components = {name: value * mask for name, value in three_factor_shapley(a0, a1, b0, b1, v0, v1).items()}
    target = (
        counterfactual["attention_write"][0, destination].float()
        - base["attention_write"][0, destination].float()
    )
    denominator = target.square().sum().clamp_min(1e-30)
    heads = model_heads = block.attn.n_head
    head_dim = block.attn.head_dim
    reads = []
    for head in range(heads):
        weight = block.attn.c_proj.weight.float()[:, head * head_dim:(head + 1) * head_dim]
        reads.append(target @ weight)
    read = torch.stack(reads)

    def projected(values):
        return (values * read[:, :, None]).sum(-1) / denominator

    edge_projection = projected(edge_delta)
    component_projection = {name: float(projected(value).sum()) for name, value in components.items()}
    head_projection = edge_projection.sum((1, 2))
    absolute_head = head_projection.abs()
    top_head = int(absolute_head.argmax())
    top3_share = float(torch.topk(absolute_head, min(3, len(absolute_head))).values.sum() / absolute_head.sum().clamp_min(1e-30))
    source_projection = {
        "before_edit": float(edge_projection[:, :, :edit_position].sum()),
        "edit": float(edge_projection[:, :, edit_position].sum()),
        "after_edit": float(edge_projection[:, :, edit_position + 1:].sum()),
    }
    self_projection = float(sum(
        edge_projection[:, relative, edit_position + relative].sum()
        for relative in range(final_position - edit_position + 1)
    ))
    flattened = edge_projection.abs().flatten()
    selected = torch.topk(flattened, min(top_edges, len(flattened))).indices
    width_source = final_position + 1
    width_destination = final_position - edit_position + 1
    edge_rows = []
    for index in selected.tolist():
        head = index // (width_destination * width_source)
        remainder = index % (width_destination * width_source)
        destination_relative = remainder // width_source
        source = remainder % width_source
        edge_rows.append({
            "head": head,
            "destination_relative": destination_relative,
            "source_relative": source - edit_position,
            "projection": float(edge_projection[head, destination_relative, source]),
        })
    predicted = F.linear(
        edge_delta.sum(2).transpose(0, 1).reshape(width_destination, -1),
        block.attn.c_proj.weight.float(),
    )
    component_sum = sum(components.values())
    shapley_predicted = F.linear(
        component_sum.sum(2).transpose(0, 1).reshape(width_destination, -1),
        block.attn.c_proj.weight.float(),
    )
    base_reference = F.linear(
        ((a0 * b0)[..., None] * v0[:, None] * mask).sum(2).transpose(0, 1).reshape(width_destination, -1),
        block.attn.c_proj.weight.float(),
    )
    counterfactual_reference = F.linear(
        ((a1 * b1)[..., None] * v1[:, None] * mask).sum(2).transpose(0, 1).reshape(width_destination, -1),
        block.attn.c_proj.weight.float(),
    )
    algebraic_target = counterfactual_reference - base_reference
    # Validate the algebraic identity in float64. The native attention write is
    # still float32 and is reported separately because changing the reduction
    # order across sources produces ~1e-5 relative roundoff in this checkpoint.
    a0d, a1d, b0d, b1d = a0.double(), a1.double(), b0.double(), b1.double()
    v0d, v1d, maskd = v0.double(), v1.double(), mask.double()
    edged = ((a1d * b1d)[..., None] * v1d[:, None] - (a0d * b0d)[..., None] * v0d[:, None]) * maskd
    componentsd = {
        name: value * maskd
        for name, value in three_factor_shapley(a0d, a1d, b0d, b1d, v0d, v1d).items()
    }
    weightd = block.attn.c_proj.weight.double()
    predictedd = F.linear(edged.sum(2).transpose(0, 1).reshape(width_destination, -1), weightd)
    shapley_predictedd = F.linear(
        sum(componentsd.values()).sum(2).transpose(0, 1).reshape(width_destination, -1), weightd,
    )
    base_referenced = F.linear(
        ((a0d * b0d)[..., None] * v0d[:, None] * maskd).sum(2).transpose(0, 1).reshape(width_destination, -1), weightd,
    )
    counterfactual_referenced = F.linear(
        ((a1d * b1d)[..., None] * v1d[:, None] * maskd).sum(2).transpose(0, 1).reshape(width_destination, -1), weightd,
    )
    algebraic_targetd = counterfactual_referenced - base_referenced
    edge_error = float((predictedd - algebraic_targetd).norm() / algebraic_targetd.norm().clamp_min(1e-30))
    shapley_error = float((shapley_predictedd - algebraic_targetd).norm() / algebraic_targetd.norm().clamp_min(1e-30))
    native_order_error = float((algebraic_target - target).norm() / target.norm().clamp_min(1e-30))
    output_delta = counterfactual["output"][0] - base["output"][0]
    attention_delta = counterfactual["attention_write"][0] - base["attention_write"][0]
    mlp_delta = counterfactual["mlp_write"][0] - base["mlp_write"][0]
    affected_output = output_delta[edit_position:final_position + 1]
    output_energy = affected_output.square().sum().clamp_min(1e-30)
    position_energy = {
        "source": float(output_delta[edit_position].square().sum() / output_energy),
        "intermediate": float(output_delta[edit_position + 1:final_position].square().sum() / output_energy),
        "final": float(output_delta[final_position].square().sum() / output_energy),
    }
    return {
        "attention_delta_energy": float(target.square().sum()),
        "residual_delta_energy": float(affected_output.square().sum()),
        "attention_to_residual_delta_norm": float(attention_delta[edit_position:].norm() / affected_output.norm().clamp_min(1e-30)),
        "mlp_to_residual_delta_norm": float(mlp_delta[edit_position:].norm() / affected_output.norm().clamp_min(1e-30)),
        "component_projection": component_projection,
        "source_projection": source_projection,
        "self_projection": self_projection,
        "head_projection": [float(value) for value in head_projection],
        "top_head": top_head,
        "top3_absolute_head_share": top3_share,
        "position_energy": position_energy,
        "top_edges": edge_rows,
        "edge_recomposition_error": edge_error,
        "shapley_recomposition_error": shapley_error,
        "native_accumulation_order_error": native_order_error,
    }


@torch.inference_mode()
def main():
    global model_heads
    config = json.loads(CONFIG.read_text())
    path_config = json.loads(Path(config["source_config"]).read_text())
    fixed = json.loads(Path(path_config["source_config"]).read_text())
    parent = json.loads(Path(fixed["source_config"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    results = {}
    maximum_error = 0.0
    maximum_native_order_error = 0.0

    for corpus_index, (corpus, corpus_config) in enumerate(fixed["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        rng = np.random.default_rng(path_config["seed"] + corpus_index)
        records = []
        for family, pairs in parent["families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows, source_token, edit_position, final_position,
                    config["contexts_per_pair"], rng,
                )
                for row_index, start in windows:
                    base_raw = np.asarray(rows[row_index, start:start + fixed["sequence_length"]], dtype=np.int64).copy()
                    counterfactual_raw = base_raw.copy()
                    counterfactual_raw[edit_position] = target_token
                    base_layers = forward_layers(model, torch.as_tensor(base_raw, device="cuda")[None])
                    counterfactual_layers = forward_layers(model, torch.as_tensor(counterfactual_raw, device="cuda")[None])
                    layer_rows = []
                    for layer in range(9):
                        row = layer_edge_atlas(
                            model.blocks[layer], base_layers[layer], counterfactual_layers[layer],
                            edit_position, final_position, config["top_edges_per_layer"],
                        )
                        maximum_error = max(maximum_error, row["edge_recomposition_error"], row["shapley_recomposition_error"])
                        maximum_native_order_error = max(maximum_native_order_error, row["native_accumulation_order_error"])
                        layer_rows.append(row)
                    records.append({"family": family, "pair": f"{source_token}->{target_token}", "row": row_index, "layers": layer_rows})
                print(corpus, family, source_token, "->", target_token, flush=True)
        layer_summary = []
        for layer in range(9):
            rows_for_layer = [record["layers"][layer] for record in records]
            top_counts = collections.Counter(row["top_head"] for row in rows_for_layer)
            most_head, most_count = top_counts.most_common(1)[0]
            layer_summary.append({
                "layer": layer,
                "component_projection": {name: median([row["component_projection"][name] for row in rows_for_layer]) for name in COMPONENTS},
                "source_projection": {name: median([row["source_projection"][name] for row in rows_for_layer]) for name in ("before_edit", "edit", "after_edit")},
                "self_projection": median([row["self_projection"] for row in rows_for_layer]),
                "top3_absolute_head_share": median([row["top3_absolute_head_share"] for row in rows_for_layer]),
                "most_common_top_head": most_head,
                "most_common_top_head_frequency": most_count / len(rows_for_layer),
                "position_energy": {name: median([row["position_energy"][name] for row in rows_for_layer]) for name in ("source", "intermediate", "final")},
                "attention_to_residual_delta_norm": median([row["attention_to_residual_delta_norm"] for row in rows_for_layer]),
                "mlp_to_residual_delta_norm": median([row["mlp_to_residual_delta_norm"] for row in rows_for_layer]),
            })
        results[corpus] = {"layers": layer_summary, "records": records}

    hypotheses = {
        "K1_edge_decomposition_is_exact": {"supported": maximum_error <= 1e-5},
        "K2_direct_source_edges_are_nontrivial": {"supported": all(max(abs(row["source_projection"]["edit"]) for row in results[c]["layers"]) >= 0.10 for c in results)},
        "K3_value_changes_drive_an_early_layer": {"supported": all(max(results[c]["layers"][layer]["component_projection"]["value"] for layer in range(4)) >= 0.50 for c in results)},
        "K4_both_qk_forms_are_used": {"supported": all(
            max(abs(row["component_projection"][name]) for row in results[c]["layers"]) >= 0.10
            for c in results for name in ("qk1", "qk2")
        )},
        "K5_a_recurrent_head_dominates_some_layer": {"supported": all(max(row["most_common_top_head_frequency"] for row in results[c]["layers"]) >= 0.30 for c in results)},
        "K6_three_heads_can_explain_half_a_layer": {"supported": all(max(row["top3_absolute_head_share"] for row in results[c]["layers"]) >= 0.50 for c in results)},
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "maximum_recomposition_error": maximum_error,
        "maximum_native_accumulation_order_error": maximum_native_order_error,
        "results": results,
    }, indent=2))

    lines = [
        "# Exact tensor-attention atlas for semantic broadcasting",
        "",
        "For every changed attention edge in layers 0–8, the exact squared-bilinear contribution `(QK1)(QK2)V` is compared between base and counterfactual contexts. A three-factor Shapley identity assigns the changed write to QK1 routing, QK2 routing, and value/OV changes. Edge writes are projected onto the actual changed attention output; their signed projections recompose it exactly.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Semantic attention atlas](layer8_semantic_attention_atlas.png)",
        "",
        "| Corpus | Layer | QK1 / QK2 / value projection | Before/edit/after source projection | Top-3 head share | Most common top head (freq.) | Residual energy source/intermediate/final |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for row in result["layers"]:
            component, source, energy = row["component_projection"], row["source_projection"], row["position_energy"]
            lines.append(
                f"| {corpus} | {row['layer']} | {component['qk1']:.3f} / {component['qk2']:.3f} / {component['value']:.3f} | "
                f"{source['before_edit']:.3f} / {source['edit']:.3f} / {source['after_edit']:.3f} | "
                f"{row['top3_absolute_head_share']:.3f} | {row['most_common_top_head']} ({row['most_common_top_head_frequency']:.3f}) | "
                f"{energy['source']:.3f} / {energy['intermediate']:.3f} / {energy['final']:.3f} |"
            )
    lines += [
        "",
        f"Maximum algebraic edge/Shapley recomposition error: `{maximum_error:.2e}`. Maximum discrepancy to the native float32 contraction from accumulation order: `{maximum_native_order_error:.2e}`.",
        "",
        "Projection is local to each layer's changed attention write. It identifies how an attention delta is constructed, not how much that layer causes the final language-model output; finite downstream edge patching is the next gate.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for corpus, result in results.items():
        layers = range(9)
        axes[0].plot(layers, [row["component_projection"]["qk1"] for row in result["layers"]], marker="o", label=f"{corpus}/QK1")
        axes[0].plot(layers, [row["component_projection"]["qk2"] for row in result["layers"]], marker="s", label=f"{corpus}/QK2")
        axes[0].plot(layers, [row["component_projection"]["value"] for row in result["layers"]], marker="^", label=f"{corpus}/value")
        axes[1].plot(layers, [row["source_projection"]["edit"] for row in result["layers"]], marker="o", label=f"{corpus}/edited source")
        axes[1].plot(layers, [row["top3_absolute_head_share"] for row in result["layers"]], marker="s", label=f"{corpus}/top-3 heads")
    axes[0].set_title("Exact tensor-factor allocation")
    axes[0].set_ylabel("signed target projection")
    axes[1].set_title("Source and head concentration")
    for axis in axes:
        axis.set_xlabel("layer")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
