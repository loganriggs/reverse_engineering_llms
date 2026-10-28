#!/usr/bin/env python3
"""Decompose the Layer-8 write that constructs the L9H8 boundary style value."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_upstream_trace import integrated_value_components
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_boundary_style_layer8_builder.json")
OUTPUT = TENSOR_ROOT / "bilin18_boundary_style_layer8_builder.json"
REPORT = TENSOR_ROOT / "BILIN18_BOUNDARY_STYLE_LAYER8_BUILDER.md"
FIGURE = TENSOR_ROOT / "bilin18_boundary_style_layer8_builder.png"
RELATIONS = ("before_cue", "cue", "between", "boundary_self")


def relation_masks(length, cue, boundary):
    positions = torch.arange(length, device=cue.device)[None]
    return {
        "before_cue": positions < cue[:, None],
        "cue": positions == cue[:, None],
        "between": (positions > cue[:, None]) & (positions < boundary[:, None]),
        "boundary_self": positions == boundary[:, None],
    }


def empty_stats(names): return {name: {"sum": 0.0, "wins": 0, "count": 0} for name in names}


@torch.inference_mode()
def collect(model, raw, records, contrasts, config):
    target_layer, builder_layer, target_head, destination = 9, 8, 8, 127
    dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads
    projection9 = model.blocks[target_layer].attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)[target_head]
    component_names = [f"h{head}_{relation}" for head in range(heads) for relation in RELATIONS]
    variant_names = ["attention_full"] + [f"relation_{relation}" for relation in RELATIONS] + [f"head_{head}" for head in range(heads)] + [f"head_{head}_cue" for head in range(heads)]
    endpoint = empty_stats(variant_names); local_inner = {name: 0.0 for name in variant_names}
    local_energy = 0.0; edge_error = value_error = 0.0
    for start in range(0, len(records), config["batch_size"]):
        batch = records[start:start + config["batch_size"]]; batch_size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch]); swapped = original.copy()
        for index, row in enumerate(batch): swapped[index, row[3]] = row[5]
        ids = torch.as_tensor(np.concatenate([original, swapped]), device="cuda")
        boundary = torch.as_tensor([row[1] for row in batch], device="cuda")
        cue = torch.as_tensor([row[3] for row in batch], device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        raw_components = None; builder_delta = None
        for layer, block in enumerate(model.blocks[:target_layer + 1]):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            pre = mixed + details["write"]; state = F.rms_norm(pre, (dimension,))
            mlp_write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
            if layer == builder_layer:
                length = ids.shape[1]; paired = torch.arange(batch_size, device="cuda")
                pattern = details["score1"] * details["score2"]
                query_index = boundary[:, None, None, None].expand(batch_size, heads, 1, length)
                p0 = pattern[:batch_size].gather(2, query_index).squeeze(2).float()
                p1 = pattern[batch_size:].gather(2, query_index).squeeze(2).float()
                v0 = details["mixed_value"][:batch_size].float().permute(0, 2, 1, 3)
                v1 = details["mixed_value"][batch_size:].float().permute(0, 2, 1, 3)
                projection8 = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
                edges0 = torch.einsum("bhs,bhsd,hdk->bhsk", p0, v0, projection8)
                edges1 = torch.einsum("bhs,bhsd,hdk->bhsk", p1, v1, projection8)
                edge_delta = edges1 - edges0
                masks = relation_masks(length, cue, boundary)
                raw_components = {}
                for head in range(heads):
                    for relation, mask in masks.items():
                        raw_components[f"h{head}_{relation}"] = (edge_delta[:, head] * mask[:, :, None]).sum(1)
                builder_delta = details["write"][paired + batch_size, boundary].float() - details["write"][paired, boundary].float()
                edge_error = max(edge_error, float((sum(raw_components.values()) - builder_delta).norm()
                                                   / builder_delta.norm().clamp_min(1e-30)))
            if layer < target_layer:
                x = pre + mlp_write; bus = details["value_bus"]; continue

            paired = torch.arange(batch_size, device="cuda")
            base_mixed = mixed[paired, boundary].float(); target_mixed = mixed[paired + batch_size, boundary].float()
            scale = block.lambdas[0].float()
            scaled = {name: scale * value for name, value in raw_components.items()}
            mapped, _ = integrated_value_components(block, base_mixed, target_mixed, scaled,
                                                     target_head, config["quadrature_points"])
            attention_value = sum(mapped.values())
            direct_component, direct_target = integrated_value_components(
                block, base_mixed, target_mixed, {"attention_l8": scale * builder_delta},
                target_head, config["quadrature_points"])
            direct_attention = direct_component["attention_l8"]
            value_error = max(value_error, float((attention_value - direct_attention).norm()
                                                 / direct_attention.norm().clamp_min(1e-30)))
            grouped = {"attention_full": attention_value}
            for relation in RELATIONS:
                grouped[f"relation_{relation}"] = sum(mapped[f"h{head}_{relation}"] for head in range(heads))
            for head in range(heads):
                grouped[f"head_{head}"] = sum(mapped[f"h{head}_{relation}"] for relation in RELATIONS)
                grouped[f"head_{head}_cue"] = mapped[f"h{head}_cue"]
            local_energy += float(attention_value.double().square().sum())
            for name, value in grouped.items(): local_inner[name] += float((value.double() * attention_value.double()).sum())

            pattern9 = (details["score1"][paired, target_head, destination, boundary]
                        * details["score2"][paired, target_head, destination, boundary]).float()
            base_attention = details["write"][:batch_size].float(); attention_variants = []
            for name in variant_names:
                edge_write = pattern9[:, None] * (1 - block.attn.lamb.float()) * grouped[name] @ projection9
                modified = base_attention.clone(); modified[paired, destination] += edge_write
                attention_variants.append(modified)
            attention = torch.cat(attention_variants); repeats = len(variant_names)
            mixed0 = mixed[:batch_size].float().repeat(repeats, 1, 1); x00 = x0[:batch_size].float().repeat(repeats, 1, 1)
            bus0 = details["value_bus"][:batch_size].float().repeat(repeats, 1, 1, 1)
            state0 = F.rms_norm(mixed0 + attention, (dimension,))
            suffix_x = mixed0 + attention + block.mlp.Down(block.mlp.Left(state0) * block.mlp.Right(state0)) + block.mlp.Down_bias
            suffix_bus = bus0
            for suffix in model.blocks[target_layer + 1:]: suffix_x, suffix_bus = suffix(suffix_x, suffix_bus, x00)
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(suffix_x, (dimension,))[:, destination]).float() / 30)
            scores = (logits @ contrasts[:, 0]).reshape(repeats, batch_size)
            native_state = F.rms_norm(mixed[:batch_size] + details["write"][:batch_size].float(), (dimension,))
            native_x = mixed[:batch_size].float() + details["write"][:batch_size].float() + block.mlp.Down(block.mlp.Left(native_state) * block.mlp.Right(native_state)) + block.mlp.Down_bias
            native_bus = details["value_bus"][:batch_size].float(); native_x0 = x0[:batch_size].float()
            for suffix in model.blocks[target_layer + 1:]: native_x, native_bus = suffix(native_x, native_bus, native_x0)
            native_logits = 30 * torch.tanh(model.lm_head(F.rms_norm(native_x, (dimension,))[:, destination]).float() / 30)
            native_score = native_logits @ contrasts[:, 0]
            direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)
            for index, name in enumerate(variant_names):
                directed = direction * (scores[index] - native_score)
                endpoint[name]["sum"] += float(directed.sum()); endpoint[name]["wins"] += int((directed > 0).sum()); endpoint[name]["count"] += batch_size
            break
    endpoint = {name: {"mean": row["sum"] / row["count"], "win": row["wins"] / row["count"]} for name, row in endpoint.items()}
    return {"edge_recomposition_error": edge_error, "value_recomposition_error": value_error,
            "local_projection_share": {name: value / max(local_energy, 1e-30) for name, value in local_inner.items()},
            "endpoint": endpoint}


def main():
    config = json.loads(CONFIG.read_text()); base = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r"); counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    pairs = [(tokenizer.encode(us, add_special_tokens=False)[0], tokenizer.encode(uk, add_special_tokens=False)[0]) for us, uk in base["spelling_pairs"]]
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results = []
    for fold in range(config["folds"]):
        target_indices = set(range(fold * config["target_pairs_per_fold"], (fold + 1) * config["target_pairs_per_fold"]))
        target_pairs = [pair for index, pair in enumerate(pairs) if index in target_indices]
        cues = [pair for index, pair in enumerate(pairs) if index not in target_indices]
        contrasts = make_contrasts(model.lm_head.weight.shape[0], target_pairs, 0, config["seed"] + fold, "cuda")
        for split, (corpus_start, corpus_end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[corpus_start:corpus_end, :128], dtype=np.int64)
            records = select_records(raw, boundary_flags, cues, counts, 127, config["examples_per_label_per_split"], config["seed"] + fold * 10 + split)
            print("builder", fold, split, len(records), flush=True)
            results.append({"fold": fold, "split": split, **collect(model, raw, records, contrasts, config)})
    max_error = max(max(row["edge_recomposition_error"], row["value_recomposition_error"]) for row in results)
    b0 = max_error <= 1e-5
    ratio = lambda row, name: row["endpoint"][name]["mean"] / max(abs(row["endpoint"]["attention_full"]["mean"]), 1e-30)
    b1 = all(ratio(row, "relation_cue") >= .25 for row in results)
    b2 = all(abs(ratio(row, "relation_boundary_self")) < .25 for row in results)
    max_heads = [max(row["local_projection_share"][f"head_{head}"] for head in range(9)) for row in results]
    b3 = all(value <= .75 for value in max_heads)
    strongest = {(row["fold"], row["split"]): max(range(9), key=lambda head: row["local_projection_share"][f"head_{head}"]) for row in results}
    stable_folds = sum(strongest[(fold, 0)] == strongest[(fold, 1)] for fold in range(3)); b4 = stable_folds >= 2
    b6 = all(row["endpoint"]["attention_full"]["win"] >= .80 for row in results)
    hypotheses = {
        "B0_edge_partition_recomposes": {"supported": b0, "max_error": max_error},
        "B1_direct_cue_read_is_material": {"supported": b1},
        "B2_boundary_self_is_not_the_source": {"supported": b2},
        "B3_one_head_is_not_the_whole_builder": {"supported": b3, "max_head_shares": max_heads},
        "B4_builder_head_identity_replicates": {"supported": b4, "stable_folds": stable_folds},
        "B5_direct_two_hop_route_is_supported": {"supported": b0 and b1 and b2 and b4},
        "B6_full_attention_component_is_causal": {"supported": b6},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Layer-8 construction of the L9H8 boundary style value", "",
             "Every Layer-8 attention edge into the selected boundary is differenced under the one-token cue swap and partitioned by native head and source relation. The exact edge writes are transported through the Layer-9 RMSNorm/value map, inserted into L9H8's boundary edge with native routing, and propagated through the remaining model.", "", "## Locked results", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Fold/split | Full win/mean | source means: before / cue / between / self | strongest head share/mean |", "|---|---:|---:|---:|"]
    for row in results:
        head = strongest[(row["fold"], row["split"])]
        relation_means = " / ".join(f"{row['endpoint'][f'relation_{relation}']['mean']:+.4g}" for relation in RELATIONS)
        lines.append(f"| {row['fold']}/{row['split']} | {row['endpoint']['attention_full']['win']:.3f}/{row['endpoint']['attention_full']['mean']:+.4g} | {relation_means} | H{head} {row['local_projection_share'][f'head_{head}']:.3f}/{row['endpoint'][f'head_{head}']['mean']:+.4g} |")
    lines += ["", "## Interpretation boundary", "",
              "The source relation describes where the Layer-8 boundary invocation reads after the cue swap; it is not yet a semantic label for the source residual state. A direct cue contribution supports a literal two-hop route. A between-position contribution instead implies that the style signal has already been relayed into other token states. These are mediated contributions to L9H8's declared spelling contrast, not complete functions of the Layer-8 heads."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained"); x = np.arange(len(results)); labels = [f"f{r['fold']}s{r['split']}" for r in results]
    bottom = np.zeros(len(results))
    for relation in RELATIONS:
        values = np.asarray([row["endpoint"][f"relation_{relation}"]["mean"] for row in results]); axes[0].bar(x, values, bottom=bottom, label=relation); bottom += values
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Directed spelling shift", title="Which source positions build the boundary value?"); axes[0].legend(fontsize=8)
    for head in range(9): axes[1].plot(x, [row["local_projection_share"][f"head_{head}"] for row in results], marker="o", label=f"H{head}")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Projection share", title="Layer-8 builder heads"); axes[1].legend(fontsize=7, ncol=3)
    for axis in axes: axis.grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
