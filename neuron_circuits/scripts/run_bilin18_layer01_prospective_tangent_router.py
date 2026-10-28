#!/usr/bin/env python3
"""Prospectively route Layer-0/1 native channels with a tensor-network JVP.

The tangent router sees the unedited state, the requested token-embedding edit,
and the frozen model weights.  It never sees the finite counterfactual product
change used as the causal patch.  Frozen identity banks and realized-change
rankings provide matched lower and upper controls.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer01_final_bilinear_readout import downstream_fidelities
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer01_prospective_tangent_router.json")
OUTPUT = TENSOR_ROOT / "layer01_prospective_tangent_router.json"
REPORT = TENSOR_ROOT / "LAYER01_PROSPECTIVE_TANGENT_ROUTER.md"


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def cosine(left, right):
    return float((left * right).sum() / (left.norm() * right.norm()).clamp_min(1e-30))


def jaccard(left, right):
    a, b = set(left.tolist()), set(right.tolist())
    return len(a & b) / max(1, len(a | b))


def normalized_embedding(model, ids):
    return F.rms_norm(model.embedding(ids).float(), (model.config.n_embd,))


def prefix_pre_from_x0(model, x0, layer, final_position):
    """Differentiable checkpoint-native prefix ending before the selected MLP."""
    x, value_bus = x0, None
    for index, block in enumerate(model.blocks[: layer + 1]):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(
            F.rms_norm(mixed, (model.config.n_embd,)), value_bus,
        )
        pre = mixed + attention_write
        if index == layer:
            return pre[:, final_position]
        x = pre + block.mlp(F.rms_norm(pre, (model.config.n_embd,)))
    raise AssertionError("unreachable")


def rms_jvp(state, tangent):
    epsilon = torch.finfo(state.dtype).eps
    inverse = torch.rsqrt(state.square().mean(-1, keepdim=True) + epsilon)
    return tangent * inverse - state * (tangent * state).mean(-1, keepdim=True) * inverse.pow(3)


def tangent_product(
    model, block, ids, target_token, layer, edit_position, final_position, path_point=0.0,
):
    base_x0 = normalized_embedding(model, ids).detach()
    target_id = torch.tensor([target_token], device=ids.device)
    target = normalized_embedding(model, target_id)[0]
    direction = torch.zeros_like(base_x0)
    direction[:, edit_position] = target - base_x0[:, edit_position]

    def function(value):
        return prefix_pre_from_x0(model, value, layer, final_position)

    # A preceding inference-mode pass may have populated the checkpoint's rotary
    # cache with inference tensors, which autograd correctly refuses to save.
    # Rebuild the small deterministic cache under ordinary grad mode.
    for prefix_block in model.blocks[: layer + 1]:
        prefix_block.attn.rotary.seq_len_cached = None
        prefix_block.attn.rotary.cos_cached = None
        prefix_block.attn.rotary.sin_cached = None
    path_x0 = base_x0 + path_point * direction
    with torch.enable_grad():
        base_pre, tangent_pre = torch.autograd.functional.jvp(
            function, (path_x0,), (direction,), create_graph=False, strict=True,
        )
    base_pre, tangent_pre = base_pre[0].detach(), tangent_pre[0].detach()
    base_state = F.rms_norm(base_pre, (model.config.n_embd,))
    tangent_state = rms_jvp(base_pre, tangent_pre)
    left, right = block.mlp.Left(base_state), block.mlp.Right(base_state)
    dleft = F.linear(tangent_state, block.mlp.Left.weight.float())
    dright = F.linear(tangent_state, block.mlp.Right.weight.float())
    return dleft * right + left * dright, tangent_pre


@torch.no_grad()
def score_rows(full, decoder):
    decoder_norm = decoder.norm(dim=0)
    write = F.linear(full, decoder)
    return {
        "activation": full.abs(),
        "state_write": full.abs() * decoder_norm,
        "local": (full * (write @ decoder)).abs(),
    }


@torch.inference_mode()
def finite_example(model, ids, target_token, layer, edit_position, final_position):
    changed = ids.clone()
    changed[:, edit_position] = target_token
    base = forward_layers(model, ids, last_layer=layer)[layer]
    counter = forward_layers(model, changed, last_layer=layer)[layer]
    block = model.blocks[layer]
    base_state = F.rms_norm(base["pre"][0, final_position].float(), (model.config.n_embd,))
    counter_state = F.rms_norm(counter["pre"][0, final_position].float(), (model.config.n_embd,))
    full = (
        block.mlp.Left(counter_state) * block.mlp.Right(counter_state)
        - block.mlp.Left(base_state) * block.mlp.Right(base_state)
    )
    base_product = block.mlp.Left(base_state) * block.mlp.Right(base_state)
    return base, counter, full, base_product


def select_indices(score, rank):
    return score.topk(rank).indices


def frozen_banks(model, arrays, data_config, pair_config, config, edit_position, final_position):
    """Fit fixed global/family identities on original FineWeb lexical pairs."""
    rows = arrays["fineweb"]
    generator = np.random.default_rng(config["seed"] - 1)
    accumulators = {
        layer: {family: [] for family in pair_config["families"]}
        for layer in config["layers"]
    }
    for family, pairs in pair_config["families"].items():
        for source_token, target_token in pairs:
            windows = select_windows(
                rows, source_token, edit_position, final_position,
                config["discovery_contexts_per_pair"], generator,
            )
            for row_index, start in windows:
                raw = np.asarray(
                    rows[row_index, start:start + data_config["sequence_length"]], dtype=np.int64,
                ).copy()
                ids = torch.as_tensor(raw, device="cuda")[None]
                for layer in config["layers"]:
                    _, _, full, _ = finite_example(
                        model, ids, target_token, layer, edit_position, final_position,
                    )
                    score = score_rows(full, model.blocks[layer].mlp.Down.weight.float())["local"]
                    accumulators[layer][family].append(score / score.sum().clamp_min(1e-30))
    banks = {}
    for layer in config["layers"]:
        family_scores = {
            family: torch.stack(values).mean(0)
            for family, values in accumulators[layer].items()
        }
        banks[layer] = {
            "family": {
                family: select_indices(score, config["rank"])
                for family, score in family_scores.items()
            },
            "global": select_indices(torch.stack(list(family_scores.values())).mean(0), config["rank"]),
        }
    return banks


@torch.no_grad()
def main():
    config = json.loads(CONFIG.read_text())
    data_config = json.loads(Path(config["data_config"]).read_text())
    pair_config = json.loads(Path(config["discovery_pair_config"]).read_text())
    evaluation_pairs = json.loads(Path(config["evaluation_pair_config"]).read_text())
    edit_position, final_position = data_config["edit_position"], data_config["causal_position"]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in data_config["evaluation"].items()
    }
    banks = frozen_banks(
        model, arrays, data_config, pair_config, config, edit_position, final_position,
    )
    random_indices = {
        layer: torch.randperm(
            model.blocks[layer].mlp.Down.weight.shape[1],
            generator=torch.Generator(device="cpu").manual_seed(config["seed"] + layer),
        )[:config["rank"]].cuda()
        for layer in config["layers"]
    }
    records = {str(layer): [] for layer in config["layers"]}

    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        generator = np.random.default_rng(config["seed"] + corpus_index)
        for family, pairs in evaluation_pairs["new_families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows, source_token, edit_position, final_position,
                    config["evaluation_contexts_per_pair"], generator,
                )
                for context_index, (row_index, start) in enumerate(windows):
                    raw = np.asarray(
                        rows[row_index, start:start + data_config["sequence_length"]], dtype=np.int64,
                    ).copy()
                    ids = torch.as_tensor(raw, device="cuda")[None]
                    for layer in config["layers"]:
                        block = model.blocks[layer]
                        decoder = block.mlp.Down.weight.float()
                        base, counter, full, base_product = finite_example(
                            model, ids, target_token, layer, edit_position, final_position,
                        )
                        predicted, tangent_pre = tangent_product(
                            model, block, ids, target_token, layer, edit_position, final_position,
                        )
                        finite_pre = (
                            counter["pre"][0, final_position].float()
                            - base["pre"][0, final_position].float()
                        )
                        actual_scores = score_rows(full, decoder)
                        predicted_scores = score_rows(predicted, decoder)
                        static = decoder.norm(dim=0)
                        base_score = base_product.abs() * static
                        selections = {
                            "random": random_indices[layer],
                            "static_decoder": select_indices(static, config["rank"]),
                            "base_active": select_indices(base_score, config["rank"]),
                            "frozen_global": banks[layer]["global"],
                            "frozen_family": banks[layer]["family"][family],
                            "tangent_activation": select_indices(predicted_scores["activation"], config["rank"]),
                            "tangent_state_write": select_indices(predicted_scores["state_write"], config["rank"]),
                            "tangent_local": select_indices(predicted_scores["local"], config["rank"]),
                            "realized_activation": select_indices(actual_scores["activation"], config["rank"]),
                            "realized_state_write": select_indices(actual_scores["state_write"], config["rank"]),
                            "realized_local": select_indices(actual_scores["local"], config["rank"]),
                        }
                        deltas = {"full": full}
                        for name, indices in selections.items():
                            selected = torch.zeros_like(full)
                            selected[indices] = full[indices]
                            deltas[name] = selected
                        fidelities = downstream_fidelities(
                            model, layer, base, deltas, final_position,
                        )
                        records[str(layer)].append({
                            "corpus": corpus,
                            "family": family,
                            "pair": f"{source_token}->{target_token}",
                            "context": context_index,
                            "fidelity": {name: fidelities[name] for name in config["methods"]},
                            "pre_cosine": cosine(tangent_pre, finite_pre),
                            "pre_norm_ratio": float(tangent_pre.norm() / finite_pre.norm().clamp_min(1e-30)),
                            "product_cosine": cosine(predicted, full),
                            "product_norm_ratio": float(predicted.norm() / full.norm().clamp_min(1e-30)),
                            "write_cosine": cosine(F.linear(predicted, decoder), F.linear(full, decoder)),
                            "support_jaccard": jaccard(
                                selections["tangent_local"], selections["realized_local"],
                            ),
                        })
                print("prospective tangent", corpus, family, source_token, target_token, flush=True)

    summaries = {}
    for layer in config["layers"]:
        summaries[str(layer)] = {}
        for corpus in arrays:
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            summaries[str(layer)][corpus] = {
                "fidelity": {
                    name: median([row["fidelity"][name] for row in selected])
                    for name in config["methods"]
                },
                "pre_cosine": median([row["pre_cosine"] for row in selected]),
                "pre_norm_ratio": median([row["pre_norm_ratio"] for row in selected]),
                "product_cosine": median([row["product_cosine"] for row in selected]),
                "product_norm_ratio": median([row["product_norm_ratio"] for row in selected]),
                "write_cosine": median([row["write_cosine"] for row in selected]),
                "support_jaccard": median([row["support_jaccard"] for row in selected]),
                "examples": len(selected),
            }

    def row(layer, corpus):
        return summaries[str(layer)][corpus]

    hypotheses = {
        "T0_tangent_predicts_finite_product_direction": {
            "supported": all(
                row(layer, corpus)["product_cosine"] >= (0.50 if layer == 0 else 0.30)
                for layer in config["layers"] for corpus in arrays
            )
        },
        "T1_tangent_router_is_causally_useful": {
            "supported": all(
                row(layer, corpus)["fidelity"]["tangent_local"] >= (0.40 if layer == 0 else 0.25)
                for layer in config["layers"] for corpus in arrays
            )
        },
        "T2_tangent_beats_state_free_weights": {
            "supported": all(
                row(layer, corpus)["fidelity"]["tangent_local"]
                - row(layer, corpus)["fidelity"]["static_decoder"] >= 0.20
                for layer in config["layers"] for corpus in arrays
            )
        },
        "T3_tangent_closes_most_of_oracle_gap": {
            "supported": all(
                row(layer, corpus)["fidelity"]["tangent_local"] >=
                0.60 * row(layer, corpus)["fidelity"]["realized_local"]
                for layer in config["layers"] for corpus in arrays
            )
        },
        "T4_tangent_router_transfers_ood": {
            "supported": all(
                row(layer, "tinystories")["fidelity"]["tangent_local"] >=
                0.75 * row(layer, "fineweb")["fidelity"]["tangent_local"]
                for layer in config["layers"]
            )
        },
        "T5_tangent_support_is_predictive": {
            "supported": all(
                row(layer, corpus)["support_jaccard"] >= 0.08
                for layer in config["layers"] for corpus in arrays
            )
        },
        "T6_mechanistic_router_beats_frozen_identity": {
            "supported": all(
                row(layer, corpus)["fidelity"]["tangent_local"] - max(
                    row(layer, corpus)["fidelity"]["frozen_global"],
                    row(layer, corpus)["fidelity"]["frozen_family"],
                ) >= 0.15
                for layer in config["layers"] for corpus in arrays
            )
        },
        "T7_base_activity_is_not_enough": {
            "supported": all(
                row(layer, corpus)["fidelity"]["tangent_local"]
                - row(layer, corpus)["fidelity"]["base_active"] >= 0.10
                for layer in config["layers"] for corpus in arrays
            )
        },
        "T8_layer0_is_more_locally_linear": {
            "supported": all(
                row(0, corpus)["product_cosine"] - row(1, corpus)["product_cosine"] >= 0.10
                for corpus in arrays
            )
        },
    }

    output = {
        "config": config,
        "hypotheses": hypotheses,
        "summaries": summaries,
        "frozen_banks": {
            str(layer): {
                "global": banks[layer]["global"].cpu().tolist(),
                "family": {
                    family: indices.cpu().tolist()
                    for family, indices in banks[layer]["family"].items()
                },
            }
            for layer in config["layers"]
        },
        "records": records,
    }
    OUTPUT.write_text(json.dumps(output, indent=2))

    lines = [
        "# Prospective tensor-tangent router for Layer-0/1 neurons",
        "",
        "The router uses the unedited state, the requested replacement embedding, and a checkpoint-native prefix JVP. It never observes the finite counterfactual product change used in the causal patch. All methods select the same 256 native channels; selected *actual* finite deltas are propagated through the complete suffix and scored in the output Fisher metric.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "| Layer | Corpus | Pre/product/write cosine | Tangent norm ratio pre/product | Support Jaccard | Random / static / base | Frozen global/family | Tangent activation/state/local | Realized activation/state/local |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in config["layers"]:
        for corpus in arrays:
            item = row(layer, corpus)
            fidelity = item["fidelity"]
            lines.append(
                f"| {layer} | {corpus} | {item['pre_cosine']:.3f}/{item['product_cosine']:.3f}/{item['write_cosine']:.3f} | "
                f"{item['pre_norm_ratio']:.3f}/{item['product_norm_ratio']:.3f} | {item['support_jaccard']:.3f} | "
                f"{fidelity['random']:.3f}/{fidelity['static_decoder']:.3f}/{fidelity['base_active']:.3f} | "
                f"{fidelity['frozen_global']:.3f}/{fidelity['frozen_family']:.3f} | "
                f"{fidelity['tangent_activation']:.3f}/{fidelity['tangent_state_write']:.3f}/{fidelity['tangent_local']:.3f} | "
                f"{fidelity['realized_activation']:.3f}/{fidelity['realized_state_write']:.3f}/{fidelity['realized_local']:.3f} |"
            )
    lines += [
        "",
        "`tangent_local` is prospective but edit-aware: it is allowed to see the target token embedding because the intervention to be explained includes that target. `realized_*` methods see the finite counterfactual and are upper controls. Frozen banks see discovery counterfactuals but are fixed before all new lexical pairs.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
