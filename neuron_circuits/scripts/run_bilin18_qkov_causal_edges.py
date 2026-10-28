#!/usr/bin/env python3
"""Causally test weight-predicted QK×OV edges into Bilin18 MLP neurons.

The experiment freezes the twelve layer-0--2 neurons selected by the earlier
atlas, finds positive and negative tail sites on two disjoint FineWeb panels,
and removes one native attention head/source edge at the target destination.
Every intervention is propagated through the rest of the 18-layer model to the
next-token distribution.  QK×OV-ranked edges are compared with routing-only,
same-head, same-source, and uniform controls, plus removal of the complete local
attention write at the destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import (
    MODEL_REPO,
    SNAPSHOT,
    TOKENIZER,
    context_for,
    load_front,
    target_layer_inputs,
)


DEFAULT_UNITS = "0:1737,1642,3574,874;1:1690,1775,1535,4361;2:1737,223,4027,3455"
DEFAULT_CORPUS = Path("artifacts/bilin18_layers_0_2/corpus/fineweb_viewer_20k.npy")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", default=DEFAULT_UNITS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--selection-start", type=int, default=800)
    parser.add_argument("--transfer-start", type=int, default=1200)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--search-batch-size", type=int, default=4)
    parser.add_argument("--tail-sites", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/bilin18_qkov_causal_edges"),
    )
    return parser.parse_args()


def parse_units(spec):
    result = {}
    for group in spec.split(";"):
        layer, values = group.split(":", 1)
        result[int(layer)] = [int(value) for value in values.split(",") if value]
    return result


@dataclass
class FullModel:
    embedding: torch.nn.Module
    blocks: torch.nn.ModuleList
    lm_head: torch.nn.Module
    config: object
    tt: object


def load_full_model(device):
    sys.path.insert(0, str(MODEL_REPO))
    import jacclust.tt_model as tt

    # The training repository treats einops as optional, but Block.forward
    # references its einsum helper. Supply the two checkpoint-native contractions
    # explicitly so the independent all-one validation can call Block.forward.
    def native_einsum(first, second, equation):
        if "-> ... n_head seq_q seq_k" in equation or "-> ... n_head seq_q2 seq_k2" in equation:
            return torch.einsum("bqhd,bkhd->bhqk", first, second)
        if "n_head seq_q seq_k" in equation and "seq_k n_head d_head" in equation:
            return torch.einsum("bhqk,bkhd->bhqd", first, second)
        raise ValueError(f"unsupported checkpoint einsum: {equation}")

    tt.einsum = native_einsum

    cfg_dict = json.loads((SNAPSHOT / "config.json").read_text())
    cfg_dict.pop("step", None)
    config = tt.GPTConfig(**cfg_dict)
    state = torch.load(
        SNAPSHOT / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True,
    )
    embedding = torch.nn.Embedding(config.vocab_size, config.n_embd)
    embedding.load_state_dict({"weight": state["transformer.wte.weight"]})
    blocks = torch.nn.ModuleList()
    for layer in range(config.n_layer):
        block = tt.Block(config)
        prefix = f"transformer.h.{layer}."
        block.load_state_dict({
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        })
        blocks.append(block)
    lm_head = tt.CastedLinear(config.n_embd, config.vocab_size, bias=False)
    lm_head.load_state_dict({"weight": state["lm_head.weight"]})
    del state
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return FullModel(
        embedding.to(device=device, dtype=dtype).eval(),
        blocks.to(device=device, dtype=dtype).eval(),
        lm_head.to(device=device, dtype=dtype).eval(),
        config,
        tt,
    )


@torch.inference_mode()
def attention_details(attention, normalized, value_bus, config, tt, edge=None):
    batch, length, dimension = normalized.shape
    heads, head_dim = config.n_head, dimension // config.n_head

    def projected(module):
        value = module(normalized).view(batch, length, heads, head_dim)
        value = F.rms_norm(value, (head_dim,))
        cosine, sine = attention.rotary(value)
        return tt.apply_rotary_emb(value, cosine, sine)

    q, k = projected(attention.c_q), projected(attention.c_k)
    q2, k2 = projected(attention.c_q2), projected(attention.c_k2)
    current_value = attention.c_v(normalized).view(batch, length, heads, head_dim)
    first_value = current_value if value_bus is None else value_bus.view_as(current_value)
    mixed_value = (1 - attention.lamb) * current_value + attention.lamb * first_value
    score1 = torch.einsum("bqhd,bkhd->bhqk", q, k) / head_dim
    score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / head_dim
    causal = torch.tril(torch.ones(length, length, device=normalized.device, dtype=torch.bool))
    pattern = (score1 * score2).masked_fill(~causal, 0)
    if edge is not None:
        pattern = pattern.clone()
        if edge[0] == "all_local":
            pattern[:, :, edge[1], :] = 0
        elif edge[0] == "scale":
            _, head, destination, source, keep = edge
            pattern[:, head, destination, source] *= keep
        else:
            _, head, destination, source = edge
            pattern[:, head, destination, source] = 0
    # Match the checkpoint's contraction layout and transpose order exactly;
    # bfloat16 accumulation is otherwise measurably order-dependent over 18 blocks.
    attended = torch.einsum("bhqk,bkhd->bhqd", pattern, mixed_value)
    attended = attended.transpose(1, 2).contiguous().view(batch, length, dimension)
    return {
        "write": attention.c_proj(attended),
        "value_bus": first_value,
        "current_value": current_value,
        "mixed_value": mixed_value,
        "score1": score1,
        "score2": score2,
        "pattern": pattern,
    }


@torch.inference_mode()
def forward_prefix(
    model, token_ids, target_layer, target_unit, edge=None, capture=False,
    product_override=None,
):
    dimension = model.config.n_embd
    x = F.rms_norm(model.embedding(token_ids), (dimension,))
    x0 = x
    value_bus = None
    captured = None
    destination = token_ids.shape[1] - 1
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        details = attention_details(
            block.attn,
            F.rms_norm(mixed, (dimension,)),
            value_bus,
            model.config,
            model.tt,
            edge=edge if layer == target_layer else None,
        )
        value_bus = details["value_bus"]
        pre_mlp = mixed + details["write"]
        mlp_input = F.rms_norm(pre_mlp, (dimension,))
        left = block.mlp.Left(mlp_input)
        right = block.mlp.Right(mlp_input)
        product = left * right
        if layer == target_layer:
            captured = {
                "left": float(left[0, destination, target_unit]),
                "right": float(right[0, destination, target_unit]),
                "product": float(product[0, destination, target_unit]),
            }
            if capture:
                captured.update({
                    "mixed": mixed,
                    "pre_mlp": pre_mlp,
                    "mlp_input": mlp_input,
                    "attention": details,
                })
            if product_override is not None:
                product = product.clone()
                product[0, destination, target_unit] = product_override
        mlp_write = block.mlp.Down(product) + block.mlp.Down_bias
        x = pre_mlp + mlp_write
    final = F.rms_norm(x, (dimension,))
    raw_logits = model.lm_head(final[:, -1]).float()
    logits = 30 * torch.tanh(raw_logits / 30)
    return logits[0], captured


@torch.inference_mode()
def reference_logits(model, token_ids):
    """Use the checkpoint's native Block.forward as an all-one validation."""
    dimension = model.config.n_embd
    x = F.rms_norm(model.embedding(token_ids), (dimension,))
    x0 = x
    value_bus = None
    for block in model.blocks:
        x, value_bus = block(x, value_bus, x0)
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return (30 * torch.tanh(raw / 30))[0]


@torch.inference_mode()
def find_sites(rows, layer, units, args):
    embedding, blocks, config, tt = load_front(layer, args.device)
    index = torch.tensor(units, device=args.device)
    width = len(units)
    k = args.tail_sites
    high_v = torch.full((k, width), -torch.inf, device=args.device)
    low_v = torch.full((k, width), torch.inf, device=args.device)
    high_i = torch.full((k, width), -1, dtype=torch.long, device=args.device)
    low_i = torch.full((k, width), -1, dtype=torch.long, device=args.device)
    offset = 0
    block = blocks[-1]
    for start in range(0, len(rows), args.search_batch_size):
        tokens = torch.as_tensor(
            rows[start:start + args.search_batch_size, :args.sequence_length].astype(np.int64),
            device=args.device,
        )
        normalized = target_layer_inputs(embedding, blocks, tokens, config, tt)
        left = block.mlp.Left(normalized)[..., index].float().reshape(-1, width)
        right = block.mlp.Right(normalized)[..., index].float().reshape(-1, width)
        product = left * right
        ids = torch.arange(offset, offset + product.shape[0], device=args.device)[:, None].expand_as(product)
        for largest, saved_v, saved_i in ((True, high_v, high_i), (False, low_v, low_i)):
            candidates = torch.cat((saved_v, product), 0)
            candidate_ids = torch.cat((saved_i, ids), 0)
            values, selected = torch.topk(candidates, k, dim=0, largest=largest)
            saved_v.copy_(values)
            saved_i.copy_(torch.gather(candidate_ids, 0, selected))
        offset += product.shape[0]
    del embedding, blocks
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return high_v.cpu(), high_i.cpu(), low_v.cpu(), low_i.cpu()


def select_control_edges(target, routes, predicted, position, rng):
    heads = routes.shape[0]
    th, ts = target
    candidates = [(h, s) for h in range(heads) for s in range(position + 1) if (h, s) != target]
    # The routing-only competitor asks whether OV/factor information adds value.
    route_edge = max(candidates, key=lambda hs: abs(float(routes[hs])))
    same_head = [(h, s) for h, s in candidates if h == th]
    same_source = [(h, s) for h, s in candidates if s == ts]
    return {
        "qkov": target,
        "route_only": route_edge,
        "same_head_random_source": rng.choice(same_head or candidates),
        "same_source_random_head": rng.choice(same_source or candidates),
        "uniform_random": rng.choice(candidates),
    }


def edge_predictions(model, captured, layer, unit):
    block = model.blocks[layer]
    attention = block.attn
    details = captured["attention"]
    position = captured["pre_mlp"].shape[1] - 1
    heads = model.config.n_head
    head_dim = model.config.n_embd // heads
    left_weight = block.mlp.Left.weight[unit].float()
    right_weight = block.mlp.Right.weight[unit].float()
    output_weight = attention.c_proj.weight.float()
    route = details["pattern"][0, :, position, :position + 1].float()
    path_left = torch.zeros_like(route)
    path_right = torch.zeros_like(route)
    # Frozen-normalization path prediction. Finite interventions below recompute
    # RMSNorm, so agreement is an empirical test rather than an identity.
    pre = captured["pre_mlp"][0, position]
    normalized = captured["mlp_input"][0, position]
    scale = float(normalized.float().norm() / pre.float().norm().clamp_min(1e-8))
    for head in range(heads):
        sl = slice(head * head_dim, (head + 1) * head_dim)
        read_left = output_weight[:, sl].T @ left_weight
        read_right = output_weight[:, sl].T @ right_weight
        values = details["mixed_value"][0, :position + 1, head].float()
        path_left[head] = route[head] * (values @ read_left) * scale
        path_right[head] = route[head] * (values @ read_right) * scale
    left, right = captured["left"], captured["right"]
    predicted_delta = (left - path_left) * (right - path_right) - left * right
    flat = int(predicted_delta.abs().argmax())
    target = divmod(flat, position + 1)
    return route, path_left, path_right, predicted_delta, target


def compare_logits(base_logits, changed_logits, target_id):
    base_logp = F.log_softmax(base_logits, -1)
    changed_logp = F.log_softmax(changed_logits, -1)
    base_p = base_logp.exp()
    delta = changed_logits - base_logits
    return {
        "target_logit_delta": float(delta[target_id]),
        "nll_delta": float(-changed_logp[target_id] + base_logp[target_id]),
        "kl_base_to_changed": float((base_p * (base_logp - changed_logp)).sum()),
        "logit_delta_rms": float(delta.square().mean().sqrt()),
        "top1_changed": int(base_logits.argmax() != changed_logits.argmax()),
    }


def context(ids, position, tokenizer, radius=8):
    start, stop = max(0, position - radius), min(len(ids), position + radius + 1)
    before = tokenizer.decode(ids[start:position].tolist()).replace("\n", "\\n")
    current = tokenizer.decode([int(ids[position])]).replace("\n", "\\n")
    after = tokenizer.decode(ids[position + 1:stop].tolist()).replace("\n", "\\n")
    return f"{before}[{current}]{after}"


def summarize(records):
    conditions = sorted({row["condition"] for row in records})
    result = {}
    for split in ("selection", "transfer"):
        result[split] = {}
        subset = [row for row in records if row["split"] == split]
        for condition in conditions:
            rows = [row for row in subset if row["condition"] == condition]
            result[split][condition] = {
                "n": len(rows),
                "median_abs_product_delta": float(np.median([abs(r["product_delta"]) for r in rows])),
                "median_abs_product_delta_fraction": float(np.median([r["abs_product_delta_fraction"] for r in rows])),
                "median_kl": float(np.median([r["kl_base_to_changed"] for r in rows])),
                "mean_nll_delta": float(np.mean([r["nll_delta"] for r in rows])),
                "median_abs_target_logit_delta": float(np.median([abs(r["target_logit_delta"]) for r in rows])),
                "top1_change_rate": float(np.mean([r["top1_changed"] for r in rows])),
            }
        qkov = [row for row in subset if row["condition"] == "qkov"]
        valid = [row for row in qkov if abs(row["predicted_product_delta"]) > 1e-8]
        result[split]["prediction_validation"] = {
            "n": len(valid),
            "sign_agreement": float(np.mean([
                np.sign(r["predicted_product_delta"]) == np.sign(r["product_delta"])
                for r in valid
            ])),
            "pearson": float(np.corrcoef(
                [r["predicted_product_delta"] for r in valid],
                [r["product_delta"] for r in valid],
            )[0, 1]),
        }
        mediated = [row for row in qkov if "mediated_logit_delta_rms" in row]
        result[split]["target_neuron_mediation"] = {
            "n": len(mediated),
            "median_mediated_logit_delta_rms": float(np.median([
                r["mediated_logit_delta_rms"] for r in mediated
            ])),
            "median_residual_edge_logit_delta_rms": float(np.median([
                r["residual_edge_logit_delta_rms"] for r in mediated
            ])),
            "median_mediated_fraction_of_additive_rms": float(np.median([
                r["mediated_logit_delta_rms"] /
                max(r["mediated_logit_delta_rms"] + r["residual_edge_logit_delta_rms"], 1e-12)
                for r in mediated
            ])),
        }
    return result


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    unit_map = parse_units(args.units)
    corpus = np.load(args.corpus, mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    panels = {
        "selection": corpus[args.selection_start:args.selection_start + args.rows],
        "transfer": corpus[args.transfer_start:args.transfer_start + args.rows],
    }
    selected_sites = []
    for split, rows in panels.items():
        for layer, units in unit_map.items():
            high_v, high_i, low_v, low_i = find_sites(rows, layer, units, args)
            for column, unit in enumerate(units):
                for tail, values, indices in (
                    ("positive", high_v, high_i), ("negative", low_v, low_i),
                ):
                    for rank in range(args.tail_sites):
                        flat = int(indices[rank, column])
                        row_offset, position = divmod(flat, args.sequence_length)
                        # The corpus includes one target token beyond the input.
                        selected_sites.append({
                            "split": split,
                            "layer": layer,
                            "unit": unit,
                            "tail": tail,
                            "rank": rank + 1,
                            "row": (args.selection_start if split == "selection" else args.transfer_start) + row_offset,
                            "position": position,
                            "selected_product": float(values[rank, column]),
                        })

    model = load_full_model(args.device)
    rng = random.Random(args.seed)
    records = []
    for site_index, site in enumerate(selected_sites):
        row = np.asarray(corpus[site["row"]], dtype=np.int64)
        position = site["position"]
        ids = torch.as_tensor(row[:position + 1], device=args.device)[None]
        target_id = int(row[position + 1])
        base_logits, captured = forward_prefix(
            model, ids, site["layer"], site["unit"], capture=True,
        )
        route, path_left, path_right, predicted, qkov_edge = edge_predictions(
            model, captured, site["layer"], site["unit"],
        )
        edge_map = select_control_edges(qkov_edge, route, predicted, position, rng)
        finite = {}
        for condition, (head, source) in edge_map.items():
            changed_logits, changed = forward_prefix(
                model, ids, site["layer"], site["unit"],
                edge=("edge", head, position, source),
            )
            finite[condition] = {
                "head": head,
                "source": source,
                "route": float(route[head, source]),
                "path_left": float(path_left[head, source].detach()),
                "path_right": float(path_right[head, source].detach()),
                "predicted_product_delta": float(predicted[head, source].detach()),
                "product_delta": changed["product"] - captured["product"],
                **compare_logits(base_logits, changed_logits, target_id),
            }
            if condition == "qkov":
                restored_logits, _ = forward_prefix(
                    model, ids, site["layer"], site["unit"],
                    edge=("edge", head, position, source),
                    product_override=captured["product"],
                )
                mediated = compare_logits(restored_logits, changed_logits, target_id)
                residual = compare_logits(base_logits, restored_logits, target_id)
                finite[condition].update({
                    "mediated_target_logit_delta": mediated["target_logit_delta"],
                    "mediated_nll_delta": mediated["nll_delta"],
                    "mediated_kl_restored_to_changed": mediated["kl_base_to_changed"],
                    "mediated_logit_delta_rms": mediated["logit_delta_rms"],
                    "residual_edge_target_logit_delta": residual["target_logit_delta"],
                    "residual_edge_nll_delta": residual["nll_delta"],
                    "residual_edge_kl": residual["kl_base_to_changed"],
                    "residual_edge_logit_delta_rms": residual["logit_delta_rms"],
                })
        changed_logits, changed = forward_prefix(
            model, ids, site["layer"], site["unit"],
            edge=("all_local", position),
        )
        finite["all_local_attention"] = {
            "head": None,
            "source": None,
            "route": None,
            "path_left": None,
            "path_right": None,
            "predicted_product_delta": None,
            "product_delta": changed["product"] - captured["product"],
            **compare_logits(base_logits, changed_logits, target_id),
        }
        denominator = max(abs(finite["all_local_attention"]["product_delta"]), 1e-6)
        for condition, values in finite.items():
            records.append({
                **site,
                "site_index": site_index,
                "condition": condition,
                "token_id": int(row[position]),
                "target_id": target_id,
                "context": context(row, position, tokenizer),
                "source_token_id": None if values["source"] is None else int(row[values["source"]]),
                "source_token": None if values["source"] is None else tokenizer.decode([int(row[values["source"]])]).replace("\n", "\\n"),
                "baseline_product": captured["product"],
                "abs_product_delta_fraction": abs(values["product_delta"]) / denominator,
                **values,
            })
        if (site_index + 1) % 8 == 0:
            print(f"completed {site_index + 1}/{len(selected_sites)} sites", flush=True)

    result = {
        "model": "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd",
        "corpus": str(args.corpus),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "config": vars(args) | {"output": str(args.output), "corpus": str(args.corpus)},
        "sites": selected_sites,
        "records": records,
        "summary": summarize(records),
    }
    validation_ids = torch.as_tensor(
        np.asarray(corpus[args.selection_start, :64], dtype=np.int64),
        device=args.device,
    )[None]
    validation_layer = min(unit_map)
    custom, _ = forward_prefix(
        model, validation_ids, validation_layer, unit_map[validation_layer][0]
    )
    native = reference_logits(model, validation_ids)
    result["all_one_validation"] = {
        "max_abs_logit_error": float((custom - native).abs().max()),
        "relative_logit_l2_error": float(
            (custom - native).norm() / native.norm().clamp_min(1e-8)
        ),
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    print(args.output / "results.json")


if __name__ == "__main__":
    main()
