#!/usr/bin/env python3
"""Trace named residual sources and local QK×OV paths into selected MLP layers.

The tracer preserves the checkpoint's embedding reinjection and shared layer-0
value bus. For selected native product neurons it finds held-out positive and
negative activation tails, expands each factor over named residual sources, and
expands the current layer's attention write over head/source-token paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import (
    CORPUS,
    TOKENIZER,
    context_for,
    load_front,
    target_layer_inputs,
)


DEFAULT_UNITS = "0:1737,1642,3574,874;1:1690,1775,1535,4361;2:1737,223,4027,3455"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", default=DEFAULT_UNITS)
    parser.add_argument("--row-start", type=int, default=400)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tail-sites", type=int, default=2)
    parser.add_argument("--paths-per-site", type=int, default=6)
    parser.add_argument("--source-terms", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/bilin18_layers_0_2/source_traces"),
    )
    return parser.parse_args()


def parse_units(spec):
    result = {}
    for group in spec.split(";"):
        layer, values = group.split(":", 1)
        result[int(layer)] = [int(value) for value in values.split(",") if value]
    return result


@torch.inference_mode()
def attention_details(attention, normalized, value_bus, config, tt):
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
    attended = torch.einsum("bhqk,bkhd->bqhd", pattern, mixed_value).reshape(
        batch, length, dimension,
    )
    write = attention.c_proj(attended)
    return {
        "write": write,
        "value_bus": first_value,
        "current_value": current_value,
        "mixed_value": mixed_value,
        "score1": score1,
        "score2": score2,
        "pattern": pattern,
    }


@torch.inference_mode()
def find_sites(embedding, blocks, config, tt, rows, units, args):
    device = args.device
    index = torch.tensor(units, device=device)
    k, width = args.tail_sites, len(units)
    high_v = torch.full((k, width), -torch.inf, device=device)
    low_v = torch.full((k, width), torch.inf, device=device)
    high_i = torch.full((k, width), -1, dtype=torch.long, device=device)
    low_i = torch.full((k, width), -1, dtype=torch.long, device=device)
    offset = 0
    block = blocks[-1]
    for start in range(0, len(rows), args.batch_size):
        tokens = rows[start:start + args.batch_size].to(device)
        mlp_input = target_layer_inputs(embedding, blocks, tokens, config, tt)
        left = block.mlp.Left(mlp_input)[..., index].float().reshape(-1, width)
        right = block.mlp.Right(mlp_input)[..., index].float().reshape(-1, width)
        product = left * right
        ids = torch.arange(offset, offset + product.shape[0], device=device)[:, None].expand_as(product)
        for largest, saved_v, saved_i in ((True, high_v, high_i), (False, low_v, low_i)):
            candidates = torch.cat((saved_v, product), 0)
            candidate_ids = torch.cat((saved_i, ids), 0)
            values, selected = torch.topk(candidates, k, dim=0, largest=largest)
            saved_v.copy_(values)
            saved_i.copy_(torch.gather(candidate_ids, 0, selected))
        offset += product.shape[0]
    return high_v.cpu(), high_i.cpu(), low_v.cpu(), low_i.cpu()


def combine_embedding(components, beta_x0):
    if "embedding" in components:
        components["embedding"] = components["embedding"] + beta_x0
    else:
        components["embedding"] = beta_x0


@torch.inference_mode()
def trace_site(embedding, blocks, config, tt, token_ids, position, unit, args):
    dimension = config.n_embd
    x0 = F.rms_norm(embedding(token_ids), (dimension,))
    x = x0
    components = {"embedding": x0}
    value_bus = None
    target_details = None
    target_components = None
    target_mixed = None
    target_pre = None
    for layer, block in enumerate(blocks):
        alpha, beta = block.lambdas[0], block.lambdas[1]
        components = {name: alpha * value for name, value in components.items()}
        combine_embedding(components, beta * x0)
        mixed = alpha * x + beta * x0
        details = attention_details(
            block.attn, F.rms_norm(mixed, (dimension,)), value_bus, config, tt,
        )
        value_bus = details["value_bus"]
        pre_mlp = mixed + details["write"]
        components[f"attention{layer}"] = details["write"]
        if layer == len(blocks) - 1:
            target_details = details
            target_components = components
            target_mixed = mixed
            target_pre = pre_mlp
            break
        mlp_write = block.mlp(F.rms_norm(pre_mlp, (dimension,)))
        components[f"mlp{layer}"] = mlp_write
        x = pre_mlp + mlp_write

    block = blocks[-1]
    left_weight = block.mlp.Left.weight[unit].float()
    right_weight = block.mlp.Right.weight[unit].float()
    mlp_input = F.rms_norm(target_pre, (dimension,))
    actual_left = float(block.mlp.Left(mlp_input)[0, position, unit])
    actual_right = float(block.mlp.Right(mlp_input)[0, position, unit])
    actual_product = actual_left * actual_right

    # In exact arithmetic RMSNorm supplies one shared scalar. The checkpoint runs
    # bf16 linear operations, so Left and Right each acquire tiny independent
    # rounding. Project named raw components, then calibrate each factor by the
    # single scalar that makes its source sum equal the actual native factor.
    raw = target_pre[0, position].float()
    # Deep bfloat16 residual additions are not exactly distributive. Preserve the
    # accumulated finite-precision remainder as an explicit named source rather
    # than silently allowing the source decomposition to drift at late layers.
    component_sum = sum(value[0, position].float() for value in target_components.values())
    rounding_residual = raw - component_sum
    target_components = dict(target_components)
    rounding_component = torch.zeros_like(target_pre, dtype=torch.float32)
    rounding_component[0, position] = rounding_residual
    target_components["rounding_residual"] = rounding_component
    component_sum = sum(value[0, position].float() for value in target_components.values())
    component_reconstruction_error = float(
        (component_sum - raw).norm() / raw.norm().clamp_min(1e-8)
    )
    raw_left, raw_right = {}, {}
    for name, component in target_components.items():
        vector = component[0, position].float()
        raw_left[name] = float(vector @ left_weight)
        raw_right[name] = float(vector @ right_weight)
    left_scale = actual_left / max(abs(sum(raw_left.values())), 1e-8)
    if sum(raw_left.values()) < 0:
        left_scale = -left_scale
    right_scale = actual_right / max(abs(sum(raw_right.values())), 1e-8)
    if sum(raw_right.values()) < 0:
        right_scale = -right_scale
    source_left = {name: value * left_scale for name, value in raw_left.items()}
    source_right = {name: value * right_scale for name, value in raw_right.items()}

    source_names = list(target_components)
    pair_terms = []
    for i, first in enumerate(source_names):
        for j in range(i, len(source_names)):
            second = source_names[j]
            if i == j:
                contribution = source_left[first] * source_right[first]
            else:
                contribution = (
                    source_left[first] * source_right[second]
                    + source_left[second] * source_right[first]
                )
            pair_terms.append({"first": first, "second": second, "contribution": contribution})
    pair_terms.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    pair_abs = sum(abs(item["contribution"]) for item in pair_terms)
    pair_sq = sum(item["contribution"] ** 2 for item in pair_terms)

    no_local = F.rms_norm(target_mixed, (dimension,))
    no_local_left = float(block.mlp.Left(no_local)[0, position, unit])
    no_local_right = float(block.mlp.Right(no_local)[0, position, unit])

    attention = block.attn
    details = target_details
    heads, head_dim = config.n_head, dimension // config.n_head
    output_weight = attention.c_proj.weight.float()
    path_left = torch.zeros(heads, token_ids.shape[1], device=token_ids.device)
    path_right = torch.zeros_like(path_left)
    ov_current_left = torch.zeros_like(path_left)
    ov_bus_left = torch.zeros_like(path_left)
    ov_current_right = torch.zeros_like(path_left)
    ov_bus_right = torch.zeros_like(path_left)
    for head in range(heads):
        sl = slice(head * head_dim, (head + 1) * head_dim)
        read_left = output_weight[:, sl].T @ left_weight
        read_right = output_weight[:, sl].T @ right_weight
        current = details["current_value"][0, :, head].float()
        bus = details["value_bus"][0, :, head].float()
        ov_current_left[head] = (1 - attention.lamb.float()) * (current @ read_left)
        ov_bus_left[head] = attention.lamb.float() * (bus @ read_left)
        ov_current_right[head] = (1 - attention.lamb.float()) * (current @ read_right)
        ov_bus_right[head] = attention.lamb.float() * (bus @ read_right)
        route = details["pattern"][0, head, position].float()
        path_left[head] = route * (ov_current_left[head] + ov_bus_left[head]) * left_scale
        path_right[head] = route * (ov_current_right[head] + ov_bus_right[head]) * right_scale

    left_den = path_left.abs().sum().clamp_min(1e-8)
    right_den = path_right.abs().sum().clamp_min(1e-8)
    importance = path_left.abs() / left_den + path_right.abs() / right_den
    importance[:, position + 1:] = -torch.inf
    flat = torch.topk(
        importance.flatten(), min(args.paths_per_site, heads * (position + 1)),
    ).indices
    paths = []
    length = token_ids.shape[1]
    for flat_index in flat.tolist():
        head, source = divmod(flat_index, length)
        paths.append({
            "head": head,
            "source_position": source,
            "relative_position": source - position,
            "source_token_id": int(token_ids[0, source]),
            "qk1": round(float(details["score1"][0, head, position, source]), 6),
            "qk2": round(float(details["score2"][0, head, position, source]), 6),
            "route": round(float(details["pattern"][0, head, position, source]), 6),
            "ov_current_left": round(float(ov_current_left[head, source] * left_scale), 5),
            "ov_bus_left": round(float(ov_bus_left[head, source] * left_scale), 5),
            "path_left": round(float(path_left[head, source]), 5),
            "ov_current_right": round(float(ov_current_right[head, source] * right_scale), 5),
            "ov_bus_right": round(float(ov_bus_right[head, source] * right_scale), 5),
            "path_right": round(float(path_right[head, source]), 5),
        })

    return {
        "actual_left": round(actual_left, 5),
        "actual_right": round(actual_right, 5),
        "actual_product": round(actual_product, 5),
        "without_local_attention_left": round(no_local_left, 5),
        "without_local_attention_right": round(no_local_right, 5),
        "without_local_attention_product": round(no_local_left * no_local_right, 5),
        "source_left": {name: round(value, 5) for name, value in source_left.items()},
        "source_right": {name: round(value, 5) for name, value in source_right.items()},
        "component_vector_reconstruction_relative_error": round(
            component_reconstruction_error, 8
        ),
        "factor_reconstruction_relative_error": round(max(
            abs(sum(source_left.values()) - actual_left) / max(abs(actual_left), 1e-8),
            abs(sum(source_right.values()) - actual_right) / max(abs(actual_right), 1e-8),
        ), 8),
        "pair_reconstruction_relative_error": round(
            abs(sum(item["contribution"] for item in pair_terms) - actual_product)
            / max(abs(actual_product), 1e-8), 8
        ),
        "pair_effective_support": round(pair_abs ** 2 / max(pair_sq, 1e-8), 5),
        "dominant_pair_abs_fraction": round(
            abs(pair_terms[0]["contribution"]) / max(pair_abs, 1e-8), 5
        ),
        "source_pairs": [
            {**item, "contribution": round(item["contribution"], 5)}
            for item in pair_terms[:args.source_terms]
        ],
        "paths": paths,
    }


def fmt(value):
    return f"{value:+.3g}"


def write_report(result, path):
    lines = [
        "# Bilin18 selected-layer named-source and QK×OV traces", "",
        f"The selected sites come from {result['tokens']:,} held-out FineWeb tokens. "
        "Each factor is expanded over the embedding, prior attention/MLP writes, and the "
        "current attention write. The current attention term is then expanded over native "
        "head/source QK×OV paths.", "",
        "`Without local attention` removes only the target block's attention write; prior "
        "context remains for layers 1–2. Source-pair terms are symmetric off-diagonal sums.", "",
    ]
    for layer_result in result["layers"]:
        lines.extend([f"## Layer {layer_result['layer']}", ""])
        for unit_result in layer_result["units"]:
            lines.extend([f"### Unit {unit_result['unit']}", ""])
            for site in unit_result["sites"]:
                lines.extend([
                    f"**{site['tail']} tail:** `{site['context']}`", "",
                    f"Product {fmt(site['actual_product'])}; without local attention "
                    f"{fmt(site['without_local_attention_product'])}. Dominant named source-pair "
                    f"share {site['dominant_pair_abs_fraction']:.1%}; effective support "
                    f"{site['pair_effective_support']:.2f}.", "",
                    "Top named source-pair terms: " + "; ".join(
                        f"`{term['first']}×{term['second']}` {fmt(term['contribution'])}"
                        for term in site["source_pairs"][:5]
                    ) + ".", "",
                    "|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|",
                    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ])
                for item in site["paths"]:
                    lines.append(
                        f"|{item['head']}|`{item['source_token']}`|{item['relative_position']}|"
                        f"{fmt(item['route'])}|{fmt(item['ov_current_left'])}|"
                        f"{fmt(item['ov_bus_left'])}|{fmt(item['path_left'])}|"
                        f"{fmt(item['ov_current_right'])}|{fmt(item['ov_bus_right'])}|"
                        f"{fmt(item['path_right'])}|"
                    )
                lines.append("")
    path.write_text("\n".join(lines))


def main():
    args = parse_args()
    unit_map = parse_units(args.units)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = torch.load(CORPUS, map_location="cpu", weights_only=False)["rows"]
    stop = args.row_start + args.rows
    rows = corpus[args.row_start:stop, :args.sequence_length].contiguous()
    result = {
        "model": "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd",
        "row_start": args.row_start,
        "rows": args.rows,
        "sequence_length": args.sequence_length,
        "tokens": args.rows * args.sequence_length,
        "layers": [],
    }
    for layer, units in unit_map.items():
        embedding, blocks, config, tt = load_front(layer, args.device)
        high_v, high_i, low_v, low_i = find_sites(
            embedding, blocks, config, tt, rows, units, args,
        )
        layer_result = {"layer": layer, "units": []}
        for column, unit in enumerate(units):
            unit_result = {"unit": unit, "sites": []}
            for tail, values, indices in (
                ("positive", high_v, high_i), ("negative", low_v, low_i),
            ):
                for rank in range(args.tail_sites):
                    flat_index = int(indices[rank, column])
                    row_index, position = divmod(flat_index, args.sequence_length)
                    tokens = rows[row_index:row_index + 1].to(args.device)
                    site = trace_site(
                        embedding, blocks, config, tt, tokens, position, unit, args,
                    )
                    site.update({
                        "tail": tail,
                        "tail_rank": rank + 1,
                        "row": args.row_start + row_index,
                        "position": position,
                        "selected_product": round(float(values[rank, column]), 5),
                        "token": tokenizer.decode([int(tokens[0, position])]).replace("\n", "\\n"),
                        "context": context_for(
                            flat_index, rows, tokenizer, args.sequence_length, radius=7,
                        ),
                    })
                    for item in site["paths"]:
                        item["source_token"] = tokenizer.decode(
                            [item["source_token_id"]]
                        ).replace("\n", "\\n")
                    unit_result["sites"].append(site)
            layer_result["units"].append(unit_result)
        result["layers"].append(layer_result)
        del embedding, blocks
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    write_report(result, args.output / "README.md")
    maximum_factor = max(
        site["factor_reconstruction_relative_error"]
        for layer in result["layers"] for unit in layer["units"] for site in unit["sites"]
    )
    maximum_pair = max(
        site["pair_reconstruction_relative_error"]
        for layer in result["layers"] for unit in layer["units"] for site in unit["sites"]
    )
    maximum_component = max(
        site["component_vector_reconstruction_relative_error"]
        for layer in result["layers"] for unit in layer["units"] for site in unit["sites"]
    )
    print(json.dumps({
        "tokens": result["tokens"],
        "sites": sum(len(unit["sites"]) for layer in result["layers"] for unit in layer["units"]),
        "max_factor_reconstruction_relative_error": maximum_factor,
        "max_pair_reconstruction_relative_error": maximum_pair,
        "max_component_vector_reconstruction_relative_error": maximum_component,
    }, indent=2))
    print(args.output / "README.md")


if __name__ == "__main__":
    main()
