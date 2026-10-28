#!/usr/bin/env python3
"""Trace exact layer-0 QK×OV paths into selected native bilinear MLP units.

For MLP product unit j, the two factors before their product are

    left_j  = L_j RMS(x_direct + attention(x_direct))
    right_j = R_j RMS(x_direct + attention(x_direct)).

At a fixed token position, each pre-RMS attention contribution has the exact
path form

    QK(h, destination, source) * <V_h(source), O_h^T L_j>

and analogously for R_j.  This script finds positive and negative held-out
activation tails, then reports the largest signed head/source paths into both
factors.  The decomposition freezes no learned attention pattern: layer 0 QK
depends only on the supplied token embeddings and positions, and is recomputed
from the checkpoint weights for every example.
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
    layer_zero_inputs,
    load_layer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", default="1737,1642,3574,3326,3069,874")
    parser.add_argument("--row-start", type=int, default=400)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tail-sites", type=int, default=2)
    parser.add_argument("--paths-per-site", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/bilin18_mlp_discriminators/layer_0_qkov"),
    )
    return parser.parse_args()


@torch.inference_mode()
def find_tail_sites(embedding, block, rows, units, args, config, tt):
    k = args.tail_sites
    n_units = len(units)
    high_v = torch.full((k, n_units), -torch.inf, device=args.device)
    high_i = torch.full((k, n_units), -1, dtype=torch.long, device=args.device)
    low_v = torch.full((k, n_units), torch.inf, device=args.device)
    low_i = torch.full((k, n_units), -1, dtype=torch.long, device=args.device)
    unit_index = torch.tensor(units, device=args.device)
    offset = 0
    for start in range(0, len(rows), args.batch_size):
        tokens = rows[start:start + args.batch_size, :args.sequence_length].to(args.device)
        mlp_input = layer_zero_inputs(embedding, block, tokens, config, tt)
        left = block.mlp.Left(mlp_input)[..., unit_index].float().reshape(-1, n_units)
        right = block.mlp.Right(mlp_input)[..., unit_index].float().reshape(-1, n_units)
        product = left * right
        indices = torch.arange(offset, offset + product.shape[0], device=args.device)
        index_matrix = indices[:, None].expand_as(product)
        for largest, values, saved_values, saved_indices in (
            (True, product, high_v, high_i),
            (False, product, low_v, low_i),
        ):
            candidates = torch.cat((saved_values, values), dim=0)
            candidate_indices = torch.cat((saved_indices, index_matrix), dim=0)
            selected_values, selected_rows = torch.topk(candidates, k, dim=0, largest=largest)
            selected_indices = torch.gather(candidate_indices, 0, selected_rows)
            saved_values.copy_(selected_values)
            saved_indices.copy_(selected_indices)
        offset += product.shape[0]
    return high_v.cpu(), high_i.cpu(), low_v.cpu(), low_i.cpu()


@torch.inference_mode()
def decompose_site(embedding, block, token_ids, position, unit, config, tt, n_paths):
    """Return fixed-input QK×OV contributions to L_unit and R_unit."""
    device = token_ids.device
    dimension = config.n_embd
    heads, head_dim = config.n_head, dimension // config.n_head
    x0 = F.rms_norm(embedding(token_ids), (dimension,))
    mixed = block.lambdas[0] * x0 + block.lambdas[1] * x0
    attention_input = F.rms_norm(mixed, (dimension,))
    attention = block.attn
    batch, length, _ = attention_input.shape
    assert batch == 1

    def qk_projection(module):
        value = module(attention_input).view(batch, length, heads, head_dim)
        value = F.rms_norm(value, (head_dim,))
        cosine, sine = attention.rotary(value)
        return tt.apply_rotary_emb(value, cosine, sine)

    query = qk_projection(attention.c_q)
    key = qk_projection(attention.c_k)
    query2 = qk_projection(attention.c_q2)
    key2 = qk_projection(attention.c_k2)
    value = attention.c_v(attention_input).view(batch, length, heads, head_dim)
    score1 = torch.einsum("bqhd,bkhd->bhqk", query, key) / head_dim
    score2 = torch.einsum("bqhd,bkhd->bhqk", query2, key2) / head_dim
    causal = torch.tril(torch.ones(length, length, device=device, dtype=torch.bool))
    pattern = (score1 * score2).masked_fill(~causal, 0)

    left_weight = block.mlp.Left.weight[unit].float()
    right_weight = block.mlp.Right.weight[unit].float()
    output_weight = attention.c_proj.weight.float()
    path_left = torch.zeros(heads, length, device=device)
    path_right = torch.zeros(heads, length, device=device)
    ov_left = torch.zeros_like(path_left)
    ov_right = torch.zeros_like(path_right)
    for head in range(heads):
        sl = slice(head * head_dim, (head + 1) * head_dim)
        read_left = output_weight[:, sl].T @ left_weight
        read_right = output_weight[:, sl].T @ right_weight
        ov_left[head] = value[0, :, head].float() @ read_left
        ov_right[head] = value[0, :, head].float() @ read_right
        path_left[head] = pattern[0, head, position].float() * ov_left[head]
        path_right[head] = pattern[0, head, position].float() * ov_right[head]

    direct_left = float(mixed[0, position].float() @ left_weight)
    direct_right = float(mixed[0, position].float() @ right_weight)
    attention_left = float(path_left.sum())
    attention_right = float(path_right.sum())

    # Validate the source expansion against the actual projected attention write.
    attended = torch.einsum("bhqk,bkhd->bqhd", pattern, value).reshape(batch, length, dimension)
    attention_write = attention.c_proj(attended)
    actual_raw_left = float((mixed[0, position] + attention_write[0, position]).float() @ left_weight)
    actual_raw_right = float((mixed[0, position] + attention_write[0, position]).float() @ right_weight)
    reconstructed_left = direct_left + attention_left
    reconstructed_right = direct_right + attention_right
    raw_vector = mixed + attention_write
    mlp_input = F.rms_norm(raw_vector, (dimension,))
    actual_left = float(block.mlp.Left(mlp_input)[0, position, unit])
    actual_right = float(block.mlp.Right(mlp_input)[0, position, unit])
    direct_only_input = F.rms_norm(mixed, (dimension,))
    direct_only_left = float(block.mlp.Left(direct_only_input)[0, position, unit])
    direct_only_right = float(block.mlp.Right(direct_only_input)[0, position, unit])

    # Rank a path highly if it explains either factor. This avoids suppressing a
    # decisive R path merely because L has larger raw units, or vice versa.
    left_denominator = path_left.abs().sum().clamp_min(1e-8)
    right_denominator = path_right.abs().sum().clamp_min(1e-8)
    importance = path_left.abs() / left_denominator + path_right.abs() / right_denominator
    importance[:, position + 1:] = -torch.inf
    flat = torch.topk(importance.flatten(), min(n_paths, heads * (position + 1))).indices
    paths = []
    for flat_index in flat.tolist():
        head, source = divmod(flat_index, length)
        paths.append({
            "head": head,
            "source_position": source,
            "relative_position": source - position,
            "source_token_id": int(token_ids[0, source]),
            "qk_factor_1": round(float(score1[0, head, position, source]), 7),
            "qk_factor_2": round(float(score2[0, head, position, source]), 7),
            "qk_product": round(float(pattern[0, head, position, source]), 7),
            "ov_to_left": round(float(ov_left[head, source]), 5),
            "ov_to_right": round(float(ov_right[head, source]), 5),
            "path_to_left": round(float(path_left[head, source]), 5),
            "path_to_right": round(float(path_right[head, source]), 5),
        })

    return {
        "actual_left": round(actual_left, 5),
        "actual_right": round(actual_right, 5),
        "actual_product": round(actual_left * actual_right, 5),
        "attention_ablated_left": round(direct_only_left, 5),
        "attention_ablated_right": round(direct_only_right, 5),
        "attention_ablated_product": round(direct_only_left * direct_only_right, 5),
        "raw_direct_left": round(direct_left, 5),
        "raw_direct_right": round(direct_right, 5),
        "raw_attention_left": round(attention_left, 5),
        "raw_attention_right": round(attention_right, 5),
        "raw_total_left": round(actual_raw_left, 5),
        "raw_total_right": round(actual_raw_right, 5),
        "reconstruction_error_left": round(abs(actual_raw_left - reconstructed_left), 6),
        "reconstruction_error_right": round(abs(actual_raw_right - reconstructed_right), 6),
        "reconstruction_relative_error_left": round(
            abs(actual_raw_left - reconstructed_left) / max(abs(actual_raw_left), 1e-8), 8
        ),
        "reconstruction_relative_error_right": round(
            abs(actual_raw_right - reconstructed_right) / max(abs(actual_raw_right), 1e-8), 8
        ),
        "attention_abs_share_left": round(float(path_left.abs().sum() / (abs(direct_left) + path_left.abs().sum()).clamp_min(1e-8)), 5),
        "attention_abs_share_right": round(float(path_right.abs().sum() / (abs(direct_right) + path_right.abs().sum()).clamp_min(1e-8)), 5),
        "paths": paths,
    }


def fmt(value):
    return f"{value:+.3g}"


def write_markdown(result, path: Path):
    lines = [
        "# Layer-0 QK×OV paths into native bilinear MLP neurons",
        "",
        f"This report uses {result['tokens']:,} held-out FineWeb token positions from rows "
        f"{result['row_start']}--{result['row_start'] + result['rows'] - 1}. For each preselected "
        "unit, activation sites are selected only within this held-out panel.",
        "",
        "For a fixed destination and source, `path L = QK1 × QK2 × OV→L`; the R path is "
        "analogous. The sum of these paths plus the direct current-token residual reconstructs "
        "the two pre-RMS factor inputs. RMSNorm supplies a shared positive scale, after which "
        "the native product is `Left × Right`.",
        "",
        "The source paths are exact fixed-input forward decompositions, not causal deletion "
        "effects. A large signed term may cancel another term.",
        "",
    ]
    for unit_result in result["units"]:
        lines.extend([f"## Unit {unit_result['unit']}", ""])
        for site in unit_result["sites"]:
            lines.extend([
                f"### {site['tail']} tail: `{site['context']}`",
                "",
                f"- Native factors: Left {fmt(site['actual_left'])}, Right {fmt(site['actual_right'])}, "
                f"product {fmt(site['actual_product'])}.",
                f"- Zeroing the entire attention-0 write at this position gives Left "
                f"{fmt(site['attention_ablated_left'])}, Right {fmt(site['attention_ablated_right'])}, "
                f"product {fmt(site['attention_ablated_product'])}.",
                f"- Raw Left: direct {fmt(site['raw_direct_left'])} + attention "
                f"{fmt(site['raw_attention_left'])} = {fmt(site['raw_total_left'])}.",
                f"- Raw Right: direct {fmt(site['raw_direct_right'])} + attention "
                f"{fmt(site['raw_attention_right'])} = {fmt(site['raw_total_right'])}.",
                f"- Attention absolute-path share: Left {100*site['attention_abs_share_left']:.1f}%, "
                f"Right {100*site['attention_abs_share_right']:.1f}%.",
                "",
                "|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for item in site["paths"]:
                lines.append(
                    f"|{item['head']}|`{item['source_token']}`|{item['relative_position']}|"
                    f"{fmt(item['qk_factor_1'])}|{fmt(item['qk_factor_2'])}|{fmt(item['qk_product'])}|"
                    f"{fmt(item['ov_to_left'])}|{fmt(item['path_to_left'])}|"
                    f"{fmt(item['ov_to_right'])}|{fmt(item['path_to_right'])}|"
                )
            lines.append("")
    path.write_text("\n".join(lines))


def main():
    args = parse_args()
    units = [int(value) for value in args.units.split(",") if value.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = torch.load(CORPUS, map_location="cpu", weights_only=False)["rows"]
    stop = args.row_start + args.rows
    if stop > len(corpus):
        raise ValueError(f"rows [{args.row_start}:{stop}] exceed corpus length {len(corpus)}")
    rows = corpus[args.row_start:stop, :args.sequence_length].contiguous()
    embedding, block, config, tt = load_layer(0, args.device)
    high_v, high_i, low_v, low_i = find_tail_sites(
        embedding, block, rows, units, args, config, tt,
    )

    result = {
        "model": "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd",
        "layer": 0,
        "row_start": args.row_start,
        "rows": args.rows,
        "sequence_length": args.sequence_length,
        "tokens": args.rows * args.sequence_length,
        "units": [],
    }
    for column, unit in enumerate(units):
        unit_result = {"unit": unit, "sites": []}
        for tail, values, indices in (("positive", high_v, high_i), ("negative", low_v, low_i)):
            for rank in range(args.tail_sites):
                flat_index = int(indices[rank, column])
                row_index, position = divmod(flat_index, args.sequence_length)
                tokens = rows[row_index:row_index + 1].to(args.device)
                decomposition = decompose_site(
                    embedding, block, tokens, position, unit, config, tt, args.paths_per_site,
                )
                decomposition.update({
                    "tail": tail,
                    "tail_rank": rank + 1,
                    "row": args.row_start + row_index,
                    "position": position,
                    "selected_product": round(float(values[rank, column]), 5),
                    "token_id": int(tokens[0, position]),
                    "token": tokenizer.decode([int(tokens[0, position])]).replace("\n", "\\n"),
                    "context": context_for(flat_index, rows, tokenizer, args.sequence_length, radius=7),
                })
                for path_item in decomposition["paths"]:
                    path_item["source_token"] = tokenizer.decode(
                        [path_item["source_token_id"]]
                    ).replace("\n", "\\n")
                unit_result["sites"].append(decomposition)
        result["units"].append(unit_result)

    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    write_markdown(result, args.output / "README.md")
    print(json.dumps({
        "units": units,
        "tokens": result["tokens"],
        "sites": sum(len(item["sites"]) for item in result["units"]),
        "max_reconstruction_relative_error": max(
            max(site["reconstruction_relative_error_left"], site["reconstruction_relative_error_right"])
            for item in result["units"] for site in item["sites"]
        ),
    }, indent=2))
    print(args.output / "README.md")


if __name__ == "__main__":
    main()
