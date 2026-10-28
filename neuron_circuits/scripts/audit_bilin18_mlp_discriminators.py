#!/usr/bin/env python3
"""Build a held-out input-discrimination atlas for one bilin18 MLP layer.

The primary object is a native product channel

    h_j(x) = Left_j(x) * Right_j(x)

and not an SAE feature or a downstream output direction.  The script streams real
FineWeb token rows, characterizes the two linear factors and their product on a
discovery split, and evaluates the frozen winning coarse token class on held-out
rows.  It also retains positive and negative activation tails for concrete examples.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


MODEL_REPO = Path("/home/loganriggs/Coding/reverse_engineer/tensor_language")
SNAPSHOT = Path(
    "/home/loganriggs/.cache/huggingface/hub/"
    "models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/"
    "snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240"
)
CORPUS = MODEL_REPO / "basis_aligned/bilinear_quotient/census_state_diverse.pt"
TOKENIZER = Path(
    "/home/loganriggs/.cache/huggingface/hub/models--gpt2/"
    "snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
)

CLASSES = (
    "determiner", "preposition", "conjunction", "pronoun", "auxiliary",
    "number", "punctuation", "capitalized", "subword", "content_word",
)
DETERMINERS = frozenset("the a an this that these those his her its their our my your some any no every each all".split())
PREPOSITIONS = frozenset("of in to for on at by with from as into about over after before between through under against within".split())
CONJUNCTIONS = frozenset("and or but nor so yet because although while if than when where since".split())
PRONOUNS = frozenset("he she it they we you i him her them us me who which what".split())
AUXILIARIES = frozenset("is was be have are not has will had can would were been could do am did may being does should".split())


def classify_token(text: str) -> int:
    stripped = text.strip()
    if not stripped or not stripped[0].isalnum():
        return CLASSES.index("punctuation")
    if stripped[0].isdigit():
        return CLASSES.index("number")
    lower = stripped.lower()
    for name, vocabulary in (
        ("determiner", DETERMINERS), ("preposition", PREPOSITIONS),
        ("conjunction", CONJUNCTIONS), ("pronoun", PRONOUNS),
        ("auxiliary", AUXILIARIES),
    ):
        if lower in vocabulary:
            return CLASSES.index(name)
    if stripped[0].isupper():
        return CLASSES.index("capitalized")
    # GPT-2 pieces without leading whitespace are usually continuations.  Whole
    # punctuation and sentence-initial words were handled above.
    if not text.startswith(" ") and stripped.isalpha():
        return CLASSES.index("subword")
    return CLASSES.index("content_word")


@dataclass
class SplitStats:
    count: int
    class_counts: torch.Tensor
    class_abs_sums: dict[str, torch.Tensor]
    class_signed_sums: dict[str, torch.Tensor]
    sums: dict[str, torch.Tensor]
    square_sums: dict[str, torch.Tensor]
    top_values: dict[str, torch.Tensor] | None = None
    top_indices: dict[str, torch.Tensor] | None = None
    bottom_values: dict[str, torch.Tensor] | None = None
    bottom_indices: dict[str, torch.Tensor] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--discovery-rows", type=int, default=160)
    parser.add_argument("--heldout-rows", type=int, default=80)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tail-size", type=int, default=8)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument(
        "--all-tail-contexts", type=int, default=0,
        help="Store this many positive/negative product-tail contexts for every channel.",
    )
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/bilin18_mlp_discriminators/layer_0"))
    return parser.parse_args()


def load_layer(layer_index: int, device: str):
    sys.path.insert(0, str(MODEL_REPO))
    import jacclust.tt_model as tt

    config_dict = json.loads((SNAPSHOT / "config.json").read_text())
    config_dict.pop("step", None)
    config = tt.GPTConfig(**config_dict)
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    embedding = torch.nn.Embedding(config.vocab_size, config.n_embd)
    embedding.load_state_dict({"weight": state["transformer.wte.weight"]})
    block = tt.Block(config)
    prefix = f"transformer.h.{layer_index}."
    block.load_state_dict({key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)})
    del state
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return embedding.to(device=device, dtype=dtype).eval(), block.to(device=device, dtype=dtype).eval(), config, tt


def load_front(layer_index: int, device: str):
    """Load the embedding and blocks 0..layer_index from one mmap'd state."""
    sys.path.insert(0, str(MODEL_REPO))
    import jacclust.tt_model as tt

    config_dict = json.loads((SNAPSHOT / "config.json").read_text())
    config_dict.pop("step", None)
    config = tt.GPTConfig(**config_dict)
    if not 0 <= layer_index < config.n_layer:
        raise ValueError(f"layer {layer_index} outside [0, {config.n_layer})")
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    embedding = torch.nn.Embedding(config.vocab_size, config.n_embd)
    embedding.load_state_dict({"weight": state["transformer.wte.weight"]})
    blocks = torch.nn.ModuleList()
    for index in range(layer_index + 1):
        block = tt.Block(config)
        prefix = f"transformer.h.{index}."
        block.load_state_dict({
            key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)
        })
        blocks.append(block)
    del state
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return (
        embedding.to(device=device, dtype=dtype).eval(),
        blocks.to(device=device, dtype=dtype).eval(),
        config,
        tt,
    )


@torch.inference_mode()
def layer_zero_inputs(embedding, block, token_ids: torch.Tensor, config, tt) -> torch.Tensor:
    """Return the exact normalized input to MLP0, including attention0."""
    if config.n_layer < 1:
        raise ValueError("model has no transformer blocks")
    dimension = config.n_embd
    x = F.rms_norm(embedding(token_ids), (dimension,))
    x0 = x
    x = block.lambdas[0] * x + block.lambdas[1] * x0
    normalized = F.rms_norm(x, (dimension,))
    attention = block.attn
    batch, length, _ = normalized.shape
    heads, head_dim = config.n_head, dimension // config.n_head

    def projected(module):
        value = module(normalized).view(batch, length, heads, head_dim)
        value = F.rms_norm(value, (head_dim,))
        cosine, sine = attention.rotary(value)
        return tt.apply_rotary_emb(value, cosine, sine)

    query, key = projected(attention.c_q), projected(attention.c_k)
    query2, key2 = projected(attention.c_q2), projected(attention.c_k2)
    value = attention.c_v(normalized).view(batch, length, heads, head_dim)
    score1 = torch.einsum("bqhd,bkhd->bhqk", query, key) / head_dim
    score2 = torch.einsum("bqhd,bkhd->bhqk", query2, key2) / head_dim
    causal = torch.tril(torch.ones(length, length, device=token_ids.device, dtype=torch.bool))
    pattern = (score1 * score2).masked_fill(~causal, 0)
    attended = torch.einsum("bhqk,bkhd->bqhd", pattern, value).reshape(batch, length, dimension)
    x = x + attention.c_proj(attended)
    return F.rms_norm(x, (dimension,))


@torch.inference_mode()
def exact_attention_forward(attention, normalized, value_bus, config, tt):
    """Checkpoint-faithful squared bilinear attention without einops dependency."""
    batch, length, dimension = normalized.shape
    heads, head_dim = config.n_head, dimension // config.n_head

    def projected(module):
        value = module(normalized).view(batch, length, heads, head_dim)
        value = F.rms_norm(value, (head_dim,))
        cosine, sine = attention.rotary(value)
        return tt.apply_rotary_emb(value, cosine, sine)

    query, key = projected(attention.c_q), projected(attention.c_k)
    query2, key2 = projected(attention.c_q2), projected(attention.c_k2)
    value = attention.c_v(normalized).view(batch, length, heads, head_dim)
    if value_bus is None:
        value_bus = value
    mixed_value = (1 - attention.lamb) * value + attention.lamb * value_bus.view_as(value)
    score1 = torch.einsum("bqhd,bkhd->bhqk", query, key) / head_dim
    score2 = torch.einsum("bqhd,bkhd->bhqk", query2, key2) / head_dim
    causal = torch.tril(torch.ones(length, length, device=normalized.device, dtype=torch.bool))
    pattern = (score1 * score2).masked_fill(~causal, 0)
    attended = torch.einsum("bhqk,bkhd->bqhd", pattern, mixed_value).reshape(
        batch, length, dimension,
    )
    return attention.c_proj(attended), value_bus


@torch.inference_mode()
def target_layer_inputs(embedding, blocks, token_ids: torch.Tensor, config, tt) -> torch.Tensor:
    """Exact normalized MLP input for the last loaded block."""
    dimension = config.n_embd
    x = F.rms_norm(embedding(token_ids), (dimension,))
    x0 = x
    value_bus = None
    for index, block in enumerate(blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = exact_attention_forward(
            block.attn, F.rms_norm(mixed, (dimension,)), value_bus, config, tt,
        )
        pre_mlp = mixed + attention_write
        if index == len(blocks) - 1:
            return F.rms_norm(pre_mlp, (dimension,))
        x = pre_mlp + block.mlp(F.rms_norm(pre_mlp, (dimension,)))
    raise ValueError("no blocks loaded")


def update_tails(values, global_indices, top_values, top_indices, largest: bool):
    candidate_values = torch.cat((top_values, values), dim=0)
    candidate_indices = torch.cat((top_indices, global_indices[:, None].expand(-1, values.shape[1])), dim=0)
    selected_values, selected_rows = torch.topk(candidate_values, top_values.shape[0], dim=0, largest=largest)
    selected_indices = torch.gather(candidate_indices, 0, selected_rows)
    return selected_values, selected_indices


@torch.inference_mode()
def collect_split(embedding, block, rows, class_lookup, args, config, tt, keep_tails: bool,
                  blocks=None) -> SplitStats:
    hidden = block.mlp.Left.out_features
    device = args.device
    names = ("left", "right", "product")
    sums = {name: torch.zeros(hidden, device=device) for name in names}
    squares = {name: torch.zeros(hidden, device=device) for name in names}
    class_sums = {name: torch.zeros(len(CLASSES), hidden, device=device) for name in names}
    class_signed_sums = {name: torch.zeros(len(CLASSES), hidden, device=device) for name in names}
    class_counts = torch.zeros(len(CLASSES), device=device)
    top_values = {name: torch.full((args.tail_size, hidden), -torch.inf, device=device) for name in names} if keep_tails else None
    bottom_values = {name: torch.full((args.tail_size, hidden), torch.inf, device=device) for name in names} if keep_tails else None
    top_indices = {name: torch.full((args.tail_size, hidden), -1, dtype=torch.long, device=device) for name in names} if keep_tails else None
    bottom_indices = {name: torch.full((args.tail_size, hidden), -1, dtype=torch.long, device=device) for name in names} if keep_tails else None
    count = 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size, :args.sequence_length].to(device)
        normalized = (
            layer_zero_inputs(embedding, block, batch, config, tt)
            if blocks is None
            else target_layer_inputs(embedding, blocks, batch, config, tt)
        )
        left = block.mlp.Left(normalized).float().reshape(-1, hidden)
        right = block.mlp.Right(normalized).float().reshape(-1, hidden)
        signals = {"left": left, "right": right, "product": left * right}
        labels = class_lookup[batch.long()].reshape(-1)
        one_hot = F.one_hot(labels, len(CLASSES)).float()
        class_counts += one_hot.sum(0)
        n = left.shape[0]
        global_indices = torch.arange(count, count + n, device=device)
        for name, values in signals.items():
            sums[name] += values.sum(0)
            squares[name] += values.square().sum(0)
            class_sums[name] += one_hot.T @ values.abs()
            class_signed_sums[name] += one_hot.T @ values
            if keep_tails:
                top_values[name], top_indices[name] = update_tails(
                    values, global_indices, top_values[name], top_indices[name], True,
                )
                bottom_values[name], bottom_indices[name] = update_tails(
                    values, global_indices, bottom_values[name], bottom_indices[name], False,
                )
        count += n
    return SplitStats(count, class_counts, class_sums, class_signed_sums, sums, squares,
                      top_values, top_indices, bottom_values, bottom_indices)


def metrics(stats: SplitStats, name: str):
    global_abs = stats.class_abs_sums[name].sum(0) / stats.class_counts.sum()
    class_means = stats.class_abs_sums[name] / stats.class_counts[:, None].clamp_min(1)
    lifts = class_means / global_abs[None, :].clamp_min(1e-8)
    winner = lifts.argmax(0)
    winning_lift = torch.gather(lifts, 0, winner[None, :]).squeeze(0)
    mean = stats.sums[name] / stats.count
    variance = stats.square_sums[name] / stats.count - mean.square()
    standard_deviation = variance.clamp_min(0).sqrt()
    signed_class_means = stats.class_signed_sums[name] / stats.class_counts[:, None].clamp_min(1)
    centered_class_means = (signed_class_means - mean[None, :]) / standard_deviation[None, :].clamp_min(1e-8)
    positive_class = centered_class_means.argmax(0)
    negative_class = centered_class_means.argmin(0)
    return winner, winning_lift, standard_deviation, lifts, positive_class, negative_class, signed_class_means


def escaped(text: str) -> str:
    return text.replace("\n", "\\n").replace("\t", "\\t")


def context_for(flat_index: int, rows: torch.Tensor, tokenizer, length: int, radius: int = 5):
    row, position = divmod(flat_index, length)
    ids = rows[row, :length]
    lo, hi = max(0, position - radius), min(length, position + radius + 1)
    before = tokenizer.decode(ids[lo:position].tolist())
    current = tokenizer.decode([int(ids[position])])
    after = tokenizer.decode(ids[position + 1:hi].tolist())
    return escaped(before + "[" + current + "]" + after)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    if args.corpus.suffix == ".npy":
        corpus = torch.from_numpy(np.load(args.corpus).astype(np.int64, copy=False))
    else:
        loaded = torch.load(args.corpus, map_location="cpu", weights_only=False)
        corpus = loaded["rows"] if isinstance(loaded, dict) else loaded
    required = args.discovery_rows + args.heldout_rows
    if required > len(corpus):
        raise ValueError(f"requested {required} rows but corpus contains {len(corpus)}")
    discovery_rows = corpus[:args.discovery_rows]
    heldout_rows = corpus[args.discovery_rows:required]

    vocab_text = [tokenizer.decode([index]) for index in range(tokenizer.vocab_size)]
    class_lookup = torch.tensor([classify_token(text) for text in vocab_text], device=args.device)
    embedding, blocks, config, tt = load_front(args.layer, args.device)
    block = blocks[-1]
    discovery = collect_split(
        embedding, block, discovery_rows, class_lookup, args, config, tt, True, blocks=blocks,
    )
    heldout = collect_split(
        embedding, block, heldout_rows, class_lookup, args, config, tt, False, blocks=blocks,
    )

    discovery_metrics = {name: metrics(discovery, name) for name in ("left", "right", "product")}
    heldout_metrics = {name: metrics(heldout, name) for name in ("left", "right", "product")}
    down_norm = block.mlp.Down.weight.detach().float().norm(dim=0)
    product_std = discovery_metrics["product"][2]
    use_score = product_std * down_norm
    active = use_score > torch.quantile(use_score, .5)

    left_class = discovery_metrics["left"][0]
    right_class = discovery_metrics["right"][0]
    product_class = discovery_metrics["product"][0]
    heldout_product_lifts = heldout_metrics["product"][3]
    frozen_heldout_lift = torch.gather(heldout_product_lifts, 0, product_class[None, :]).squeeze(0)
    discovery_lift = discovery_metrics["product"][1]
    product_positive_class = discovery_metrics["product"][4]
    product_negative_class = discovery_metrics["product"][5]
    heldout_signed_means = heldout_metrics["product"][6]
    heldout_std = heldout_metrics["product"][2]
    heldout_positive_mean = torch.gather(heldout_signed_means, 0, product_positive_class[None, :]).squeeze(0)
    heldout_negative_mean = torch.gather(heldout_signed_means, 0, product_negative_class[None, :]).squeeze(0)
    heldout_signed_gap = (heldout_positive_mean - heldout_negative_mean) / heldout_std.clamp_min(1e-8)

    # Half the examples emphasize causally usable channels; half emphasize unusually
    # coherent partitions.  Deduplicate while preserving rank order.
    use_candidates = torch.topk(use_score, args.examples).indices.tolist()
    coherent_score = frozen_heldout_lift * (use_score / use_score.median().clamp_min(1e-8)).sqrt()
    coherent_candidates = torch.topk(coherent_score, args.examples).indices.tolist()
    candidates = []
    for unit in use_candidates[:args.examples // 2] + coherent_candidates:
        if unit not in candidates:
            candidates.append(unit)
        if len(candidates) == args.examples:
            break

    units = []
    for unit in candidates:
        tails = {}
        for name in ("left", "right", "product"):
            positive = discovery.top_indices[name][:, unit].detach().cpu().tolist()
            negative = discovery.bottom_indices[name][:, unit].detach().cpu().tolist()
            tails[name] = {
                "positive_values": [round(float(value), 5) for value in discovery.top_values[name][:, unit].detach().cpu()],
                "positive_contexts": [context_for(index, discovery_rows, tokenizer, args.sequence_length) for index in positive],
                "negative_values": [round(float(value), 5) for value in discovery.bottom_values[name][:, unit].detach().cpu()],
                "negative_contexts": [context_for(index, discovery_rows, tokenizer, args.sequence_length) for index in negative],
            }
        units.append({
            "unit": unit,
            "left_class": CLASSES[int(left_class[unit])],
            "right_class": CLASSES[int(right_class[unit])],
            "product_class": CLASSES[int(product_class[unit])],
            "product_positive_class": CLASSES[int(product_positive_class[unit])],
            "product_negative_class": CLASSES[int(product_negative_class[unit])],
            "left_right_same": bool(left_class[unit] == right_class[unit]),
            "discovery_product_lift": round(float(discovery_lift[unit]), 4),
            "heldout_product_lift": round(float(frozen_heldout_lift[unit]), 4),
            "heldout_signed_class_gap_sd": round(float(heldout_signed_gap[unit]), 4),
            "product_std": round(float(product_std[unit]), 5),
            "down_norm": round(float(down_norm[unit]), 5),
            "use_score": round(float(use_score[unit]), 5),
            "tails": tails,
        })

    all_channels = []
    for unit in range(config.expansion_factor * config.n_embd):
        channel = {
            "unit": unit,
            "left_class": CLASSES[int(left_class[unit])],
            "right_class": CLASSES[int(right_class[unit])],
            "product_class": CLASSES[int(product_class[unit])],
            "product_positive_class": CLASSES[int(product_positive_class[unit])],
            "product_negative_class": CLASSES[int(product_negative_class[unit])],
            "heldout_product_lift": round(float(frozen_heldout_lift[unit]), 4),
            "heldout_signed_class_gap_sd": round(float(heldout_signed_gap[unit]), 4),
            "product_std": round(float(product_std[unit]), 5),
            "down_norm": round(float(down_norm[unit]), 5),
            "use_score": round(float(use_score[unit]), 5),
        }
        if args.all_tail_contexts:
            count = min(args.all_tail_contexts, args.tail_size)
            positive_indices = discovery.top_indices["product"][:count, unit].detach().cpu().tolist()
            negative_indices = discovery.bottom_indices["product"][:count, unit].detach().cpu().tolist()
            channel["positive_tail_values"] = [
                round(float(value), 5)
                for value in discovery.top_values["product"][:count, unit].detach().cpu()
            ]
            channel["positive_tail_contexts"] = [
                context_for(index, discovery_rows, tokenizer, args.sequence_length)
                for index in positive_indices
            ]
            channel["negative_tail_values"] = [
                round(float(value), 5)
                for value in discovery.bottom_values["product"][:count, unit].detach().cpu()
            ]
            channel["negative_tail_contexts"] = [
                context_for(index, discovery_rows, tokenizer, args.sequence_length)
                for index in negative_indices
            ]
        all_channels.append(channel)

    active_lifts = frozen_heldout_lift[active]
    active_gaps = heldout_signed_gap[active]
    summary = {
        "model": "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd",
        "parameters": 545_902_902,
        "layer": args.layer,
        "channels": config.expansion_factor * config.n_embd,
        "discovery_tokens": discovery.count,
        "heldout_tokens": heldout.count,
        "token_classes": list(CLASSES),
        "left_right_same_class_fraction_all": round(float((left_class == right_class).float().mean()), 4),
        "left_right_same_class_fraction_active_half": round(float((left_class[active] == right_class[active]).float().mean()), 4),
        "active_half_heldout_lift_median": round(float(active_lifts.median()), 4),
        "active_half_fraction_heldout_lift_ge_1_5": round(float((active_lifts >= 1.5).float().mean()), 4),
        "active_half_fraction_heldout_lift_ge_2": round(float((active_lifts >= 2).float().mean()), 4),
        "active_half_signed_gap_sd_median": round(float(active_gaps.median()), 4),
        "active_half_fraction_signed_gap_ge_1_sd": round(float((active_gaps >= 1).float().mean()), 4),
        "selected_units": units,
        "all_channels": all_channels,
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# Bilin18 layer-{args.layer} native MLP discriminator atlas", "",
        "This is a first-pass input-selectivity audit of all 4,608 native bilinear product channels. "
        f"Each channel computes `Left_j(x) * Right_j(x)` after attention {args.layer} and RMS normalization. "
        "A token-class label is selected on disjoint discovery rows and its absolute-activation lift is evaluated on held-out rows.", "",
        "## Aggregate", "",
        f"- Discovery tokens: {discovery.count:,}; held-out tokens: {heldout.count:,}.",
        f"- Left/right factors choose the same coarse class for {summary['left_right_same_class_fraction_all']:.1%} of all channels and {summary['left_right_same_class_fraction_active_half']:.1%} of the more usable half.",
        f"- Among the more usable half, median held-out class lift is {summary['active_half_heldout_lift_median']:.2f}; {summary['active_half_fraction_heldout_lift_ge_1_5']:.1%} exceed 1.5× and {summary['active_half_fraction_heldout_lift_ge_2']:.1%} exceed 2×.", "",
        f"- Their median held-out signed class-to-class gap is {summary['active_half_signed_gap_sd_median']:.2f} product standard deviations; {summary['active_half_fraction_signed_gap_ge_1_sd']:.1%} exceed one standard deviation.", "",
        "`Lift` is mean absolute product activation on the frozen preferred class divided by its mean over all held-out tokens. `Use score = product standard deviation × Down-column norm`; it is only a first-order availability score, not a causal effect on loss.", "",
        "## Representative native channels", "",
        "| unit | Left | Right | signed product separator | held-out gap (SD) | lift | strongest + / − contexts |",
        "|---:|---|---|---|---:|---:|---|",
    ]
    for unit in units[:20]:
        positive = "; ".join(f"`{text}`" for text in unit["tails"]["product"]["positive_contexts"][:2])
        negative = "; ".join(f"`{text}`" for text in unit["tails"]["product"]["negative_contexts"][:2])
        lines.append(
            f"| {unit['unit']} | {unit['left_class']} | {unit['right_class']} | "
            f"{unit['product_positive_class']} → + / {unit['product_negative_class']} → − | "
            f"{unit['heldout_signed_class_gap_sd']:.2f} | {unit['heldout_product_lift']:.2f} | "
            f"+ {positive}<br>− {negative} |"
        )
    lines += ["", "## Red-team limits", "",
              "- The ten token classes are deliberately coarse and incomplete. Low lift means only that this vocabulary failed, not that a channel is incoherent.",
              "- Selection and top contexts come from discovery rows; only the class lift is held out.",
              f"- Layer {args.layer} already includes attention-{args.layer} and all earlier blocks, so a channel can separate contextual states rather than current-token identity.",
              "- The signed separator names are frozen on discovery data and evaluated as a standardized class-mean gap on held-out data. They remain coarse descriptions, not exhaustive meanings.",
              "- Activation selectivity is descriptive. Finite channel ablations and cross-context transfer are required before assigning a causal verb.",
              "- Native product channels have bilinear scaling and swap gauges. Interpret Left/Right labels as a pair, not as independently canonical neurons."]
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("selected_units", "all_channels")}, indent=2))
    print(args.output / "README.md")


if __name__ == "__main__":
    main()
