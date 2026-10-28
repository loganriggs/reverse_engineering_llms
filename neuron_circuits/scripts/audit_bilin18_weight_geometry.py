#!/usr/bin/env python3
"""Weight-first census of native bilinear MLP neurons in bilin18 layers 0--2.

The input side folds the normalized GPT-2 embedding table through each native
Left/Right pair.  The output side contracts each Down column with the next
block's QK, V, and bilinear-factor readers, plus the final unembedding.  These
are availability measurements: they say what distinctions and consumers the
weights make possible before activation frequency or causal use is considered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import CLASSES, SNAPSHOT, TOKENIZER, classify_token


MODEL_REPO = Path("/home/loganriggs/Coding/reverse_engineer/tensor_language")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", default="0,1,2")
    parser.add_argument("--vocab-chunk", type=int, default=512)
    parser.add_argument("--tail-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/bilin18_layers_0_2/weight_geometry"),
    )
    return parser.parse_args()


def load_weights(layers, device):
    sys.path.insert(0, str(MODEL_REPO))
    import jacclust.tt_model as tt

    cfg_dict = json.loads((SNAPSHOT / "config.json").read_text())
    cfg_dict.pop("step", None)
    cfg = tt.GPTConfig(**cfg_dict)
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    embedding = F.rms_norm(state["transformer.wte.weight"].float(), (cfg.n_embd,))
    unembedding = state["lm_head.weight"].float()
    out = {}
    for layer in layers:
        prefix = f"transformer.h.{layer}.mlp."
        out[layer] = {
            "left": state[prefix + "Left.weight"].float(),
            "right": state[prefix + "Right.weight"].float(),
            "down": state[prefix + "Down.weight"].float(),
        }
        if layer + 1 < cfg.n_layer:
            next_prefix = f"transformer.h.{layer + 1}."
            out[layer]["next"] = {
                "qk": torch.cat([
                    state[next_prefix + f"attn.{name}.weight"].float()
                    for name in ("c_q", "c_k", "c_q2", "c_k2")
                ], 0),
                "v": state[next_prefix + "attn.c_v.weight"].float(),
                "mlp_left": state[next_prefix + "mlp.Left.weight"].float(),
                "mlp_right": state[next_prefix + "mlp.Right.weight"].float(),
            }
    del state
    return embedding.to(device=device, dtype=dtype), unembedding, out, cfg


def update_extreme(values, token_ids, saved_values, saved_ids, largest):
    ids = token_ids[:, None].expand_as(values)
    candidate_values = torch.cat((saved_values, values), 0)
    candidate_ids = torch.cat((saved_ids, ids), 0)
    selected_values, selected_rows = torch.topk(
        candidate_values, saved_values.shape[0], dim=0, largest=largest,
    )
    return selected_values, torch.gather(candidate_ids, 0, selected_rows)


@torch.inference_mode()
def embedding_fold(embedding, left, right, class_lookup, args):
    units = left.shape[0]
    device = embedding.device
    count = 0
    sums = {name: torch.zeros(units, device=device) for name in ("left", "right", "product")}
    squares = {name: torch.zeros(units, device=device) for name in sums}
    class_abs = {name: torch.zeros(len(CLASSES), units, device=device) for name in sums}
    class_signed = {name: torch.zeros(len(CLASSES), units, device=device) for name in sums}
    class_counts = torch.zeros(len(CLASSES), device=device)
    top_v = torch.full((args.tail_size, units), -torch.inf, device=device)
    top_i = torch.full((args.tail_size, units), -1, dtype=torch.long, device=device)
    low_v = torch.full((args.tail_size, units), torch.inf, device=device)
    low_i = torch.full((args.tail_size, units), -1, dtype=torch.long, device=device)
    for start in range(0, embedding.shape[0], args.vocab_chunk):
        ids = torch.arange(start, min(start + args.vocab_chunk, embedding.shape[0]), device=device)
        emb = embedding[ids]
        lval = F.linear(emb, left.to(device=device, dtype=emb.dtype)).float()
        rval = F.linear(emb, right.to(device=device, dtype=emb.dtype)).float()
        signals = {"left": lval, "right": rval, "product": lval * rval}
        labels = class_lookup[ids]
        one_hot = F.one_hot(labels, len(CLASSES)).float()
        class_counts += one_hot.sum(0)
        for name, values in signals.items():
            sums[name] += values.sum(0)
            squares[name] += values.square().sum(0)
            class_abs[name] += one_hot.T @ values.abs()
            class_signed[name] += one_hot.T @ values
        top_v, top_i = update_extreme(signals["product"], ids, top_v, top_i, True)
        low_v, low_i = update_extreme(signals["product"], ids, low_v, low_i, False)
        count += len(ids)

    metrics = {}
    for name in sums:
        mean = sums[name] / count
        sd = (squares[name] / count - mean.square()).clamp_min(0).sqrt()
        abs_mean = class_abs[name].sum(0) / count
        class_abs_mean = class_abs[name] / class_counts[:, None].clamp_min(1)
        lift = class_abs_mean / abs_mean[None].clamp_min(1e-8)
        winner = lift.argmax(0)
        signed_mean = class_signed[name] / class_counts[:, None].clamp_min(1)
        centered = (signed_mean - mean[None]) / sd[None].clamp_min(1e-8)
        metrics[name] = {
            "mean": mean,
            "sd": sd,
            "winner": winner,
            "lift": torch.gather(lift, 0, winner[None]).squeeze(0),
            "positive": centered.argmax(0),
            "negative": centered.argmin(0),
            "signed_gap_sd": centered.max(0).values - centered.min(0).values,
        }
    return metrics, top_v, top_i, low_v, low_i


@torch.inference_mode()
def consumer_scores(down, next_weights, unembedding, device, vocab_chunk):
    # down: [D, U]; every column is one channel's fixed write direction.
    d = down.to(device)
    dnorm = d.norm(dim=0).clamp_min(1e-8)
    scores = {"down_norm": dnorm}
    for name, reader in next_weights.items():
        response = reader.to(device) @ d
        scores[f"next_{name}_read"] = response.norm(dim=0) / dnorm
        del response
    # Direct logit-lens norm, streamed over the padded vocabulary.
    square_sum = torch.zeros(d.shape[1], device=device)
    for start in range(0, unembedding.shape[0], vocab_chunk):
        reader = unembedding[start:start + vocab_chunk].to(device)
        square_sum += (reader @ d).square().sum(0)
    scores["unembedding_read"] = square_sum.sqrt() / dnorm
    return scores


def summary(values, active):
    selected = values[active]
    return {
        "median_active_half": round(float(selected.median()), 5),
        "q10_active_half": round(float(selected.quantile(.1)), 5),
        "q90_active_half": round(float(selected.quantile(.9)), 5),
    }


def main():
    args = parse_args()
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    embedding, unembedding, weights, cfg = load_weights(layers, args.device)
    vocab_size = tokenizer.vocab_size
    embedding = embedding[:vocab_size]
    vocab_text = [tokenizer.decode([index]) for index in range(vocab_size)]
    class_lookup = torch.tensor([classify_token(text) for text in vocab_text], device=args.device)
    results = {
        "model": "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd",
        "layers": layers,
        "vocab_size": vocab_size,
        "classes": list(CLASSES),
        "layer_results": [],
    }
    for layer in layers:
        layer_weights = weights[layer]
        left = layer_weights["left"]
        right = layer_weights["right"]
        down = layer_weights["down"]
        folded, top_v, top_i, low_v, low_i = embedding_fold(
            embedding, left, right, class_lookup, args,
        )
        consumers = consumer_scores(
            down, layer_weights["next"], unembedding, args.device, args.vocab_chunk,
        )
        left_d = left.to(args.device)
        right_d = right.to(args.device)
        factor_cosine = (left_d * right_d).sum(1) / (
            left_d.norm(dim=1) * right_d.norm(dim=1)
        ).clamp_min(1e-8)
        use = folded["product"]["sd"] * consumers["down_norm"]
        active = use >= torch.quantile(use, .5)
        channels = []
        for unit in range(left.shape[0]):
            channels.append({
                "unit": unit,
                "factor_cosine": round(float(factor_cosine[unit]), 6),
                "embedding_left_class": CLASSES[int(folded["left"]["winner"][unit])],
                "embedding_right_class": CLASSES[int(folded["right"]["winner"][unit])],
                "embedding_product_class": CLASSES[int(folded["product"]["winner"][unit])],
                "embedding_positive_class": CLASSES[int(folded["product"]["positive"][unit])],
                "embedding_negative_class": CLASSES[int(folded["product"]["negative"][unit])],
                "embedding_lift": round(float(folded["product"]["lift"][unit]), 5),
                "embedding_signed_gap_sd": round(float(folded["product"]["signed_gap_sd"][unit]), 5),
                "embedding_product_sd": round(float(folded["product"]["sd"][unit]), 5),
                "down_norm": round(float(consumers["down_norm"][unit]), 5),
                "use_score": round(float(use[unit]), 5),
                "next_qk_read": round(float(consumers["next_qk_read"][unit]), 5),
                "next_v_read": round(float(consumers["next_v_read"][unit]), 5),
                "next_mlp_left_read": round(float(consumers["next_mlp_left_read"][unit]), 5),
                "next_mlp_right_read": round(float(consumers["next_mlp_right_read"][unit]), 5),
                "unembedding_read": round(float(consumers["unembedding_read"][unit]), 5),
                "top_embedding_tokens": [vocab_text[int(index)] for index in top_i[:, unit]],
                "top_embedding_values": [round(float(value), 5) for value in top_v[:, unit]],
                "bottom_embedding_tokens": [vocab_text[int(index)] for index in low_i[:, unit]],
                "bottom_embedding_values": [round(float(value), 5) for value in low_v[:, unit]],
            })
        result = {
            "layer": layer,
            "channels": channels,
            "aggregate": {
                "left_right_same_class_all": round(float((folded["left"]["winner"] == folded["right"]["winner"]).float().mean()), 5),
                "left_right_same_class_active_half": round(float((folded["left"]["winner"][active] == folded["right"]["winner"][active]).float().mean()), 5),
                "embedding_lift": summary(folded["product"]["lift"], active),
                "embedding_signed_gap_sd": summary(folded["product"]["signed_gap_sd"], active),
                "factor_cosine": summary(factor_cosine, active),
                **{name: summary(values, active) for name, values in consumers.items()},
            },
        }
        results["layer_results"].append(result)
        print(json.dumps({"layer": layer, **result["aggregate"]}, indent=2), flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    print(args.output / "results.json")


if __name__ == "__main__":
    main()
