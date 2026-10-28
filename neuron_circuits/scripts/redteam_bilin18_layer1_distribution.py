#!/usr/bin/env python3
"""Red-team the apparent distributed layer-1 channel requirement.

Tests whether the sharp small-subset failure is specific to the frozen ranking,
whether behavior changes smoothly as contextual variation is attenuated, and
whether a marginally realistic but context-mismatched layer-1 product vector is
less damaging than the positional mean.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audit_bilin18_layer1_channels import (
    evaluate_curve,
    forward,
    freeze,
    positional_product_mean,
    rows129,
)
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--mean-start", type=int, default=12000)
    parser.add_argument("--mean-rows", type=int, default=400)
    parser.add_argument("--eval-start", type=int, default=14000)
    parser.add_argument("--eval-rows", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--ks", default="512,1024,1536,2048,2560,3072,3584,4096")
    parser.add_argument("--random-seeds", default="11,29,47,83")
    parser.add_argument("--scales", default="0,.125,.25,.375,.5,.625,.75,.875,1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--ood-dir", type=Path, default=Path("artifacts/bilin18_ood_corpora"))
    parser.add_argument("--iteration1", type=Path,
                        default=Path("artifacts/bilin18_layer1_deep/iteration1_channels.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/bilin18_layer1_deep/iteration2_redteam.json"))
    return parser.parse_args()


@torch.inference_mode()
def evaluate_special(model, rows, mean, args, panel):
    rng = np.random.default_rng(20261019)
    permutation = rng.permutation(len(rows))
    scales = [float(value) for value in args.scales.split(",")]
    metrics = {f"scale_{scale:g}": {"nll": 0., "kl": 0., "same": 0, "n": 0}
               for scale in scales}
    metrics.update({
        "row_shuffle": {"nll": 0., "kl": 0., "same": 0, "n": 0},
        "position_shuffle": {"nll": 0., "kl": 0., "same": 0, "n": 0},
    })
    baseline_nll = 0.
    products = []
    # Collect intact layer-1 products.  They are kept on CPU in float16; each
    # shuffled intervention uses real joint channel vectors rather than an
    # independently sampled Gaussian or per-channel marginal.
    for start in range(0, len(rows), args.batch_size):
        raw = rows[start:start + args.batch_size]
        ids = torch.as_tensor(raw[:, :-1], device=args.device)
        _, leaf = forward(model, ids, args.layer, capture_product=True)
        products.append(leaf.detach().to("cpu", torch.float16))
    products = torch.cat(products)
    position_permutation = rng.permutation(args.sequence_length)
    for start in range(0, len(rows), args.batch_size):
        raw = rows[start:start + args.batch_size]
        ids = torch.as_tensor(raw[:, :-1], device=args.device)
        labels = torch.as_tensor(raw[:, -1], device=args.device)
        native, _ = forward(model, ids, args.layer)
        native_logp = F.log_softmax(native, -1)
        native_p = native_logp.exp()
        baseline_nll += float(F.nll_loss(native_logp, labels, reduction="sum"))
        variants = {}
        for scale in scales:
            variants[f"scale_{scale:g}"], _ = forward(
                model, ids, args.layer, mean, replacement_scale=scale)
        donor_indices = permutation[start:start + len(raw)]
        donor = products[donor_indices].to(args.device)
        variants["row_shuffle"], _ = forward(
            model, ids, args.layer, product_override=donor)
        # Same example, wrong position: preserves each document's overall style
        # and exact product-vector manifold while breaking token alignment.
        wrong_position = products[start:start + len(raw), position_permutation].to(args.device)
        variants["position_shuffle"], _ = forward(
            model, ids, args.layer, product_override=wrong_position)
        for name, logits in variants.items():
            logp = F.log_softmax(logits, -1)
            item = metrics[name]
            item["nll"] += float(F.nll_loss(logp, labels, reduction="sum"))
            item["kl"] += float((native_p * (native_logp - logp)).sum())
            item["same"] += int((native.argmax(-1) == logits.argmax(-1)).sum())
            item["n"] += len(raw)
        if (start // args.batch_size + 1) % 16 == 0:
            print(f"{panel} context controls: {start + len(raw)}/{len(rows)}", flush=True)
    baseline_ce = baseline_nll / len(rows)
    return {
        "examples": len(rows),
        "baseline_ce": baseline_ce,
        "conditions": {
            name: {
                "ce_delta": item["nll"] / item["n"] - baseline_ce,
                "kl_from_native": item["kl"] / item["n"],
                "top1_agreement": item["same"] / item["n"],
            }
            for name, item in metrics.items()
        },
    }


def summarize_random(replicates):
    result = {}
    for panel in replicates[0]:
        result[panel] = {}
        for mode in ("necessity", "sufficiency"):
            by_k = {}
            for replicate in replicates:
                for row in replicate[panel][mode]:
                    by_k.setdefault(row["k"], []).append(row)
            result[panel][mode] = [{
                "k": k,
                "ce_delta_mean": float(np.mean([x["ce_delta"] for x in rows])),
                "ce_delta_min": float(np.min([x["ce_delta"] for x in rows])),
                "ce_delta_max": float(np.max([x["ce_delta"] for x in rows])),
                "kl_mean": float(np.mean([x["kl_from_native"] for x in rows])),
                "top1_agreement_mean": float(np.mean([x["top1_agreement"] for x in rows])),
            } for k, rows in sorted(by_k.items())]
    return result


def main():
    args = parse_args()
    started = time.time()
    corpus = np.load(args.corpus, mmap_mode="r")
    mean_rows = rows129(corpus, args.mean_start, args.mean_rows, args.sequence_length)
    panels = {
        "fineweb": rows129(corpus, args.eval_start, args.eval_rows, args.sequence_length),
        "wikitext103": rows129(np.load(args.ood_dir / "wikitext103_1200x129.npy", mmap_mode="r"),
                                300, args.eval_rows, args.sequence_length),
        "tinystories": rows129(np.load(args.ood_dir / "tinystories_1200x129.npy", mmap_mode="r"),
                                300, args.eval_rows, args.sequence_length),
    }
    model = load_full_model(args.device)
    freeze(model)
    mean = positional_product_mean(model, mean_rows, args)
    iteration1 = json.loads(args.iteration1.read_text())
    ranked_order = torch.tensor(
        np.argsort(iteration1["fineweb_absolute_scores"])[::-1].copy(), device=args.device)
    ranked = {panel: evaluate_curve(model, rows, mean, ranked_order, args, panel)
              for panel, rows in panels.items()}
    random_replicates = []
    for seed in [int(value) for value in args.random_seeds.split(",")]:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(ranked_order), generator=generator).to(args.device)
        print(f"random subset seed {seed}", flush=True)
        random_replicates.append({
            panel: evaluate_curve(model, rows, mean, order, args, f"{panel}/random-{seed}")
            for panel, rows in panels.items()
        })
    context_controls = {
        panel: evaluate_special(model, rows, mean, args, panel)
        for panel, rows in panels.items()
    }
    result = {
        "ranked_subset_curves": ranked,
        "random_subset_summary": summarize_random(random_replicates),
        "random_subset_replicates": random_replicates,
        "context_controls": context_controls,
        "elapsed_seconds": time.time() - started,
        "notes": [
            "Whole-vector row shuffling preserves the observed joint channel vector at each position but mismatches it to the residual stream.",
            "Position shuffling preserves the example and observed joint vectors but assigns them to the wrong token positions.",
            "Neither shuffle is guaranteed to be on-manifold conditional on the receiving residual stream; both are stronger controls than a constant mean.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(args.output), "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
