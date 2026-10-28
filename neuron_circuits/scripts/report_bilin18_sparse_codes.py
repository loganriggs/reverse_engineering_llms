#!/usr/bin/env python3
"""Report and red-team sparse-code alternatives for the Bilin18 layer-1→2 path."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from audit_bilin18_mlp_discriminators import SNAPSHOT
from train_bilin18_sparse_codes import SPLITS, TopKAutoencoder, fidelity


ROOT = Path("artifacts/bilin18_sparse_codes")


def fmt(value):
    return "nan" if not np.isfinite(value) else f"{value:.3f}"


@torch.inference_mode()
def derived_checks(data):
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    left = state["transformer.h.2.mlp.Left.weight"].float()
    right = state["transformer.h.2.mlp.Right.weight"].float()
    del state

    train_delta = torch.from_numpy(
        data["causal__fineweb_train__input_delta"].astype(np.float32))
    scale = train_delta.std(0).clamp_min(1e-4)
    sae = TopKAutoencoder(1152, 2048, 32, 3)
    sae.load_state_dict(torch.load(ROOT / "causal_delta_sae.pt", map_location="cpu",
                                   weights_only=True))
    sae.eval()
    derived = {"exact_equation": {}, "sae_by_variant": {}, "target_pca": {}}
    for split in SPLITS:
        context = torch.from_numpy(data[f"causal__{split}__context"].astype(np.float32))
        delta = torch.from_numpy(data[f"causal__{split}__input_delta"].astype(np.float32))
        target = torch.from_numpy(data[f"causal__{split}__target_delta"].astype(np.float32))
        dl, dr = delta @ left.T, delta @ right.T
        prediction = dl * (context @ right.T) + (context @ left.T) * dr - dl * dr
        derived["exact_equation"][split] = fidelity(target, prediction)

        reconstruction, _ = sae(delta / scale, 32)
        reconstructed_delta = reconstruction * scale
        rdl, rdr = reconstructed_delta @ left.T, reconstructed_delta @ right.T
        product = (rdl * (context @ right.T) + (context @ left.T) * rdr - rdl * rdr)
        variants = data[f"causal__{split}__variant"]
        derived["sae_by_variant"][split] = {}
        for variant, name in enumerate(("top1_remove", "top4_remove", "top16_remove",
                                        "quiet8_add_2z")):
            mask = torch.from_numpy(variants == variant)
            derived["sae_by_variant"][split][name] = fidelity(target[mask], product[mask])

    # Does a fixed dense output subspace work? It is a control for the claim that
    # the sparse transcoders fail only because their dictionaries are sparse.
    train_target = torch.from_numpy(
        data["causal__fineweb_train__target_delta"].astype(np.float32))
    target_scale = train_target.std(0).clamp_min(1e-4)
    _, _, vectors = torch.pca_lowrank(
        (train_target / target_scale).cuda(), q=512, center=False, niter=4)
    vectors = vectors.cpu()
    for rank in (4, 16, 32, 128, 512):
        basis = vectors[:, :rank]
        derived["target_pca"][str(rank)] = {}
        for split in SPLITS:
            target = torch.from_numpy(
                data[f"causal__{split}__target_delta"].astype(np.float32))
            standardized = target / target_scale
            prediction = ((standardized @ basis) @ basis.T) * target_scale
            derived["target_pca"][str(rank)][split] = fidelity(target, prediction)
    return derived


def plot(results, derived, large):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ks = np.asarray([32, 128, 512])
    ax = axes[0, 0]
    native = [results["activation_sae"]["native"][str(k)]["fineweb_test"]
              ["write"]["median_fidelity"] for k in ks]
    ax.plot(ks, native, "o-", linewidth=2.3, label="native neurons")
    for seed in ("1", "2"):
        values = [results["activation_sae"]["seeds"][seed]["write_reconstruction"]
                  [str(k)]["fineweb_test"]["median_fidelity"] for k in ks]
        ax.plot(ks, values, "o--", label=f"SAE seed {seed}")
    large_values = [large["seeds"][seed]["evaluation"]["fineweb_test"]
                    ["write"]["median_fidelity"] for seed in ("101", "102")]
    ax.scatter([128, 128], large_values, marker="*", s=150, zorder=5,
               label="large-data SAE (2 seeds)")
    ax.set(title="Natural layer-1 write", xlabel="retained channels/features",
           ylabel="median explained energy", xscale="log")
    ax.legend()

    ax = axes[0, 1]
    ks2 = np.asarray([4, 16, 32, 128])
    for name, label, style in (("ranks", "causal-delta SAE", "o-"),
                               ("pca", "PCA", "s--"),
                               ("native", "native coordinates", "^--")):
        source = (results["causal_delta_sae"][name] if name == "ranks" else
                  results["causal_delta_sae"]["baselines"][name])
        values = [source[str(k)]["fineweb_test"]["product"]["median_fidelity"]
                  for k in ks2]
        ax.plot(ks2, values, style, linewidth=2, label=label)
    ax.set(title="Causal layer-2 effect from compressed input delta",
           xlabel="rank / active features", ylabel="median explained energy", xscale="log")
    ax.legend()

    ax = axes[1, 0]
    names = list(results["transcoders"])
    labels = ["source", "source + ctx", "input + ctx", "bilinear\ninput × ctx"]
    x = np.arange(len(names))
    width = .25
    for offset, split in enumerate(SPLITS):
        values = [results["transcoders"][name]["transfer"][split]
                  ["median_fidelity"] for name in names]
        ax.bar(x + (offset - 1) * width, values, width, label=split)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(title="Sparse transcoder transfer (K=32)", ylabel="median explained energy",
           xticks=x, xticklabels=labels)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    native_categories = large["native_categories"]
    sae_categories = large["seeds"]["101"]["categories"]
    categories = sorted(set(native_categories) & set(sae_categories))
    nx, sy = [], []
    for category in categories:
        n = native_categories[category]["auc"]["fineweb_test"]
        s = sae_categories[category]["auc"]["fineweb_test"]
        if np.isfinite(n) and np.isfinite(s):
            nx.append(n); sy.append(s)
            ax.annotate(category.replace("suffix_", "-"), (n, s), fontsize=7,
                        xytext=(3, 2), textcoords="offset points")
    ax.scatter(nx, sy, s=34)
    ax.plot([.45, 1], [.45, 1], "k--", linewidth=1)
    ax.set(xlim=(.45, 1.02), ylim=(.45, 1.02), xlabel="native best-neuron AUC",
           ylabel="SAE best-feature AUC", title="Held-out coarse semantic selectivity")

    fig.suptitle("Bilin18 sparse codes: compression improves before features stabilize", fontsize=14)
    fig.tight_layout()
    fig.savefig(ROOT / "sparse_code_summary.png", dpi=180)
    plt.close(fig)


def report(results, derived, large):
    native128 = results["activation_sae"]["native"]["128"]["fineweb_test"]["write"]
    sae128 = results["activation_sae"]["seeds"]["1"]["write_reconstruction"]["128"]
    stability = results["activation_sae"]["stability"]
    causal = results["causal_delta_sae"]["ranks"]["32"]["fineweb_test"]["product"]
    causal_native = results["causal_delta_sae"]["baselines"]["native"]["32"] \
        ["fineweb_test"]["product"]
    causal_pca = results["causal_delta_sae"]["baselines"]["pca"]["32"] \
        ["fineweb_test"]["product"]
    exact = derived["exact_equation"]["fineweb_test"]
    pca512 = derived["target_pca"]["512"]["fineweb_test"]
    large_native = large["native"]["fineweb_test"]["write"]
    large_sae = large["seeds"]["101"]["evaluation"]["fineweb_test"]["write"]
    large_stability = large["stability"]

    transfer_rows = []
    for name, row in results["transcoders"].items():
        transfer_rows.append(
            f"| `{name}` | " + " | ".join(
                fmt(row["transfer"][split]["median_fidelity"]) for split in SPLITS) + " |")
    variant_rows = []
    for name, row in derived["sae_by_variant"]["fineweb_test"].items():
        variant_rows.append(f"| {name} | {fmt(row['median_fidelity'])} | "
                            f"{fmt(row['median_cosine'])} |")

    text = f"""# Sparse codes for the Bilin18 layer-1→2 path

## Bottom line

The result is **compression without a canonical concept basis**. The 8,192-position
pilot initially favored native neurons. After scaling to 131,072 training positions,
a Top-K SAE with 128 active features explains **{large_sae['median_fidelity']:.1%}**
of the layer-1 downstream write energy, versus **{large_native['median_fidelity']:.1%}**
for the best 128 native neurons. So a learned sparse oblique basis is genuinely more
efficient at reconstruction.

However, two SAE seeds achieve essentially identical fidelity while learning very
different dictionaries: median best decoder cosine is only
**{large_stability['median_best_cosine']:.3f}**, and the fraction above 0.8 is
**{large_stability['fraction_above_0_8']:.1%}**. Native neurons also retain cleaner
known morphology controls. The SAE has found interchangeable compression coordinates,
not a unique list of neuron-like concepts.

There is nevertheless a useful intermediate code. A Top-K SAE of the *causal
post-attention input change* explains **{causal['median_fidelity']:.1%}** of the
next bilinear MLP's finite product change with 32 features. That beats 32 native
input coordinates ({causal_native['median_fidelity']:.1%}) and rank-32 PCA
({causal_pca['median_fidelity']:.1%}), but it is far from a circuit-level
replacement. It is a lossy causal coordinate system, not a solved interpretation.

The weight-derived equation is the decisive positive control: it predicts the
saved finite layer-2 product change with median explained energy
**{exact['median_fidelity']:.6f}** and cosine **{exact['median_cosine']:.6f}**.
Thus we understand the local computation exactly, while failing to compress its
many context-dependent outcomes into a small fixed dictionary. Even a rank-512
output PCA subspace explains only **{pca512['median_fidelity']:.1%}** on held-out
FineWeb.

![Sparse-code summary](sparse_code_summary.png)

## What was compared

- Natural layer-1 product activations: native coordinate Top-K versus a 6,144-feature
  Top-K SAE trained at K=128, with two seeds.
- Causal input changes entering layer 2: native-coordinate Top-K, PCA, and a
  2,048-feature Top-K SAE, evaluated by the *finite downstream bilinear effect*.
- Sparse K=32 transcoders: layer-1 source only, source plus context, post-attention
  input plus context, and an interaction-aware transcoder with explicit
  `input_delta × context` gates.
- FineWeb holdout and two corpus shifts: WikiText-103 and TinyStories.

All explained-energy numbers use `1 - ||target-prediction||² / ||target||²` per
example, then take the median. Zero means predicting no change; negative means
worse than predicting no change. This is stricter than cosine because it penalizes
the wrong magnitude.

## Natural activation result

### Large-data replication (primary)

| 128-feature comparison | FineWeb | WikiText-103 | TinyStories |
|---|---:|---:|---:|
| Native neurons, downstream write | {fmt(large['native']['fineweb_test']['write']['median_fidelity'])} | {fmt(large['native']['wikitext103']['write']['median_fidelity'])} | {fmt(large['native']['tinystories']['write']['median_fidelity'])} |
| SAE seed 101, downstream write | {fmt(large['seeds']['101']['evaluation']['fineweb_test']['write']['median_fidelity'])} | {fmt(large['seeds']['101']['evaluation']['wikitext103']['write']['median_fidelity'])} | {fmt(large['seeds']['101']['evaluation']['tinystories']['write']['median_fidelity'])} |

Both SAE seeds agree within 0.1 percentage point on all three corpora. Their feature
directions do not agree. This is the central multiplexing result: the represented
subspace/computation is reproducible, but its sparse factorization is not.

For known token morphology, the best native channel remains better on held-out
FineWeb: `-ing` AUC is 0.924 native versus 0.851 SAE, and `-ed` is 0.952 versus
0.829. The SAE improves `newline` (0.911 versus 0.845), so neither basis dominates
every semantic partition.

### Small-data pilot (red-team history)

| 128-feature comparison | FineWeb | WikiText-103 | TinyStories |
|---|---:|---:|---:|
| Native neurons, downstream write | {fmt(native128['median_fidelity'])} | {fmt(results['activation_sae']['native']['128']['wikitext103']['write']['median_fidelity'])} | {fmt(results['activation_sae']['native']['128']['tinystories']['write']['median_fidelity'])} |
| SAE seed 1, downstream write | {fmt(sae128['fineweb_test']['median_fidelity'])} | {fmt(sae128['wikitext103']['median_fidelity'])} | {fmt(sae128['tinystories']['median_fidelity'])} |

This pilot conclusion does not survive the larger-data reconstruction test: the SAE
reverses the write-fidelity ordering. Keeping the pilot visible is important because
it demonstrates how an undertrained SAE can falsely make a native basis look optimal.

## Causal compression result

At K=32 on held-out FineWeb:

| Representation of post-attention input change | Input explained energy | Finite product-effect explained energy |
|---|---:|---:|
| PCA | {fmt(results['causal_delta_sae']['baselines']['pca']['32']['fineweb_test']['input']['median_fidelity'])} | {fmt(causal_pca['median_fidelity'])} |
| Native coordinates | {fmt(results['causal_delta_sae']['baselines']['native']['32']['fineweb_test']['input']['median_fidelity'])} | {fmt(causal_native['median_fidelity'])} |
| Causal-delta SAE | {fmt(results['causal_delta_sae']['ranks']['32']['fineweb_test']['input']['median_fidelity'])} | {fmt(causal['median_fidelity'])} |

The SAE advantage over coordinate sparsity is real but modest. It says causal
changes are somewhat more compressible in a learned oblique basis. It does not say
that the resulting features have stable meanings.

| Intervention family | K=32 product fidelity | Product cosine |
|---|---:|---:|
{chr(10).join(variant_rows)}

## Transcoder result

| K=32 model | FineWeb | WikiText-103 | TinyStories |
|---|---:|---:|---:|
{chr(10).join(transfer_rows)}

The interaction-aware model is less wrong than the additive models, but none
generalizes. This is consistent with severe undersampling and a genuinely broad
context-dependent output family. It is **not** evidence that bilinear sparse
features cannot work with more data or a weight-tied construction.

## What this means for multiplexed codes

The useful decomposition is now clearer:

1. **Native discriminators:** layer-1 bilinear neurons are already sparse and often
   semantically legible. The SAE compresses their aggregate write more efficiently,
   but does not replace their stable individual identities.
2. **A low-dimensional transported perturbation:** earlier canonicalization found
   that one post-attention input direction captures much of a particular causal
   perturbation family. The causal-delta SAE extends this into several sparse
   perturbation directions.
3. **A context-indexed expansion:** the next bilinear layer turns those directions
   into many different output-channel changes depending on the base context. This
   family is high-rank across examples. A single small, context-independent output
   dictionary is therefore the wrong explanatory object.

So the multiplexing is not best described as “one SAE feature equals one hidden
concept.” The SAE result is compatible with many equally good sparse rotations.
A better causal object is a **conditional operator**: a sparse upstream event,
the context directions it interacts with, and the native downstream discriminators
that receive the product. This is closer to a sparse tensor factorization or a
weight-tied bilinear transcoder than a vanilla SAE.

## Red-team limits

- The primary natural SAE saw 131,072 positions and 512,000 sampled training
  exposures for roughly 57M dictionary parameters. This resolves the grossly
  undertrained pilot, but a million-token run, validation-based checkpointing, and
  a feature-count/sparsity sweep are still warranted.
- The SAE was trained at K=128. K=32 and K=512 curves are diagnostic extrapolations;
  the matched claim is the K=128 comparison.
- Coarse token-label AUC is a narrow semantic audit with multiple feature search.
  It is useful for known controls such as suffixes, not a general monosemanticity
  score.
- The transcoders have millions of parameters but only 4,096 causal training
  examples. Their negative held-out fidelity diagnoses this implementation, not
  the whole method class.
- The interventions are deliberate removals/additions, not the full distribution
  of naturally occurring upstream differences.
- Exact local algebra does not yet provide a short global account of which circuits
  matter for model loss or behavior.

## Next decisive experiment

Fit a **weight-tied conditional dictionary**:
reuse the actual layer-2 Left/Right directions, cluster or factor their input pairs,
and learn only a sparse router over upstream perturbations and contexts. Evaluate
the router by held-out finite effects and cross-corpus loss recovery. In parallel,
increase the natural SAE to roughly one million positions and test whether feature
instability persists under feature-count, K, regularization, and seed sweeps. The
present evidence says SAE compression is real; the unresolved question is whether
any data/regularization regime makes its individual features canonical.
"""
    (ROOT / "REPORT.md").write_text(text)


def main():
    data = np.load(ROOT / "data.npz")
    results = json.loads((ROOT / "results.json").read_text())
    large = json.loads((ROOT / "large_sae_results.json").read_text())
    derived = derived_checks(data)
    (ROOT / "derived_results.json").write_text(json.dumps(derived, indent=2))
    plot(results, derived, large)
    report(results, derived, large)
    print(f"wrote {ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
