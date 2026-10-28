#!/usr/bin/env python3
"""Aggregate and red-team bilinear SAE / sparse bilinear layer results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from audit_bilin18_mlp_discriminators import SNAPSHOT
from train_bilin18_bilinear_codes import (ROOT, SparseBilinearDictionary,
                                           evaluate_replacement)
from train_bilin18_sparse_codes import SPLITS


SPARSE_ROOT = Path("artifacts/bilin18_sparse_codes")


def load_dictionary(path, input_dim, features, output_dim, k):
    model = SparseBilinearDictionary(input_dim, features, output_dim, k, 0)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def best_form_matches(model, native_left, native_right, chunk=64):
    a = F.normalize(model.left.detach().float().cpu(), dim=1)
    b = F.normalize(model.right.detach().float().cpu(), dim=1)
    c = F.normalize(native_left.float().cpu(), dim=1)
    d = F.normalize(native_right.float().cpu(), dim=1)
    norm1 = (.5 * (1 + (a * b).sum(1).square())).sqrt()
    norm2 = (.5 * (1 + (c * d).sum(1).square())).sqrt()
    maxima = []
    for start in range(0, len(a), chunk):
        similarity = .5 * ((a[start:start + chunk] @ c.T) *
                           (b[start:start + chunk] @ d.T) +
                           (a[start:start + chunk] @ d.T) *
                           (b[start:start + chunk] @ c.T))
        similarity = similarity.abs() / (norm1[start:start + chunk, None] * norm2[None])
        maxima.append(similarity.max(1).values)
    values = torch.cat(maxima)
    return {"median": float(values.median()), "mean": float(values.mean()),
            "fraction_above_0_8": float((values > .8).float().mean()),
            "fraction_above_0_95": float((values > .95).float().mean())}


@torch.inference_mode()
def usage(model, values, batch_size=256):
    counts = torch.zeros(len(model.left), dtype=torch.float64)
    model.cuda()
    for start in range(0, len(values), batch_size):
        code = model.encode(values[start:start + batch_size].cuda())
        counts += (code != 0).sum(0).cpu()
    total = counts.sum()
    effective = float(total.square() / counts.square().sum().clamp_min(1e-20))
    return {"features_ever_used": int((counts > 0).sum()),
            "fraction_ever_used": float((counts > 0).double().mean()),
            "effective_feature_support": effective,
            "top_32_usage_fraction": float(counts.topk(32).values.sum() / total),
            "top_128_usage_fraction": float(counts.topk(128).values.sum() / total)}


def make_plot(local, evaluation, large, scratch):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    widths = [128, 512, 2048]
    for split in ("fineweb_test", "wikitext103", "tinystories"):
        values = [local["replacements"][f"m{width}_k128"]["evaluation"][split]
                  ["median_fidelity"] for width in widths]
        ax.plot(widths, values, "o-", label=split)
    ax.axhline(large["native"]["fineweb_test"]["write"]["median_fidelity"],
               color="black", linestyle="--", label="native Top-128 (FineWeb)")
    ax.set(xscale="log", xlabel="candidate bilinear features (128 active)",
           ylabel="median write explained energy", title="Local sparse replacement")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    names = ["linear_sae", "bilinear_sae", "m128_k128", "m512_k128", "m2048_k128",
             "m2048_k128_scratch"]
    labels = ["linear\nSAE", "bilinear\nSAE", "M128\nK128", "M512\nK128",
              "M2048\nK128", "M2048\nscratch"]
    xloc = np.arange(len(names)); width = .25
    for offset, split in enumerate(("fineweb", "wikitext103", "tinystories")):
        values = [evaluation["end_to_end"][split]["replacements"][name]
                  ["cross_entropy_delta"] for name in names]
        ax.bar(xloc + (offset - 1) * width, values, width, label=split)
    ax.set(xticks=xloc, xticklabels=labels, ylabel="cross-entropy increase",
           title="Full-model layer-1 replacement")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ks = [32, 64, 128, 256, 512, 2048]
    for split in ("fineweb", "wikitext103", "tinystories"):
        values = []
        for k in ks:
            name = "m2048_k128" if k == 128 else f"m2048_evalk{k}"
            values.append(evaluation["end_to_end"][split]["replacements"][name]
                          ["cross_entropy_delta"])
        ax.plot(ks, values, "o-", label=split)
    ax.set(xscale="log", xlabel="features enabled at inference",
           ylabel="cross-entropy increase", title="Is extra activity harmful?")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    categories = ["suffix_ing", "suffix_ed", "suffix_ly", "newline",
                  "code_operator", "word_continuation"]
    sources = {
        "native": large["native_categories"],
        "linear SAE": large["seeds"]["101"]["categories"],
        "bilinear SAE": evaluation["semantics"]["bilinear_sae"],
        "sparse bilinear": evaluation["semantics"]["replacement_m2048_k128"],
        "scratch bilinear": evaluation["semantics"]["replacement_m2048_scratch"],
    }
    xloc = np.arange(len(categories)); width = .15
    for index, (name, rows) in enumerate(sources.items()):
        values = [rows[category]["auc"]["fineweb_test"] for category in categories]
        ax.bar(xloc + (index - 2) * width, values, width, label=name)
    ax.set(ylim=(.45, 1.02), xticks=xloc,
           xticklabels=["-ing", "-ed", "-ly", "newline", "operator", "continuation"],
           ylabel="held-out FineWeb AUC", title="Known semantic controls")
    ax.legend(fontsize=8)
    fig.suptitle("Bilin18 layer 1: sparse bilinear computation", fontsize=14)
    fig.tight_layout()
    fig.savefig(ROOT / "bilinear_code_summary.png", dpi=180)
    plt.close(fig)


def write_report(local, evaluation, large, scratch, derived):
    e = evaluation["end_to_end"]
    rows = []
    for name in ("linear_sae", "bilinear_sae", "m128_k128", "m512_k128", "m2048_k128",
                 "m2048_k128_scratch"):
        rows.append("| `" + name + "` | " + " | ".join(
            f"{e[split]['replacements'][name]['cross_entropy_delta']:.3f}"
            for split in ("fineweb", "wikitext103", "tinystories")) + " |")
    krows = []
    for k in (32, 64, 128, 256, 512, 2048):
        name = "m2048_k128" if k == 128 else f"m2048_evalk{k}"
        krows.append(f"| {k} | " + " | ".join(
            f"{e[split]['replacements'][name]['cross_entropy_delta']:.3f}"
            for split in ("fineweb", "wikitext103", "tinystories")) + " |")
    text = f"""# Bilinear SAE and sparse bilinear layer — layer 1

## Result in one sentence

A bilinear feature parameterization recovers semantic discriminators, but the best
practical compression is a **2,048-candidate sparse bilinear layer with 128 active
features per token**: it uses {local['replacements']['m2048_k128']['parameters']:,}
parameters instead of {derived['original_parameters']:,}, explains
{local['replacements']['m2048_k128']['evaluation']['fineweb_test']['median_fidelity']:.1%}
of held-out layer-write energy, and increases full-model FineWeb cross-entropy by
only {e['fineweb']['replacements']['m2048_k128']['cross_entropy_delta']:.3f} nats.

![Bilinear-code summary](bilinear_code_summary.png)

## What was trained

The bilinear SAE uses

`z_k(x) = gain_k (a_k·x)(b_k·x)`

followed by signed Top-K selection and a normalized decoder into the original
4,608-dimensional product code. It has 2,048 candidate features and K=128.

The sparse replacement uses the same encoder but decodes directly into the
1,152-dimensional residual write. Models with 128, 512, and 2,048 candidate
features were trained; at most 128 features are retained for each token. The largest
replacement is 44.5% of the original MLP parameter count. It still computes all
candidate projections in this prototype, so the parameter and active-code reductions
must not be confused with a realized 36× wall-clock speedup.

Factor comparisons quotient out reciprocal scaling, left/right swap, simultaneous
sign changes, and feature permutation by comparing normalized symmetric bilinear
forms.

## Local reconstruction

| Model | FineWeb write fidelity | WikiText-103 | TinyStories |
|---|---:|---:|---:|
| Native per-token Top-128 | {large['native']['fineweb_test']['write']['median_fidelity']:.3f} | {large['native']['wikitext103']['write']['median_fidelity']:.3f} | {large['native']['tinystories']['write']['median_fidelity']:.3f} |
| Linear SAE, K=128 | {large['seeds']['101']['evaluation']['fineweb_test']['write']['median_fidelity']:.3f} | {large['seeds']['101']['evaluation']['wikitext103']['write']['median_fidelity']:.3f} | {large['seeds']['101']['evaluation']['tinystories']['write']['median_fidelity']:.3f} |
| Sparse bilinear M128/K128 | {local['replacements']['m128_k128']['evaluation']['fineweb_test']['median_fidelity']:.3f} | {local['replacements']['m128_k128']['evaluation']['wikitext103']['median_fidelity']:.3f} | {local['replacements']['m128_k128']['evaluation']['tinystories']['median_fidelity']:.3f} |
| Sparse bilinear M512/K128 | {local['replacements']['m512_k128']['evaluation']['fineweb_test']['median_fidelity']:.3f} | {local['replacements']['m512_k128']['evaluation']['wikitext103']['median_fidelity']:.3f} | {local['replacements']['m512_k128']['evaluation']['tinystories']['median_fidelity']:.3f} |
| Sparse bilinear M2048/K128 | {local['replacements']['m2048_k128']['evaluation']['fineweb_test']['median_fidelity']:.3f} | {local['replacements']['m2048_k128']['evaluation']['wikitext103']['median_fidelity']:.3f} | {local['replacements']['m2048_k128']['evaluation']['tinystories']['median_fidelity']:.3f} |
| M2048/K128 from scratch | {scratch['seeds']['2401']['evaluation']['fineweb_test']['median_fidelity']:.3f} | {scratch['seeds']['2401']['evaluation']['wikitext103']['median_fidelity']:.3f} | {scratch['seeds']['2401']['evaluation']['tinystories']['median_fidelity']:.3f} |

The bilinear SAE itself reconstructs the product activation less accurately than the
linear SAE: {local['bilinear_sae']['201']['evaluation']['fineweb_test']['median_fidelity']:.3f}
versus {large['seeds']['101']['evaluation']['fineweb_test']['activation']['median_fidelity']:.3f}.
Its advantage is structural and semantic, not raw MSE.

## Full-model replacement

Cross-entropy increases relative to the unmodified 18-layer model, over 8,192 tokens
per corpus:

| Replacement | FineWeb | WikiText-103 | TinyStories |
|---|---:|---:|---:|
{chr(10).join(rows)}

M2048/K128 slightly beats the linear SAE on all three panels while bypassing the
original 4,608-channel product and Down projection. M512/K128 is a more aggressive
11.1%-parameter replacement, but its FineWeb penalty rises to
{e['fineweb']['replacements']['m512_k128']['cross_entropy_delta']:.3f} nats.

The random-initialization control fails despite receiving 2,000 rather than 1,600
updates: its FineWeb write fidelity is only
{scratch['seeds']['2401']['evaluation']['fineweb_test']['median_fidelity']:.3f}, and
its full-model cross-entropy penalty is
{e['fineweb']['replacements']['m2048_k128_scratch']['cross_entropy_delta']:.3f} nats.
The two scratch runs agree in performance but have median symmetry-corrected
cross-seed form cosine {scratch['stability']['median_best_form_cosine']:.3f}.

## Is the code actually sparse?

On 16,384 held-out FineWeb positions, all
{derived['usage']['features_ever_used']:,} of 2,048 candidates are used at least once.
The corpus-level effective support is {derived['usage']['effective_feature_support']:.0f};
the 32 most frequently selected features account for only
{derived['usage']['top_32_usage_fraction']:.1%} of selections. Each token is sparse,
but the vocabulary of available computations is broad across the corpus.

Changing only inference-time K gives:

| Active K | FineWeb ΔCE | WikiText-103 ΔCE | TinyStories ΔCE |
|---:|---:|---:|---:|
{chr(10).join(krows)}

There is no catastrophic “too many neurons on” transition. FineWeb is best near the
trained K=64–128 region, while WikiText benefits from enabling more features. The
amount of useful sparsity is distribution-dependent; extra activation produces a
small mismatch/noise cost on FineWeb and TinyStories, not explosive interference.

## Meaning and identifiability

The random-initialized bilinear SAE recovers much clearer morphology than the linear
SAE: held-out FineWeb `-ing` AUC is
{evaluation['semantics']['bilinear_sae']['suffix_ing']['auc']['fineweb_test']:.3f}
and `-ed` is
{evaluation['semantics']['bilinear_sae']['suffix_ed']['auc']['fineweb_test']:.3f}.
Those are close to the native controls ({large['native_categories']['suffix_ing']['auc']['fineweb_test']:.3f}
and {large['native_categories']['suffix_ed']['auc']['fineweb_test']:.3f}).

But individual bilinear SAE features are still non-identifiable. Across two seeds,
the median best symmetry-corrected bilinear-form cosine is
{local['bilinear_sae_stability']['median_best_form_cosine']:.3f}, with
{local['bilinear_sae_stability']['fraction_above_0_8']:.1%} above 0.8. Their best
matches to native neuron forms have median cosine
{derived['native_alignment']['bilinear_sae_seed201']['median']:.3f}. The learned
features can implement similar semantic partitions without converging to the same
quadratic forms.

The successful sparse replacement is different: it was initialized from high-impact native
neurons, and {derived['native_alignment']['replacement_m2048']['fraction_above_0_8']:.1%}
of its learned forms remain above 0.8 cosine with some native form. Its interpretability
is partly inherited, not independently rediscovered. The scratch model's median best
native-form cosine is only
{derived['native_alignment']['replacement_m2048_scratch']['median']:.3f}.

## Red-team conclusions

- **Validated:** a substantially smaller sparse bilinear layer can replace layer 1
  with a modest language-model penalty, including on two corpus shifts.
- **Validated:** bilinear features recover known morphology better than the linear
  SAE at the same coarse audit, despite worse activation MSE.
- **Falsified:** simply matching the bilinear primitive does not make the learned
  feature basis unique across seeds.
- **Not yet established:** that M2048/K128 preserves rare behaviors, long contexts,
  or generation quality. The full-model panels contain only 8,192 tokens each.
- **Falsified under this recipe:** “the compact result is readily rediscovered from
  scratch.” With the same architecture, data, objective, and comparable training
  budget, the scratch model is dramatically worse locally and end-to-end.
- **Metric caution:** Top-K selection is sparse in activation space, but all 2,048
  factor pairs are evaluated before selection in this implementation.

## Interpretation

This supports a hybrid picture. Native bilinear neurons are meaningful discriminators;
a larger collection of them is reused across contexts, while only a small subset is
important for any one token. A learned sparse layer can rotate and compress the
aggregate computation, but there is no evidence that its individual coordinates are
the uniquely correct ontology. For reverse engineering, the stable object is likely
the bilinear function/subspace plus its causal input-output behavior—not whichever
sparse basis one optimizer happens to choose.
"""
    (ROOT / "REPORT.md").write_text(text)


def main():
    local = json.loads((ROOT / "local_results.json").read_text())
    evaluation = json.loads((ROOT / "evaluation_results.json").read_text())
    large = json.loads((SPARSE_ROOT / "large_sae_results.json").read_text())
    scratch = json.loads((ROOT / "scratch_results.json").read_text())
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    left = state["transformer.h.1.mlp.Left.weight"].float()
    right = state["transformer.h.1.mlp.Right.weight"].float()
    down = state["transformer.h.1.mlp.Down.weight"].float()
    bias = state["transformer.h.1.mlp.Down_bias"].float()
    original_parameters = left.numel() + right.numel() + down.numel() + bias.numel()
    models = {
        "bilinear_sae_seed201": load_dictionary(
            ROOT / "bilinear_sae_seed201.pt", 1152, 2048, 4608, 128),
        "bilinear_sae_seed202": load_dictionary(
            ROOT / "bilinear_sae_seed202.pt", 1152, 2048, 4608, 128),
        "replacement_m2048": load_dictionary(
            ROOT / "replacement_m2048_k128.pt", 1152, 2048, 1152, 128),
        "replacement_m2048_scratch": load_dictionary(
            ROOT / "replacement_m2048_k128_scratch2401.pt", 1152, 2048, 1152, 128),
    }
    alignment = {name: best_form_matches(model, left, right)
                 for name, model in models.items()}
    input_archive = np.load(SPARSE_ROOT / "layer1_inputs_large.npz")
    test_x = torch.from_numpy(input_archive["fineweb_test__input1"].astype(np.float32))
    usage_result = usage(models["replacement_m2048"], test_x)
    derived = {"original_parameters": original_parameters,
               "native_alignment": alignment, "usage": usage_result}
    (ROOT / "derived_results.json").write_text(json.dumps(derived, indent=2))
    make_plot(local, evaluation, large, scratch)
    write_report(local, evaluation, large, scratch, derived)
    print(f"wrote {ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
