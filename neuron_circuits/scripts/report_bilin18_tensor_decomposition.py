#!/usr/bin/env python3
"""Report direct CP/LL1 factorization and the two follow-up iterations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/bilin18_tensor_decomposition")


def plot(weight, activation, routed, full):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    direct_names = ["cp_r128", "cp_r512", "cp_r2048", "ll1_r64_l8",
                    "ll1_r128_l4", "ll1_r256_l8", "ll1_r512_l4"]
    labels = ["CP128", "CP512", "CP2048", "LL1\n64×8", "LL1\n128×4",
              "LL1\n256×8", "LL1\n512×4"]
    ax = axes[0, 0]
    colors = ["tab:blue"] * 3 + ["tab:orange"] * 4
    ax.bar(np.arange(len(labels)), [weight["models"][name]["after_tensor_probe"]
                                   ["median_fidelity"] for name in direct_names],
           color=colors)
    ax.set(xticks=np.arange(len(labels)), xticklabels=labels,
           ylabel="isotropic tensor-probe fidelity", title="Direct weight decomposition")

    ax = axes[0, 1]
    stages = {
        "CP512 direct": (weight["models"]["cp_r512"]["after_tensor_probe"]["median_fidelity"],
                         weight["models"]["cp_r512"]["natural"]["fineweb_test"]["median_fidelity"]),
        "CP512 activation": (activation["models"]["cp_r512"]["after_tensor"]["median_fidelity"],
                             activation["models"]["cp_r512"]["natural"]["fineweb_test"]["median_fidelity"]),
        "CP2048 direct": (weight["models"]["cp_r2048"]["after_tensor_probe"]["median_fidelity"],
                          weight["models"]["cp_r2048"]["natural"]["fineweb_test"]["median_fidelity"]),
        "CP2048 activation": (activation["models"]["cp_r2048"]["after_tensor"]["median_fidelity"],
                              activation["models"]["cp_r2048"]["natural"]["fineweb_test"]["median_fidelity"]),
        "routed behavior": (routed["models"]["routed_behavior"]["after_tensor"]["median_fidelity"],
                            routed["models"]["routed_behavior"]["natural"]["fineweb_test"]["median_fidelity"]),
        "routed mixed": (routed["models"]["routed_mixed"]["after_tensor"]["median_fidelity"],
                         routed["models"]["routed_mixed"]["natural"]["fineweb_test"]["median_fidelity"]),
    }
    for name, (tensor, natural) in stages.items():
        ax.scatter(tensor, natural, s=60)
        ax.annotate(name, (tensor, natural), xytext=(4, 3), textcoords="offset points",
                    fontsize=8)
    ax.set(xlabel="tensor-probe fidelity", ylabel="natural write fidelity",
           title="Weight fidelity versus behavior fidelity")

    ax = axes[1, 0]
    names = ["cp512_weight", "cp2048_weight", "cp512_activation",
             "cp2048_activation", "ll1_activation", "routed_behavior", "routed_mixed"]
    labels = ["CP512\nweight", "CP2048\nweight", "CP512\nactivation",
              "CP2048\nactivation", "LL1\nactivation", "routed\nbehavior",
              "routed\nmixed"]
    xloc = np.arange(len(names)); width = .25
    for offset, corpus in enumerate(("fineweb", "wikitext103", "tinystories")):
        values = [full["corpora"][corpus]["replacements"][name]["mean_kl_from_full"]
                  for name in names]
        ax.bar(xloc + (offset - 1) * width, values, width, label=corpus)
    ax.set(xticks=xloc, xticklabels=labels, ylabel="mean KL from original",
           title="Full-model behavioral fidelity")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ranks = np.asarray([8, 16, 32, 64, 128, 256, 512])
    explained = np.asarray([.0330, .0572, .0987, .1711, .2938, .4854, .7329])
    ax.plot(ranks, explained, "o-")
    ax.set(xscale="log", xlabel="output-mode rank", ylabel="sampled energy explained",
           title="Why low-block-count LL1 struggles")
    fig.suptitle("Bilin18 layer-1 tensor factorization", fontsize=14)
    fig.tight_layout()
    fig.savefig(ROOT / "tensor_decomposition_summary.png", dpi=180)
    plt.close(fig)


def report(weight, activation, routed, full):
    wm = weight["models"]; am = activation["models"]; rm = routed["models"]
    text = f"""# Direct tensor factorization of Bilin18 layer 1

## Bottom line

The original layer tensor can be written implicitly as

`T = Σ_j d_j ⊗ sym(l_j ⊗ r_j)`.

A direct CP truncation to 2,048 terms (7.08M parameters) preserves
**{wm['cp_r2048']['after_tensor_probe']['median_fidelity']:.1%}** of held-out
isotropic tensor-probe energy and **{wm['cp_r2048']['natural']['fineweb_test']['median_fidelity']:.1%}**
of natural FineWeb write energy. Inserted into the full model without activation
distillation, it changes FineWeb cross-entropy by
**{full['corpora']['fineweb']['replacements']['cp2048_weight']['cross_entropy_delta']:.3f} nats**
and has KL **{full['corpora']['fineweb']['replacements']['cp2048_weight']['mean_kl_from_full']:.3f}**
from the original model.

The best sparse compromise is the second follow-up, a CP-2048 factorization with
Top-K 128 routing and a mixed tensor-plus-natural objective. It preserves
**{rm['routed_mixed']['after_tensor']['median_fidelity']:.1%}** tensor-probe energy,
**{rm['routed_mixed']['natural']['fineweb_test']['median_fidelity']:.1%}** natural
write energy, and gives FineWeb ΔCE
**{full['corpora']['fineweb']['replacements']['routed_mixed']['cross_entropy_delta']:.3f}**.

![Tensor decomposition summary](tensor_decomposition_summary.png)

## Decompositions

### CP

The CP student is

`f(x) = Σ_k d'_k (a_k·x)(b_k·x)`.

Native terms are ranked by their exact individual tensor Frobenius norm, retained,
and optimized on isotropic bilinear probes `(x,y)`. The probe estimator avoids
materializing the 1,152×1,152×1,152 tensor. It measures the Frobenius objective in
expectation, rather than computing every tensor entry.

| Direct model | Parameters | Tensor fidelity | Natural FineWeb write fidelity |
|---|---:|---:|---:|
| CP rank 128 | {wm['cp_r128']['parameters']:,} | {wm['cp_r128']['after_tensor_probe']['median_fidelity']:.3f} | {wm['cp_r128']['natural']['fineweb_test']['median_fidelity']:.3f} |
| CP rank 512 | {wm['cp_r512']['parameters']:,} | {wm['cp_r512']['after_tensor_probe']['median_fidelity']:.3f} | {wm['cp_r512']['natural']['fineweb_test']['median_fidelity']:.3f} |
| CP rank 2048 | {wm['cp_r2048']['parameters']:,} | {wm['cp_r2048']['after_tensor_probe']['median_fidelity']:.3f} | {wm['cp_r2048']['natural']['fineweb_test']['median_fidelity']:.3f} |

The direct optimizer does not materially improve the selected native components.
This suggests that native terms already provide a strong local CP parameterization,
or that improving a large nonconvex CP decomposition requires a better optimizer.

### LL1 block-term decomposition

An LL1 model groups `L` input bilinear factors behind one output vector:

`f(x) = Σ_r c_r Σ_(l=1)^L (a_rl·x)(b_rl·x)`.

This is the rank-`(L,L,1)` block-term model. It is attractive if many input
discriminators share an output direction. That assumption is poorly matched here.

| LL1 model | Parameters | Tensor fidelity | Natural FineWeb write fidelity |
|---|---:|---:|---:|
| R64/L8 | {wm['ll1_r64_l8']['parameters']:,} | {wm['ll1_r64_l8']['after_tensor_probe']['median_fidelity']:.3f} | {wm['ll1_r64_l8']['natural']['fineweb_test']['median_fidelity']:.3f} |
| R128/L4 | {wm['ll1_r128_l4']['parameters']:,} | {wm['ll1_r128_l4']['after_tensor_probe']['median_fidelity']:.3f} | {wm['ll1_r128_l4']['natural']['fineweb_test']['median_fidelity']:.3f} |
| R256/L8 | {wm['ll1_r256_l8']['parameters']:,} | {wm['ll1_r256_l8']['after_tensor_probe']['median_fidelity']:.3f} | {wm['ll1_r256_l8']['natural']['fineweb_test']['median_fidelity']:.3f} |
| R512/L4 | {wm['ll1_r512_l4']['parameters']:,} | {wm['ll1_r512_l4']['after_tensor_probe']['median_fidelity']:.3f} | {wm['ll1_r512_l4']['natural']['fineweb_test']['median_fidelity']:.3f} |

The sampled output mode is high-rank: its first 128 directions explain about 29%
of tensor-probe energy and 512 directions explain about 73%. Within each output
direction, the associated symmetric input matrix also needs more than L=4–8
directions. LL1 therefore compresses along precisely the mode that is broad in this
layer.

## Follow-up iteration 1: activation weighting

Starting from the direct decompositions and optimizing natural layer writes gives:

| Model | Tensor fidelity after refinement | FineWeb write fidelity | FineWeb full-model ΔCE | FineWeb KL |
|---|---:|---:|---:|---:|
| CP512 | {am['cp_r512']['after_tensor']['median_fidelity']:.3f} | {am['cp_r512']['natural']['fineweb_test']['median_fidelity']:.3f} | {full['corpora']['fineweb']['replacements']['cp512_activation']['cross_entropy_delta']:.3f} | {full['corpora']['fineweb']['replacements']['cp512_activation']['mean_kl_from_full']:.3f} |
| CP2048 | {am['cp_r2048']['after_tensor']['median_fidelity']:.3f} | {am['cp_r2048']['natural']['fineweb_test']['median_fidelity']:.3f} | {full['corpora']['fineweb']['replacements']['cp2048_activation']['cross_entropy_delta']:.3f} | {full['corpora']['fineweb']['replacements']['cp2048_activation']['mean_kl_from_full']:.3f} |
| LL1 R256/L8 | {am['ll1_r256_l8']['after_tensor']['median_fidelity']:.3f} | {am['ll1_r256_l8']['natural']['fineweb_test']['median_fidelity']:.3f} | {full['corpora']['fineweb']['replacements']['ll1_activation']['cross_entropy_delta']:.3f} | {full['corpora']['fineweb']['replacements']['ll1_activation']['mean_kl_from_full']:.3f} |

Activation refinement is extremely effective behaviorally, but it rotates away from
the global tensor. CP2048 reaches 87.0% natural fidelity and nearly preserves model
behavior; CP512 loses almost all isotropic tensor fidelity while becoming a good
natural-data approximation. This is a concrete distinction between explaining the
weight function globally and explaining the part exercised by language.

## Follow-up iteration 2: sparse routing

Both routed models retain 128 of 2,048 CP factors per example.

| Routed CP2048/K128 | Tensor fidelity | FineWeb write fidelity | WikiText write fidelity | FineWeb ΔCE | WikiText ΔCE |
|---|---:|---:|---:|---:|---:|
| Behavior-only | {rm['routed_behavior']['after_tensor']['median_fidelity']:.3f} | {rm['routed_behavior']['natural']['fineweb_test']['median_fidelity']:.3f} | {rm['routed_behavior']['natural']['wikitext103']['median_fidelity']:.3f} | {full['corpora']['fineweb']['replacements']['routed_behavior']['cross_entropy_delta']:.3f} | {full['corpora']['wikitext103']['replacements']['routed_behavior']['cross_entropy_delta']:.3f} |
| Mixed tensor + behavior | {rm['routed_mixed']['after_tensor']['median_fidelity']:.3f} | {rm['routed_mixed']['natural']['fineweb_test']['median_fidelity']:.3f} | {rm['routed_mixed']['natural']['wikitext103']['median_fidelity']:.3f} | {full['corpora']['fineweb']['replacements']['routed_mixed']['cross_entropy_delta']:.3f} | {full['corpora']['wikitext103']['replacements']['routed_mixed']['cross_entropy_delta']:.3f} |

The mixed objective is the better result: it retains much more weight-level behavior,
has only a small natural-fidelity cost, and improves the full-model result on FineWeb
and WikiText. All 2,048 factors are used somewhere; effective corpus support is about
{rm['routed_mixed']['usage']['fineweb_test']['effective_support']:.0f}. The code is
sparse per token, not globally pruned to 128 universal components.

## Red-team conclusions

- **Validated:** the layer has a useful lower-rank CP description. CP2048 preserves
  54% of the global tensor and already gives a small full-model perturbation without
  activation distillation.
- **Validated:** language behavior occupies a substantially easier subdistribution
  than isotropic tensor space. Activation-refined CP2048 reaches 87% natural-write
  fidelity while preserving only 36% of isotropic probe energy.
- **Validated:** an explicit CP factorization can support sparse, input-dependent
  routing. The mixed CP2048/K128 model is the best weight/behavior/sparsity compromise.
- **Falsified at tested budgets:** LL1 is a competitive direct description of this
  layer. It underperforms CP despite comparable or larger parameter counts.
- **Caution:** the CP initialization selects native components; this is an explicit
  factorization and refinement of the trained tensor, not proof that an unrelated
  optimizer finds a unique canonical decomposition.
- **Caution:** “tensor fidelity” is estimated with held-out isotropic Gaussian
  bilinear probes. It is an unbiased route to a Frobenius-style objective, but the
  reported median per-probe fidelity is not an exact closed-form tensor norm.
- **Caution:** hard Top-K makes the routed model nonlinear and input-dependent, so it
  is no longer one fixed rank-2048 tensor after routing.

## Interpretation

There are now three distinct explanatory objects:

1. **Global weight tensor:** CP factors describe what the layer could compute on
   arbitrary residual directions.
2. **Language-supported computation:** activation refinement describes the much
   smaller region actually visited by text.
3. **Conditional sparse program:** Top-K routing selects a small set of factors for
   each token while drawing from a broad corpus-level vocabulary.

LL1 asks for reusable output modules shared by several input discriminators. Its
failure says that layer 1 does not compress naturally in that particular modular
direction. CP plus routing fits the evidence better: many bilinear discriminators,
many output directions, and sparse conditional use.

## Method reference

LL1 here follows the standard rank-`(L,L,1)` block-term definition described in
[De Lathauwer's block-term decomposition work](https://epubs.siam.org/doi/10.1137/17M1150062)
and the [Tensorlab LL1 documentation](https://www.tensorlab.net/doc/ll1.html).
"""
    (ROOT / "REPORT.md").write_text(text)


def main():
    weight = json.loads((ROOT / "weight_factorization_results.json").read_text())
    activation = json.loads((ROOT / "activation_refinement_results.json").read_text())
    routed = json.loads((ROOT / "sparse_routing_results.json").read_text())
    full = json.loads((ROOT / "full_model_results.json").read_text())
    plot(weight, activation, routed, full)
    report(weight, activation, routed, full)
    print(f"wrote {ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
