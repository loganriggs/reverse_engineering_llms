# Iterated L1/prune on the varied bilinear language model

The earlier basis-aligned experiment found that L1, iterative magnitude pruning, and an unregularized final finetune could expose a small, legible QK/OV circuit when the solution was already in a favorable basis. This is a direct transfer test on a trained **two-layer, four-head bilinear language model** with a 317-token vocabulary. It uses the varied full-language generator with known lexical, previous-place, cue, and induction rules. No ground-truth basis is supplied to the pruning process.

## Protocol

The implementation is [`sparsify_varied_bilinear.py`](sparsify_varied_bilinear.py). Each round trains for 150 AdamW steps with ordinary next-token cross-entropy, the exact-oracle verb soft-target term (weight 2), and L1 on selected, still-active weights (coefficient `3e-5`). It then measures KL to the generator's exact next-token distribution on a fixed 512-stream panel. If KL remains within **original model KL + 0.020 overall, + 0.025 at verbs, and + 0.035 at induction verbs**, it permanently zeros the smallest 15% of remaining selected weights and repeats. On the first failed round, it restores the last passing weights and mask and trains another 150 steps with **no L1**. Hard masks remain enforced during every optimizer step and the final finetune. A run reaching the round cap also restores the last tested mask and performs the no-L1 finetune.

The `attention_only` arm selects both blocks' QK1, QK2, V, and output matrices (49,152 scalar weights). The `all_linear` arm additionally selects the token embedding and LM head (89,728 weights). Their reported densities use **different denominators**. Both start from the same trained checkpoint; a continuation of attention-only starts from its saved model and hard mask while keeping the original model as the KL reference. The selection panel is seed 20261125, fresh panel seed 20261126, and low-parenthetical panel seed 20261127. These are generated independently. All use 512 streams of length 72. These panels test behavior; they were not used to fit a sparse basis.

| Arm | Selected density after no-L1 finetune | Fresh overall KL (base .0503) | Fresh verb KL (base .0628) | Fresh induction-verb KL (base .0770) |
|---|---:|---:|---:|---:|
| Attention only, 12-round cap | 16.74% | .0495 | .0729 | .0953 |
| Attention only, continued to first failed round | **14.23%** | **.0495** | **.0734** | **.0966** |
| All linear, first failed round | **37.72%** | **.0508** | **.0664** | **.0868** |

The continued attention run passed at 14.23%; an attempted 12.10% state breached the verb and induction KL limits and was reverted. The no-L1 finetune of the restored 14.23% model reduced selection-panel verb KL from .0792 to .0725 and induction-verb KL from .1038 to .0964. The all-linear run passed at 37.72%; its attempted 32.06% state breached limits and was reverted. Its no-L1 finetune reduced selection-panel verb KL from .0721 to .0660 and induction-verb KL from .0954 to .0898. Thus the user's suggested final unregularized step is material in both arms.

## Does sparse weight structure reveal rule paths?

We separately tested **causal edit fidelity**, **complete-head path ablations**, and an **exact first-layer token-space fold**. All checkpoints passed a direct first-block replay: folding every embedding through RMSNorm into per-token QK1/QK2/V vectors, retaining RoPE by position, multiplying the two score branches, and summing head OV paths plus the direct residual reproduces native block output to relative L2 error below `1e-7`. This construction uses the full vocabulary and no sampled token pairs; the generated streams only test the equality and measure observed use.

On a separate 512-stream causal-edit panel (seed 20261128), effect-L1 error for the original, continued attention-only, and all-linear checkpoints is:

| Rule edit | Original | Attention 14.23% | All linear 37.72% |
|---|---:|---:|---:|
| Lexical verb | .271 | .268 | .260 |
| Induction | .254 | .279 | .274 |
| Previous place | .228 | .254 | .239 |
| Cue | .216 | .249 | .224 |
| Nuisance determiner | .013 | .022 | .018 |
| Nuisance adjective | .016 | .024 | .029 |

The pruned models remain close in aggregate KL but lose some rule-edit fidelity, especially induction and previous-place. Lower nuisance error would be better; the pruned models are worse there too.

Raw attention matrix zeros did **not** produce a sparse token-pair score function. At lag one, all 317×317 QK1, QK2, and product tables per head have no exact zeros even after pruning. The top 1% of absolute product scores contain 32.2% of total magnitude in the original, 29.2% in the continued attention model, and 30.3% in the all-linear model. On observed causal token pairs, corresponding shares are 40.2%, 35.3%, and 37.7%. OV token-to-residual maps concentrate somewhat, but the score product becomes less concentrated by this metric. Complete-head ablations on another independent panel identify important heads but do not yield a cleaner one-head-per-rule account after pruning; these ablations also cannot separate QK1, QK2, OV, and direct residual paths.

**Conclusion:** This reproduces the earlier *behavior-preserving pruning procedure*, including the beneficial no-L1 final finetune, but does **not** yet reproduce the earlier **legible computation-path recovery**. Low weight density is an insufficient interpretability criterion here. The next comparison should explicitly optimize a path-level or token-function-level sparsity objective, measure causal-rule specificity, and compare at matched KL and causal-edit fidelity. The exact token-space fold supplies a data-free first-layer object on which to define that objective; a distribution-weighted version should use much broader held-out streams and task mixtures. Later blocks require contextual activations rather than a simple vocabulary table.

Raw outputs: [`attention-only`](surface_mixture_v2_varied_prune_attention/), [`continued attention-only`](surface_mixture_v2_varied_prune_attention_continued/), [`all-linear`](surface_mixture_v2_varied_prune_all_linear/), and [`original checkpoint`](surface_mixture_v2_varied_soft_models/). Each pruned directory contains `pruning_results.json`, `model_depth2.pt`, `masks.pt`, `causal_edits_n512.json`, `head_path_ablation_n256.json`, and `first_layer_fold_n64.json`.
