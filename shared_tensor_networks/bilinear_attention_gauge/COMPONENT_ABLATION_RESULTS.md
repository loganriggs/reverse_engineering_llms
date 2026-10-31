# Do sparse component bases yield selective full-model effects?

The [exact-gauge experiment](EXACT_GAUGE_SPARSITY_RESULTS.md) made pair contributions more concentrated without changing the attention function. A [post-fit token-form label test](COMPONENT_LABEL_TRANSFER_RESULTS.md) did not show better held-out category prediction. This experiment tests another possible virtue: perhaps deleting one sparse-basis component affects a narrower set of contexts in the **full 18-layer model**.

The starting point is the same natural sequence-fitted rank-64 cut of layer 0, head 0. We compare six exact component gauges: SVD, one random orthogonal rotation, factor-weight L1, natural-pair activation sparsity, QK2-intervention-weighted activation sparsity, and rare-balanced activation sparsity. For each gauge, select its **two highest pair-response-energy components** on the prior FineWeb label-selection sample. The semantic labels assigned in that study are not used for component selection or this evaluation. Each full deletion changes the bond from \(X=\sum_i a_ib_i^{\mathsf T}\) to \(X-a_ib_i^{\mathsf T}\). KL is measured relative to the **rank-64 cut's** full-model distribution, isolating the component edit from baseline compression error. FineWeb test documents 17,500–19,999 and separate WikiText-103 and TinyStories samples contribute eight length-128 windows per seed at seeds 23 and 29: 16 documents per corpus. The native-model CE gate is exactly zero in the recorded run.

## Why full deletions need calibration

The top components in activation-sparse bases carry far more of the head response than top components in a random rotation. Under full deletion, averaged across the two components and three corpora:

| Basis | Mean relative head-output error | Mean full-model KL per token (×10⁻⁶) |
|---|---:|---:|
| SVD | .0235 | 8.30 |
| Random | .0075 | 3.07 |
| Factor-weight L1 | .0184 | 6.76 |
| Natural activation sparsity | .0465 | 20.88 |
| Intervention-weighted activation sparsity | .0467 | 19.84 |
| Rare-balanced activation sparsity | .0384 | 14.87 |

The larger KL for sparse bases is therefore not evidence of greater downstream importance per unit head change; their chosen components are simply larger. The full-deletion records are saved, but the fairer localization comparison scales each edit.

On **64 separate FineWeb calibration documents** from 15,000–17,499, choose a partial-deletion fraction for each component so that its aggregate relative head-output error is **.003**. These fractions range from **.269 to .828** and are fixed before testing. All are genuine partial deletions (fraction ≤1). This calibration uses no final FineWeb test document or external-corpus document. Test-set head-output errors vary with distribution, as expected: across basis averages they range about .0038–.0044, not exactly .003.

Define the effective fraction of affected documents for a component's per-document KL values \(k_j\) as

\[
F_{\rm eff}=\frac{(\sum_j k_j)^2}{N\sum_j k_j^2}.
\]

It ranges from \(1/N\) for one affected document to 1 for equal effects everywhere. **Lower** means a more localized effect. The table averages the two components × three corpora (six values per basis); KL is mean nats per token across the same six cases.

| Basis, calibrated partial deletion | Mean test head error | Mean KL (×10⁻⁶) | Effective document fraction |
|---|---:|---:|---:|
| SVD | .0038 | 1.50 | .585 |
| Random | .0044 | 1.76 | **.502** |
| Factor-weight L1 | .0043 | 1.55 | .553 |
| Natural activation sparsity | .0041 | 1.95 | .681 |
| Intervention-weighted activation sparsity | .0039 | 1.64 | .729 |
| Rare-balanced activation sparsity | .0040 | 1.70 | .724 |

**There is no downstream localization advantage for the activation-sparse gauges in this assay.** Their effects are more diffuse by this metric than the random rotation's, despite having fewer effective active components per *pair*. Pair-level component concentration and context-selective downstream causality are different properties. The full per-component, per-corpus, per-document results are saved; the aggregate hides substantial variation, including one SVD component with a very concentrated FineWeb effect.

This is an exploratory negative result, not a general theorem. It uses only one head, one rank, one random gauge, one fitted seed for each sparse prior, two selected components per basis, 16 documents per corpus, and small perturbations whose KL is around 10⁻⁶. The calibration matches head-output error on FineWeb selection documents, not downstream KL or each OOD corpus separately. More components, seeds, tasks, and explicit semantic interventions would be needed before ruling out a useful sparse gauge. It does show that neither a low activation-support score nor a few attractive token exemplars is sufficient evidence of a mechanistic explanation.

Reproduce with `flagship_component_ablation.py --n-docs 8 --top-components 2`, then `--target-head-error .003 --calibration-docs 64` for the matched run. JSON files are `full_width_results/head0_component_ablation_n8.json` and `head0_component_ablation_n8_matched.json`. The external model and corpora remain referenced by path rather than copied into this repository.
