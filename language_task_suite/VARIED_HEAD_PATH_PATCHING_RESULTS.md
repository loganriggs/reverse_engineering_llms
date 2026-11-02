# Causal head paths in the varied joint-rule bilinear model

The [functional sparsity experiment](VARIED_FUNCTIONAL_SPARSITY_RESULTS.md) increased concentration of first-layer QK1×QK2 score tables while preserving most predictive and causal-edit behavior. This assay asks whether those models actually have cleaner **rule-carrying attention paths**. It compares the original checkpoint, an ordinary finetuning control, and the uniform- and observed-pair score-penalty checkpoints on identical, grammar-valid one-token edits.

[`eval_varied_head_path_patching.py`](eval_varied_head_path_patching.py) records the edited and original pre-output-projection value of every complete attention head. It then runs the original prefix with **one edited head path patched in at every position**. The reverse intervention runs the edited prefix with one original head path patched in. Both are evaluated at the next-verb prediction. The remaining heads and the direct residual come from the target run. The script checks that self-patching leaves logits *exactly* unchanged; all saved runs passed with maximum difference zero. The head path includes the native QK1×QK2×V calculation and its output projection, but the assay does not separately isolate those factors.

The metric below is the projection of the patch-induced probability shift onto the model's **full edit effect**, divided by that effect's squared norm, averaged across edits. Thus 1 means the patch carries the full effect *in its direction*; it is not a fraction of probability mass, need not be additive across heads, and does not by itself measure agreement with the exact oracle. The model's full-effect oracle error is recorded separately in each JSON. Forward and reverse values agree closely for the strongest paths.

Two disjoint generated panels use seeds 20261203 (64 streams, 957 supported pairs) and 20261204 (128 streams, 1,916 pairs). The core original-model results are:

| Rule edit | Patched path | Forward alignment, panel 1 / panel 2 | Reverse alignment, panel 1 / panel 2 |
|---|---|---:|---:|
| Lexical subject | All first-block heads | .873 / .876 | .861 / .849 |
| Induction from earlier verb | All second-block heads | .894 / .903 | .906 / .905 |
| Previous place | Second-block head 1 | .867 / .833 | .850 / .832 |
| Frame cue | All first-block heads | .433 / .384 | .433 / .360 |

At the single-head level, first-block head 1 carries .411 of the lexical edit's aligned shift on panel 1. Second-block heads 0 and 1 carry .415 and .405 of induction, respectively; second-block head 1 also carries .867 of previous-place, so it is **not** a one-rule-only component. First-block all-head patching carries only .177 of induction and .078 of previous-place, whereas second-block all-head patching carries .894 and .985. The cue rule is less localized: both blocks mediate substantial, interacting parts of its effect. The nuisance edits have very small full-model effects, so ratios to those effects are unstable and should not be used to name a circuit.

The observed-pair score-penalty model preserves this division of labor rather than making it more selective:

| Rule edit and path | Forward alignment, original panel 1 / panel 2 | Forward alignment, observed-penalty panel 1 / panel 2 |
|---|---:|---:|
| Lexical, first-block all heads | .873 / .876 | .871 / .871 |
| Induction, second-block all heads | .894 / .903 | .889 / .902 |
| Previous place, second-block head 1 | .867 / .833 | .858 / .817 |
| Cue, first-block all heads | .433 / .384 | .392 / .320 |

The ordinary-finetuned and uniform-penalty checkpoints were also run on panel 1. Their main assignments are similar: previous-place alignment from second-block head 1 is .862 in both, and induction alignment from all second-block heads is .897 and .895. This is a useful mechanistic result about the **original model**: the layer/head location of several rule effects can be predicted and intervened on, and it replicates across two panels and both patch directions. It also shows that making the first-layer score table concentrated did not clarify the causal organization.

This is still a coarse path map. Complete-head patches mix QK, V, and OV, and injecting an edited activation into the original context creates a hybrid state. Head effects can interact nonlinearly, so the alignment numbers do not sum to a unique circuit decomposition. The current panels have only 84/166 induction edits and mostly ordinary source distances; they do not establish a long-range circuit. Next, test **within-head source-position and QK/OV path selectivity** on held-out sentence frames and distance-balanced induction edits, then compare any learned sparsity objective against that causal map rather than against score concentration alone.

Raw outputs: original [panel 1](surface_mixture_v2_varied_soft_models/head_path_patching_both_n64.json) and [panel 2](surface_mixture_v2_varied_soft_models/head_path_patching_both_n128_seed2.json); observed-pair penalty [panel 1](surface_mixture_v2_varied_functional_observed_p10/head_path_patching_both_n64.json) and [panel 2](surface_mixture_v2_varied_functional_observed_p10/head_path_patching_both_n128_seed2.json). The control and uniform-penalty panel-1 forward-patch outputs are in their respective checkpoint directories.
