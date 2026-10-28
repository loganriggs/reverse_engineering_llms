## H831 — Crossed MLP states affect final predictions, but block-feature downstream dominance is heterogeneous

H830's local derivative analysis showed stronger angular changes from native MLP state variation than incoming-direction variation. H831 tests whether that distinction reaches final predictions, preserving natural attention and skip paths.

### Intervention design

All13 groups,52 recipients and both frozen block/random directions are retained. For each recipient, at the first downstream MLP's final token, replace output by its native MLP output plus signed amplitude times an analytic H830 response. Evaluate own baseline/own input, zero MLP response, each of three other baselines with own input, and own baseline with each of three other incoming inputs. Signs are +/-0.1; no reference is selected for favorable outcomes. Baseline means the state used to CALCULATE the response, not a replacement of the recipient's actual activation baseline. Attention and skip responses remain natural for that recipient.

KL compares the own analytic-response intervention distribution with each candidate's full-vocabulary distribution. The zero-response arm deletes only the tested MLP response; it retains the upstream intervention and other paths. Ratios to zero-response KL contextualize errors, not percentages of all model computation explained. The target remains faithfulness, not CE loss or task correctness.

### Results

Equal-weight group mean KL and ratios:

| Injection site | Direction | Zero-response KL | Other-baseline KL / zero | Other-input KL / zero | Groups where baseline substitution hurts more |
|---|---|---:|---:|---:|---:|
| after block0 | block |2.57e-6|1.344|.292|3/5|
| after block0 | random |8.82e-7|1.853|.082|5/5|
| after block1 | block |7.60e-6|1.642|1.352|6/8|
| after block1 | random |6.53e-6|1.388|.665|8/8|

Changing the state used to calculate the response has measurable final-distribution consequences. Mean errors exceed those from replacing incoming direction in all four site/mode summaries, and baseline replacements exceed zero-response error on average. A wrong response can therefore be worse than deleting that component. This does not imply the tested component is globally important: absolute divergences are small.

For block directions, the mean baseline-minus-input KL contrast is +2.70e-6 after block0, with descriptive group-bootstrap95% interval[-1.76e-8,7.95e-6], and +2.21e-6 after block1[-1.52e-6,6.93e-6]. Both intervals include zero. Some groups reverse the ordering; one group drives much of the block0 aggregate. Thus H830's all-group local angular result does NOT establish uniformly stronger downstream baseline effects.

For random directions the contrasts are +1.56e-6 [1.00e-7,4.40e-6] and +4.71e-6 [5.95e-7,1.02e-5], positive in all5/8 groups. These are descriptive4096-resample group intervals, not multiplicity-corrected proof or independent evidence of semantic features. Random controls displaying the same sensitivity reinforce that this can be a generic property of state-dependent computation.

### Limits and controls

Responses change both direction and magnitude. Alternative-baseline response norms range.596–1.667 times own, median.995; alternative-input norms range.730–1.478, median.998. Thus this is not a norm-matched angular-only comparison. These changes are part of the crossed operator's response but complicate attribution of downstream errors to angles alone. The signs and all reference choices are retained in records.

The intervention uses local linear response replacements in otherwise nonlinear runs. Own analytic responses replay natural interventions with mean KL9.12e-9 and maximum4.06e-7. Relative to each matched zero-response KL, own-to-natural replay error has median.000418,90th percentile.002209 and maximum.008576. Replay discrepancies are therefore below0.86% of the deletion-reference error for every evaluated signed case; this supports interpreting the much larger crossed-response effects while preserving the approximation distinction.

1716 whole-model forwards:1664 signed nonzero runs and52 zero-amplitude controls. Zero-amplitude patched logits reproduce native logits exactly. All tracked hashes unchanged, metrics finite, minimum KL3.91e-13. Full outputs are saved. No reserved-test evaluations or new documents; natural-document pool26,112.

### Research consequence and next test

State-dependent MLP computation affects actual model predictions, but local geometric alignment is an incomplete proxy for downstream fidelity. These results favor studying compact state-conditioned operators while retaining fixed-operator and flat-subspace controls. They do not recover semantic units, validate a manifold ontology, or show that BSF labels identify the relevant state conditions.

The next useful test should predict rather than keep explaining the same52 contexts: fit a compact operator-state representation using only training documents, then evaluate local responses on independent recipient occurrences and fixed perturbation directions. Compare a fixed/global operator, token-conditioned fixed operators, low-rank state-conditioned operators and BSF/shared-group state codes at explicit resource budgets. Keep original RMS computation and any state information available to each baseline explicit, so an apparent feature advantage cannot come from unreported access to the full baseline. Evaluate response fidelity first, then a frozen subset through full-model output KL. Reserve genuinely unused occurrences for confirmation and keep diverse competent-task transfer as an outstanding requirement. The observed baseline sensitivity supplies a target for this comparison; compression alone will not establish meaningful computation.
