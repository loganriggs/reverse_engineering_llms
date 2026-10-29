# Early interaction components, downstream behavior, and featurizer fidelity

M109 completes the bounded M108 repetition follow-up. It is a diagnostic on eight reused prompt cubes, not a new-task discovery or model-wide coverage estimate. All sequence-copy examples remain excluded from featurizer training.

## What changed conceptually

The large three-way pattern in the final two-value margin is not carried to an appreciable extent by the already-three-way part of the final-position writes in MLP0–2, tested one layer at a time. Removing those components has very small downstream effects. Early lower-order signals may still be essential and combine nonlinearly later; earlier token positions and joint removals were not tested.

Sparse featurizers can preserve one selected behavioral contrast reasonably well while distorting local computation and changing many full-vocabulary predictions. A contrast is therefore an insufficient equivalence criterion for a proposed computational unit. This does not establish native removable neurons, duplicated computation, semantic blocks, or cancellation-dominated MLPs.

## Theory and fixed experiment

For the three binary repetition factors, write the native write as w(s)=sum_A c_A product_{j in A}s_j. Uniform-cube orthogonality gives c_123=mean_s[s1*s2*s3*w(s)]. Subtracting s1*s2*s3*c_123 at each vertex removes precisely that local coefficient in exact arithmetic. Downstream nonlinear computation can mix degrees, so we measure every output-margin coefficient and full output KL.

We removed this final-position component separately in MLP0, MLP1, and MLP2. Three fixed signed-permutation controls preserve its norm and vertex sign pattern. All 64 prompts were retained; no prompts, gains, directions, or seeds were selected after results. This is a cube-informed off-manifold diagnostic, not a deployable representation or native channel subset.

An affine encoder/decoder maps each nonconstant cube coefficient linearly and cannot create a degree absent from its input. Sparse input-dependent gates do not have this guarantee: they can mix degrees. Comparing frozen sparse encoders with dense least-squares encoding in the same dictionaries tests whether the sparse encoding is needed for the observed fidelity; the dense control uses up to 128 coordinates versus 32 and is not sparsity-matched.

## Native cubic removal

| Layer | Mean output KL | Margin-change RMS | Top1 changes / 64 |
|---|---:|---:|---:|
| MLP0 | 6.55e-08 | 0.000565 | 0 |
| MLP1 | 7.69e-08 | 0.000622 | 0 |
| MLP2 | 8.94e-07 | 0.002743 | 0 |

No removal or scrambled control changed either candidate preference or top1. MLP2 removal has larger KL than its three scrambled controls (controls approximately 0.97–1.30e-7), but its absolute effect remains tiny. Three controls do not support a significance claim. The large original output cubic coefficients remain almost unchanged. Individual low impact does not rule out compensation, joint dependence, or causal use of lower-order components.

## Frozen featurizer comparison

Only MLP1 is reconstructed, at the final position, preserving its separate bias and every earlier position. Four fixed states: two seeds each for intact-KL training and KL plus encoded causal-response supervision. No retraining. Dense controls use the identical learned dictionaries. Errors below are pooled squared error divided by pooled native energy; “variation” excludes the constant coefficient.

| Objective | Seed | Encoding | Local variation error | Margin variation error | Full-output KL | Top1 changes / 64 |
|---|---:|---|---:|---:|---:|---:|
| KL | 71 | sparse | 1.369 | 0.0517 | 0.1486 | 32 |
| KL | 71 | dense | 0.638 | 0.0403 | 0.0771 | 28 |
| KL | 72 | sparse | 1.477 | 0.0214 | 0.0700 | 25 |
| KL | 72 | dense | 0.635 | 0.0211 | 0.0378 | 20 |
| KL + causal | 71 | sparse | 1.458 | 0.0528 | 0.1535 | 32 |
| KL + causal | 71 | dense | 0.638 | 0.0405 | 0.0782 | 28 |
| KL + causal | 72 | sparse | 1.792 | 0.0208 | 0.0619 | 24 |
| KL + causal | 72 | dense | 0.635 | 0.0201 | 0.0363 | 20 |

Sparse local cubic coefficient relative errors are 153–235, versus 0.694–0.703 for dense controls. The native local cubic coefficient is tiny, so these ratios must not be interpreted as dominant overall write errors. They are consistent with gate-dependent interaction distortion, not proof that native computation uses those extracted interactions. A frozen-gate counterfactual would distinguish gating from fixed linear-map effects directly.

Dense controls improve local variation error, margin variation error, and KL for all four states. This does not identify meaningful blocks, and using more coordinates is a material advantage. Causal supervision does not uniformly improve these metrics across seeds. Even dense replacements change 20–28 top predictions. The selected margin alone understates the discrepancy in the complete output distribution.

## Validation and limits

832 native-panel forwards and 576 feature-panel forwards; 128 exact baseline replays. Independent general linear solves audit the cubic directions and all 768 edits. NumPy independently reconstructs all 512 feature outputs (largest relative discrepancy 1.23e-7). Native GPT/Block.forward hooks independently reproduce 20 fixed intervention outputs bit-for-bit, without using the trace runner. These 20 are a sample, not exhaustive independent execution. No statistical generalization claim is made from the small reused vocabulary panel.

The producer receipts retain their historical pending-audit status; completed audits are cube_causal_analysis.json, cube_featurizer_analysis.json, and cube_direct_forward_audit.json in validation/. Full logits and arrays remain in the local artifact cache, referenced with hashes by the receipts.

## Consequence for the research direction

Close this repetition detour. Its featurizer deliverable is a verified factor-response evaluation with a clear separation between local geometry, a selected behavioral contrast, and full distribution fidelity. The next discriminating question is whether nonlinear sparse selection introduces misleading interaction structure across diverse tasks, rather than only this repetition panel. Use fixed gate counterfactuals with known affine/switching toys, followed by all existing competent held-out task families, with no new fits or group selection. Reuse of those holdouts is development evidence, not pristine confirmation. Do not infer semantic manifolds or native redundancy from this study.
