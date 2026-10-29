# M56: a competent color label does not guarantee a useful early-MLP causal target

**Native opposite-answer swaps at MLP1's final position have very small effects, and all18 frozen candidates are worse than zero-response prediction on both pooled full-distribution response tests.** This does not negate native color competence. It shows that the chosen early-MLP site and these featurizers do not yet provide faithful causal supervision for that behavior.

## Fixed test

All eight object-color completion endpoints from M55, including native errors, are retained. Three donor conditions: opposite endpoint in the same contrast pair; same object in the explicit-QA format; identity. At native MLP1's final position, replace only the product write with the donor final-position product write. Bias is retained and downstream computation runs normally. Unequal prompt lengths are allowed because only final positions are aligned; this is not full-sequence semantic alignment.

For all18 M37 initial/KL/KL+response scalar/block/shared models, reconstruct the original entire MLP1 write, then add the exact same saved native delta at the final position. Responses are measured relative to each candidate's own intact output. This restores common causal targets; no recipient magnitude oracle or candidate-defined target is used. Format swaps are diagnostics, not assumed task-preserving nuisance interventions.

## Native effects

| Quantity, pooled over8recipients | Opposite-answer donor | Same-object format donor |
|---|---:|---:|
| Sum squared write-delta norm | 1,636,671 | 87,819,906 |
| Sum native weighted logprob response energy | 0.000885 | 0.717955 |
| Mean output KL | 0.0000552 | 0.04714 |
| Mean absolute answer-margin change | 0.00685 | 0.15688 |
| Changes reducing recipient intended-answer margin | 3/8 | 2/8 |

Format deltas have about54times more write energy and812times more response energy. These are unmatched interventions; do not attribute the difference solely to downstream sensitivity, semantic relevance, or a computational strategy. Task donors have tiny, inconsistent answer-margin effects, despite swapping between different intended colors. Other positions/layers may carry the behavior; this experiment cannot localize the whole color circuit or establish absence of MLP1 involvement.

## Every frozen candidate retained

Errors are summed before dividing by native target energy; zero response=1. Task-margin errors and full-distribution errors measure different properties. Tiny native target energy makes normalized errors particularly sensitive to small absolute mistakes; raw energies remain in JSON.

| Representation | Seed | Objective | Task response | Task margin | Format response | Format margin |
|---|---:|---|---:|---:|---:|---:|
| scalar | 71 | initial | 1.077 | 1.126 | 2.223 | 1.998 |
| block | 71 | initial | 1.443 | 0.658 | 2.444 | 2.608 |
| shared_fixed | 71 | initial | 3.182 | 1.811 | 4.547 | 2.992 |
| scalar | 72 | initial | 1.091 | 0.900 | 1.786 | 1.914 |
| block | 72 | initial | 1.804 | 0.513 | 2.537 | 2.007 |
| shared_fixed | 72 | initial | 1.585 | 0.737 | 2.317 | 1.976 |
| scalar | 71 | intact_KL | 1.633 | 1.031 | 2.636 | 2.202 |
| scalar | 71 | intact_KL_plus_native_response | 1.640 | 1.108 | 2.437 | 2.238 |
| block | 71 | intact_KL | 1.340 | 0.637 | 3.193 | 3.562 |
| block | 71 | intact_KL_plus_native_response | 1.431 | 0.356 | 2.360 | 3.619 |
| shared_fixed | 71 | intact_KL | 4.965 | 2.389 | 7.319 | 6.419 |
| shared_fixed | 71 | intact_KL_plus_native_response | 3.703 | 2.409 | 6.457 | 5.830 |
| scalar | 72 | intact_KL | 1.005 | 0.914 | 1.478 | 1.619 |
| scalar | 72 | intact_KL_plus_native_response | 1.075 | 0.966 | 2.026 | 2.428 |
| block | 72 | intact_KL | 1.749 | 0.408 | 2.867 | 2.977 |
| block | 72 | intact_KL_plus_native_response | 1.752 | 0.527 | 2.252 | 2.168 |
| shared_fixed | 72 | intact_KL | 1.733 | 1.675 | 3.132 | 2.845 |
| shared_fixed | 72 | intact_KL_plus_native_response | 2.252 | 1.287 | 3.180 | 2.684 |

All36 nonidentity full-distribution aggregates exceed1, including all initial states. Some answer-margin errors are below1, illustrating that matching a chosen answer contrast can coexist with poor full-distribution response fidelity. No seed, architecture, or condition is selected for further tuning.

## Conceptual consequence

The featurizer loop needs **competent behaviors plus informative, faithful causal targets**, not just more labels. A supervised task name does not identify where its computation lives. These weak final-position task responses are useful negative evidence: faithful representations should not invent large response changes there. They are not strong evidence of an early color-computation unit or a reason to train a semantic group automatically.

This reinforces three separate questions: what structure appears in the MLP's writes, which native parts carry the task-changing signal, and which aspects the downstream reader uses. Distributed implementation, token-conditioned geometry, and small causal impact at one position can coexist. None implies literal removal redundancy or cancellation without the corresponding native interventions.

## Verification and stopping point

464forward passes;24native conditions,144candidate baselines,432candidate responses. All16 color-format native baseline vectors replay M55 exactly. Independent NumPy audit rescores every saved full-vocabulary final vector, all native delta constructions, candidate target hashes, zero controls, and54summaries. Maximum absolute numeric discrepancy is1.86e-9, including large write-energy quantities. Audit does not independently reimplement the transformer.

This closes the fixed color-only comparison. No amplitude, prompt, layer, or candidate search follows from its outcome. No training expansion has been performed or validated.

Next perform a **broad training-only causal-coverage inventory** across the existing approved task families, before further training: audit pair alignment and native source-token versus final-position interventions at the same predeclared early layers for every family. The deliverable is a common-target map showing task effects, format/context confounds, and null responses, not a best-site search for color. Preserve whole-family exclusions and retain competence failures, ineligible alignments, and zero effects. Freeze the exact inventory and intervention semantics before forwards; do not infer semantic alignment from equal length alone. This will tell us which task/site targets actually constrain a featurizer and which should serve as null controls.
