# Matched continuous-block adaptation: natural fidelity is worse; causal transfer pending

M112 completes four fixed adaptation runs and their natural-text evaluation. The proposed continuous block map eliminates hard selection jumps, but after this matched adaptation it preserves native outputs substantially less well than the hard-block controls. Task intervention-response evaluation remains pending; this report does not close the joint comparison.

## What was held fixed

Hard and continuous maps start from identical historical M36 block-model parameters within each seed71/72. Both have296,192trainable parameters,32groups of4coordinates, and an eight-block budget. Both receive the same132Adam updates:66natural-document updates alternating66task updates, with both task endpoints averaged. The training tasks span11families.

The objective and calibration are unchanged from M69: natural output KL; task intact KL plus0.0930050359836775 times family/site-balanced encoded-change response error. Optimizer defaults, learning rate, seed-specific examples and ordering, dictionary normalization, and natural-data split are identical. No architecture-specific gain or weight calibration and no extra training steps were used.

This is matched adaptation from an already hard-trained starting representation, not a from-scratch architecture comparison. Equal parameters do not mean equal initial functions or equally favorable initial conditions. The continuous map starts attenuated and has different gradients. The result cannot establish a universal advantage of hard selection or convergence limits of continuous maps.

Both rerun hard controls match the historical M69 final checkpoint tensors bit-for-bit. This provides a strong check that the comparison really used the earlier training procedure. All four runs retained every final state; no seed or checkpoint selection occurred.

## Gradient and training verification

Four full-parameter directional checks pass through the native Transformer causal loss, with three fixed finite-difference scales per model. One continuous71 trial crosses a branch boundary and is explicitly retained as nonsmooth; all four models have stable trials. Stable relative derivative discrepancies are below0.31%. Threshold derivatives remain in the graph, the native model stays frozen, and parameters are restored exactly before training.

The first check used an overstrict ordered Top9 signature: reordering within selected groups does not change the hard map. It failed before any training. The signature was corrected to the selected set plus threshold-block identity only for the continuous map, without changing examples, epsilons, or tolerances. A subsequent output-serialization error was fixed by using the existing string/Path-compatible hash helper. Both failures are recorded in validation/; they are not hidden successful runs.

The training audit verifies528updates, schedules, all11families per state, objective arithmetic, historical optimizer defaults, final/progress state agreement, finite tensors and unit decoder columns. All recorded training examples use32active coordinates for both maps. Online first/last losses use different examples and are not a convergence measure.

## Held-out natural-text results

All32reserved natural documents (4096positions) were evaluated, replacing MLP1 product writes at all positions while preserving the separate bias. These are existing development evaluation documents, not a newly collected confirmation set.

| Seed | Map | Output KL | CE increase | Top1 agreement | Local write error | Mean code energy |
|---|---|---:|---:|---:|---:|---:|
| 71 | hard | 0.1655 | 0.1588 | 78.71% | 0.4933 | 3.7106 |
| 71 | continuous | 0.4636 | 0.4662 | 65.80% | 0.5723 | 1.4089 |
| 72 | hard | 0.1811 | 0.1829 | 77.37% | 0.4947 | 3.6633 |
| 72 | continuous | 0.4971 | 0.4974 | 64.65% | 0.5800 | 1.3503 |

KL is native-output-to-replacement-output KL, averaged over positions and documents. CE increase is relative to native mean CE3.2603. Local error is squared reconstruction error divided by native write energy per document, then averaged; it is not a causal-response metric. Code energy is mean squared latent coordinate, not decoded output energy.

For each seed, continuous substitution is worse on every one of the32documents for output KL, CE increase, and local reconstruction error. Thus the aggregate difference is not caused by a single document. Both maps still use32active coordinates; the continuous models’ code energy remains substantially smaller. This is consistent with an amplitude/fidelity tradeoff but does not by itself identify the cause of the fidelity loss.

The natural evaluation made160forwards; all32native write tensors replay the historical cache exactly, and all64hard-model/document metric rows reproduce the historical evaluation exactly. Independent NumPy audits cover all128output/reconstruction conditions and1024metrics. Maximum metric discrepancy3.56e-15; maximum independent reconstruction discrepancy2.69e-7. All native logits and reconstructions are retained in the local artifact cache for reproducibility.

## Conceptual consequence

Continuity is a meaningful prior because it rules out an observed extractor artifact. It is not a sufficient prior for a faithful computational representation. Removing the artificial boundary can sacrifice amplitudes or other directions that downstream computation uses. The current result argues against promoting this continuous candidate based on geometric cleanliness alone.

It says nothing yet about native redundant channels, cancellation, semantic manifolds, or true reusable units. It also does not refute continuous featurizers generally: initialization, fixed adaptation budget, and the rank-adaptive shrinkage parameterization limit the scope.

## Required remaining evaluation

The predeclared evaluation remains unchanged: a fixed11-family training diagnostic, all14source-group holdout pairs, and all56whole-task pairs; source and final sites separately; all7whole-task families with the prespecified competent5 also shown. Measure intact retention separately from native-background encoded-change response fidelity. Report pooled and equal-family errors.

For each seed/pair/site, include the hard encoded change scaled to the continuous change norm using only the candidate vectors. This is required to distinguish a better direction from merely a smaller intervention. Retain zero cases and explicit ineligibility. No loss-weight retuning, extra updates, or favorable-seed selection in response to the natural result. The task families remain excluded from feature fitting.

Relevant artifacts: validation/continuous_training_protocol.json, continuous_training_evaluation_protocol.json, continuous_causal_gradient_checks.json, continuous_training_results.json, continuous_training_audit.json, continuous_natural.json, and continuous_natural_audit.json. Full checkpoints and outputs remain local with hashes in the receipts.
