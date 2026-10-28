# Direct-objective information grid

All 18 fits frozen before collecting test. Scores concern four fixed L8 coordinates, not the whole MLP.

| Run | Selected step | Natural R2 | Even | Odd | Half | Pairs | Gaussian | Isotropic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_32 | 0 | 0.763 | 0.434 | 0.418 | 0.391 | 0.419 | 0.308 | 0.409 |
| baseline_64 | 0 | 0.844 | 0.573 | 0.549 | 0.525 | 0.546 | 0.424 | 0.520 |
| baseline_128 | 0 | 0.918 | 0.730 | 0.700 | 0.686 | 0.698 | 0.562 | 0.637 |
| r32_lambda0_spectral | 400 | 0.846 | 0.459 | 0.469 | 0.437 | 0.465 | 0.278 | 0.360 |
| r32_lambda0_perturbed_spectral | 400 | 0.846 | 0.457 | 0.472 | 0.438 | 0.466 | 0.280 | 0.360 |
| r32_lambda1_spectral | 400 | 0.817 | 0.517 | 0.446 | 0.440 | 0.453 | 0.299 | 0.398 |
| r32_lambda1_perturbed_spectral | 400 | 0.816 | 0.519 | 0.449 | 0.443 | 0.455 | 0.301 | 0.396 |
| r32_lambda4_spectral | 400 | 0.793 | 0.517 | 0.436 | 0.433 | 0.445 | 0.305 | 0.409 |
| r32_lambda4_perturbed_spectral | 400 | 0.792 | 0.520 | 0.438 | 0.436 | 0.446 | 0.308 | 0.408 |
| r64_lambda0_spectral | 360 | 0.893 | 0.599 | 0.580 | 0.558 | 0.575 | 0.408 | 0.484 |
| r64_lambda0_perturbed_spectral | 360 | 0.893 | 0.598 | 0.581 | 0.558 | 0.575 | 0.406 | 0.485 |
| r64_lambda1_spectral | 400 | 0.882 | 0.628 | 0.571 | 0.561 | 0.571 | 0.414 | 0.500 |
| r64_lambda1_perturbed_spectral | 400 | 0.881 | 0.628 | 0.570 | 0.560 | 0.571 | 0.411 | 0.501 |
| r64_lambda4_spectral | 400 | 0.866 | 0.627 | 0.563 | 0.555 | 0.565 | 0.420 | 0.514 |
| r64_lambda4_perturbed_spectral | 400 | 0.865 | 0.626 | 0.563 | 0.554 | 0.564 | 0.416 | 0.514 |
| r128_lambda0_spectral | 280 | 0.937 | 0.737 | 0.723 | 0.706 | 0.719 | 0.565 | 0.624 |
| r128_lambda0_perturbed_spectral | 280 | 0.937 | 0.737 | 0.720 | 0.705 | 0.717 | 0.566 | 0.620 |
| r128_lambda1_spectral | 400 | 0.935 | 0.756 | 0.714 | 0.707 | 0.714 | 0.561 | 0.624 |
| r128_lambda1_perturbed_spectral | 320 | 0.934 | 0.756 | 0.710 | 0.705 | 0.711 | 0.562 | 0.623 |
| r128_lambda4_spectral | 200 | 0.924 | 0.754 | 0.705 | 0.699 | 0.706 | 0.563 | 0.638 |
| r128_lambda4_perturbed_spectral | 200 | 0.923 | 0.754 | 0.702 | 0.698 | 0.705 | 0.562 | 0.635 |

Each rank is per output scalar. The decoder is a single symmetric quadratic per scalar and has no source identity input. Exact mean-state linear terms and discovery mean calibration are retained. Objective weights use per-axis discovery scales; table scores pool unnormalized squared errors. Two nearby starts do not rule out optimization failures. Row-bootstrap intervals in JSON condition on trained models and fixed payload draws.


## H229 — Direct response fitting mainly learns the supplied examples and sources

**Question.** Does a poor spectral objective explain the missing source responses?
An information grid tested ranks 32/64/128, response weights 0/1/4, and spectral
or 5%-perturbed spectral initialization: 18 fits plus three unchanged baselines.
All checkpoints were selected using new-text validation and even heads, then frozen
before test collection. Every decoder is one globally integrable symmetric
quadratic per output, with no source identity input. Signed coefficients can
change sign during optimization; the parameterization does not fix inertia.

**Results.** On the fresh 512-site test panel, the spectral-initialization runs give:

| Rank | Spectral natural R2 / odd response | Value-only natural R2 / odd response | Value+response natural R2 / odd response | Value+response even response |
|---|---|---|---|---:|
| 32 | .763 / .418 | .846 / .469 | .817 / .446 | .517 |
| 64 | .844 / .549 | .893 / .580 | .882 / .571 | .628 |
| 128 | .918 / .700 | .937 / .723 | .935 / .714 | .756 |

Value+response uses weight 1. Weight 4 yields odd-head fidelity .436/.563/.705.
Both nearby initializations show the same pattern. No response-trained condition
passes the locked >=.10 odd-head gain plus <=.05 natural-loss gate against its
rank-matched spectral baseline. Compared with corresponding value-only training,
all 12 response-trained fits have *lower* odd-head fidelity. Post-result paired
row-bootstrap intervals for weight-1 spectral runs are [-.035,-.011],
[-.016,-.003], and [-.016,-.003]. These are conditional intervals, not
multiplicity-corrected hypothesis tests.

Training even-head fidelity reaches .882/.958/.987 for weight 1, while validation
on new text reaches .475/.519/.643. On final test text, even-head fidelity is
.517/.628/.756. Thus there is substantial example generalization error in addition
to weaker transfer to untrained heads. Value-only fitting improves natural
prediction but can hurt Gaussian/isotropic responses: at rank32, Gaussian fidelity
falls .308→.278 and isotropic .409→.360. Weight-1 response fitting yields
.299/.398, still below their unchanged baselines. Full results and histories are
retained rather than selecting a test winner.

**Interpretation.** T08's strong easy-fix version weakens: directly fitting finite
responses does not close transfer gaps in this setup. Data/support coverage
(T07 broadly construed) and capacity (T11) remain live. Larger ranks consistently
help, but no lower bound is established. Two nearby starts are stable, yet 11/18
runs choose the last allowed checkpoint; longer optimization remains unresolved.
The large train/validation gap makes sample coverage a more informative next
axis than merely increasing optimizer steps. Source-declared hybrids are not
proven intrinsically necessary, and this experiment does not identify semantics.

**Controls and cost.** Native coefficient replay relative RMS 8.30e-7; finite
response replay 9.45e-6; head-sum errors <4e-7. Three new tests verify finite
integrability/cycle closure, the mean-calibrated pair-difference objective, and
response gradients. Rank is per scalar: up to 4(r+1) input reads across four
outputs, not a shared r-dimensional module. Mean and exact linear terms are fixed;
full Q matrices supply training targets, not decoder inputs. Training-history
'natural' scores on the intervention-discovery panel are not scores on the
separate 2048-site natural-value fitting panel. Response training scores are
in-sample. Per-axis normalization in the fitting objective differs from the
pooled-error fidelity in the table.

**Next information experiment.** A 2×2 grid: 128 versus 512 training text rows,
and five even heads versus all nine heads. Keep rank64, objective weight1, and
optimizer update budget fixed; inspect longer validation curves as a secondary
optimization check. Use nested row subsets and new validation/test rows. Evaluate
formerly withheld odd heads separately, explicitly noting they are only unseen
sources in the five-head condition. More rows helping across heads implicates
example coverage; adding heads selectively helping odd heads implicates support
coverage; neither helping despite low training error motivates regularization or
synthetic support supervision. Hold out isotropic payloads throughout. Report
unique examples, updates, validation gaps, all cells, and read/storage costs.
Suggested unused continuation rows: training18536–19047, validation19048–19175,
test19176–19303; reserve in a new config before fitting.

**Record.** BILIN18_L8_DIRECT_OBJECTIVE_GRID.md, full JSON histories, frozen factor
arrays, row-error arrays, and post-result contrasts JSON. Validation rows
18280–18407; test18408–18535; original training panels reused. Only four frozen
L8 axes and local coefficients, with no new full-suffix or whole-MLP claim.
