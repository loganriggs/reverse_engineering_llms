# Learned local coordination versus bilinear geometry

Every score targets its own surrogate function on identical trained upstream states/head writes. Global bases are refit separately.

| Local MLP | Matched16 coverage | Wrong-context16 | Global16 | Matched64 | Full tangent | PCA64 cross prediction | PCA256 cross prediction |
|---|---:|---:|---:|---:|---:|---:|---:|
| native | 0.452 | 0.059 | 0.108 | 0.760 | 0.962 | 0.314 | 0.646 |
| joint | 0.452 | 0.059 | 0.108 | 0.760 | 0.962 | 0.314 | 0.646 |
| right0 | 0.297 | 0.025 | 0.034 | 0.692 | 0.976 | 0.263 | 0.599 |
| right1 | 0.292 | 0.029 | 0.040 | 0.681 | 0.976 | 0.266 | 0.601 |
| right2 | 0.283 | 0.028 | 0.038 | 0.675 | 0.976 | 0.266 | 0.604 |
| down0 | 0.338 | 0.026 | 0.036 | 0.712 | 0.973 | 0.275 | 0.610 |
| down1 | 0.334 | 0.027 | 0.033 | 0.708 | 0.973 | 0.273 | 0.609 |
| down2 | 0.324 | 0.027 | 0.031 | 0.714 | 0.974 | 0.277 | 0.611 |
| direction0 | 0.364 | 0.036 | 0.054 | 0.722 | 0.969 | 0.283 | 0.616 |
| direction1 | 0.361 | 0.037 | 0.055 | 0.719 | 0.969 | 0.282 | 0.616 |
| direction2 | 0.348 | 0.036 | 0.052 | 0.723 | 0.969 | 0.285 | 0.616 |

Locked gates: {'invariant_control': True, 'right_pairing_specificity': True, 'down_binding_specificity': True, 'context_rotation_survives': True, 'broad_context_survives': True}. Joint permutation write replay relative RMS 3.82e-07; largest pooled metric change 1.1e-06.

Same local counterfactual task as primary, but fresh test19848–19879. Trained upstream states remain fixed. Nulls depend on the declared canonical convention; no semantic, whole-untrained-network, or downstream language-quality claim.


## H237 — Context rotation is generic, while native channel bindings add response concentration

**Primary null comparison.** At fresh19816–19847 (128 sites, all nine heads),
compare the native local L8 bilinear function with three Right-row shuffles and
three Down-column shuffles. Each matrix's spectrum/marginal weights are preserved,
but different channel bindings are broken. A joint Left/Right-row plus matching
Down-column permutation preserves the function. Every function gets its own
response-derived global basis fit on the same discovery states. All upstream
states/head writes remain from the trained network. Scores target each function's
own responses, not preservation of the original language model's behavior.

| Local function | Matched16 full coverage | Wrong-context16 | Global16 | PCA-context64 cross fidelity |
|---|---:|---:|---:|---:|
| Native | .559 | .063 | .137 | .296 |
| Joint permutation | .559 | .063 | .137 | .296 |
| Right shuffles,3 replicates | .258–.281 | .022–.030 | .033–.035 | .250–.260 |
| Down shuffles,3 replicates | .387–.408 | .029–.032 | .052–.058 | .261–.270 |

Context-specific charts still beat mismatched contexts in every shuffled function,
and all require broad shared context codes. Those properties therefore do not
by themselves establish learned features. Native matched16 coverage exceeds each
Right and Down control by >=.10, passing both predeclared specificity gates.
Uniform-head averages also differ: native .460, Right .243–.263, Down .344–.355;
this is not solely a change in which heads dominate pooled response energy.
Joint permutation replay error3.85e-7, maximum pooled metric change1.27e-6.

**Identified confound.** Bilinear channel factors admit arbitrary scaling and
sign redistribution while preserving the function; the two input factors can
also be exchanged. Separate permutations can mix these arbitrary parameter gains.
H238 therefore repeats the discrimination after canonicalizing those freedoms,
and adds a direction-only output shuffle that retains each channel's gain.
The primary finding is not interpreted as orientation-specific until that check.

## H238 — The coordination effect survives factor canonicalization and gain-preserving controls

**Adaptive follow-up, locked before fresh test19848–19879.** Normalize input
factor rows, fix their signs using the largest-magnitude coordinate, transfer the
signed norms into Down, and order each unordered input pair lexicographically.
This preserves the native function and removes signed-scaling and input-pair
exchange ambiguities. Repeat three canonical second-factor and Down-column
shuffles; add three Down-direction shuffles preserving each original canonical
column norm. The latter do NOT preserve Down's spectrum. Other shuffles preserve
spectra of canonical matrices, not necessarily the original parameter matrices.
These are declared null conventions, not a unique physically privileged gauge.

| Local function | Matched16 full coverage | Wrong-context16 | Global16 | PCA-context64 cross fidelity |
|---|---:|---:|---:|---:|
| Native canonical | .452 | .059 | .108 | .314 |
| Joint invariant | .452 | .059 | .108 | .314 |
| Second-factor shuffles | .283–.297 | .025–.029 | .034–.040 | .263–.266 |
| Whole-output-binding shuffles | .324–.338 | .026–.027 | .031–.036 | .273–.277 |
| Output-direction shuffles, gains kept | .348–.364 | .036–.037 | .052–.055 | .282–.285 |

Native-minus-second-factor coverage is .155–.170 across permutations; their
conditional paired row-bootstrap95% intervals all exclude zero (e.g. replicate0
[.122,.199]). Whole-output-binding gaps are .114–.129. Gain-preserving output-
direction gaps are .088–.105, with intervals [.072,.109], [.077,.111], [.088,.122].
The direction-only comparison is descriptive, not an additional passed >=.10
all-replicate gate. Uniform-head coverage is native .420, second-factor .262–.282,
whole-output .304–.317, direction-only .338–.352. The native difference between
primary and follow-up panels is sampling variation, not a canonicalization effect.
Canonical native function replay relative RMS1.35e-6; joint replay3.82e-7 and
maximum pooled metric change1.10e-6. Twenty tests pass, including signed scaling,
pair exchange, channel permutation, spectrum and direction-norm controls.

**Interpretation.** Two claims separate cleanly. Rotating context-dependent maps
and broad context requirements survive disrupted local coordination: generic
bilinear geometry plus trained upstream inputs can produce them. Native bindings
nevertheless make actual source responses more concentrated in their strongest
local directions. This depends on both input-factor coordination and output binding,
and cannot be reduced to arbitrary factor scaling or output gain placement alone.
It does not identify semantic neurons, unique tensor factors, or a small global
feature list. The upstream network remains trained; these are not comparisons to
an entirely untrained model. Three permutations are sensitivity checks, not a
calibrated randomization distribution or independent trained-model replications.

**What remains ambiguous.** Macro isotropic source-tangent rank16 energy is .251
for native versus about .232–.240 for canonical shuffles, smaller differences than
those for actual responses. This suggests execution-specific alignment, but the
isotropic-spectrum and actual-response scores weight states differently. Their
difference is not by itself a pure directional-alignment estimate.

**Next discriminator, not another decoder-fitting exercise.** Reuse frozen
canonical weights/permutations on new rows19880–19911. Within each identical state
and head, compare actual head payloads, different-row same-position shuffled
payloads, and isotropic Gaussian payloads in the same raw source span. Match each
payload's norm to the actual head write, use at least two fixed sham realizations,
and evaluate both small/tangent and full-edit responses. Compare rank16/64 local
chart coverage across native and shuffled MLPs. Report pooled, per-head and
per-site coverage to expose reweighting effects. If native's extra concentration
appears mainly for actual payloads, learned source–state alignment is implicated;
if it persists equally for Gaussian payloads, intrinsic local-operator concentration
is the better account. Same-position shuffles distinguish contextual pairing from
marginal source anisotropy while controlling position. No changed-output selection,
no fitted decoder, no semantic naming. All normalization, chart cost, and trained-
upstream limitations remain explicit. Only120 continuation-fresh rows remain
before the verified20000-row corpus bound; do not silently exceed/recycle them.

**Records.** Primary and canonical JSON/NPZ/logs/reports, frozen permutations and
per-surrogate global bases, per-head results, paired row arrays/contrasts, macro
head audit and PNG/SVG comparison. All current experiments completed; goal open.
