# Joint verb, object, and nested-bracket language

This versioned generator combines three exact visible-prefix tasks in the
same streams: contested verb prediction from v2, contested object prediction
from v3, and variable-length nested-bracket closing. The first two tasks
each mix several possible sources; the third requires the latest unclosed
opener. It tests whether a decomposition that separated verb and bracket
behavior on the earlier two-task model still works when another contested
rule site competes for the model's capacity. It is not a realism claim.

[`surface_mixture_v3_nested_varlen.py`](surface_mixture_v3_nested_varlen.py)
composes the existing oracles using cooperative multiple inheritance. The
joint sampler uses the 319-token nested vocabulary and labels verb, object,
optional closer, deterministic inner closer, and deterministic outer closer
positions. The object lexicon is restricted to eight tokens for the first
model capability test; this does not restrict the model's 319-token output
vocabulary.

The [generator audit](surface_mixture_v3_nested_varlen_audit.json) replays
every prefix in 1,000 streams of length 128. It independently recomputes the
object mixture, checks the unchanged parent verb oracle, and verifies the
stack's deterministic closer. It finds 6,409 verb sites, 6,239 object sites,
450 deterministic inner closes, and 943 deterministic outer closes. Outer
opener-to-closer distances are 10, 14, and 18 tokens. Existing builders
produce valid verb, object, opener, and nuisance edit pairs across 15 kinds.
The original v3 and variable-length nested audits still pass after the
constructor extension.

The [frozen FineWeb form comparison](surface_mixture_v3_nested_varlen_fineweb_form.json)
is strongly negative. Across 480 nominally 128-token windows, the toy data
has 311 distinct alphabetic words versus 8,616 in the FineWeb panel. Square
brackets occur in 69.4% of toy windows versus 1.5% of FineWeb windows; the
toy data has no newlines versus 86.0% of the FineWeb windows. Tokenizers
differ, so these are coarse form checks, not semantic task prevalence. The
bracket task remains an artificial computational probe.

## First model capability test

A width-64, four-head, two-block bilinear transformer with RMSNorm trained
for 2,000 steps at batch 32 and length 128. Ordinary next-token CE is
augmented with exact-oracle expected CE at verb and object slots and at
deterministic bracket closes, each with weight one. The [training record](surface_mixture_v3_nested_varlen_depth2/results.json)
and checkpoint are retained. On an independent 256-stream panel:

| Site | Exact-oracle KL (nats) |
|---|---:|
| All positions | 0.0744 |
| Verb | 0.1571 |
| Verb with induction available | 0.2004 |
| Object | 0.1607 |
| Object with induction available | 0.2146 |
| Inner bracket closer | 0.0005 |
| Outer bracket closer | 0.0003 |

The [held-out edit evaluation](surface_mixture_v3_nested_varlen_depth2/heldout_edits_seed20261105.json)
contains 120 inner and 236 outer relevant opener edits, each with a
corresponding no-effect control. Mean effect-L1 error is 0.0016 and 0.0008
for inner and outer relevant edits, with 100% correct edited closer.
Nuisance bracket edits have effect-L1 below 0.0005. This model is highly
competent on the bracket task.

Verb and object causal effects are less faithful. Representative exact-oracle
versus model total-variation effects and their L1 errors are:

| Valid edit | Oracle TV | Model TV | Effect L1 error |
|---|---:|---:|---:|
| Lexical verb | 0.566 | 0.603 | 0.372 |
| Verb induction | 0.250 | 0.159 | 0.350 |
| Previous-place verb | 0.179 | 0.232 | 0.310 |
| Object adjective | 0.456 | 0.471 | 0.307 |
| Object induction | 0.250 | 0.114 | 0.402 |
| Previous-event object | 0.178 | 0.218 | 0.284 |
| Nuisance determiner / object-place | 0 / 0 | 0.010 / 0.015 | 0.020 / 0.030 |

The model predicts all three site types, but it does not reproduce the full
verb and object rule mixtures. In particular, a decomposition could appear
to have a clean induction circuit by explaining a weak approximation rather
than the intended rule. Keep exact-oracle competence and valid-edit fidelity
as gates before testing canonicalization or sparse circuit extraction on
this checkpoint. A deeper capability run is being evaluated separately.

This is task prediction and causal-fidelity evidence on **known** synthetic
tasks. It is not reconstruction evidence, a unique canonical circuit, or
genuine discovery of a new task. Future basis and sparse-factor comparisons
need independent fitting streams, a development panel for choosing modes,
held-out edits, and whole-task-family holdouts if claiming discovery.

## First-layer score-basis comparison

The [shared-basis evaluation](surface_mixture_v3_nested_varlen_depth2/shared_token_basis_seed20261110.json)
uses one query-token and one key-token eigenspace per head, shared across all
relative lags of the exact first-layer token-pair score table. It changes
only that table; the second layer and output remain native. A full-table
replay gives mean teacher KL below 1e-6 on 128 independent evaluation
streams, validating the replay up to small rotary-rounding differences.
Basis fitting uses 2,048 separate unlabeled streams. The no-bracket fit uses
the same v3 verb/object language but no nested tails. Evaluation uses a
third stream seed and an independent 64-stream edit panel.

| Rank-8 weighting | Whole-model teacher KL | Verb oracle KL | Object oracle KL | Outer opener-edit L1 |
|---|---:|---:|---:|---:|
| Native / full table | ~0 | 0.1605 | 0.1607 | 0.0010 |
| Uniform vocabulary, data-free | 0.01189 | 0.1725 | 0.1634 | 0.0279 |
| Unlabeled joint streams | **0.00208** | **0.1632** | **0.1610** | **0.0022** |
| Unlabeled streams without brackets | 0.00497 | 0.1633 | 0.1616 | 0.0921 |

At rank 16, outer edit errors are 0.0034, 0.0011, and 0.0225 for uniform,
joint, and no-bracket weighting. This replicates the importance of the
specified input metric in a deeper model with a third contested task. It
does **not** establish a uniquely interpretable basis. Verb and object edit
errors remain close to the native model's, including its weak induction
effects; matching an incomplete teacher computation is not oracle recovery.
The data-aware fit is task-blind but not task-held-out: its unlabeled streams
contain bracket tokens, and the trained model saw bracket supervision.

## Three-block capability and basis check

A separate three-block model trained 3,000 steps under the same task weights,
width, and batch setup. This is a stronger capability run, not an isolated
depth comparison: both depth and number of optimization steps changed. On a
fresh 256-stream panel, [exact-oracle and edit evaluation](surface_mixture_v3_nested_varlen_depth3/heldout_edits_seed20261115.json)
gives all-position KL **0.0440**, verb KL **0.0828**, object KL **0.1100**,
inner-close KL **0.0009**, and outer-close KL **0.0001**. Relevant inner and
outer opener edits have effect-L1 0.0022/0.0005, with 100% edited accuracy.

Several verb/object rule edits improve: lexical verb effect-L1 0.277,
previous-place verb 0.219, and object adjective 0.254. Object induction is
still the decisive failure: its oracle TV is 0.250, model TV 0.123, and
effect-L1 0.389 on 566 supported edits. The depth-three checkpoint is more
competent at average prediction but still an incomplete object-induction
target. A causal-supervision continuation and an extra object-supervision
control are the next comparison.

The same [shared-basis evaluation](surface_mixture_v3_nested_varlen_depth3/shared_token_basis_seed20261110.json)
on this checkpoint again favors unlabeled joint-stream weighting at rank 8:

| Rank-8 weighting | Whole-model teacher KL | Verb oracle KL | Object oracle KL | Outer opener-edit L1 |
|---|---:|---:|---:|---:|
| Native / full table | ~0 | 0.0822 | 0.1113 | 0.0004 |
| Uniform vocabulary | 0.00507 | 0.0880 | 0.1165 | 0.0018 |
| Unlabeled joint streams | **0.00112** | **0.0848** | **0.1127** | **0.0005** |
| Unlabeled streams without brackets | 0.00778 | 0.0847 | 0.1139 | 0.0053 |

The bracket gap is smaller than for the two-block model because all three
approximations are fairly good here, but the ordering persists. This is a
first-layer function-level reduction with a specified input metric, not a
claim of a unique or sparse physical circuit.

## First causal-supervision continuation

The depth-three model's weak object-induction response led to an explicit
counterfactual objective. For a current object site with a prior same-subject
sentence, [`object_induction_counterfactuals`](train_surface_mixture.py)
changes that prior object's token to another supported value and computes
the exact new next-object distribution. A smoke audit verified the edited
prefixes stay in the generator's support and the targets equal its oracle.
The original object-slot loss remains active. All continuations below start
from the same 3,000-step checkpoint, use the same new training-data seed,
and take 1,000 further updates. The counterfactual arm has an extra forward
pass, so step matching is not compute matching.

| Continuation | Object oracle KL | Object induction model TV (oracle 0.250) | Induction effect L1 | Adjective effect L1 |
|---|---:|---:|---:|---:|
| Original 3,000-step model | 0.110 | 0.123 | 0.389 | 0.254 |
| Object soft weight 2, no counterfactual | **0.093** | 0.123 | 0.366 | **0.223** |
| Object soft weight 1, counterfactual weight 1 | 0.142 | **0.166** | **0.320** | 0.310 |

All values come from the same independent 256-stream, 566-object-induction-
edit panel, with [control outputs](surface_mixture_v3_nested_varlen_depth3_object_soft_control/heldout_edits_seed20261115.json)
and [counterfactual outputs](surface_mixture_v3_nested_varlen_depth3_causal_continue/heldout_edits_seed20261115.json).
The targeted objective improves the intended causal dependency beyond
additional ordinary object supervision, but it sacrifices average object
prediction and the adjective rule. Thus neither continuation dominates.
A smaller counterfactual term with ordinary object supervision is the next
balance check. Because the weight choice responds to this panel, treat it as
development data and use a fresh edit seed for the actual validation.

That validation used [a new shared 256-stream panel](surface_mixture_v3_nested_varlen_depth3_causal025_continue/heldout_edits_seed20261125.json)
with 592 object-induction edits. All arms use the same panel and pair set:

| Model | Object KL | Induction model TV (oracle 0.250) | Induction effect L1 | Adjective effect L1 |
|---|---:|---:|---:|---:|
| Original | 0.115 | 0.131 | 0.383 | 0.258 |
| Extra ordinary object supervision | **0.096** | 0.128 | 0.355 | **0.229** |
| Counterfactual weight 1 | 0.143 | **0.178** | **0.309** | 0.314 |
| Extra ordinary object supervision + counterfactual weight 0.25 | 0.099 | 0.132 | 0.353 | 0.247 |

The weight-1 intervention improves the specific causal dependency on both
panels, but at a substantial cost elsewhere. Weight 0.25 largely returns
to the ordinary-supervision control; it does not combine the best of both.
Stop this local weight sweep. A future method should explicitly constrain
the original object distribution or rule-specific effects while learning
the edited induction response, then use new development and validation
panels. These are still known-task supervision experiments, not evidence
of unsupervised task discovery.
