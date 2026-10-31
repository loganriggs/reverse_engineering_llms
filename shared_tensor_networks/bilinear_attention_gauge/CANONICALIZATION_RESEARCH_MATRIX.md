# Canonical reduction and legibility: experiment matrix

## Question and objects

There are two choices to make independently. The **reduction metric** says which
changes to the attention function count as expensive. The **legibility prior**
says which coordinates or modules we prefer among sufficiently faithful
reductions. Weighted trace gauge supplies reproducible coordinates for one
specified network and metric, but cannot alone select a unique decomposition
of the function across QK branches or heads.

The first-layer object is the exact multi-head query/key/output response,
including both QK paths, RoPE, and OV. For one-hot tokens this can be studied
without sampled sequences. For the 546M model a dense full-vocabulary
query–key–output tensor is prohibitive, but the factors support implicit
contractions and Monte Carlo estimates with explicit coverage/error checks.
Sequence weighting is a **different objective**, not a computational
requirement for defining attention.

For a declared measure `mu`, fit a reduced function `F_r` by minimizing
`E_mu ||F-F_r||²` at each rank/resource budget. In a second pass, optimize a
legibility cost `L(F_r, coordinates)` while bounding the increase in that
same function error. Exact gauge rotations make the bound zero. Any claimed
gain must then survive the *other* measures and held-out interventions; the
training objective is not itself the validation criterion.

## Stage 1: choose what reduction preserves

| Metric | What it optimizes | Necessary challenge set |
|---|---|---|
| Uniform token pairs | Function on a broad token domain, including rare tokens | Independent uniform pairs; rare-token and position-gap strata |
| Natural corpus pairs | Expected response on observed text | Disjoint corpora, documents, and topic/grammar shifts |
| Floored or mixed measure | Trade broad coverage against natural usage | Sweep floor/mixing weight; report both uniform and natural error |
| Branch-intervention ensemble | Preserve QK1/QK2 edits as well as native response | Hold out edit strengths, compositions, and edit families |
| Downstream logits or KL | Preserve final model behavior | Separate pair/head-output fidelity from final KL and task accuracy |

The uniform metric is data-free in the sense of needing only weights and a
declared token domain. Corpus and downstream metrics depend on a data
distribution. For each rank and parameter budget, report the complete
cross-metric Pareto frontier rather than naming one metric universally best.

## Stage 2: choose what makes components legible

Evaluate these at a **fixed reduced function first** using exact gauge changes
when possible; then allow the reduced function to change within a stated error
budget. This isolates coordinate choice from lost information.

| Prior or objective | Intended structure | Main failure check |
|---|---|---|
| Native weight sparsity | Few physical coefficients | Exact gauge and QK-branch-swap sensitivity |
| Canonical-bond sparsity | Sparse coordinates after bond gauge fixing | Seed stability; cross-head redistribution |
| Token or pair-response sparsity | Each axis affects a compact set of token pairs | Frequency-matched controls and held-out token forms |
| Group sparsity over shared dictionary | Reused subspaces with selective gates | Whole-task holdouts, group-count sensitivity, independent starts |
| Intervention-response sparsity | Axes with selective causal effect profiles | Unseen edits, dose matching, downstream propagation |
| Computation or description-length penalty | Few reusable nodes/edges across heads | Same-function alternative factorizations and graph recovery |

No sparsity score is itself an interpretability score. Report reconstruction,
task prediction, causal fidelity, axis/group stability, known-rule recovery,
and discovery of withheld rules separately. Compare priors at matched
reconstruction error, parameter count, rank, and intervention dose. Model
selection may see training tasks and validation edits; discovery claims require
whole withheld task families and fresh test interventions.

## Testbed ladder and decision rule

1. **Hand-built attention controls:** two heads with common and rare rules,
   shared versus duplicated computation, branch swapping, and exact token
   function. These prove algebraic invariances and expose obvious biases but
   cannot demonstrate discovery in a trained network.
2. **Trained toy language:** one- and two-layer bilinear attention models with
   multiple nonredundant outputs, rare compositional combinations, variable
   token counts, and at least one context-dependent interaction. Hold out
   complete combinations and intervention types. Recover known rules without
   using their labels for unsupervised fitting or selection.
3. **Small real checkpoints:** vocabulary-small models where all token pairs
   can be enumerated, to benchmark sampling and distinguish approximation
   artifacts from genuine metric tradeoffs.
4. **546M model:** first-layer all-head dictionary, then full 18-layer
   propagation on FineWeb, WikiText-103, TinyStories, rare-token strata, and
   edited inputs. Measure whether proposed components predict selective
   behavior better than random, frequency-matched, and same-gauge controls.

Advance a prior only if it retains fidelity on the declared broad and natural
domains, improves rule recovery on trained toys, and makes a prediction about
new inputs/interventions that survives controls in the real model. If local
pair labels fail in the full model, report the failure rather than optimizing
the same label on the test documents.

## What the existing experiments already show

The [factorial toy](FACTORIAL_PLAN_AND_TOY_RESULTS.md) shows that uniform and
usage-weighted rank-one cuts retain different common/rare rules; the prior
sweep finds a real fidelity–intervention tradeoff, not a free gain. The
[trained toy sequence](THREE_COUNT_CANONICAL_RESULTS.md) shows rank competition
and differing uniform, frequency, and supervised-CE outcomes. The
[nonlinear-readout follow-up](THREE_COUNT_NONLINEAR_METRIC_RESULTS.md) shows
that optimizing a downstream bilinear output at rank two can raise overall
accuracy while eliminating a withheld rare-combination behavior across ten
trained attention checkpoints. The
[all-head real-model comparison](ALL_HEAD_SHARED_KEY_RESULTS.md) shows useful
shared-key compression; the [function-aware fit](FUNCTION_AWARE_SHARED_KEY_RESULTS.md)
improves natural fidelity but loses much of its advantage under a QK2 edit.
The [exact-gauge prior comparison](ALL_HEAD_GAUGE_PRIOR_RESULTS.md) and
[label transfer test](ALL_HEAD_LABEL_TRANSFER_RESULTS.md) show that sparse
coordinates and label enrichment can disagree. The
[full-model causal test](ALL_HEAD_COUNTERFACTUAL_CAUSALITY_RESULTS.md) finds a
small, reproducible punctuation-edit effect for one selected component;
whether its proposed query/key direction generalizes is a separate test.

The immediate experiment is a held-out four-cell query/key form factorial for
two opposite-labelled components. Its selected-pair and full-model
key-edit-response readouts must be reported separately. If they disagree,
the next legibility target should be a named task or a full-context response
signature, rather than another pair-form sparsity proxy.
