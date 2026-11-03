# Full-path weight score versus causal head necessity

This completes the bounded single-rule comparison proposed in the
[delimiter pilot](CORPUS_DELIMITER_MODEL_AND_WEIGHT_PILOT.md). The purpose is
to ask whether a calculation from the *trained weights* identifies the heads
that support a **behaviorally verified** punctuation-pair rule. It does not
establish a general unsupervised rule-discovery method.

Five independent, width-64, two-block bilinear models were trained on the
same whole-document-split natural-text transformation, with no normalization
and 50× loss weight at the known delimiter sites. Their held-out paired
round/square two-way accuracies were 88.3%, 88.6%, 76.0%, 87.4%, and 89.5%.
They learned a *relative* closer preference; full-vocabulary top-1 remains
much lower. All use the same known rule and train/test source split, so the
five seeds are a robustness check, not five independent tasks.

## What the weight calculation includes

The [full-path probe](probe_delimiter_full_weight_path.py) first computes a
token table from the embedding and first bilinear MLP, with the first
attention output fixed to zero. For a hypothetical sequence containing one
opener, zero middle positions, and a query token, it exactly evaluates the
second attention's bilinear QK product, OV transport, residual, final
bilinear MLP, and the `)`-versus-`]` readout. It measures how much each
second-layer head contributes to the change in closer margin when ` (` is
replaced by ` [`.

This calculation uses no sampled activations for its uniform-vocabulary
version, but it **is task-conditioned**: it specifies the opener and closer
token pairs. A second version weights query tokens and distances by the
training rule sites. The script checks its algebra against direct model
forwards for distances 1 and 8. An intermediate QK×OV→readout score omits
the final MLP to locate where ranking information enters.

The [causal probe](probe_delimiter_weight_vs_causal_heads.py) separately
removes each attention head and each whole block on untouched paired test
documents. Four seeds pass a prespecified approximation gate: at least 70%
paired accuracy, at least 90% correct-direction opener shifts, and at most
three percentage points change when the *entire* first attention block is
removed. Seed 71413 fails the last gate: each first-layer head alone costs
at most 1.8 points, but removing all four costs 9.0. That is a concrete
redundancy/interaction warning; single-head ablations cannot justify
discarding a whole layer.

The primary causal comparison is restricted to **152** held-out pairs with
the exact leading-space opener tokens and pure one-character closer tokens
used in the weight calculation. The broader 334-pair ablations are also
recorded in JSON; they give the same top-head match counts and accuracy
ranking percentages.

| Score for second-attention heads | Correct top causal head, four eligible seeds | Pairwise agreement with token-matched accuracy-drop ranking | Pairwise agreement with token-matched margin-drop ranking |
| --- | ---: | ---: | ---: |
| QK opener-difference RMS | 1/4 | 69.6% | 75.0% |
| QK difference divided by overall score RMS | 1/4 | 52.2% | 45.8% |
| QK×OV→readout, uniform vocabulary | 3/4 | 78.3% | 91.7% |
| QK×OV→final MLP→readout, uniform vocabulary | 3/4 | 82.6% | 95.8% |
| Same full path, training-site query/distances | 3/4 | 82.6% | 95.8% |

These are tiny rank comparisons: four heads per seed, and 23 usable
accuracy-ranking head pairs after ties. The table is descriptive, not a
statistical claim of method superiority. QK×OV/readout supplies much of
the improvement over QK magnitude; the final MLP helps the aggregate
pairwise order but changes which seed has the top-head mistake. Weighting
by training-site contexts did not change aggregate rankings in this pilot.
The data-free version uses a uniform vocabulary and four relative
distances, which is an artificial prior. The full-path score still omits
ordinary natural context and cannot prove the exact circuit in documents.

The most useful conclusion is methodological: **canonicalizing or ranking
only the attention pattern can miss the computation.** A task-conditioned
weight score that carries a signal through OV, downstream nonlinear
bilinear computation, and readout has a better chance of locating causal
heads. It must be paired with layer-level and head-level interventions,
and its approximation assumptions must be checked on each model. Here,
one of five seeds invalidated a seemingly safe approximation despite
small individual head effects.

The full per-seed [comparison JSON](results/corpus_delimiter_weight_method_comparison_fiveseed.json)
records the eligibility gate, rankings, and pairwise counts. The code and
one checkpoint are in this repository; the reconstructable FineWeb token
cache and the other four checkpoints remain local.

**Stopping criterion met:** the single-task full-path comparison is done.
Next experiments should use a broader, variable language DGP with several
distinct rules and hyperparameter settings, gate each rule separately on
held-out behavior, and compare whether canonicalization, sparsification,
and intervention-based scores recover *new* structure across tasks. The
natural-text delimiter transform is one form-realism control, not the main
known-world language generator.
