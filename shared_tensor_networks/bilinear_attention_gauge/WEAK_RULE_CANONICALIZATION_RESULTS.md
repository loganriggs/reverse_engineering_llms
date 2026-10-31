# A weak competent rule defeats global fidelity gates, while output balancing protects it

We trained **ten independent four-head bilinear-attention models** on four
count-sign rules. The ordinary query outputs A/B/C/D; a rarer query rotates
the first three outputs to B/C/A and keeps D. The rare query with
A=0, B=1, C=0 was excluded from model training, and all ten models solve
that combination on an independent test. We extracted each model's exact
`2 query × 8 key token × 4 output` function from its weights; the table
reproduces native logits within `4.3e-14` in float64.

To make a **competent low-energy rule**, we multiply only the D output
logit and its bias by `0.1` *after training*. This leaves every deterministic
decision unchanged, but reduces the median absolute D margin from `10.43`
to `1.04` logits. Under an analytic independent Gaussian perturbation to
that logit, the expected D accuracy is `98.4%` at noise standard deviation
`0.5` and `88.6%` at standard deviation `1` (versus essentially 100% before
scaling). This is a controlled calibration stress, not evidence that these
trained models naturally choose a weak D scale.

We then fit rank-three and rank-four functions under four **answer-free
reduction metrics**: uniform weight-only SVD, unlabeled usage-weighted
least squares from 100,000 training-law sequences, their half-uniform
mixture, and a weight-only SVD that normalizes each output's table energy.
The usage measures use no answer labels. The table reports means over ten
models on a separate fresh 30,000-sequence test with D calibrated to `0.1`.

| Reduction metric | Rank-three D accuracy | Rank-three all-four accuracy | Rank-three global edit error | Rank-three worst-output edit error | Rank-four all-four accuracy |
|---|---:|---:|---:|---:|---:|
| Uniform weight-only | 50.1% | 50.1% | `0.0038` | `0.999` | 100% |
| Empirical usage | 50.0% | 50.0% | `0.0040` | `0.999` | 100% |
| Half-uniform | 50.0% | 50.0% | `0.0038` | `0.999` | 100% |
| Output-balanced weight-only | 98.4% | 66.9% | `0.2947` | `0.445` | 100% |

At rank three, the ordinary metrics largely delete D while reporting less
than `0.5%` *global* all-pair edit error. The output-balanced cut keeps D
but sacrifices part of another rule: changing the fit metric does not create
four independent directions in a three-direction budget. At rank four,
every metric retains 100% joint and rare-held-out accuracy in every seed.
This is why reduction fidelity must be checked before optimizing a legible
gauge.

The rank-selection failure is exact and repeatable at a `1%` error
threshold. For each of uniform, empirical, and half-uniform reduction,
global all-28-token-pair edit fidelity selects **rank three in 10/10
models**, as does the global criterion using only 15 pairs among the A/B/C
tokens and **withholding both D tokens**. Both choices lose D. A
per-output relative edit-error gate, or a per-output relative weight-table
error gate, selects **rank four in 10/10** for all four reduction metrics.
Without D calibration, the global all-pair gate also selects rank four:
the failure is caused by allowing a small but competent output to be
averaged away. Per-output normalization is an explicit prior that the
named output channels matter separately; it is not invariant to arbitrary
rotations of output space.

At the faithful rank-four function, we compared **exact orthogonal
coordinate changes** for human legibility. Eight-start Powell searches
minimize key-dictionary L1, all-pair edit L1, or edit L1 with every direct
D-token edit excluded. SVD and 32 random orthogonal rotations per cut are
controls. All changes reconstruct their fixed reduced function within
`1.8e-14`; this stage cannot repair a bad rank-three cut.

| Exact gauge, 40 fixed cuts | Mean matched A/B/C/D axis cosine | Mean D-axis cosine | Accuracy on other three rules after removing matched D axis |
|---|---:|---:|---:|
| SVD | `0.9113` | `0.9996` | 100% |
| Key L1 | `0.9996` | `0.9995` | 100% |
| All-pair edit L1 | `0.9996` | `0.9996` | 100% |
| Edit L1 excluding D tokens | `0.9996` | `0.9997` | 100% |
| Random rotation, averaged | `0.7189` | `0.3874` | 96.0% |

Removing the matched D component from the SVD or sparse gauges takes D
accuracy to about 50% while preserving the other three rules; these are
**functional rank-one component deletions**, not physical attention-head
interventions. The SVD already isolates D because its small scale gives it
a distinct singular direction. The no-D-edit prior mostly straightens
A/B/C and leaves D as the orthogonal remainder; it does **not** establish
discovery of D from unrelated examples. The sparse objectives have local
minima, and the best of eight starts is not a global-optimum certificate.
The usage-weighted low-rank fits are also nonconvex; their recorded
first-order optimality is at most `3.2e-6`, but we do not claim global
optimality.

This toy supplies a sharp metric counterexample: low global reconstruction
and edit error can coexist with losing a competent rule. It also shows a
usable alternative when output coordinates have meaning: require fidelity
per output before optimizing sparsity within the surviving function. A
language model has far more output coordinates, and treating every token
logit equally may be inappropriate; the real-model analogue needs
behavioral groups or intervention-defined outputs and independent
validation. A stronger discovery toy should mix the weak variable with
the strong variables in its token code, so SVD no longer isolates it by
scale, and should withhold its edit family from both gauge fitting and
selector tuning.

Reproduce training with `train_four_count_weak_rule.py` for seeds 0–9,
then run `four_count_weak_rule_reduction.py`,
`four_count_weak_rule_gauges.py`, and
`analyze_four_count_weak_rule.py`. Per-model reductions, full cut matrices,
gauge outcomes, random controls, rank gates, and noise calculations are in
the matching JSON/NPZ artifacts.
