# Rule-aware rank-4 OV fit on corpus-grounded list and bracket rules

The two depth-2, width-64 bilinear transformers from the numbered-list pilot
learned item-3/4/5 transitions and the separate bracket closure rule. This
follow-up asks whether a *rule-site distillation objective* finds a better
rank-4 OV representation than weight-only or input-covariance SVD. It does not
fit QK, the embedding, unembedding, MLP, or an entire attention function.

## Design

- Each of four heads in each attention layer retains four of sixteen OV
  coordinates. The omitted OV coordinates stay exactly zero. All non-OV
  parameters stay frozen.
- The two SVD baselines use either the OV weight matrix alone or the activation
  covariance of separate-rule training windows. Rule-aware fits initialize
  from weight SVD and optimize teacher next-token KL at the marked site, plus
  0.1 times teacher KL at sampled background positions.
- `list` fit uses list item 3/4/5 windows, both edited arms, 64 source rows per
  arm and ordinal. `bracket` fit uses 64 independent bracket windows. `both`
  samples these two kinds equally. Each scope gets 300 Adam steps, batch 4.
- The heldout list, bracket, and 20-document four-arm list×bracket panels are
  never used for fitting. No joint edited windows enter the fitting objective.
  Source partitions and duplicate screening are as in
  [the preceding result](CORPUS_LIST_BILINEAR_AND_OV_RESULT.md).

## Results

Here `list` is paired full-vocabulary top-1 on the list test; `bracket` is
paired two-way accuracy; `joint exact` requires full-vocabulary top-1 for both
rules in all four arms of a document. The two KL columns compare the rank-4
student with the *dense teacher* at marked sites (lower is better).

| Model | OV method | List | Bracket | Joint exact | List KL | Bracket KL |
|---|---|---:|---:|---:|---:|---:|
| 71503 | dense teacher | .933 | .934 | .35 | 0 | 0 |
| 71503 | weight SVD | .922 | .934 | .45 | .047 | .823 |
| 71503 | data SVD | .922 | .907 | .35 | .046 | .087 |
| 71503 | list-site fit | .900 | .913 | .30 | .021 | .226 |
| 71503 | bracket-site fit | .911 | .928 | .40 | .046 | .074 |
| 71503 | both-site fit | .922 | .937 | .40 | .014 | .082 |
| 71504 | dense teacher | .867 | .880 | .15 | 0 | 0 |
| 71504 | weight SVD | .833 | .886 | .25 | .701 | .478 |
| 71504 | data SVD | .889 | .865 | .20 | .386 | .096 |
| 71504 | list-site fit | .867 | .886 | .15 | .042 | .314 |
| 71504 | bracket-site fit | .889 | .889 | .20 | .533 | .091 |
| 71504 | both-site fit | .867 | .868 | .15 | .045 | .115 |

The both-site fit strongly improves *list teacher KL* relative to either SVD
on both seeds (.014/.045 versus .046/.386 for data SVD) while preserving or
slightly lowering separate-rule discrete behavior. It does not improve strict
joint exact behavior over the best SVD. With just 20 joint documents, the joint
fractions are noisy; importantly, neither fitted model is a robust exact
joint-rule solver. Background KL also deteriorates against the data SVD in
both seeds, despite the 0.1 regularizer: for the both-site fit, list background
KL is .046/.081 versus .026/.041 for data SVD, and bracket background KL is
.051/.078 versus .029/.029. This shows a real tradeoff between rule-site
teacher fidelity and broad output fidelity.

**Interpretation.** Causal/task supervision changes which low-rank OV
approximation is preferred; weight error and activation error alone do not
select it. But low teacher KL and marked-site behavior do not establish a
human-legible or modular circuit. The fits are supervised by known rules and
inherit the dense teacher's algorithm. They do not discover a new rule or
recover a unique canonical form. The more important next step is to test
whether a similarly restricted representation preserves *causal responses*
to targeted edits across more than two tasks and larger joint panels, while
keeping background fidelity and whole-task holdouts separate. The sampled
language DGP also still needs broader rules and document structures.

The aggregate [JSON results](results/corpus_list_bracket_rule_aware_ov_rank4.json)
include paired two-way scores, edit directions, and both background KLs. The
[script](fit_corpus_list_bracket_ov.py) reproduces the fit from the local model
checkpoints and reconstructable token caches.
