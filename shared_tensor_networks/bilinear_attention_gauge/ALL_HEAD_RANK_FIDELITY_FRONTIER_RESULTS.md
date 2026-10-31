# Aggregate fidelity can hide lost edit families in the 546M model

The [36-head joint-edit atlas](ALL_HEAD_JOINT_EDIT_GEOMETRY_RESULTS.md)
found stable intervention-defined axes but only 38% mean rank-four
**head-state** response-energy coverage. That number alone does not tell us
how much of the head's contribution survives its learned output projection:
discarded state directions may be suppressed by `c_proj`. We therefore
tested stage-one reduction fidelity directly on the saved native edit
responses, for all 36 heads in layers 9, 11, 13, and 15.

For each head, the exact weight-only matrix is the head's block of
`c_proj`. We form its 128×128 weight Gram matrix and compare its leading
eigenspace with eigenspaces of training-edit covariances. The edit panels
contain 80 examples of each of ten input-substitution types. Three edit
measures are tried: pooled sample energy, equal trace for each type in
head-state space, and equal energy for each type *after* `c_proj`.
We also mix each measure with the weight Gram at fractions 0, .1, .5,
.9, and 1. Ranks are 4, 8, 16, 32, 64, 80, 96, 112, and 128.
These are spectral fitting surrogates; the equal-output-type variant
changes covariance weights but does not solve the exact worst-type or
output-aware projector optimization problem.

Every candidate is an orthogonal projector fitted without answer labels
or final-model gradients. On the independent 800-edit validation panel,
we measure both the head-state edit MSE and the MSE of the immediate
`c_proj` residual response. Neither is a final-logit or KL measurement.
The **worst-type** MSE is the maximum of ten separately normalized
relative MSEs, so a low-energy but competent type cannot disappear in
an aggregate numerator.

| Rank / fitting measure | Mean aggregate immediate-output MSE | Mean worst-type immediate-output MSE |
|---|---:|---:|
| 4 / weight-only | .813 | .936 |
| 4 / pooled edit energy | .592 | .922 |
| 4 / equal input-type energy | .642 | .904 |
| 4 / equal output-type energy | .669 | .904 |
| 16 / pooled edit energy | .360 | .736 |
| 16 / equal input-type energy | .398 | .688 |
| 64 / pooled edit energy | .095 | .277 |
| 64 / equal input-type energy | .106 | .268 |
| 96 / pooled edit energy | .034 | .108 |
| 96 / equal input-type energy | .036 | .111 |

Balancing types trades some aggregate error for a smaller worst-type
error, but the low-rank gap remains large. Under a **fixed** mixed
equal-output-type measure (50% weight Gram, 50% balanced edit moment),
the median rank needed to get every type below 10% immediate-output
relative MSE is **96**; 20/36 heads first pass at rank 96, seven at
rank 112, eight at rank 80, and one at rank 64. For pooled edit PCA
alone, the median is 112. These are ranks required *within the tested
spectral family*, not a lower bound on every possible projector. The
rank rule is evaluated on validation, so it is an audit rather than a
deployable model-selection procedure.

The known bracket head (layer 13/head 8) shows the central distinction.
Its rank-four pooled-edit cut misses **6.4%** of total head-state response
but only **2.74%** of aggregate immediate output response. That does
correct the impression that head-state coverage alone makes this cut
unfaithful. Yet the same cut misses **93–97%** of the immediate output
response for eight non-bracket/non-punctuation edit types individually.
The bracket edit's projected response energy is roughly 3,300 times the
color edit's in this balanced-count panel, so its aggregate fit dominates.
The best tested rank-four method still has .928 worst-type error for
this head. At rank 112, a type-balanced mixture reduces that to .069.

For punctuation head 13/3, rank-four pooled-edit fit has .161 aggregate
output error but .980 worst-type error. Equal-type balancing does not
rescue rank four; at rank 96 a 50:50 balanced mixture reaches about .091
worst-type error. The previous [finite-path red team](PUNCTUATION_HEAD_FINITE_RESULTS.md)
found this head's punctuation path small, so preserving its input-side
geometry would not by itself establish a strong behavioral mechanism.

**Decision.** The first-stage metric must be chosen for the question:
pooled edit fit preserves dominant response families efficiently; a
worst-type gate protects diverse counterfactuals but costs much more
rank under these spectral fits. A sparse or jointly diagonal gauge cannot
repair an edit family omitted by its fixed subspace. The next real-model
experiment should select a **specific** set of behaviorally relevant
families and validate a weighted reduction on disjoint token identities,
documents, and final logits. Uniform protection of all ten synthetic
edit types may be unnecessary; treating aggregate fidelity as universal
coverage is equally unjustified.

Reproduce with `flagship_all_head_rank_frontier.py`. The saved
`full_width_results/all_head_rank_frontier.json` contains every
head, rank, fitting measure, mixture, and edit-type MSE.
