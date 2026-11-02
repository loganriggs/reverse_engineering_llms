# Sparse featurizers at the final attention bond

The [whole-association benchmark at the final residual](V8_SECTION_COMBO_FEATURIZER_RESULTS.md)
was unusually easy for a PCA-initialized scalar dictionary. This follow-up
uses the **pre-output activation of the last attention layer** at the section
answer position. The [path screen](V8_SECTION_ATTENTION_PATHS.md) showed that
answer-to-query-key attention in this layer affects section-key edits in
both trained teachers. The same [exact-support association split](results/v8_section_combo_holdout_seed81001.json)
is reused: fit/development exclude eight section-ID×color associations
from *every fact* in their prefixes, while held-out evaluation queries all
eight. The language models themselves were trained on all associations.

The updated [featurizer runner](train_v8_section_combo_featurizers.py) captures
the last block's incoming residual `h`, pre-output attention bond `z`, and
native answer distribution. Because this block has no MLP, replacing only
`z` at the final answer position has an exact downstream replay:

`p(answer | prefix) = softmax(LMHead(FinalNorm(h + W_O z)))`.

Native endpoint replay error is below `7.2e-7` in both teachers across fit,
development, seen, and held-out panels. The existing final-residual smoke
metrics reproduce exactly after the runner extension. All models use the
same PCA initialization, 1,200 optimizer steps, and development-only
checkpoint selection. Reconstruction-only and teacher-edit-effect losses
are compared. Scalar Top-8/10 match the overlapping group's actual 8–10
active atoms; scalar and group Top-16 ceilings, physical coordinates, and
weight-only SVD are additional controls. Two independently trained teachers
and two featurizer initializations per teacher were evaluated.

The table reports **queried-heldout section-key edits**. KL is teacher→
reconstructed-model KL at both answer endpoints. Edit error is the mean L1
difference in answer-probability *change* from the native teacher. Lower is
better. A/B are two featurizer initializations.

| Teacher, method | Active atoms | Bond relative MSE | Teacher KL | Key-edit effect error |
|---|---:|---:|---:|---:|
| Seed 1, scalar Top-8 recon | 8 / 8 | .034 / .035 | .0049 / .0047 | .059 / .052 |
| Seed 1, scalar Top-10 recon | 10 / 10 | .026 / .025 | .0036 / .0041 | .044 / .048 |
| Seed 1, shared groups recon | 8.7 / 8.9 | .088 / .135 | .0104 / .0267 | .069 / .104 |
| Seed 1, shared groups causal | 7.8 / 9.1 | .118 / .109 | .0138 / .0091 | .090 / .069 |
| Seed 4, scalar Top-8 recon | 8 / 8 | .074 / .069 | .0086 / .0057 | .075 / .069 |
| Seed 4, scalar Top-10 recon | 10 / 10 | .047 / .047 | .0037 / .0037 | .052 / .052 |
| Seed 4, shared groups recon | 8.5 / 9.3 | .231 / .204 | .0501 / .0145 | .160 / .091 |
| Seed 4, shared groups causal | 9.9 / 9.9 | .206 / .226 | .0218 / .0275 | .098 / .136 |

The shared-group candidates remain behind ordinary scalar Top-8/10 on
held-out teacher KL and key-edit fidelity at roughly matched active support.
Teacher-effect training does not consistently improve held-out fidelity
over reconstruction-only training. Group models also have worse bond
reconstruction, so the present result is an implementation/optimization
and inductive-bias comparison, **not** evidence that reusable groups do not
exist in the trained transformer. The rank-16 scalar model again selects
its PCA initialization at step 0 and gives held-out teacher KL about
`.0007` and key-edit error `.022` in both teachers. This bond is harder
than the final residual at rank 8–10 but still compresses well by rank 16.

A weight-only functional baseline is notable. Right singular directions
of `W_O`, with Top-16 coordinate selection, give seed-4 held-out bond
relative MSE `.073` and teacher KL `.0017`, versus scalar Top-10 MSE
`.047` and KL `.0037`. The poorer reconstruction has *better* functional
fidelity. In seed 1, weight SVD MSE `.139`, KL `.0029`, and key-edit error
`.045` also compete with scalar Top-10 MSE `.026`, KL `.0036`, error `.044`.
These comparisons reinforce that reconstruction, output KL, and causal
edit preservation are distinct metrics; selecting a canonical basis
requires an explicit functional objective.

We tested one further **data-free** choice: SVD of the composed linear
matrix `LMHead.weight @ W_O`. At rank 16 it is worse than `W_O` SVD on both
held-out panels: teacher KL `.0045` versus `.0029` in seed 1 and `.0032`
versus `.0017` in seed 4. The actual readout includes a residual `h` and
RMS normalization, so the composition is only a logit-linear heuristic.
The experiment rejects this simple improvement claim; it does not identify
which nonlinear or distributional term causes the difference. The
[weight-baseline records for seed 1](results/v8_section_combo_bond_weight_baselines_modelseed1.json)
and [seed 4](results/v8_section_combo_bond_weight_baselines_modelseed4.json)
contain ranks 8/10/16 and all edit families without fitting any featurizer.

This is a **local last-bond intervention**, not a sparse whole transformer,
a human-legible manifold, or new-task discovery. It also does not directly
train on the measured answer-to-query-key *edge contribution*: the causal
loss matches the teacher's complete section-key answer response. The next
experiment should use the earlier attention bonds where the two teachers
allocate answer-to-color computation differently, and compare path-targeted
supervision to reconstruction and whole-output losses. Keep the same
whole-association split, actual-support matching, and separate teacher/oracle
fidelity. Stop a group variant if it cannot first achieve useful in-fit
fidelity at this site.

Trained records and states: [seed-1 featurizer seed 1](results/v8_section_combo_bond_modelseed1_featseed1_1200.json),
[seed-1 featurizer seed 101](results/v8_section_combo_bond_modelseed1_featseed101_1200.json),
[seed-4 featurizer seed 4](results/v8_section_combo_bond_modelseed4_featseed4_1200.json),
and [seed-4 featurizer seed 104](results/v8_section_combo_bond_modelseed4_featseed104_1200.json).
