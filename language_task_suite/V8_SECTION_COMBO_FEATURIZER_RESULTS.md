# Featurizers with whole section–color association holdouts

The [section attention path screen](V8_SECTION_ATTENTION_PATHS.md) found
causally relevant routes but no [sufficient hand-picked circuit](V8_SECTION_PATH_SUFFICIENCY.md).
This experiment returns to the featurizer loop. It asks whether a sparse
shared dictionary with overlapping group gates transfers section-answer
structure to **unseen section-ID × color associations**, compared with
ordinary scalar sparse coding and fixed bases. The language model is frozen;
the holdout applies to **featurizer fitting only**.

The [panel builder](build_v8_section_combo_holdout.py) withholds the eight
associations where `(section_index + color_index) mod 3 = 0`. Every one of
the four section IDs and six colors appears individually in fit. To prevent
indirect association leakage, a fit/development prefix is accepted only if
**all** of its facts use one of the other 16 associations. Fit has 240
exact-support section-key edits; development has 80. Separate test streams
provide 120 key edits, 80 relevant-color edits, and 80 irrelevant-color
edits each for fully seen and queried-heldout associations. The test-heldout
side covers all eight excluded queried associations. The serialized
[manifest](results/v8_section_combo_holdout_seed81001.json) records every
prefix, edit, association, and split audit. Both independent balanced v8
teachers use exactly the same panels.

The [training runner](train_v8_section_combo_featurizers.py) captures the
64-dimensional final residual at each original/edited answer endpoint and
checks that the frozen final readout reproduces native probabilities below
`7.2e-7` maximum error. It fits five principal families:

* Scalar Top-8/10/16: independently select the largest encoded coordinates
  and decode a shared learned dictionary.
* Disjoint groups: 16 groups of four atoms, four active groups, with fixed
  non-overlapping membership; 16 active atoms.
* Overlapping shared groups: one 64-atom dictionary, 16 learned groups with
  four memberships each, four selected groups; shared atoms are decoded
  once. Its actual union support is about 8–10 atoms.
* Physical-coordinate and output-weight SVD Top-16 are non-trained controls.

All learned models start from fit-endpoint PCA. Reconstruction-only and
teacher-edit-effect objectives are compared for scalar and overlapping
group models; the disjoint control uses the edit-effect objective. The
causal objective adds the squared difference between reconstructed and
native answer-probability *changes* on fit key edits. Development data
selects checkpoints; test edit families and held-out associations never
enter optimization or checkpoint selection. We extended the run from 400
to **1,200 steps** because group development curves were still improving.
Each teacher was then repeated with a second featurizer initialization.
Active support, reconstruction, teacher KL, exact answer accuracy/CE, and
edit-to-teacher/oracle errors are reported separately.

The table shows the **queried-heldout key-edit** result after 1,200 steps.
KL is the mean teacher→featurizer KL at both answer endpoints; effect error
is L1 difference in the edit response. Lower is better. Each cell contains
two featurizer initializations. Scalar Top-8 is the nearest support match
to the overlapping model; Top-10 brackets its support in seed 4.

| Teacher, method | Mean active atoms | Teacher KL | Key-edit effect error |
|---|---:|---:|---:|
| Seed 1, scalar Top-8 reconstruction | 8 / 8 | .0024 / .0021 | .036 / .033 |
| Seed 1, scalar Top-10 reconstruction | 10 / 10 | .0017 / .0014 | .032 / .026 |
| Seed 1, shared groups reconstruction | 9.2 / 8.3 | .0170 / .0222 | .069 / .062 |
| Seed 1, shared groups causal | 8.2 / 8.6 | .0139 / .0154 | .062 / .081 |
| Seed 4, scalar Top-8 reconstruction | 8 / 8 | .0199 / .0138 | .117 / .091 |
| Seed 4, scalar Top-10 reconstruction | 10 / 10 | .0074 / .0067 | .079 / .066 |
| Seed 4, shared groups reconstruction | 9.9 / 9.3 | .0442 / .1074 | .128 / .207 |
| Seed 4, shared groups causal | 8.7 / 9.8 | .0906 / .0540 | .140 / .156 |

The overlap model improves substantially between 400 and 1,200 updates,
but at the tested architecture and objective it still loses to scalar
Top-8/10 at comparable active support on both teachers and both
featurizer seeds. The same direction holds for held-out relevant- and
irrelevant-color edit effects. For example, seed-1/first-featurizer-seed
relevant-color effect error is .047 for scalar Top-8 versus .133 for
reconstruction shared groups and .090 for causal shared groups; in seed 4
it is .138 versus .193 and .186. The group models also have larger
reconstruction error, so their causal gap cannot be assigned solely to the
objective metric. Causal training helps some group runs but does not
consistently improve their held-out causal fidelity.

An important positive result is how easy this *specific final readout site*
is for an ordinary scalar dictionary: scalar Top-16 has held-out teacher KL
`.0003`/`.0011` and key-edit effect error `.011`/`.025` in teacher seeds
1/4. Both selected their PCA initialization at step 0. This is strong
local functional compression, **not** evidence that those coordinates are
human-legible, that the full model is sparse, or that we discovered a new
task. The native teachers themselves perform worse on queried-heldout
combinations than seen ones: key-edit paired answer accuracy is .883 versus
.942 in seed 1 and .854 versus .938 in seed 4. The language models were
trained on every combination, so this is a distribution-relative
featurizer transfer test rather than an unseen-task test for the LM.

The next featurizer comparison should move from the already-compressible
final residual to the **intermediate attention bonds** identified by the
path screen, and use path-effect supervision at the teacher-specific
answer-to-color layer and the shared query-key/header route. Preserve the
whole-association split and compare scalar, disjoint, and overlapping
groups at matched actual active support and reconstruction/teacher-KL
budgets. Stop a variant if it fails in-fit fidelity before interpreting
its transfer or naming its groups. Genuine new-task discovery still needs
an unlabeled held-out rule family and an independent ground-truth test.

Full records and checkpoints: [seed-1 featurizer seed 1](results/v8_section_combo_featurizers_seed1_1200.json),
[seed-1 featurizer seed 101](results/v8_section_combo_featurizers_modelseed1_featseed101_1200.json),
[seed-4 featurizer seed 4](results/v8_section_combo_featurizers_seed4_1200.json),
and [seed-4 featurizer seed 104](results/v8_section_combo_featurizers_modelseed4_featseed104_1200.json).
