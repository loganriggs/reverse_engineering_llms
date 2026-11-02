# Does direct color-edge supervision improve local featurizers?

The [earlier-bond benchmark](V8_SECTION_EARLY_BOND_FEATURIZER_RESULTS.md) found compact scalar representations at the teacher-specific answer-to-color attention layer, while overlapping shared groups lagged. Its causal objective matched the *whole* answer response to section-key edits. This follow-up asks whether supervising the measured **answer-to-all-fact-colors edge contribution** changes that comparison.

The [runner](train_v8_section_combo_featurizers.py) now captures the incoming attention state and analytically computes the sum of pre-output bond contributions from every structurally parsed fact-color key to the answer query. The selector includes all fact colors, without using the matching section's color as a training label. It computes the native edge response by removing that contribution at the selected layer and rerunning all downstream blocks. A separate remove-edge forward hook agrees with the local replay to maximum answer-probability error `1.1e-6` on sampled original/edited prefixes from both teachers. The refactored path-ablation runner reproduces its pre-change seed-1 panel results within `5e-7`. Full native endpoint replay is below `6e-7` over the complete panels. The native edge response is substantial: on queried-heldout key edits its mean endpoint L1 magnitude is `.971` in seed 1 and `.489` in seed 4.

For each endpoint bond `z` and selected-edge contribution `c`, the featurizer is applied **twice**, to `z` and `z-c`. The training loss is full-bond reconstruction MSE plus ten times squared error in the answer-probability difference between those two featurized states. This matters: subtracting the native `c` *after* featurizing `z` would leave the edge signal uncompressed and test a different question. An early pilot using that different formulation was discarded; only the two-state records below support the conclusions. Fit/development contain only section-key edits and exclude eight section-ID × color associations from every fact. Held-out panels query all eight associations and include unseen relevant- and irrelevant-color edit families. The frozen teachers saw every association during language-model training.

The table compares 1,200-step development-selected featurizers on **queried-heldout key edits**. Edge error is mean L1 disagreement with the teacher's answer-to-all-colors removal response. Output KL and edit error measure different endpoints. Scalar candidates have exactly 8 or 10 active atoms. Shared-group support is the actual union of active atoms, not four groups times four memberships.

| Teacher/site, candidate | Active atoms | Edge error | Output KL | Key-edit error |
|---|---:|---:|---:|---:|
| Seed 1 / layer 1, scalar Top-8 reconstruction | 8 | .0770 | .00099 | .0199 |
| Seed 1 / layer 1, scalar Top-8 edge | 8 | **.0574** | .00117 | .0211 |
| Seed 1 / layer 1, scalar Top-10 reconstruction | 10 | .0499 | .00041 | .0141 |
| Seed 1 / layer 1, scalar Top-10 edge | 10 | **.0363** | .00048 | .0153 |
| Seed 4 / layer 0, scalar Top-8 reconstruction | 8 | .0738 | .00101 | .0214 |
| Seed 4 / layer 0, scalar Top-8 edge | 8 | **.0560** | .00121 | .0217 |
| Seed 4 / layer 0, scalar Top-10 reconstruction | 10 | **.0330** | .00021 | .0097 |
| Seed 4 / layer 0, scalar Top-10 edge | 10 | .0384 | .00047 | .0142 |

The scalar Top-8 edge gain replicates across the two teacher-specific layers, about 25% lower edge error at the cost of slightly worse ordinary teacher KL and key-edit fidelity. It also transfers from fit key edits to held-out **color edits** in seed 1: relevant-color edge error `.085`→`.055`, irrelevant-color `.060`→`.051`. Top-10 is mixed, so direct edge supervision is a way to choose a desired causal tradeoff, not a universal improvement. A rank-16, **weight-only** SVD basis of each selected layer's `W_O` has heldout key-edit edge error `.0677` and `.0663` for seeds 1/4, respectively, without data fitting. Its output KL remains good, but rank 8 loses substantially to scalar Top-8 on the edge metric.

For the overlapping shared groups, random gate/membership initialization materially affects the comparison. An unpaired edge run in seed 1 looked better than a separately initialized reconstruction run. We therefore repeated the two objectives from **identical** group states and identical minibatch indices, verifying equal step-zero reconstruction and actual support. These paired results are the decisive comparison:

| Teacher/site, paired group objective | Active atoms | Edge error | Output KL | Key-edit error |
|---|---:|---:|---:|---:|
| Seed 1 / layer 1, reconstruction | 7.78 | **.1143** | **.01075** | **.0571** |
| Seed 1 / layer 1, edge | 8.76 | .1191 | .01122 | .0620 |
| Seed 4 / layer 0, reconstruction | 7.52 | **.1036** | **.01105** | **.0648** |
| Seed 4 / layer 0, edge | 7.73 | .1187 | .01921 | .0852 |

At this architecture, initialization, and 1,200-step budget, edge supervision does **not** improve the shared-group model; both paired edge candidates remain behind scalar Top-8 at comparable support. This is a stopping point for this signed-linear gate variant on the single section task. It is not evidence against overlapping directions in the teacher. The group encoder selects by signed linear scores, whereas the scalar control selects by coordinate magnitude; a sign-invariant energy gate and structured shared-membership initialization deserve separate synthetic validation before making a broader architectural claim.

As a broader-task diagnostic, the [cross-rule evaluator](eval_v8_section_bond_cross_rule.py) applied section-fitted checkpoints to three source/speaker edit families on fresh v8 streams. The local native replay passed (`<4e-7`); weight-only rank-16 SVD had mean teacher KL `.00044` in seed 1 and `.00009` in seed 4. But **the teachers themselves are not competent source-task models** on this panel: native answer accuracy across the three source edit families is `.149/.117/.453` in seed 1 and `.140/.164/.066` in seed 4. Their edit effects are tiny. Therefore output preservation here is a distribution-shift diagnostic, not evidence of a reusable *correct* language mechanism or new-task discovery. This agrees with the existing joint-rule evaluation, where source-answer oracle KL remains around 2 for the section-competent teachers.

The next useful step is a competence-gated multi-rule teacher: retain section-key and relevant-color behavior while making source binding and other toy-language rules genuinely correct on fresh exact-support panels. Only then compare scalar, improved overlapping-group, and weight-only bases on whole-rule holdouts. Keep reconstruction, teacher KL, exact-oracle task performance, targeted causal fidelity, and genuine unlabeled new-task discovery separate. The toy language's similarity to FineWeb remains a distinct, unfinished validation question.

Records: [seed-1 edge-trained](results/v8_section_color_edge_both_states_seed1_1200.json), [seed-4 edge-trained](results/v8_section_color_edge_both_states_seed4_1200.json), [seed-1 reconstruction rescored](results/v8_section_color_edge_seed1_pretrained_1200_corrected.json), [seed-4 reconstruction rescored](results/v8_section_color_edge_seed4_pretrained_1200_corrected.json), [seed-1 paired groups](results/v8_section_color_edge_paired_seed1_1200.json), [seed-4 paired groups](results/v8_section_color_edge_paired_seed4_1200.json), [seed-1 source transfer](results/v8_section_bond_cross_rule_seed1_both_states_85001.json), and [seed-4 source transfer](results/v8_section_bond_cross_rule_seed4_both_states_85001.json).
