# M37: native transfer and ordinary outputs after common shared training

2026-10-28. Follow-up to [M35–M36](JOINT_SHARED_TRAINING_RETENTION.md).

The common-training causal objective improves native log-probability response fidelity in 7/12 architecture/seed/region comparisons and answer-margin response fidelity in 8/12. All18 initial/final states were evaluated on identical within-seed external interventions and the full ordinary-output inventory. These are development transfer results, not semantic-unit validation.

## What is compared

Six M7 reconstruction-trained initial states and12 final states: scalar, disjoint block and
fixed overlapping groups, seeds71/72, intact-output KL versus KL plus native-response loss.
Every model of a seed receives BOTH original M18 benchmark generators (scalar and block),
using the same exact delta tensors. This corrects the architecture-specific generator difference
in older comparisons. The original704 native effects replay; perturbation construction is frozen.
Group membership is not deleted here: these are external changes to the reconstructed/native
MLP1 write, identical across architectures. Shared-group gate deletion is a different experiment.

There are22 evaluation pairs across three lexical/template quadrants,44 endpoints,704 native
directions including96 zero directions,6336 student intervention responses and792 intact
baselines. Natural benchmark noise/zeros are retained, not filtered for convenient effect size.
The native probe uses6916forwards. The output inventory separately uses1152 inputs per state
(32 natural evaluation documents,160 original task endpoints,960 expanded endpoints),20736
student rows and21888forwards. No new fits, prompts or checkpoint selection.

## Paired causal-objective effects

Ratios below are joint/control normalized log-probability response errors. Lower than1 favors
causal supervision. Both generators are pooled; errors are summed before normalization by
native response energy. A zero-response predictor has normalized error1, whereas these table
ratios compare two trained predictors. Directions and native denominators are identical within
each paired comparison. Seeds do not have identical perturbation sets.

| Architecture / seed | Prefix joint/control | Task joint/control | Joint prefix error | Joint task error |
|---|---:|---:|---:|---:|
| scalar 71 | 1.042 | 1.197 | 1.317 | 0.929 |
| scalar 72 | 0.836 | 0.826 | 0.964 | 0.521 |
| block 71 | 1.070 | 0.927 | 1.363 | 0.866 |
| block 72 | 0.773 | 0.925 | 1.196 | 0.827 |
| shared_fixed 71 | 0.890 | 0.713 | 1.496 | 1.027 |
| shared_fixed 72 | 1.457 | 1.011 | 3.168 | 1.231 |

Shared seed71 improves log-probability fidelity in both regions (ratios .890 and .713),
while shared seed72 worsens in both (1.457 and1.011), despite its strong training-target gain.
Shared joint absolute normalized errors exceed1 in all four region/seed aggregates: by this
metric the zero-response baseline is better. This does not make the entire reconstruction
useless; it limits the causal-response claim. The same native direction set is used for scalar
and block comparisons, so these differences do not arise from architecture-specific probes.

The complete JSON also reports answer-margin response fidelity, each source generator
separately, each family and each lexical/template quadrant. Aggregate success is not equivalent
to all-family transfer or preservation of a semantic unit. Exact active dimension is unmatched:
shared groups can use fewer than32 atoms, despite equal trainable parameter counts.

## Ordinary prediction and natural-text effects

Mean intact-output KL (lower is better), evaluated over all token positions per input, then
averaged equally over inputs. The jointly excluded quadrant means both lexical split and
wrapper split were excluded from training; these examples have been examined in earlier
research and are not a pristine test set.

| Architecture / seed | Natural initial | Natural KL-only | Natural joint | Jointly excluded KL-only | Jointly excluded joint |
|---|---:|---:|---:|---:|---:|
| scalar 71 | 0.30385 | 0.14898 | 0.16362 | 0.08499 | 0.09951 |
| scalar 72 | 0.29158 | 0.13198 | 0.15415 | 0.06961 | 0.09467 |
| block 71 | 0.42411 | 0.17677 | 0.19322 | 0.10517 | 0.10521 |
| block 72 | 0.43366 | 0.18755 | 0.21947 | 0.08682 | 0.10164 |
| shared_fixed 71 | 0.56679 | 0.23966 | 0.26871 | 0.08492 | 0.10052 |
| shared_fixed 72 | 0.52366 | 0.22384 | 0.26619 | 0.08268 | 0.10815 |

Joint training has worse mean KL than its matched KL-only control for all six starts on
both natural text and jointly excluded lexical/template examples (block71 is nearly tied
on the latter). All these joint scores nevertheless improve over their M7 initial states.
Thus the added causal target costs some ordinary-prediction improvement, rather than causing
a return to worse-than-initial performance on these aggregates.

Full results retain observed next-token CE differences, top1 agreement, task binary/full-vocabulary
competence, local write error, native competence failures and actual support costs. A worse
natural KL than control can mean foregone improvement, rather than forgetting relative to the
initial state; initial scores are shown to distinguish these.

## Interpretation and remaining requirements

This is the first common-training native-transfer comparison here to restore fixed overlap
alongside scalar and block models. Fixed random memberships are an implementation baseline,
not learned concepts. Training improvement, native causal transfer, meaningful manifolds and
new task discovery remain separate claims. Neither a favorable response metric nor intact
classification alone proves that a component is a reusable computational unit.

The larger learned-membership architecture remains outstanding in this native-write setup.
Earlier recovery tests supplied gates or found membership concentration, so their existence
does not justify treating learned sharing as solved. The next bounded representation experiment should isolate grouping quality: freeze a common
trained scalar dictionary/coordinate encoder, derive balanced overlapping groups from training
coactivation only, and compare them with balanced random groups on the same fixed dictionary.
Keep32 groups of8, four active groups and every atom in exactly two groups; report actual union
size. This prevents the prior membership-coverage collapse and avoids confounding a new grouping
with new decoder learning. Test output/causal transfer and selective shared-atom versus group-gate
removal; do not call grouping scores manifold validation. Fix the construction/optimization budget
before outcomes and retain the random control. No optimizer or wrapper sweep is warranted.

Preserve whole-task training exclusions sequence_copy, spatial_inverse, weekday_successor,
role_binding and tense; past_tense is distinct. No genuine new tasks have been discovered.

## Evidence and reproduction

- [Model manifest](validation/joint_shared_models.json)
- [Native responses](validation/joint_shared_native_fidelity.json), [audit](validation/joint_shared_native_audit.json)
- [Ordinary outputs](validation/joint_shared_outputs.json), [audit](validation/joint_shared_outputs_audit.json)
- [All paired comparisons](validation/joint_shared_transfer_comparison.json)

Scripts: probe_joint_shared_native.py, evaluate_joint_shared_outputs.py,
audit_joint_shared_native.py, audit_joint_shared_outputs.py, analyze_joint_shared_transfer.py.
Established pinned model/local state caches are required; scalar results and scripts are shared,
large native/model/sample caches remain at hash-linked local paths. Both probes checkpoint
progress and refuse to overwrite a completed output. Do not start duplicate live jobs.

The output probe first exited132 (illegal instruction in apply_rotary_emb). Its unchanged
restart resumed the saved768-input checkpoint; no examples, model settings or outcomes were
selected or dropped. The independent output audit checks the complete final inventory.

Audits check complete model/input/direction inventories, every original native effect, identical
cross-architecture delta hashes and all aggregates. Independent NumPy audits rescore sampled
full vectors and decoder outputs, including boolean union semantics for shared supports. Native
transformer forward computation is not independently reimplemented by the audits.
