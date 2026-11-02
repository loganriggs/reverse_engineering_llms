# Independent v8 section teacher: conditional replication

The [first curriculum and balancing run](V8_SCOPE_BALANCED_CONTINUATION.md)
found a key-sensitive section model whose other language rules improved
after task-balanced continuation. The [five-seed short-stage study](V8_SHORT_CURRICULUM_SEED_REPLICATION.md)
showed that the short curriculum itself is optimization-sensitive. This
follow-up takes independent seed 4, whose **predeclared** 1,500-step short
run reached 100% short-test accuracy, through the same 1,200-step joint
stage and 1,200-step balancing stage. It tests **downstream replication
conditional on a successful short checkpoint**, not the unconditional
success rate of the entire procedure.

The model and v8 generator match seed 1. Joint training uses LR .003,
scope-answer soft weight 2 and scope counterfactual weight 1. Balancing uses
LR .001, exact-oracle soft weights 2 on verb, object, object-place answer,
source answer and scope answer, plus supported counterfactual weights 1 on
question, source and scope answers. Both have batch 32, length 256 and
1,200 updates. The seed-4 runs used one PyTorch CPU thread because earlier
attempts faulted before optimization; this changes the runtime setting,
not the loss, data generator or model architecture.

On the **same** two fresh 256-stream panels (seeds 49101/49102), lower
exact-oracle next-token KL is better:

| Rule site | Seed-4 joint KL A/B | Seed-4 balanced KL A/B |
|---|---:|---:|
| Verb | 1.000 / .991 | **.566 / .563** |
| Object | 1.920 / 1.935 | **1.260 / 1.288** |
| Object-place answer | 6.151 / 6.030 | **4.041 / 3.988** |
| Source answer | 5.013 / 5.851 | **2.136 / 2.087** |
| Section answer | .400 / .398 | **.286 / .300** |

Seed 1 balanced, on these same streams, gets verb .559/.555, object
1.098/1.112, source 2.088/1.981 and section .296/.225. The same qualitative
multi-rule repair appears twice, with material model-to-model variation.

Paired exact-support section edits on two further independent 200-question
panels (seeds 49001/49002) test whether answer improvement reflects the
requested key. Key-blind Bayes accuracy is .456/.443. TV measures the
change in the model's answer distribution under the edit; exact-oracle TV
is 1 for key/relevant-color edits and 0 for an irrelevant-color edit.

| Seed-4 checkpoint | Accuracy A/B | Key-edit TV A/B | Relevant-color TV A/B | Irrelevant-color TV A/B |
|---|---:|---:|---:|---:|
| Joint | .825 / .810 | .741 / .676 | .717 / .657 | .176 / .226 |
| Balanced | **.880 / .820** | **.812 / .727** | **.773 / .744** | **.127 / .191** |

The second seed confirms nontrivial keyed retrieval before and after
balancing. It is still imperfect: on the balanced checkpoint, key-edit
effect L1 from the oracle is .631/.787, and irrelevant-color response
remains nonzero. There is no compute-matched scratch control for seed 4;
the seed-1 [scratch continuation control](V8_SCOPE_COMPUTE_CONTROL_AND_READOUT.md)
showed that the same balancing stage did not teach key use from scratch.

We also repeated the [local readout-basis test](eval_v8_rule_holdout_readout.py)
on the independent balanced teacher. A rank-16 combined-residual basis
fitted only to **120 section-key edits** has teacher-effect L1 error .0048
on 120 independent key edits, .0574 on 57 held-out relevant-color edits,
and .0451 on 50 irrelevant-color edits. The same-rank weight SVD errors
are 1.011/.864/.140; task-blind stream PCA errors .482/.447/.083.
Endpoint replay error is <8e-7. The teacher-effect optimizer again selects
the key-edit PCA initialization. This replicates a low-dimensional *local
output-response subspace*, not a unique semantic feature set or a
whole-model canonicalization.

The next model-training experiment should improve the **unconditional**
short-stage success rate by separating initialization and data-order seeds
and testing a staged context-length curriculum. For interpretability, use
both balanced teachers to test weight-only versus distribution-aware
first-layer attention reductions and shared-dictionary/group selectors,
with whole-combination holdouts and separate prediction, reconstruction,
causal-fidelity and discovery measures. The toy surface remains much more
task-rich than natural web text.

Artifacts: [seed-4 joint checkpoint and causal panels](results/v8_scope_curriculum_seed4/),
[seed-4 balanced checkpoint and rule panels](results/v8_scope_balanced_continue_seed104/),
and [independent-teacher readout basis](results/v8_scope_keyfit_colorholdout_rank16_modelseed4_seed50001.json).
