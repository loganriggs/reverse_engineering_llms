# Does sparse extra depth actually participate?

The first [sparse-depth pilot](SPARSE_DEPTH_DISTILLATION_PILOT.md) left identity-initialized added blocks dead; the split-initialized four-layer model used them but lost much fidelity. This follow-up tests a smaller perturbation and a pruning rule that cannot delete an added projection wholesale. On the competent two-layer verb-and-object teacher, **neither route yields useful active extra depth at 29,491 attention edges**. The equal-quota route keeps physical connections in all four layers, but its inserted layers' functional contribution decays to numerical zero.

The teacher is [`surface_mixture_v3_object8`](surface_mixture_v3_object8/): width 64, four heads, two bilinear attention layers, with known lexical/induction/cue/previous-place rules at verb slots and adjective/induction/cue/previous-event rules at object slots. Its held-out exact-oracle KL is about .096 at verbs and .101 at objects. This is a **compression** experiment: the teacher and both students have seen the task families, and the expanded-object panel is a teacher-behavior shift on which the teacher itself is not oracle-competent.

The four-layer student starts as an exact embedding of the teacher in blocks 0 and 2, with newly inserted blocks 1 and 3 given small random output maps (`extra_init_scale=.03`); initial maximum teacher-logit difference on a probe is .0174. It gets 300 no-L1 warmup steps. Each prune stage uses 100 L1/distillation steps, pruning, then 100 fixed-mask no-L1 steps, at edge budgets 68,813, 49,152, 39,322 and 29,491. A final 300-step no-L1 recovery follows. The flat control uses the same per-matrix pruning rule at 39,322 and 29,491 edges, 100+100 steps per stage, and the same 300-step recovery. Thus both have the same final physical attention-edge budget and recovery length, but **not equal training compute**: the depth arm gets more updates and a warmup. Embeddings and LM head remain dense; at the final edge budget, total active parameters are 69,875 for depth versus 69,747 for flat because of extra RMSNorm scalars.

Two pruning rules expose the failure mode. Global magnitude pruning deletes both inserted output maps by the first prune to 68,813 edges; at 29,491 edges, blocks 1 and 3 have zero active attention weights. Equal per-matrix quotas reserve 1,228–1,229 edges in each of the 24 attention matrices at the final budget. This guarantees physical edges but does **not** force their use. The inserted output-map Frobenius norms drop from .436/.416 after warmup to about .0027/.0027 after final recovery.

| Student at 29,491 attention edges | Ordinary teacher KL | Expanded-object teacher KL | Exact-oracle verb KL | Exact-oracle object KL |
|---|---:|---:|---:|---:|
| Flat, equal per-matrix quotas, recovered | **.00117** | **.00430** | **.101** | **.106** |
| Four-layer, equal per-matrix quotas, recovered | .00759 | .02306 | .134 | .133 |
| Four-layer, global magnitude pruning, before recovery | .00162 | .00555 | — | — |

The globally pruned four-layer model behaves like a sparse two-layer model, so its lower KL than the equal-quota depth arm is **not** evidence for useful extra stages. The equal-quota model's extra blocks remain inert despite their retained weights. On 256 independent streams with 6,204 valid edits, its probability-effect L1 errors against the exact oracle exceed the flat student's on every representative rule below; lower is better.

| Valid edit family | Teacher | Flat sparse | Four-layer equal-quota sparse |
|---|---:|---:|---:|
| Lexical verb | .276 | **.279** | .343 |
| Verb induction | .312 | **.315** | .335 |
| Object adjective | .237 | **.246** | .284 |
| Object induction | .374 | **.378** | .386 |
| Previous-event object | .217 | **.219** | .264 |
| Nuisance object-place | .046 | **.048** | .058 |

The decisive participation check removes each block's attention output while leaving the rest of the trained student intact. On the 256-stream ordinary test panel, native-to-ablated KL is **1.105** for block 0 and **1.350** for block 2, but below displayed precision for blocks **1 and 3**; the same zero/nonzero pattern holds on the expanded-object panel. Thus surviving edges and nonzero output weights do not imply a computational stage. The added stages are not explanatory features in this run.

This is a negative result for **this initialization, magnitude/L1 schedule, and equal-matrix quota**, not a limit on deeper sparse bilinear transformers. The equal quota is a participation control, not an optimal edge allocator. Optimization may still favor the exact two-layer teacher embedding because the extra stages start as nearly identity residuals. More no-L1 recovery did not reverse that tendency. Stop this local depth-pruning sweep unless a new method explicitly transfers substantial function into the inserted blocks and beats the flat control on held-out rule effects. The next immediate experiment should instead train a sparse or factored student with a **task-balanced causal/path objective** on several joint rules, preserve a whole-rule-family fitting holdout, and test whether a small set of named paths is sufficient and necessary for each effect. Count total edges and verify interventions; low KL and nonzero weights alone are insufficient.

Reproduction: [`distill_sparse_depth_vs_factor.py`](distill_sparse_depth_vs_factor.py) now accepts `--balanced-matrix-prune`, and [`eval_sparse_depth_participation.py`](eval_sparse_depth_participation.py) performs the block ablations. Saved runs are [`v3_object8_sparse_depth_curriculum`](v3_object8_sparse_depth_curriculum/), [`v3_object8_sparse_depth_balanced`](v3_object8_sparse_depth_balanced/), [`v3_object8_sparse_flat_balanced`](v3_object8_sparse_flat_balanced/), and their corresponding `_recovery` directories. The edit JSON files and per-block participation JSON are in the recovery directories.
