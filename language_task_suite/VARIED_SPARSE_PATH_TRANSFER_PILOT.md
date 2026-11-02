# Does supervising sparse source paths improve a sparse student's behavior?

**Pilot result: no, under this objective.** A task-output-aware path loss makes the flat and factored sparse students match the dense teacher's selected second-layer source contributions more closely, including on an edit family withheld from path fitting. It does not improve teacher KL or the causal probability changes; larger path weight worsens both. This separates learning a chosen internal decomposition from preserving the function the decomposition was meant to explain.

## Setup

[`train_varied_sparse_path_transfer.py`](train_varied_sparse_path_transfer.py) resumes the previously saved flat and sparse-factored students in [`sparse_architecture_recovery_pilot`](sparse_architecture_recovery_pilot/). Both have **29,491 active attention weights and 70,259 total active parameters**; embeddings, unembedding, and normalization remain dense. The frozen teacher is the competent two-layer, width-64 [`surface_mixture_v2_varied_soft_models`](surface_mixture_v2_varied_soft_models/) checkpoint. This run does not test new four-layer students because the earlier [participation audit](SPARSE_DEPTH_PARTICIPATION_RESULTS.md) found inserted layers functionally inert under the available initialization/pruning schedule.

Training minimizes ordinary teacher KL on fresh full-mixture language streams plus a weighted paired path loss. For lexical, induction, and previous-place edits, the teacher ranks source positions by the magnitude of their contribution to centered verb-output rows, retains its four strongest per head, and supplies the **before/after change** in each selected `QK1 × QK2 × V → output projection → verb contrast` path. The student is compared at those same source positions. This is differentiable through both student attention layers. A direct replay check found zero maximum difference between the new path-signature implementation and the existing exact second-layer source decomposition on a probe. The path loss uses the teacher's head and source decomposition as a gauge anchor; it is not an invariant comparison between all functionally equivalent circuits.
These signatures are linearized output contrasts **before** final RMSNorm; they do not equal each source's independent probability effect.

Cue edits are wholly omitted from paired path fitting. They remain in ordinary language KL, and both original teacher and resumed sparse checkpoints were trained on the full task mixture. Thus this is a **causal-target family holdout**, not a whole-task training holdout or discovery of an unseen rule. Training uses one seed, 128 fit streams, 150 AdamW steps, batch 16, sequence length 72, fixed attention masks, and no L1 penalty. The no-path and two path-weight arms share the same language and edit schedule. An independent 128-stream panel has 347 lexical, 182 induction, 401 previous-place, and 164 cue valid edit pairs after exact-pair overlap exclusion. Path-response error below is measured on the first 24 test pairs per family; behavioral edit error uses all available test pairs.

| Student / path weight | Teacher KL | Test path relative error: lexical / induction / place / held-out cue | Teacher-effect L1: lexical / induction / place / held-out cue |
|---|---:|---|---|
| Flat / 0 | **.000949** | .0481 / .0534 / .0300 / .0430 | **.0790 / .0531 / .0690 / .0715** |
| Flat / .001 | .000977 | .0189 / .0379 / .0152 / .0301 | .0805 / .0553 / .0729 / .0731 |
| Flat / .01 | .001022 | **.0108 / .0264 / .0049 / .0207** | .0828 / .0642 / .0836 / .0757 |
| Factor / 0 | **.003020** | .1561 / .1200 / .0939 / .1271 | **.1597 / .0950 / .1349 / .1330** |
| Factor / .001 | .003029 | .0807 / .0916 / .0614 / .1005 | .1611 / .0950 / .1335 / .1327 |
| Factor / .01 | .003139 | **.0497 / .0701 / .0329 / .0727** | .1698 / .1046 / .1536 / .1398 |

Path relative error is squared student–teacher difference divided by teacher path-change energy for the selected sources; smaller means closer **selected internal response**. Teacher-effect L1 is the absolute difference between student and teacher signed next-token probability changes under the edit; smaller means better **functional causal fidelity**. Teacher KL is on a separate full-mixture stream panel. Exact-oracle edit metrics are also in the result JSON files; none support a claim of improved task correctness. The path loss succeeds on its direct measurement but consumes capacity or steers the student toward an internal representation that is not optimal under the sparse mask. That is a plausible explanation, **not** proven by this pilot.

The finding is one seed, one density, one teacher, one local teacher-gauge target, and two coefficients. It does not show that sparse paths are unhelpful generally. It **does** reject the simple next step of adding this path-matching term to the existing fixed-mask distillation objective as a demonstrated interpretability or fidelity improvement. Stop tuning these two coefficients on this panel. A follow-up should either learn an output-behavior-preserving path objective with stronger circuit constraints or train an active-depth architecture from an independently sparse compositional teacher, while retaining matched nonzero budgets, whole-family causal-target holdouts, exact-oracle effects, and block-ablation participation checks.

Results and checkpoints are in six `varied_sparse_path_transfer_*` directories. Each run is reproducible by invoking the script with one arm and one path weight, for example:

```bash
/home/loganriggs/Coding/neuron_circuits/.venv/bin/python \
  language_task_suite/train_varied_sparse_path_transfer.py \
  --output-dir language_task_suite/varied_sparse_path_transfer_flat001 \
  --arms flat --path-weights .001 --steps 150 \
  --fit-streams 128 --test-streams 128 --batch 16 --length 72 --device cuda
```
