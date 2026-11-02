# Can a small selected source set carry the teacher's behavior?

**Result: not reliably with this fixed teacher-selected gate.** A sparse-gated output-distillation objective can improve its own loss on sampled training edits, but it does not reliably improve sparse-gated predictions on new edits of the same rule families. Native signed causal-edit fidelity worsens in both the initial and broader runs. This is a more functional target than matching source vectors, yet it still does not turn the sparse student into a smaller faithful circuit.

## Experiment and correctness checks

[`train_varied_sparse_path_transfer.py`](train_varied_sparse_path_transfer.py) now offers `--path-objective sufficiency`. For each valid edit pair, the dense two-layer varied-language teacher chooses the four second-layer sources per head whose `QK1×QK2×V` contributions most affect centered verb-output rows across the original and edited prefixes. The same positions are held fixed for both versions. The sparse student's **actual logits** are recomputed using only those sources, its first-layer residual, native output projection, final RMSNorm, and unembedding. The new loss is `KL(full teacher || sparse-gated student)` on both prefixes, added to ordinary full-sequence teacher KL. The selected-position gate is supplied by the teacher and is not learned. This is a dynamic local source mask, not additional parameter sparsity.

An independent check found zero error in the per-source vectors against the prior exact decomposition, and 4.8e-6 maximum logit error when the sparse executor retains every source. Thus the measured gate gap is due to discarded sources, rather than a changed forward equation. Fixed-mask checkpoints were validated after training: each has exactly **29,491 active attention weights**, and every masked-out parameter remains zero. The flat and factored architectures have the same **70,259 total active parameters**. The teacher, starting checkpoints, and generator are the same as the preceding [path-transfer pilot](VARIED_SPARSE_PATH_TRANSFER_PILOT.md).

Lexical, induction, and previous-place edits provide balanced gate supervision. **Cue** edits are excluded from this paired loss, although ordinary language KL and prior model training include cues. The first run uses 128 fit streams, 150 updates, seed 0 and a separate 128-stream edit panel; the broader run uses 512 fit streams, 600 updates, seed 1 and a separate 256-stream edit panel. Within each run, the no-gate-loss and gate-loss arms share the training stream, edit schedule, mask, and optimizer settings. The gate metrics below use the first 24 independent test edit pairs per family; native edit effects use all 164 cue pairs in run 1 and all 307 in run 2. Exact-pair overlaps with the fit panel are removed from the test panel. This remains a **causal-target holdout**, not a whole-task model-training holdout.

| Run / student / gate-loss weight | Native teacher KL ↓ | Sparse-gated cue teacher KL ↓ | Native cue-effect L1 ↓ | Sparse-gated cue-effect L1 ↓ |
|---|---:|---:|---:|---:|
| 150 steps / flat / 0 | **.000949** | **.00981** | **.0715** | .1147 |
| 150 steps / flat / .1 | .001023 | .01157 | .0855 | **.1068** |
| 150 steps / flat / 1 | .001363 | .01129 | .1170 | .1066 |
| 150 steps / factor / 0 | **.003020** | **.01986** | **.1330** | .1617 |
| 150 steps / factor / .1 | .003111 | .02198 | .1462 | **.1302** |
| 150 steps / factor / 1 | .004040 | .02956 | .1796 | .1584 |
| 600 steps / flat / 0 | .001093 | **.00808** | **.0633** | **.1018** |
| 600 steps / flat / .1 | **.001066** | .01013 | .0991 | .1147 |
| 600 steps / factor / 0 | **.002595** | **.02030** | **.1247** | **.1573** |
| 600 steps / factor / .1 | .002936 | .02332 | .1439 | .1740 |

The slight sparse-gated cue-effect improvement at 150 steps does **not** replicate in the broader run. On the 150-step flat student, average gated KL over the three supervised *test* families rises from .01015 without the gate objective to .01319 at weight .1, even though the loss on a 24-pair sample of the **fit** pool falls for induction (.01326→.00787) and previous place (.01058→.00777). The 600-step run still worsens sparse-gated KL on lexical and cue test edits for both architectures. Some induction or previous-place gate scores improve, but the overall native edit response degrades. The dense teacher's own four-source gate has nonzero cue KL (.00583 in run 1; .00486 in run 2), so this choice of sources has a fidelity floor even before student distillation.

The discrepancy is consistent with fitting too few or unrepresentative paired contexts, source-set instability, or a fixed gate that excludes necessary context and cancellation paths. These explanations are hypotheses, not established mechanisms. A larger fit pool and four times as many updates did not produce the required across-family transfer. Stop this fixed teacher-top-four sufficiency objective on the current varied-language task. The next experiment should broaden the task distribution and compare **learned** gates or independently compositional sparse teachers, requiring held-out rule coverage, actual path necessity/sufficiency, native and gated oracle effects, and total nonzero budgets. More coefficient tuning on the inspected panels would not answer the main question.

## Fresh-panel source removal

[`eval_varied_sparse_source_gates.py`](eval_varied_sparse_source_gates.py) performs the missing read-only necessity check on a **new** 256-stream panel (seed 52102), excluding 30 exact pairs found in either fit pool and evaluating 64 pairs per edit family. It compares native predictions, retained-only predictions, and predictions after removing exactly those retained sources. For cue edits, teacher-selected four-source removal changes the teacher's own native distribution by KL **.779**, and changes its signed probability effect by L1 **.549**. The retained-only teacher is closer to native (KL .00794, edit L1 .067). Thus the selected paths matter substantially, but they are not exactly sufficient even in the teacher.

| Broad-run model | Cue retained-only KL to full teacher ↓ | Cue KL after removing selected sources, relative to native model ↑ | Native cue-edit L1 to teacher ↓ | Retained-only cue-edit L1 to teacher ↓ |
|---|---:|---:|---:|---:|
| Flat, no gate loss | .01275 | .811 | **.069** | **.096** |
| Flat, gate loss .1 | **.00986** | .736 | .091 | .104 |
| Factor, no gate loss | .02106 | .768 | **.114** | **.154** |
| Factor, gate loss .1 | **.02004** | .658 | .141 | .157 |

This fresh panel shows that the gate objective **can** improve retained-only KL, even on the cue family omitted from paired fitting. The improvement is accompanied by worse native edit behavior, worse retained-only cue-edit fidelity, and a smaller removed-source KL. The conclusion is therefore more precise than “the gate never works”: it can concentrate some predictive behavior in the chosen paths, but this implementation does not yield a smaller faithful causal circuit. All six edit-family rows and teacher controls are in [`varied_sparse_sufficiency_gate_audit_seed52102.json`](varied_sparse_sufficiency_gate_audit_seed52102.json). Removing a source set changes the residual context, so the removal KL is an intervention under this specified mask, not a unique attribution score.

The ten result/checkpoint pairs live in the `varied_sparse_sufficiency_*` directories. For example, the broader flat gate-loss arm can be reproduced from the repository root with:

```bash
/home/loganriggs/Coding/neuron_circuits/.venv/bin/python \
  language_task_suite/train_varied_sparse_path_transfer.py \
  --output-dir language_task_suite/varied_sparse_sufficiency_broad_flat_p1 \
  --arms flat --path-weights .1 --path-objective sufficiency \
  --steps 600 --fit-streams 512 --test-streams 256 \
  --batch 16 --length 72 --seed 1 --device cuda
```
