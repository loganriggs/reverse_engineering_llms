# Minimal keyed-alias capability control

The [parenthetical-alias joint language](SURFACE_ALIAS_RESULTS.md) trained models that read definition content but averaged over two referents instead of selecting the one named by the current alias. This control asks whether the same two-layer attention-only bilinear architecture can learn keyed binding at all. It is an artificial answer-only lookup task, **not** a claim about language realism or circuit recovery.

Each example has the form `person_3 means alias_5 . person_1 means alias_2 . ask alias_5 :` with the next token `person_3`. The two people and two aliases are distinct and resampled across cases. Querying the other alias changes the answer. Reversing the definition clauses does not. Swapping alias labels or people between definitions changes the answer. [The generator and trainer](minimal_alias_lookup.py) enumerate the full small case space and train the original bilinear transformer shell with two blocks, width 64, four heads, RMSNorm, no block MLP, answer-only CE, AdamW, 3,000 updates and batch 256. A control replaces the bilinear attention score with standard masked softmax QK attention in the same shell. Its Q2/K2 weights are inactive, so this is **not** a parameter-matched architecture comparison.

## Split and oracle audit

An alias–person pair is held out when `(alias_index + person_index) % 4 == 0`. There are 16 such pairs and 48 allowed pairs. Among allowed pairs, a deterministic whole-sequence hash creates a further sequence holdout. The exhaustive panels contain **2,844 unique training sequences**, **708 unique unseen sequences using allowed pairs**, and **1,152 unique sequences whose queried pair was excluded from training**. There is zero exact-input overlap across the training and both test panels. All 48 allowed query pairs occur in training and the unseen-sequence test; all 16 held-out query pairs occur equally often in the pair-holdout test. Query positions are balanced. The script asserts these properties and checks the edit oracle: query, alias-label and person swaps change the answer; definition-order reversal does not.

The earlier v1 pilot used a random 4,096-case training cache and a random allowed-pair test. **1,408/2,048** test inputs exactly matched training inputs. Its near-perfect allowed-pair score was therefore contaminated. The v2 results below use the exhaustive disjoint split and supersede that pilot; the v1 artifact is retained only as a record of the corrected mistake.

## Five-seed result

All values below come from saved final checkpoints for seeds 0–4. Accuracy is on the next person token; total variation (TV) and L1 error use the full predicted token distribution. The oracle query-edit TV is one, and its order-edit TV is zero.

| Attention | Unseen allowed-pair accuracy | Held-out-pair accuracy | Held-out query-edit effect L1 error | Held-out order-edit model TV |
| --- | ---: | ---: | ---: | ---: |
| Bilinear | **1.000** in all seeds | **.925** mean (.868–.982) | **.247** mean | **.128** mean |
| Softmax control | .980 mean (.961–.990) | .356 mean (.124–.437) | 1.490 mean | .530 mean |

The bilinear model's mean model TV under the held-out query edit is **.940**, and under the held-out alias-label swap it is **.958**. It is effectively invariant to definition order on unseen allowed-pair sequences (mean TV **.0001**). The softmax control also learns many allowed-pair cases but transfers much less well to unseen pairings and has larger order sensitivity. On one seed, extending softmax training from 3,000 to 10,000 updates raises held-out-pair accuracy only from **.412** to **.463**; four softmax blocks at 3,000 updates reach **.443**; adding a block MLP to two softmax blocks reaches **.199**. Those are single-seed capability checks, not matched sweeps.

This proves **capability** for the tested bilinear architecture on the isolated lookup: it can retrieve a person by a variable alias and transfer to alias–person combinations absent from training. It does not prove perfect compositional generalization: its held-out-pair accuracy varies by seed, and its mean order-edit TV there is .128 rather than zero. The five-seed softmax gap is an empirical inductive-bias result for these exact architectures, optimizer and split; active parameter count, attention normalization, and training dynamics still differ. Do not infer a universal advantage for bilinear attention.

## Consequence for the joint language

The joint-rule models' pooled-referent shortcut is not an obvious expressibility barrier in the two-layer bilinear architecture. The next informative experiment is a **same-vocabulary curriculum transfer**: first train the short keyed-lookup task using the joint language's alias/person tokens and model shape, then continue on the two-definition readable stream while preserving a whole alias–person training holdout. Evaluate the full stream's exact verb KL, alias-key edit, irrelevant-definition edit, and order edit before applying any tensor gauge or sparse decomposition. A successful transfer would produce a competent, causally testable alias circuit; failure would identify interference, long context, or training objective as the remaining bottleneck. Keep the original joint-rule task and separate task-family coverage in view rather than treating this isolated control as the final benchmark.

Reproduce with `python language_task_suite/minimal_alias_lookup.py --attention bilinear --steps 3000 --batch 256 --device cuda --seed 0 --output-dir language_task_suite/minimal_alias_bilinear_v2`, substituting `--attention softmax` and seeds 1–4 for controls. Each output directory contains `results.json` with per-edit evaluations and `model.pt`.
