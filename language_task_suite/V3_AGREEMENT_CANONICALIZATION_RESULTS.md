# Agreement task: realism check and local canonicalization baselines

The [v3 agreement generator](MULTI_GENRE_V3_AGREEMENT_RESULTS.md) supplies exact visible-prefix labels and valid subject, attractor, and no-effect edits. This report asks whether a compact **first- or last-attention-output bond basis** preserves that behavior when Agreement is held out from a data-aware basis fit. The models themselves *were trained on Agreement*. A successful holdout here is basis-transfer evidence, not new-task discovery or a recovered circuit.

## Surface realism and joint-model competence

The [FineWeb form audit](AGREEMENT_FINEWEB_FORM_NOTES.md) scanned 864 separately cached 513-token windows. It found no exact short `near/beside` or `that the N VERB is/are` frames from the generator. A broader intervening-preposition proxy occurred 25 times, 21 with `of`. These are regex surface cues, not parsed dependencies. The v3 surface is useful for a causal probe but **not calibrated to FineWeb**.

A depth-3, width-64, four-head RMSNorm model trained on the five active genres at learning rate 0.001 for 4,500 steps, without extra agreement loss, improved over the earlier unstable mixed run. Two independent 1,024-stream test panels give exact-oracle KL: report verb 0.261, policy action 0.191, FAQ copy 0.013, FAQ place recall **0.558**, and agreement main-conflict **0.0009** nats. On fresh agreement edits, main-subject and embedded-attractor effect L1 errors are about 0.003. FAQ recall remains well short of the previously competent four-genre benchmark, so this is not yet a fully competent joint teacher.

The local-bond study instead uses two shorter-context mixed models with the **same training recipe and independent seeds** (`multi_genre_v3_mixed`, seed 0; `multi_genre_v3_mixed_seed1`, seed 1). Both learn agreement, FAQ copying, and some report/policy behavior, but not FAQ place recall (fresh KL roughly 2.8/3.2). The second model's fresh agreement main-conflict KL is 0.0021 and main-subject edit L1 about 0.012; the first is more accurate on agreement. Do not use their weak FAQ-recall outputs as target circuits.

## Whole-genre basis-fit holdout

At one attention block's `attn.out` input, four orthonormal bases were compared: native coordinates, a fixed random basis, right singular vectors of the **weight** matrix, and activation PCA fitted on Reports/Guide/FAQ/Policy only. The last fit excludes the entire Agreement genre. At width *k*, the bond input is projected onto the first *k* basis vectors; model weights are frozen. Predictions are scored on new mixed streams and grammar-valid Agreement edits. Full width replays the native model to numerical precision.

At the **last bond, width 8/64**, on the same fresh 256-stream prediction and 64-stream edit panels:

| Model seed | Basis | Main-conflict model→truncated KL | Main-subject edit L1 vs oracle | FAQ-copy model→truncated KL |
| --- | --- | ---: | ---: | ---: |
| 0 | Weight SVD | ~0.00001 | 0.00008 | 1.313 |
| 0 | Heldout PCA | 0.035 | 0.073 | 3.509 |
| 1 | Weight SVD | 0.00049 | 0.0097 | 4.658 |
| 1 | Heldout PCA | 1.055 | 0.965 | 3.728 |

Native main-subject edit L1 is about 0.00005/0.0086 for seeds 0/1. Weight SVD transfers agreement across the task-family basis-fit holdout in both seeds, while the same eight dimensions do **not** preserve the whole model. The FAQ-copy damage in particular rules out calling this a faithful eight-dimensional model compression. At the **first bond**, seed 0's rank-8 weight basis retains agreement and heldout PCA does not (subject-edit L1 0.00012 versus 1.213 on a larger panel); in seed 1, even raw or random rank-4 projections barely alter agreement. Thus a first-bond task dependency is not stable across seeds. Both models depend strongly on the final bond.

The larger 512-stream, 64-edit-stream first-bond replication for seed 0 explains the data-aware failure: heldout PCA at rank 8 captures 88% of fit-genre activation energy but 35% of Agreement activation energy. Weight SVD captures only 38% of fit-genre activation energy, 54% of Agreement activation energy, and 71% of output-weight Frobenius energy, yet preserves agreement edits. When Agreement is included in the PCA fit, its rank-8 subject-edit error drops from 1.213 to 0.0053 on the same panel. A different, slower-trained model also favors weight SVD at matched rank, but its sufficient rank changes (weight SVD about 12–16; heldout PCA about 20 for subject-edit L1 near the native 0.003 level). There is no universal “rank eight” claim.

## Does the weight basis yield legible units?

Removing individual top weight-SVD directions at the final bond shows only partial rule selectivity. In seed 0, removing axis 0 raises main-conflict oracle KL by **2.17 nats** but also raises FAQ-copy KL by **4.43**; it mixes two known behaviors. In seed 1, removing axis 0 raises main-conflict KL by **7.98** while FAQ-copy KL changes by less than 0.001. Other axes carry the embedded-verb behavior in different combinations. Axis numbers, task sharing, and causal depth therefore vary across independent models. A weight-SVD basis can be a strong **fidelity prior** without being a canonical semantic decomposition.

## Supervised legibility search within the weight subspace

`search_v3_agreement_causal_rotation.py` kept the top-eight final-bond weight-SVD subspace fixed. For each model, it compared its eight original axes with 64 randomly rotated unit directions. The development objective rewards damage to the **signed main-subject edit response** under axis removal, and penalizes FAQ-copy, embedded-verb, policy-action, and four no-effect edit damages. It selects the best direction in each family on development panels; independent held-out panels contain 128 streams per prediction genre and 64 Agreement edit streams. This is supervised causal axis search, not unlabeled feature discovery.

| Seed | Best original SVD axis: dev / holdout score | Best sampled rotation: dev / holdout score | Held-out interpretation |
| --- | ---: | ---: | --- |
| 0 | axis 2: 0.667 / 0.429 | rotation 38: 1.332 / **1.105** | Rotation improves the declared score, but FAQ-copy KL damage rises from 0.224 to 0.411. |
| 1 | axis 1: 1.083 / **1.152** | rotation 8: 0.420 / 0.404 | Rotation loses; its no-effect edit damage is 1.752 versus 0.444 for the SVD axis. |

Thus a constrained rotation can improve one model's selected causal tradeoff, but the gain does **not** replicate across independent seeds. It is not enough to claim a more interpretable gauge. The search covers 64 random directions and one declared objective, so it does not rule out better gradient-based or graph-structured objectives. Per the predeclared stopping criterion, stop tuning this local random search. Return to richer task coverage and a more competent joint teacher before another interpretability-prior sweep.

The next comparison should keep weight SVD as a fidelity baseline and use whole task-family basis-fit holdouts. For any supervised sparse or shared-group representation, report reconstruction, teacher KL, exact-oracle prediction, signed causal edits, and no-effect invariances separately. A direction that merely destroys Agreement logits without preserving specific signed effects is not a recovered rule.

Reproduction: `audit_fineweb_agreement_forms.py`, `eval_multi_genre_fresh.py`, `eval_v3_agreement_bond_holdout.py`, `eval_v3_agreement_svd_axes.py`, and `search_v3_agreement_causal_rotation.py`; their saved JSONs and training configurations are in `results/`. The bond experiments test local linear projections in a nonlinear network, not a canonical form of the complete attention function.
