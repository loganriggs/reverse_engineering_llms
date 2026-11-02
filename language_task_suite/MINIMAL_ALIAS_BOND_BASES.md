# Local bond bases on the keyed-alias control

The [minimal alias lookup](MINIMAL_ALIAS_LOOKUP_RESULTS.md) supplies five competent two-layer bilinear checkpoints with whole alias–person training holdouts. [This experiment](eval_minimal_alias_bond_bases.py) asks whether an orthogonal basis at an attention output bond can compress their function while preserving the **held-out key-switch intervention**. It is a method diagnostic, not a unique tensor canonical form or recovered alias circuit.

For each model seed, fit bases using only its 2,844 allowed-pair training cases. The weight-SVD basis uses the local attention output matrix only. The all-output basis uses the product of the full output head with that matrix; the person-output basis uses only the eight known answer rows and therefore has a task label prior. Activation PCA uses bond inputs from original/query-edited training cases. Query-response SVD uses their differences and is explicitly intervention-supervised. Random and coordinate bases are controls. At test time, project **only the final-position input** to the selected output bond onto the first `k` directions, leaving source positions unchanged. Evaluate all 1,152 held-out-pair cases and their query, definition-order, alias-label and person edits. Width 64 exactly replays each reference model (mean output TV below 1e-5).

The table reports means over five independent model seeds. `Reference TV` is the mean total variation between compressed and unmodified model outputs. `Query effect L1` compares the compressed model's probability change under a query edit with the exact one-hot oracle change; the unmodified mean is **.247**. Thus it contains the model's own remaining held-out errors.

| Bond / basis | Width | Held-out accuracy | Reference TV | Query effect L1 |
| --- | ---: | ---: | ---: | ---: |
| Final bond, unmodified | 64 | .925 | 0 | .247 |
| Final bond, zeroed | 0 | .108 | .872 | 3.179 |
| Final bond, weight SVD | **8** | **.925** | **.0023** | **.248** |
| Final bond, all-output weights | 8 | .925 | .0018 | .248 |
| Final bond, activation PCA | 8 | .871 | .0839 | .385 |
| Final bond, query-response SVD | 8 | .876 | .0842 | .394 |
| Final bond, random | 8 | .423 | .590 | 2.010 |
| First bond, zeroed at final position | 0 | .481 | .513 | 2.051 |
| First bond, weight SVD | **8** | **.918** | **.0221** | **.258** |
| First bond, all-output weights | 8 | .891 | .0639 | .374 |
| First bond, activation PCA | 8 | .915 | .0313 | .277 |
| First bond, query-response SVD | 8 | .870 | .0900 | .449 |
| First bond, random | 8 | .605 | .400 | 1.525 |

At the final bond, an eight-direction weight-only basis almost perfectly preserves the checkpoint and the causal query response. The all-output baseline does as well, so the most parsimonious explanation is the small effective answer-output space, **not** an eight-dimensional alias circuit. The person-output basis also matches at width eight but uses knowledge of the target label set. At the first bond, local weight SVD at width eight is stronger than the final-output-weight baseline and close to activation PCA; this is evidence of a useful local weight prior upstream of the second block, still only a compression result. Query-response SVD is not automatically best: fitting a difference subspace alone omits coordinates needed to preserve the base computation, especially in the first block.

The bond interventions establish that these sites matter: zeroing the final-position input to the final bond drops held-out accuracy from .925 to .108 on average. They do not establish selectivity for alias binding over other possible computations, and the model has no other labeled tasks. A genuine circuit-recovery claim would require a basis selected without the target answers that predicts **which alias–person binding changes** under held-out definition and query edits, selectively mediates that change, and transfers to the richer joint-rule language. The next comparison should include the exact first-layer token/relative-position attention tensor and planted key-match relation; a low-rank output bond alone cannot supply that interpretation.

Reproduce with `python language_task_suite/eval_minimal_alias_bond_bases.py --model-dir language_task_suite/minimal_alias_bilinear_v2 --layer 1 --device cuda --output language_task_suite/minimal_alias_bond_bases_seed0_layer1.json`, then repeat `--layer 0` and seeds 1–4. Each JSON records every width, basis, held-out behavioral score and the full-width replay check.
