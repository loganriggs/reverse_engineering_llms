# Broader causal coverage helps modestly, but the source/final tradeoff persists

M98 completed the fixed eight-fit comparison: two initial dictionaries, full-pool versus previously visited training contexts, and KL-only versus joint causal supervision. All models retain all 4608 native bilinear products and decode through a fixed 128-dimensional output dictionary. This tests supervision of a compressed replacement, not removal of native channels.

The full pool increased distinct source prefixes from 72/80 to 110. Matched controls used the same update budget, task-family sequence, natural documents, optimizer, initial state and causal weight. Token counts and source eligibility were not exactly matched; the design report records those differences. All eight finals and two initial references were evaluated without selection.

## Main result

Below is the effect of adding causal supervision to KL-only training on **equal-family relative causal response error**, with both the background activation and intervention change supplied by the replacement. Lower is better; positive means causal supervision worsened this particular test. A relative error of 1 is the zero-response predictor, not a faithfulness threshold.

| Evaluation | Seed | Source: control | Source: full pool | Final: control | Final: full pool |
|---|---:|---:|---:|---:|---:|
| Seven held-out task families | 71 | +0.002679 | +0.002463 | +0.000021 | −0.000067 |
| Seven held-out task families | 72 | +0.004520 | +0.003497 | +0.000024 | −0.000154 |
| Held-out source groups | 71 | +0.003368 | +0.001414 | −0.005447 | −0.009245 |
| Held-out source groups | 72 | +0.001863 | +0.001447 | −0.008059 | −0.009340 |

The coverage-by-objective interaction is favorable in all eight entries: broader coverage reduces the source penalty and improves the final-position effect. **The source penalty remains positive in every row.** Broader coverage therefore helps this contrast without resolving the intervention-site tradeoff.

An improved interaction does not guarantee a better final model in absolute terms. Whole-task joint source errors were 0.754965 versus 0.754938 (full versus control) for seed71, and 0.681378 versus 0.682042 for seed72. Held-out source-group joint source errors improved in both seeds: 0.725840 versus 0.727182, and 0.801492 versus 0.805349. Source-group final errors improved to 1.484466/1.368043, but still exceed the zero-response predictor's error of 1.

Aggregation matters. On the five native contrast-qualified task families, equal-family source error worsens under causal supervision in both seeds, but pooled source error improves for seed72. The complete comparison retains pooled, equal-family, margin, local-write, family and format results. The background-only and change-only corners also remain separate; not every corner has a favorable coverage interaction.

Natural-text retention differences were tiny and mixed: full-pool versus control joint KL changed by −0.0000523 for seed71 and +0.0000080 for seed72. Contrast accuracy stayed unchanged across all conditions: 95% on the five qualified families and 100% on source-group holdouts. These accuracy scores conceal substantial causal-response errors.

## What this changes conceptually

**Coverage-limited learning remains a contributing explanation.** More distinct training contexts improved the causal-versus-KL contrast, with architecture and update count held fixed. This is stronger evidence than simply adding steps or choosing a better-performing seed. It is still a two-seed result on historical evaluation cohorts, with small token/eligibility mismatches.

**A single average objective can hide different computational demands.** Preserving an activation background and reproducing the effect of changing an input are distinct requirements. Source and final-position interventions remain different even after broader coverage. The persistent pattern makes objective-component compatibility a concrete next question; it does not prove that its gradients conflict.

**Output compression is not computational redundancy.** These replacements use only 128 output coordinates but still calculate every native product. The results cannot tell us that most native products are removable or that those coordinates are reusable semantic units. Low-dimensional output geometry, distributed implementation, and downstream tolerance must remain separate claims.

**This is not evidence of native signed cancellation.** Errors from an approximate background and an approximate change may interact downstream. That is different from native channels cancelling each other's writes. Likewise, unchanged answer accuracy is downstream tolerance on these tests, not proof of equivalent computation.

## Verification and boundary

All 320 natural full-output vectors, 9840 whole-task response vectors and 2520 source-group response vectors were independently rescored. Prediction algebra was independently checked using float64 NumPy; the Transformer itself does not have a second independent implementation. Native baselines and intervention targets replayed exactly. The complete evaluation used 10324 native forwards.

All 4320 causal contrast quantities were recomputed directly from audited endpoint records; maximum discrepancy was 4.44e-16. No case, model, family or failed competence group was removed. A separate documentation correction clarifies that historical native-forward guards came from M95, while M98 checked all ten prediction adapters.

M98 is closed. The next bounded diagnostic should measure compatibility of the existing training objective components before any new optimizer or rank search. This would test a specific explanation of the remaining source/final tradeoff. Manifold validity, reusable native components and genuine new-task discovery remain unestablished.
