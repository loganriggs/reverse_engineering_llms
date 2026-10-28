# M4: natural text weakens the token-dominated description

2026-10-28. Fixed cohort selected before model forwards: 64 documents from the existing H681 BSF training split, 128 tokens per window. Seed2026102804; all selected source-array and window hashes checked against the historical manifest. These are reused research documents, not untouched validation or model-training holdouts. Reserved BSF validation/test document hashes and original whole-task exclusions remain unused.

## Why replay was needed

The inspected early BSF cache producer stores persistent values and post-block residuals at 16 positions. Those are not the native MLP writes required by M3. Therefore replay the pinned native model on 64 windows, collecting all positions for MLP0/1/2 and excluding Down bias. The verified checkpoint receipt is in the result JSON. No featurizer fit, pruning, or additional toy sweep.

There are 8,192 occurrences, 8,184 distinct causal prefixes and 3,033 current token types. Of these types, 2,060 have only one distinct prefix. Restricting to repeated-context tokens leaves 973 types and 6,124 unique-prefix rows. Windows are existing sampled text spans, not necessarily document beginnings. Exact prefix deduplication does not remove near-duplicate text.

## Same observational decomposition, different result

Within-token fraction of centered native write energy, using unique causal prefixes:

| Layer | Short task prompts, repeated-context tokens (M3) | Natural text, repeated-context tokens | Natural text, all tokens |
|---|---:|---:|---:|
| MLP0 | 10.1% | 29.5% | 23.3% |
| MLP1 | 13.8% | 40.8% | 30.8% |
| MLP2 | 19.0% | 53.8% | 39.2% |

The contextual share is substantially larger on natural text, and exceeds half the centered variation at MLP2 among repeated-context tokens. Thus the short-prompt description of a small contextual correction does not transfer quantitatively to this cohort. This does not isolate the cause: vocabulary composition, context length, position and task distributions all differ. Neither panel is a representative population estimate, and the token-group means are still fitted and scored in sample. Singleton groups fit perfectly by construction.

## Cross-document prediction exposes an additional gap

Split the 64 selected training documents into two fixed groups of 32. Predict writes in each half using current-token means learned only from the other half. Evaluate only tokens covered by the training half. Compare squared error to the training-half global-mean predictor on exactly the same covered evaluation rows. Deduplicate complete causal prefixes before splitting; shared exact prefixes are assigned to their first occurrence.

| Layer | Token-mean error / global-mean error, first direction | Reverse direction |
|---|---:|---:|
| MLP0 | 0.450 | 0.440 |
| MLP1 | 0.666 | 0.647 |
| MLP2 | 0.889 | 0.869 |

Coverage is 60.2% in the first direction and 57.8% in reverse; uncovered tokens receive no prediction score. MLP0 token means reduce covered-position error by roughly 55–56% relative to the global baseline, MLP1 by 33–35%, and MLP2 by only 11–13%. These are cross-document prediction scores within reused training data, not causal effects or pristine scientific holdout performance. They are not directly comparable to in-sample explained variance because the baseline and evaluated populations differ.

## Research consequence and next bounded causal test

Do not impose a token-table-plus-small-correction ontology on the early MLPs from M3. Use separate token-conditioned and contextual diagnostics, but let their importance be measured on natural text. In particular, output reconstruction and causal relevance must remain separate.

Next freeze a finite native intervention on this same cohort: use each document's opposite-half token means to replace one layer's native product write at covered positions, retain Down bias and preserve native writes at uncovered positions. Compare full-output KL, top1 agreement and next-token CE against native replay. Include a per-position norm-only control that scales the live native product write to the predicted table-write norm, separating magnitude changes from the additional direction change without treating differences as additive causal shares. Test MLP0/1/2 separately; no table refit or new holdout exposure. Report coverage and local vector error. This asks whether the measured contextual/prediction residual matters downstream; it does not equate table replacement with deleting native channels or identify semantic features by itself.

## Validation and reproducibility

The independent Torch64 auditor checks all 12 NumPy energy partitions and six cross-document predictions, reconstructs prefix indices, checks cohort/cache hashes, and verifies reserved BSF document-hash exclusion. Maximum relative statistic discrepancy: 1.19e-15. This is an independent statistic audit, not a second full-model implementation; native tracing was validated in earlier packet checks.

Run `probe_natural_token_context.py prepare`, then `probe_natural_token_context.py run`, then `audit_natural_token_context.py` in the research environment. Preparation requires the historical manifest and source arrays. Large cached writes remain in the primary artifact directory as `emlp_m4_natural_writes.pt`; small cohort token IDs, provenance, results and audit are bundled here. Do not rerun preparation with altered selection settings and call it the same fixed experiment.

[Cohort and pre-forward protocol](validation/natural_token_context_cohort.json), [results](validation/natural_token_context.json), [audit](validation/natural_token_context_audit.json), [M3](TOKEN_CONTEXT_PARTITION.md).
