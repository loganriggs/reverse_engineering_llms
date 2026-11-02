# Multi-genre toy language: first model and interpretation results

The [project proposal](https://www.lesswrong.com/posts/GncmyuKmp526RJpmj/ambitious-mech-interp-w-tensor-transformers-on-toy-languages) asks for an exact joint language DGP and ground-truth tests of recovered computations. This stream adds structural transfer tasks to the existing mixed-rule benchmark: visible `Reports`, `Guide`, `FAQ`, and `Policy` headers select a report grammar with competing verb rules, numbered list continuation, question-answer object copying and place recall, or conditional actions whose condition, object and prior action can conflict. It has an exact visible-prefix next-token oracle. The header makes these task families easier to separate than the original joint rule competition; use this as a transfer benchmark, not a replacement for that benchmark.

`audit_multi_genre_language.py` checked every prefix of 4,000 generated 72-token streams, normalization and support of the oracle, and independent list/FAQ/policy invariants. The checks passed. The 480-window, 72-token comparison in `multi_genre_audit.json` remains strongly negative on realism: 157 distinct alphabetic words versus 6,305 in FineWeb prefixes, mean sentence lengths 10.5 versus 16.3 words, and p90 16 versus 29. The two streams have different tokenizers; mean words per window are similar (58.5 versus 54.3). Naturalness, lexical breadth, and sentence diversity are not established.

## Baseline learning

`train_multi_genre.py` trained 1–4-layer, width-64, four-head, attention-only bilinear transformers with RMSNorm for **3,000 AdamW steps**, batch 64, on all four genres. Each model has one training seed (`seed=0`). The following numbers are expected KL to the exact oracle, in nats, on **four independent test streams of 1,024 sequences** each (`fresh_eval_4x1024.json`). Easy frame tokens are excluded from the task-specific columns.

| Layers | Report verb mixture | Policy action mixture | FAQ copy | FAQ place recall | FAQ new place | List number |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.217 | 0.241 | 0.014 | 0.779 | 0.271 | 0.001 |
| 2 | 0.200 | 0.243 | 0.008 | 0.537 | 0.232 | 0.001 |
| 3 | 0.198 | 0.236 | 0.009 | **0.164** | **0.027** | 0.001 |
| 4 | **0.189** | **0.228** | 0.010 | 0.242 | 0.041 | 0.001 |

The third layer sharply helps within-document place recall in this run. Four layers improve the report/policy mixtures a little but have worse FAQ recall after equal training steps. A second 3/4-layer seed gives fresh-test FAQ recall KL **0.180 / 0.165** (versus seed 0's **0.164 / 0.242**) and causal recall effect L1 error **0.226 / 0.228** (versus **0.214 / 0.382**). Thus both deeper models can learn recall, but the apparent 3-versus-4 ordering reverses between seeds. This is **not** a general depth conclusion: there are only two deep-model seeds and optimization difficulty differs by depth.

`eval_multi_genre_edits.py` then makes grammar-valid, supported single-token edits and compares model probability changes with the exact oracle changes on 1,024 new generated streams (`paired_edits_n1024.json`). For the 193 FAQ place-recall pairs, effect L1 error drops from **1.383 / 1.018** at one/two layers to **0.214** at three and **0.382** at four. All depths copy the FAQ question object well: 824 copy-pair errors are 0.029–0.040. On 1,115 policy-condition edits, the three-layer effect has high direction cosine (**0.952**) but still L1 error **0.505**, and on 366 previous-action induction edits its error is **0.406**. Thus good policy next-token KL does not imply a quantitatively correct rule-weight circuit. A policy-person edit is an oracle no-effect control; its model effect shrinks from L1 **0.229** at one layer to **0.069** at three.

## Weight-only compression and task transfer

At the **first attention-output bond** of the two-layer model, `eval_multi_genre_bond_gauges.py` fits activation PCA on Reports and Policy only, then tests every genre on fresh streams. Weight-only right singular vectors of the attention output matrix need no fitting text. The table reports KL from the original model to its 16/64-coordinate truncation; lower means more faithful **model compression**. The FAQ and Guide columns are task-family holdouts for PCA fitting, not model training.

| 16/64-coordinate bond basis | Report verb | Policy action | FAQ place recall | List number |
| --- | ---: | ---: | ---: | ---: |
| Original coordinate order | 0.206 | 0.523 | 1.244 | 6.808 |
| Activation PCA on Reports/Policy | 0.049 | 0.018 | 0.531 | 1.996 |
| Weight-only SVD | **0.010** | **0.006** | **0.010** | **0.002** |

At 32/64, weight-only SVD is nearly lossless on these groups (model-to-truncated KL ≤0.0002). PCA fitted on all genres also handles lists well at 16/64 (0.002), confirming that the large held-out-list loss above comes from the fitting distribution. In seed 1's three-layer model, 16/64 weight-only SVD at the first bond also gives held-out FAQ-recall/list model-to-truncated KL **0.001 / below 0.001**, versus **0.060 / 0.001** for Reports/Policy-fitted PCA. This is a useful weight-only transfer result, but the SVD targets the local linear output map: low singular values and faithful compression do not prove canonicalization of the entire attention function or identification of a language rule.

## Where edits are locally mediated

`eval_multi_genre_edit_bases.py` fits contrast-PCA bases on valid edits and tests them on fresh sequences whose **object words are disjoint** from the basis-fitting objects. For each three-layer model, 512 fitting and 512 test streams yielded 818 and 855 selected edit pairs. Patching the donor's **final-token activation** at a full attention-output bond gives a local mediation ceiling. Mean L1 residual between the full-bond-patched probability change and the complete edited-model change is:

| Training seed and edit | Middle attention bond (layer index 1) | Final attention bond (index 2) |
| --- | ---: | ---: |
| Seed 0, FAQ answer copy | **0.021** | 1.984 |
| Seed 0, FAQ earlier-place recall | 1.937 | **0.0004** |
| Seed 0, policy condition | 1.244 | **0.002** |
| Seed 1, FAQ answer copy | 1.822 | **0.199** |
| Seed 1, FAQ earlier-place recall | 1.844 | **0.107** |
| Seed 1, policy condition | 1.116 | **0.001** |

The corresponding seed-0 mean model effect magnitudes are 1.992, 1.937, and 1.244. Full-sequence patching at the first and middle bonds did not materially alter seed 0's allocation; all-position patching at seed 1's middle bond likewise leaves a **1.775** FAQ-copy residual. Seed 0 routes answer copying through the middle bond, while seed 1 routes most of it through the final bond. Both learn similar copy and recall behavior. This seed dependence is direct evidence that **where** a task is locally mediated is not a stable description of the function across these two trained models.

Within seed 0, a 16-coordinate FAQ-recall contrast basis fitted on the first half of objects retains the final bond's recall response on the second half with patch-to-full L1 **0.008**; the FAQ-copy contrast basis has error **1.398** on that same recall response. A policy-condition contrast basis has error **0.024** on its own final-bond response. These are task-related *local response subspaces*, not an end-to-end circuit decomposition. A low patch-to-full error is uninformative when the full bond barely mediates that edit; the mediation table must be checked before interpreting contrast scores. Cross-seed alignment of components remains untested.

## Next discriminating tests

1. Add further seeds and longer optimization, then stratify FAQ recall by distance and repeat count. A depth-dependent memory claim requires that control.
2. On the **joint** mixed-rule stream, hold out entire collision patterns and use a visible rule-strength cue to measure the exact counterfactual log-odds shift. Compare weight-only, activation, sparse, and supervised bases on both model fidelity and oracle-effect fidelity.
3. At the multi-genre bonds, test whether the compact response directions remain selective under interventions on every residual path and with larger object/task holdouts. Only then claim a reusable component rather than a convenient local basis.
4. Expand language diversity with observed FineWeb forms and manually annotated dependency rates, keeping the exact prefix oracle. This generator is still a diagnostic toy, not realistic web text.
