# Corpus-grounded nonce-copy pilot

## Question and scope

Can a small bilinear language model learn a genuinely document-local literal-copy rule in varied human prose, and will it transfer to new source documents and unused code combinations? This pilot is a behavioral prerequisite for later weight-based circuit analysis, not yet a canonicalization or sparsity result.

The generator selects FineWeb documents containing a repeated, three-letter uppercase organization-style acronym. It consistently substitutes a sampled, opaque three-letter fictional alias throughout each selected document. Two versions of the same source differ only in that alias, so the later occurrence has a paired causal target. The opaque codes each tokenize as three BPE pieces; the model must predict all three in order. The natural surrounding text is retained as language-model training context, but only the marked copy sites have an exact rule-based answer. Source facts are not asserted to remain historically true after alias substitution.

This task is deliberately narrow. The entity-source whitelist and strict exclusions remove obvious unit/currency/economic acronyms that would make the substitution semantically misleading. The whitelist is a surface prior, not an entity recognizer. Two independent 1,000-document screens found 12 and 11 eligible/aligned documents. We reviewed examples during construction; the selected population is only about 1% of FineWeb, so this is not a full natural-language DGP.

## Data and checks

The source split uses nonoverlapping FineWeb stream offsets: validation 95,000–104,999, training 115,000–214,999, and test 335,000–344,999. The base cache selected 177, 1,846, and 166 source documents respectively. Selected documents were screened for exact hash matches and near duplicates at SimHash distance ≤3 across splits. Training codes come from a 256-code pool; the test's second panel uses a disjoint 256-code pool on the *same held-out source documents*. This isolates code-combination transfer from document transfer. The code-level holdout does not hold out individual letters.

The validation fixture checks paired token alignment, prefix-only causality at the target, three-piece tokenization, and exact reversibility. The evaluation teacher-forces alternative candidate sequences after the same visible prefix and also measures unrestricted greedy generation. Full-sequence preference between the two sampled aliases is an easier diagnostic than exact generation; it is not evidence that the model can emit an arbitrary code.

The local `.pt` caches reconstruct FineWeb text and remain outside git. Aggregate manifests contain counts and source hashes, but no source prose.

## Results before the eight-draw comparison

| Model | Training | Test seen-code paired preference | Test seen-code exact greedy, both worlds | Test unseen-code paired preference | Test unseen-code exact greedy, both worlds |
|---|---|---:|---:|---:|---:|
| Depth 2, width 64 | nonce/list/bracket mixture, 3,000 steps | 3.0% | 0% | 3.0% | 0% |
| Depth 2, width 64 | nonce-focused, 6,000 steps | 85.5% | 0.6% | 68.1% | 0% |
| Depth 4, width 128, RMSNorm | nonce-focused from scratch, 6,000 steps | 98.8% | 24.7% | 93.4% | 6.0% |

The larger model's training-first-256 exact greedy score was 91.8%, compared with 30.5% on validation and 24.7% on test familiar codes. That gap, and the 6.0% score on unused code combinations, disqualify a claim of robust open-vocabulary copying. Its held-out seen-code first/second/third piece top-1 accuracies were 50.6%, 83.7%, and 80.1%. The first piece is the main bottleneck. The depth-2 focused model also forgot list and bracket behavior inherited from its initial checkpoint, so the focused experiment is diagnostic rather than a competent three-rule model.

The controlled comparison reuses exactly the same selected source documents, validation set, test set, and unused-code panel, but draws eight alias pairs per training source. It has 14,768 training pairs rather than 1,846. The build resumed successfully from an atomic partial cache after an illegal-instruction process crash. Tensor equality confirms the held-out panels match the base cache exactly. Both conditions use the same depth-4 width-128 RMSNorm architecture and 6,000-step budget; each seed is compared with itself.

| Seed | Training aliases/source | Validation exact greedy, both worlds | Test seen-code exact greedy, both worlds | Test unseen-code exact greedy, both worlds | Test seen-code sequence NLL | Test unseen-code sequence NLL |
|---|---:|---:|---:|---:|---:|---:|
| 72104 | 1 | 30.5% | 24.7% | 6.0% | 4.68 | 7.63 |
| 72104 | 8 | 50.8% | 39.2% | 15.7% | 3.11 | 5.30 |
| 72105 | 1 | 22.6% | 25.3% | 4.8% | 4.60 | 8.45 |
| 72105 | 8 | 46.9% | 45.2% | 18.1% | 3.02 | 5.22 |

Eight-draw training improves exact generation in both seeds, but still leaves most unused-code combinations incorrectly generated. Seen-code paired *two-choice* sequence preference was 98.8%/99.4% in the two augmented seeds, well above exact generation; evaluating only that easier measure would overstate competence. The first-256 training score is not directly comparable between caches because the augmented cache's first 256 pairs come from fewer distinct source documents. The extra pairs change sampling frequency over alias realizations and reduce repeat exposure to any particular pair; they do not add independent source documents. Two seeds support a directional effect, not a precise population estimate.

If exact held-out generation remains weak, stop this narrow copy-task detour and broaden the multirule DGP rather than interpreting a nonexistent robust circuit. If it improves, repeat across seeds and put copying back into a capable joint model, then apply weight-only canonicalization and sparse decomposition under behavioral and causal fidelity checks. Track behavioral competence, reconstruction, task prediction, causal fidelity, and new-rule discovery separately.

## Three-rule joint model and a task-blind head probe

Two depth-4 width-128 RMSNorm models trained from scratch on 50% augmented nonce-copy, 25% numbered-list, and 25% bracket windows, 6,000 steps with site weights 200/500/500. They received all-token next-token CE in every window. The list and bracket source documents are separate rule windows; no nonce × list × bracket document was used for training or held-out testing. The list × bracket four-arm test panel remains an unseen combination. The second seed's first process crashed before its first checkpoint; a clean retry with restricted native threading finished. The trainer now writes resumable model/optimizer/RNG state every 500 steps.

| Joint model seed | List paired exact | Bracket paired two-way | Nonce familiar-code exact, both worlds | Nonce unused-code exact, both worlds | List × bracket all-arm exact / two-way |
|---|---:|---:|---:|---:|---:|
| 72104 | 92.2% | 95.5% | 21.7% | 7.2% | 23.5% / 88.0% |
| 72105 | 91.1% | 96.1% | 19.3% | 9.0% | 42.7% / 88.0% |

The nonce-only models at the matched seeds and architecture got 39.2%/15.7% and 45.2%/18.1% exact on familiar/unused-code panels. Joint training recovers list and bracket while weakening nonce exactness, especially code-combination transfer. This is a partial three-rule model, not robust joint mastery. The list × bracket exact score is seed-variable despite equal two-way score; tokenization and background logits still matter.

As a first weight-method check, each attention head was ranked without task examples by the product of the Frobenius norms of its two composed QK maps and composed OV map. The ranking was frozen before ablating one whole head at a time on the **full 166 nonce, 90 list, and 334 bracket held-out source panels**. These are original-plus-edited pairs, so the measured drop requires both worlds to remain correct. The task labels enter only after weight ranking.

| Seed | Rule | Most necessary head, layer/head | Weight-only rank of that head | Drop in exact paired generation (nonce/list) or paired two-way (bracket) | Largest cross-rule drop |
|---|---|---|---:|---:|---:|
| 72104 | Nonce | 6/0 | 1/16 | 21.7 points | 1.1 points |
| 72104 | List | 4/2 | 14/16 | 92.2 points | 0.6 points |
| 72104 | Bracket | 6/3 | 2/16 | 95.2 points | 0.0 points |
| 72105 | Nonce | 6/0 | 1/16 | 19.3 points | 1.1 points |
| 72105 | List | 0/3 | 14/16 | 91.1 points | 0.6 points |
| 72105 | Bracket | 6/3 | 2/16 | 94.9 points | 0.0 points |

The nonce head also reduces two-code sequence preference by 85.5/92.8 points, a stronger signal than its exact-generation drop because native exactness is only 21.7/19.3%. The list-critical head changes location across seeds but is ranked 14th in both. This is a replicated failure of the simple weight-only ranking to prioritize one clearly learned rule. It does not show that weight-only canonicalization in general fails. Whole-head ablation establishes necessity in the intact model, not a unique circuit or sufficient path. In particular, the three rules have not been tested in one crossed document; task-specific heads may partly reflect the disjoint training-window distributions.

## Matched-budget OV reductions

We next canonicalized each head's composed linear value-output map, `O_h @ V_h`, by SVD. Weight-only SVD uses the ordinary Frobenius metric; data SVD first weights input directions by an activation covariance fitted on 384 separate-rule **training** windows (64 distinct nonce source documents per edited arm, 64 list rows per arm, and 128 bracket rows). No held-out or list × bracket joint document enters this fit. In both methods, each of 16 heads retains rank 8, 16, 24, or its native 32; QK, MLPs, embeddings, and readout remain fixed. At rank 32, both refactorizations reproduce probe logits to the script's `<0.001` tolerance. The measured mean input-weighted OV error is lower for data SVD at every compressed rank in both seeds, as its objective predicts. That lower local reconstruction error does not consistently improve exact-rule behavior. At rank 8, both methods retain list and bracket behavior while nonce exact generation falls to 4.8–7.8%; at rank 16, familiar nonce exactness is 19.9–21.7%, near the native 19.3–21.7% range.

For the same retained OV channel budget, we also kept the highest-ranked heads at full width and zeroed all other heads. Keeping 8 of 16 full-width heads uses `8×32=256` OV channels, matching rank 16 in every head. Keeping 12 uses 384 channels, matching rank 24 everywhere. Each retained channel carries the same O and V factor dimensions in these comparisons. The head ranking uses the task-blind QK1×QK2×OV norm product above.

| Seed | OV allocation, 256-channel budget | List paired exact | Bracket paired two-way | Familiar nonce exact, both worlds | List × bracket all-arm exact |
|---|---|---:|---:|---:|---:|
| 72104 | Rank 16/head, weight SVD | 92.2% | 94.9% | 19.9% | 18.8% |
| 72104 | Rank 16/head, data SVD | 92.2% | 95.2% | 21.7% | 20.1% |
| 72104 | Top 8 full heads | **0%** | 95.5% | 19.9% | **0%** |
| 72105 | Rank 16/head, weight SVD | 91.1% | 96.4% | 21.1% | 42.7% |
| 72105 | Rank 16/head, data SVD | 91.1% | 96.4% | 20.5% | 42.7% |
| 72105 | Top 8 full heads | **0%** | 95.5% | 22.9% | **0%** |

At the 384-channel budget, rank-24 truncation still retains 91.1–92.2% list exactness, while top-12 full-head pruning leaves only 0–1.1% and zero all-arm list × bracket exactness. On a 64-row held-out list-site KL probe, the top-12 pruned models have teacher KL 4.20/.33 versus ≤.001 for rank-24 students. The qualitative result is replicated across seeds: a global head norm allocates zero capacity to a low-ranked but necessary list head, whereas spreading rank across heads preserves it. This does **not** make uniform rank universally optimal; the result depends on this model, metric, and budget. Neither local OV SVD nor pruning is a canonical decomposition of the full bilinear attention function or a proof of human-legible features. The [aggregate comparison](results/corpus_three_rule_ov_weight_vs_data_and_pruning_seed72104_72105.json) separates reconstruction, rule behavior, and teacher KL; the [script](eval_corpus_three_rule_ov_reduction.py) reproduces it.

## Constraint on the next crossed DGP

An exact-hash intersection of the existing selected source manifests finds only 18 documents eligible for **both** nonce-copy and numbered-list edits among 100,000 training FineWeb sources, and one among 10,000 validation sources. These figures reflect the current strict eligibility and alignment screens, not the prevalence of all acronym/list co-occurrences on the web. The nonce and list test caches use different source-offset ranges, so their intersection is intentionally reported as incomparable. There is no adequate naturally co-occurring test panel for a three-rule crossed-document claim here. The [overlap audit](results/corpus_nonce_list_source_overlap.json) is a stop signal for simply intersecting the current caches; the next DGP should use more common naturally grounded rule combinations or sampled compositions whose form and semantics are independently audited.
