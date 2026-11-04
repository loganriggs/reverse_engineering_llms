# Corpus-grounded list rule: learning, length transfer, and OV reduction

The numbered-list DGP edits only line-initial list markers in real FineWeb prose; a nearby parenthetical can be edited independently. The local cache contains reconstructable source text and is not committed. Its source-document split is 95k–105k for development, 115k–215k for training, and the previously unseen 215k–225k for test. An exact-hash and 5-gram SimHash Hamming≤3 screen rejects duplicates among selected eligible sources. There are 1,187 distinct list-training documents, 113 development and 90 test list pairs, and 29/20 development/test four-arm list × bracket documents. The separate bracket training cache has 1,801 FineWeb source windows from offsets 44k–50k. **No joint list × bracket document was used for training.**

Two depth-2, width-64, four-head bilinear transformers began from different bracket-trained checkpoints. Fine-tuning used ordinary next-token cross-entropy over all valid positions, with 50% of each batch drawn from list-only and 50% from bracket-only windows. Rule sites received extra loss weight (100 for list, 50 for bracket). The bracket starts had already learned a `)`-versus-`]` preference, but neither initially handled the list edit.

## What was learned

Training only at the **third** list marker (base `3`, shifted `4`) produces high paired full-vocabulary exact accuracy on new third-item pages: .978 and .922 for the two seeds. On a further disjoint 225k–235k panel, those models score .965/.886 at item 3 but **zero** paired exact at items 4, 5, and 6. This is a learned local transition, not evidence of general counting.

We then held architecture, source split, optimizer, steps, and bracket mixture fixed while adding weighted list sites at items 4 and 5. There were 1,187/715/407 distinct training source documents contributing item-3/4/5 site pairs; some documents contribute to multiple positions. On the original document-disjoint test panel, paired exact list accuracy for the two new seeds was:

| Target item | Seed 71503 | Seed 71504 |
| --- | ---: | ---: |
| 3 | .933 | .867 |
| 4 | .875 | .958 |
| 5 | .926 | .852 |

On the further fresh 225k–235k panel, item-3/4/5 paired exact is .930/.862/.938 and .921/.879/.875. At **item 6**, neither seed succeeds on the 13 available pairs. Seed 71503 gets all 13 unshifted sixth items as top-1 but none of their shifted seventh items. Seed 71504 gets 10/13 unshifted and zero shifted. Thus broader supervision extends the *learned range* but still does not show extrapolation beyond it. All-token loss exposed other numbers, so the contrast is about the effect of weighted rule sites under this training regime, not a proof that the architecture cannot count.

On the 20 new joint documents, the multi-position models have paired exact list accuracy .85/.75 and bracket candidate two-way accuracy .85/.70. Requiring **both rules to prefer the correct counterfactual in all four arms** gives .80/.55; requiring every answer to be full-vocabulary top-1 gives only .35/.15. The first seed's standalone bracket two-way test improved from .886 to .934; the second changed .895 to .880. The joint candidate result is encouraging but too small and seed-variable to call a robust exact two-rule circuit. The list rule is behaviorally competent over the trained range; bracket remains a candidate-preference rule whose full-vocabulary accuracy is much lower.

## OV method check

For the two multi-position models, we fitted attention-input covariance on 576 **separate-rule training** windows. We compared rank-4/8/16 per-head approximations of each OV product, either ordinary weight SVD or SVD after whitening by that input covariance. QK, MLPs, embeddings, and unembedding stayed frozen. At full rank 16, balanced refactorization changed probe logits by only about `2e-5` in both seeds, verifying an exact OV gauge up to floating-point error.

At rank 8, data-aware truncation lowered the mean input-weighted OV error in both seeds (for example .328→.249 in seed 71503) while increasing ordinary weight error (.375→.404). It also lowered teacher KL at the bracket site from .444→.015 and .236→.030; list-site KL fell .024→.013 and .064→.020. Yet paired list exactness was unchanged (.922 and .867), and bracket two-way changed .919→.925 and .892→.886. Rank-4 behavior and joint all-arm preference similarly vary by seed/method. **Lower average reconstruction error and lower site KL do not uniquely select the best discrete causal rule score.** This remains an OV-only reduction, not a canonical form of the complete attention function or an interpretable sparse circuit.

The next method comparison should evaluate task-specific sparsity or rule-aware OV optimization against list and bracket candidate margins, background KL, and causal edits on new documents. In parallel, the DGP needs more genuinely different rules and stronger whole-combination coverage; the item-6 failure warns against labeling the current numeral behavior as an algorithmic counter.

Code: `build_corpus_numbered_list_cache.py`, `train_corpus_list_delimiter_mixture.py`, `eval_corpus_list_length_transfer.py`, `eval_corpus_list_joint_recheck.py`, `ov_linear_reduction.py`, and `eval_corpus_list_ov_reduction.py`. Aggregate JSON reports are under `results/corpus_numbered_list_*`, `results/corpus_list_*`; model checkpoints and token caches stay local.
