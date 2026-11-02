# Shared-dictionary featurizers on whole-genre toy-language holdouts

The two trained three-layer, width-64 bilinear language models perform FAQ place recall. This experiment asks whether a shared dictionary with overlapping group gates gives a sparse, transferable description of the final attention bond. At this site and training budget, **it does not**. A learned scalar Top-16 dictionary is much better at reconstruction and in-fit output fidelity; the model's original 16 physical coordinates preserve held-out FAQ recall and its causal edit effect best. Adding exact downstream teacher KL to the fitting loss does not consistently close the transfer gap.

`train_multi_genre_shared_groups_holdout.py` fits featurizers to the layer-2 pre-output attention bond. The language models were trained on all four genres; **only the featurizer** withholds FAQ and Guide. Its 512 fit and 128 validation streams contain Reports and Policy only (36,352 and 9,088 token-position vectors); its evaluation uses 1,024 fresh streams across all genres and 179 valid historical-place edits from an independent 1,024-stream panel. Two independently trained model seeds share these evaluation panels. The 16/64 native-coordinate and weight-SVD gates select the largest 16 coordinates per token without fitting. Scalar Top-8/Top-16 learn an encoder and dictionary; fixed disjoint and learned overlapping groups use 16 groups of four atoms with four active groups, so their union can contain at most 16 atoms. The overlapping model pays a membership-coverage penalty. All learned candidates use PCA initialization, the same fit/validation vectors and 1,200 optimizer steps. They are **not parameter-count matched**: the comparison fixes the active-coordinate ceiling, and the disjoint-group control isolates part of the group-selection cost.

The table gives relative bond squared error on the full test stream, native-model-to-gated-model KL on FAQ recall sites, and L1 error of the model's probability *change* under a valid previous-place edit against the exact generator oracle. Lower is better. The native, ungated edit error is 0.219/0.234 on this edit panel for model seeds 0/1.

| Fit objective; model seed; 16 active coordinates max | Bond relative MSE | FAQ model→gate KL | FAQ recall-edit L1 | Mean active atoms |
|---|---:|---:|---:|---:|
| No fit; seed 0 physical axes | .039 | **.0057** | **.202** | 16 |
| No fit; seed 0 weight SVD | .116 | .0876 | .490 | 16 |
| Reconstruction; seed 0 scalar Top-16 | **.019** | .148 | .551 | 16 |
| Reconstruction; seed 0 disjoint groups | .109 | 5.107 | 3.476 | 16 |
| Reconstruction; seed 0 shared groups | .139 | 3.390 | 3.320 | 13.52 |
| No fit; seed 1 physical axes | .044 | **.0206** | **.328** | 16 |
| No fit; seed 1 weight SVD | .129 | .0366 | .389 | 16 |
| Reconstruction; seed 1 scalar Top-16 | **.032** | .0805 | .554 | 16 |
| Reconstruction; seed 1 disjoint groups | .181 | 1.918 | 2.465 | 16 |
| Reconstruction; seed 1 shared groups | .216 | 3.329 | 2.409 | 14.42 |

The distinction between reconstruction and functional fidelity matters here: scalar Top-16 reconstructs the bond *better* than physical Top-16 in both seeds, yet physical coordinates preserve FAQ behavior much better. Group models are also weak on the **in-fit Policy** task: with reconstruction fitting, their Policy model→gate KL is .166/.215 for disjoint and .366/.302 for shared, versus .005/.006 for scalar Top-16. Their poor FAQ scores therefore cannot be attributed only to held-out-genre shift. The overlapping decoder covers 43/46 of the 64 atoms on the two test panels and activates 13.52/14.42 per vector; it is neither a matched-16 support nor an entirely collapsed dictionary. Scalar Top-8 is a lower-support control and also loses much FAQ recall (model→gate KL .715/.569).

For an output-aware control, the fit loss adds the **exact** downstream KL from native to reconstructed-bond logits to standardized bond MSE, at coefficient 1. This is possible because the final block is attention-only: replaying `final_norm(residual + out(bond))` reproduces the native logits within 4.8e-6/3.8e-6 maximum error. The objective uses only Reports/Policy streams, and the FAQ/Guide panel remains untouched during fitting and checkpoint selection. It is token-averaged over all positions, not balanced toward rare rule sites or causal edits.

| Output-aware fit; model seed | FAQ model→gate KL | FAQ recall-edit L1 | In-fit Policy model→gate KL |
|---|---:|---:|---:|
| 0; scalar Top-16 | .209 | .641 | .0059 |
| 0; disjoint groups | 4.041 | 3.394 | .145 |
| 0; shared groups | 3.333 | 3.144 | .258 |
| 1; scalar Top-16 | .0272 | .531 | .0058 |
| 1; disjoint groups | 1.958 | 2.218 | .160 |
| 1; shared groups | 1.748 | 2.283 | .289 |

Seed 1 gains some held-out FAQ KL from the output loss, but seed 0 does not, and the group models remain far behind the native-coordinate intervention on both edit panels. This does **not** show that overlapping groups cannot represent meaningful features. Hard gate optimization, group layout, initialization and the mismatch between generic output KL and rare causal effects remain plausible bottlenecks. The result is enough to stop tuning this particular final-bond reconstruction setup: it has failed both the in-fit fidelity prerequisite and the whole-genre causal-transfer test.

The next comparison should use a **path-level or explicitly causal** objective on a teacher competent at multiple joint rules, with whole rule-family fit holdouts. Report reconstruction, teacher KL, exact-oracle task prediction, and valid-edit causal fidelity separately. A putative new task discovered without its label must be tested against withheld generator rules, not just assigned a plausible name. This returns the experiment to the diverse-task → causal supervision → featurizer → unlabeled discovery loop, rather than extending a single-bond hyperparameter sweep. For the parallel sparse-transformer branch, the [two-site verb/object replication](V3_SPARSE_FACTOR_TRANSFER_RESULTS.md) already found faithful weight-sparse factorization without a clear circuit-legibility advantage; adding depth should clear the same behavior and causal-selectivity bars at matched total active edges.

Reproduce with `train_multi_genre_shared_groups_holdout.py --model-dir language_task_suite/multi_genre_models_v1 --nfit 512 --nval 128 --ntest 1024 --nedit-streams 1024 --steps 1200 --output-dir language_task_suite/multi_genre_shared_groups_holdout_seed0`. Substitute `multi_genre_models_seed1` for the second model; add `--output-kl-weight 1` and use the `multi_genre_shared_groups_output1_seed*` directories for the output-aware arms. Each directory holds `results.json` and the learned featurizer states.
