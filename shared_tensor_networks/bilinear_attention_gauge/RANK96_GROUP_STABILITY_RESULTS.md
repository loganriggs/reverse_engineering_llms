# Stable functional groups exist where sparse axes are unstable

The [rank-96 single-axis panel](RANK96_LEGIBILITY_PRIORS.md) found that sparse pair-coordinate gauges transfer numerically but align weakly across seeds and fail the coarse token-form label test. This experiment tests the narrower hypothesis that an **energy-selected group of axes** is reproducible even when its members are not. It runs on both the uniform weight-only QK1 cut and the FineWeb-weighted QK1 cut of the 546M model's layer-0/head-0 attention.

The grouping rule is deliberately simple and label-free: rank the 96 exact-gauge axes by their QK2-and-OV-weighted squared pair contribution on 16,384 FineWeb selection pairs, then take the top *k* for *k* = 4, 8, 16, 32, 48, or 64. This **supplies the group size and extracts one high-energy group**; it does not infer a taxonomy or discover an unknown number of concepts. All rotations leave the underlying rank-96 cut function unchanged to relative bond error below `1.6e-14`.

For each group we compare (i) its projector in the 96-dimensional SVD factor coordinate space and (ii) the QK2-and-OV-weighted cosine of its group QK1 score on 8,192 held-out pairs per law. There are three optimizer seeds. A random orthogonal gauge, an SVD group selected by the same energy rule, and a deterministic factor-weight gauge are controls. We also optimize the intervention-aware activation objective for 600 steps from **independent random orthogonal starts** (three seeds), then compare all nine identity-start × random-start pairs. Selection pairs, held-out FineWeb pairs, WikiText-103 pairs, and TinyStories pairs are separate.

## Four-axis group stability

| Cut / start comparison | Mean projector overlap | Mean held-out FineWeb response cosine | Mean WikiText cosine | Mean TinyStories cosine |
|---|---:|---:|---:|---:|
| Uniform / identity starts | .701 | .929 | — | — |
| Uniform / random starts | .605 | .917 | — | — |
| Uniform / cross-start | **.657** | **.924** | **.880** | **.919** |
| Corpus / identity starts | .588 | .872 | — | — |
| Corpus / random starts | .671 | .921 | — | — |
| Corpus / cross-start | **.664** | **.914** | **.883** | **.920** |

The within-family cosines in this table use the held-out FineWeb corpus-pair law; cross-start uses the same held-out FineWeb document range. The reference overlap of independent random four-dimensional subspaces is 4/96 = .0417; energy-selected random rotations gave .055 on the uniform cut and .038 on the corpus cut. Individual intervention-aware axis signatures had mean cross-seed matched absolute correlation only .196/.192. Random starts retain that axis instability (.192/.194), but the group remains stable.

The effect is **not simply the top SVD group**. A four-axis sparse group overlaps the energy-selected SVD group by only .28/.18 in the identity-start fits (uniform/corpus); random starts give .26/.18. On held-out corpus pairs, the identity-start intervention-aware groups carry mean squared response energy fraction .127/.161, versus .057/.043 for the four-axis SVD group and .019/.023 for energy-selected random-gauge groups. These fractions are ratios to the complete cut's weighted squared score; they are not percentages of downstream logit variance. At *k* = 8, cross-seed within-family projector overlap remains about .54–.61, well above the random reference .083. The result is a stable *high-energy subspace*, not a unique coordinate basis.

## Does the group have a reproducible full-model effect?

We remove each top-four group partially, calibrating the deletion to `.001` aggregate relative layer-0 head-output error on 64 separate FineWeb documents. Calibration fractions range .092–.268 across all groups and cuts. We then propagate through all 18 layers on **two seeds × eight documents per corpus**, comparing each edited model with its unchanged rank-96 base cut. The identity-bond logit gate is zero. The outcome is the centered correlation of per-document KL effect vectors between independently selected groups; centering removes a shared mean effect but does not guarantee semantic selectivity.

| Cut / corpus | Random groups, within family | Sparse groups, identity starts | Sparse groups, random starts | Sparse groups, cross-start |
|---|---:|---:|---:|---:|
| Uniform / FineWeb test | .459 | .813 | .978 | **.896** |
| Uniform / WikiText-103 | .388 | .842 | .695 | **.806** |
| Uniform / TinyStories | -.071 | .933 | .823 | **.884** |
| Corpus / FineWeb test | .493 | .760 | .872 | **.792** |
| Corpus / WikiText-103 | .622 | .772 | .891 | **.811** |
| Corpus / TinyStories | .621 | .927 | .884 | **.905** |

Each within-family cell averages the three seed pairs; each cross-start cell averages nine pairs. The random controls use the **same top-four selection rule** before calibration. Mean group-deletion KL is only about 0.25–0.94 × 10⁻⁶ nats/token for the random and sparse families across these corpora, so the causal-profile test is sensitive to small changes. The sparse-group effects are more reproducible than the random controls in every row, but only 16 documents per corpus and one head were tested; document outliers and the shared selection distribution can inflate correlations. This is stronger than pair-response stability alone, and weaker than identifying a selective computation.

## Interpretation and next discriminating test

The data support **group-level reproducibility within a chosen physical bond and fidelity prior**. They do not establish semantic meaning, a unique number of groups, recovery of the trained model's physical computation, or transfer to another head or layer. In particular, the group-selection rule chooses one dominant aggregate response, so it could be a sensitivity mode rather than a concept.

The next experiment should partition or cover the bond with **multiple** groups without supplying the group count, then test whether independent fits recover corresponding groups and whether deleting one group has a selective, predictable effect on a newly nominated task or context. Use the full-model effects and QK2 branch edits as functional signatures; hold out entire task families or contexts from both group construction and label nomination. If no selective groups emerge, retain the stable high-energy subspace as a useful compression object rather than calling it an interpretation.

Reproduce with `flagship_exact_gauge_sparsity.py --cut-name <cut> --n-pairs 16384 --n-ood 8192 --steps 600 --batch-size 2048 --seeds 0 1 2 --kinds intervention_activation --random-start` for each cut; `flagship_rank96_group_stability.py` with its default, `--random-start`, and `--cross-start` modes; then `flagship_rank96_group_ablation.py --n-docs 8`. The `full_width_results/head0_rank96_group_*` files and random-start gauge JSON/NPZ files contain the per-seed records. The model checkpoint and corpora are referenced externally.
