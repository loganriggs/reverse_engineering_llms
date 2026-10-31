# A held-out QK2 edit tests whether branch-aware reduction transfers

The [first full-model branch test](FULL_MODEL_BRANCH_RESPONSE_RESULTS.md)
trained a rank-256 first-layer QK1-key basis partly on a QK2-to-one edit,
then evaluated the same edit family. Here the fitted bases are frozen and
the edit family changes: each QK2 score for key position `j` is interpolated
5% toward the score of key `max(j−1,0)` for the same query and head. The
replacement is made before QK1·QK2 multiplication. It is causal and still
requires the QK1 dictionary; a QK1-to-constant edit would largely bypass
the dictionary being tested. The ordinary unedited model is evaluated
separately, and the score compares each reduced model's *edited-minus-
unedited final-logit response* with that of the native 18-layer model.

All four bases have the same rank and were fitted before these evaluation
documents were chosen: uniform key-table SVD (weight-only), usage-weighted
key SVD, ordinary function-aware fitting, and fitting to a 50/50 ordinary
plus QK2-to-one pair-output objective (branch-aware). There are 24 disjoint
128-token documents each from held-out FineWeb, external WikiText-103, and
external TinyStories, using the same documents as the QK2-to-one test. The
metric is pooled squared error in the *full-vocabulary* logit response,
divided by pooled native-response energy. Lower is better. An identity
rank-1152 projection changes ordinary or edited logits by at most `3.0e-5`
in the two panels.

| Rank-256 QK1-key basis | FineWeb | WikiText-103 | TinyStories |
|---|---:|---:|---:|
| Uniform key-table SVD | .02680 | .05791 | .04391 |
| Usage-weighted key SVD | .01168 | .03216 | .01418 |
| Ordinary function-aware | .01088 | **.01726** | **.00637** |
| QK2-branch-aware | **.00999** | .01844 | .00677 |

The paired branch-aware minus ordinary-function differences are `−.00088`
on FineWeb (95% document-bootstrap interval `[−.00154, −.00024]`, 20/24
document wins), `+.00117` on WikiText (`[−.00115, +.00351]`, 10/24 wins),
and `+.00039` on TinyStories (`[−.00002, +.00085]`, 9/24 wins). These are
within-corpus bootstrap intervals from 10,000 resamples, not a claim that
the corpus itself was randomly sampled. Native response-energy effective
sample sizes are 18.6, 17.8, and 19.8 documents respectively. The edit
produces a measurable response, and the identity-projection error is much
smaller than it.

The QK2-to-one training objective therefore improves response fidelity on
FineWeb under a different QK2 edit, but there is **no established
external-corpus improvement** over ordinary function-aware fitting. On
both external corpora the ordinary basis has the lower point estimate.
Function-aware fitting beats uniform weight-only SVD on all three corpora;
usage weighting also improves over uniform SVD but gives up accuracy to
function-aware fitting here. This is a statement about full-model *response
prediction at fixed rank*, not proof that either basis gives human-readable
causal components. The current training objective may preserve common
QK2-branch response directions without yielding a corpus-general causal
coordinate system.

The next experiment should treat **edit family and corpus as independent
holdouts** when learning condition-indexed response signatures. It should
gate proposed sparse components on ordinary-function fidelity, response
fidelity, and full-model ablation selectivity, then check whether the same
components predict new contexts or edits. This is a stopping criterion for
another branch-only optimization sweep: without selective transferred
components, further pair-level improvements are compression results, not
mechanistic identification.

Reproduce by running `flagship_shared_key_branch_response.py` with
`--edit-family previous_key_score --dose .05 --length 128` for
`--n-docs 8 --seed 891` and `--n-docs 16 --seed 892`, then run
`analyze_flagship_branch_response.py --files` on the two resulting JSON
files with `--output-name all_head_shared_key_branch_response_summary_n24_d50_previous_key.json`.
The raw records and bootstrap summary live in `full_width_results/`.
