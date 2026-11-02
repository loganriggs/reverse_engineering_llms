# A compact source-readout subspace is conditional on the fit distribution

The [source curriculum](V6_SOURCE_CURRICULUM_TRANSFER.md) yields a two-layer
bilinear language model whose 2,000-update checkpoint is nearly exact on
source-attribution edits, including unseen speaker–claim-object combinations.
This experiment compares local bases at the **last position of the second
attention block's pre-output bond**. It asks how many of that bond's 64
directions must be retained to preserve the native model's source-answer
distribution and valid source edits. It does not canonicalize the whole
attention function or recover an end-to-end circuit.

The data-aware bases fit only on the restricted v6 generator, which excludes
`(speaker_index + object_index) % 4 == 0` from source statements. We fit on
1,309 valid source-edit pairs drawn from 1,000 restricted streams. The
task-conditioned activation PCA uses original and edited **source contexts**;
the edit-response SVD uses their activation differences. A separate
task-blind stream PCA uses 4,096 randomly selected positions from 512
restricted streams, without selecting source contexts or using labels. The
weight SVD uses the output projection matrix only. The speaker-output basis
uses the rows of the LM head for the eight *known* speaker tokens composed
with that output projection; it is weight-only but explicitly task-informed.
All bases are orthonormalized; the 64-dimensional projection exactly replays
the native model within 1e-5 TV in every edit family.

Two independent unrestricted evaluation streams provide 619 and 633 valid
edits whose **queried statement pair** was withheld from basis fitting and
model training. The table reports mean L1 error between the projected
model's query-object probability *change* and the exact generator's change;
lower is better. This is an effect metric, distinct from activation MSE and
ordinary prediction accuracy.

| Basis, retained dimensions | Test seed 24102 | Test seed 24104 |
|---|---:|---:|
| Native model, all 64 | .056 | .029 |
| Task-blind stream PCA, 8 | 2.943 | 2.941 |
| Task-blind stream PCA, 16 | 2.524 | 2.414 |
| Source-context activation PCA, 8 | **.046** | **.022** |
| Source-edit response SVD, 8 | .049 | .024 |
| Known-speaker output prior, 8 | .073 | .060 |
| Weight-only bond SVD, 16 | .092 | .055 |
| Random directions, 16 | 1.360 | 1.247 |

The same ordering holds when comparing the projected effect with the
**native model's** effect: source-context PCA at dimension 8 has L1 error
.017/.012, speaker-output has .022/.035, weight-only SVD at dimension 16 has
.037/.027, and task-blind stream PCA at dimension 8 has 2.896/2.915.
For the no-effect edit that changes an unrelated speaker, native oracle-effect
L1 is .023/.023. Source-context PCA at dimension 8 is .023/.023;
speaker-output is .028/.032, weight SVD at dimension 16 is .036/.045, and
task-blind stream PCA at dimension 8 is .068/.077. The task-blind basis's
small no-effect error does not rescue its failed query effect.

The key finding is a **distributional prior**, not a unique semantic axis.
The source computation fits in a small local subspace when fit examples
focus on source questions, but generic high-variance directions from the
same language streams miss it. The source-conditioned PCA's oracle-effect
error is slightly below the native model's; truncation can remove some
native error, so this is not evidence that it implements the exact source
algorithm. Nor do these results identify which earlier-layer paths build the
speaker–claim binding. A decisive interpretability test would require small
units or paths sufficient and necessary for the source effect, little impact
on other competent rules, and transfer across source surfaces and model
seeds. The current v6 verb/object losses remain too high for that strong
joint-rule claim.

## Transfer from quoted fit contexts to unquoted held-out pairs

After [mixed-form adaptation](V6_SOURCE_SURFACE_TRANSFER.md), the model is
competent on both quoted and unquoted source statements. We froze its
development-selected 2,000-update checkpoint and fitted **all** data-aware
bases on quoted, allowed speaker–claim pairings only. The independent test
panels select questions whose queried statement is **unquoted** and whose
speaker–claim pairing was withheld from model training and basis fitting.
The model saw the unquoted *rule family* during its own adaptation, so this
is a **featurizer fit-form holdout**, not model-training form holdout or new
task discovery. The two panels contain 483 and 468 valid edits, including
133 and 126 query-object edits.

| Basis, retained dimensions | Query-effect oracle L1, seed 25317 | Seed 25318 | Query effect error from native, seed 25317/25318 |
|---|---:|---:|---:|
| Native model, all 64 | .192 | .173 | 0 / 0 |
| Task-blind quoted-stream PCA, 8 | 2.891 | 2.751 | 2.788 / 2.623 |
| Quoted source-context PCA, 8 | **.177** | **.153** | .044 / .032 |
| Quoted source-edit response SVD, 8 | .175 | .151 | .046 / .034 |
| Known-speaker output prior, 8 | .208 | .181 | .033 / .021 |
| Weight-only bond SVD, 16 | .613 | .504 | .450 / .352 |
| Weight-only bond SVD, 32 | .202 | .185 | .017 / .018 |
| Random directions, 16 | 1.609 | 1.518 | 1.467 / 1.381 |

The quote-fit source-context PCA transfers well at eight dimensions,
including the no-effect quote-toggle edit: its model TV is .058/.026 versus
native .062/.037. The known-speaker output prior also preserves source
effects compactly, but is explicitly supervised by the answer-token set.
Generic weight SVD needs about 32 dimensions for comparable transfer here;
at 16 it had been adequate on the quoted-only model's quoted test panel but
fails on this adapted model's unquoted test panel. These are different model
checkpoints and test conditions, so the latter contrast is not a controlled
causal estimate of the form alone. The same-checkpoint comparison above is
the valid basis ranking. Task-conditioned PCA uses source-selected inputs,
and edit-response SVD uses causal pairs; neither is task-blind discovery.
The result supports context selection as a useful prior for a compact local
readout, not a semantic axis or a recovered attention circuit.

Reproduce with `eval_v6_source_bond_bases.py`; full records are
`results/v6_source_bond_bases_step2000_seed24102.json` and
`results/v6_source_bond_bases_step2000_seed24104.json`, plus
`results/v6_source_mixed_adapt_basis_quote_fit_unquoted_test_seed25317.json`
and `results/v6_source_mixed_adapt_basis_quote_fit_unquoted_test_seed25318.json`.
