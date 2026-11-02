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

Reproduce with `eval_v6_source_bond_bases.py`; full records are
`results/v6_source_bond_bases_step2000_seed24102.json` and
`results/v6_source_bond_bases_step2000_seed24104.json`.
