# M3: token-conditioned geometry plus contextual variation

2026-10-28. Exact descriptive partition of cached M2 native MLP0/1/2 product writes, excluding Down bias. No new model forwards, fits, pruning, or whole-task holdout exposure.

## Question and accounting

Does the low-dimensional write geometry mainly distinguish current tokens, or does it describe changes across contexts for the same token? For write Y, global sample mean mu and current-token sample mean mu_t, decompose Y = mu + (mu_t - mu) + (Y - mu_t). Between-token and within-token terms are orthogonal over the sampled rows, so their squared energies sum exactly to centered write energy.

Means and scores use the SAME already-opened training panel. This is an in-sample variance partition, not validated token-table prediction, causal context removal, or a discovery of semantic features. Within-token variation includes position, task and context covariation.

The panel has 1,278 occurrences, 705 distinct complete causal prefixes and 123 current token types. Of these, 53 types have only one distinct prefix; their contextual residual is zero by construction. The repeated-context subset contains 70 types, 951 occurrences and 652 distinct prefixes. Repeated identical occurrences do not count as additional contexts.

## Within-token fraction of centered write energy

| Layer | All occurrences | All unique prefixes | Repeated-context occurrences | Repeated-context unique prefixes |
|---|---:|---:|---:|---:|
| MLP0 | 7.73% | 9.26% | 12.06% | 10.13% |
| MLP1 | 6.89% | 10.84% | 18.15% | 13.83% |
| MLP2 | 14.13% | 15.88% | 24.54% | 18.98% |

Token-conditioned means dominate on this panel, but the contextual residual is not zero and its relative share increases with layer. Deduplication and exclusion of singleton-context tokens materially affect the percentages. These are write-energy fractions, not CE stake recovery, output KL, or removable fractions of computation.

## The smaller contextual component spans more directions

For the all-token unique-prefix view, r90/r99 count singular directions carrying 90%/99% of each component's own squared energy:

| Layer | Between-token r90 / r99 | Within-token r90 / r99 |
|---|---:|---:|
| MLP0 | 33 / 92 | 59 / 220 |
| MLP1 | 42 / 93 | 62 / 236 |
| MLP2 | 37 / 87 | 73 / 245 |

This supports a candidate description of token-conditioned geometry plus a smaller, more dimensionally spread contextual component. It does not prove two separable learned mechanisms: finite token-group counts constrain the between-token rank, and the two components are constructed statistically. A high-energy token component may itself re-cluster tokens rather than copy an earlier representation. Low-energy residuals can still matter strongly downstream.

## Consequence for featurizers and next test

A reconstruction-only featurizer could preferentially fit token differences and underweight contextual changes. Preserve separate scores for both components when comparing featurizers, alongside downstream causal fidelity. Do not impose this decomposition as the desired ontology until it survives broader data and interventions.

Next audit existing natural-text sources and caches, then freeze a finite replication of this same decomposition. The canonical FineWeb corpus is already referenced by numerous prior experiment builders; label any use as reused or provenance-uncertain, not untouched. Prefer compatible cached native writes, but verify model revision, site, token/prefix alignment and prior use before reuse. Keep whole-task exclusions reserved. Do not reopen the completed toy hyperparameter grid.

## Reproduction and independent audit

Run `analyze_token_context.py` and `audit_token_context.py` in the research Python environment with NumPy, Torch and tiktoken. They require M2 metadata in `validation/empirical_spectra.json` and the cached matrices at `/tmp/emlp_m2_matrices.pt` (also archived in the primary research artifacts directory). The large matrices are not included in this shareable packet.

The analyzer uses NumPy; the independent auditor uses Torch float64 grouping, alternative sum-of-squares identities and SVD. All 12 energy partitions and 12 spectra pass; maximum relative errors are 6.42e-16 and 9.34e-16 respectively. Result, source and M2 metadata hashes were checked before publication.

[Results](validation/token_context_partition.json), [audit](validation/token_context_partition_audit.json), [M2 source experiment](EMPIRICAL_OUTPUT_SPECTRA.md).
