# Joint-pair-aware reduction of a first-layer attention tensor

The earlier [first-layer token-tensor experiment](MULTI_GENRE_TOKEN_TENSOR_CANONICALIZATION.md) compared weight-only SVD with a data-aware SVD based on separate query and key token marginals. That weighting assumes the two token identities are conditionally independent at each lag. This follow-up fits an equally low-rank score matrix to the **actual joint query–key pair frequencies**. It asks whether the more faithful data prior improves held-out language behavior or merely fits the observed genre more tightly.

The two teachers are existing three-layer, width-64 bilinear multi-genre models that perform FAQ place recall. Their native FAQ-recall exact-oracle KL is 0.095/0.107 nats on the present fresh panel; the untruncated first-layer token-table replay adds at most about 1e-5 model KL. This is a competent but rigid toy language with a documented large FineWeb gap. All four genres were present during *model* training; FAQ/Guide are held out only from the tensor **factor fit**.

## Fit and controls

For each attention head and relative lag, let $S(q,k)$ be the weight-derived product of the two QK scores after folding the embedding and RoPE, as in the previous experiment. At rank 8, the new method minimizes

$$\sum_{q,k} w_{\ell}(q,k)\,[S_{\ell}(q,k)-U_{\ell}(q,:)V_{\ell}(k,:)^\top]^2.$$

The weights are 90% empirical joint-pair frequencies from 1,024 independently generated fit streams plus a 10% uniform-vocabulary floor. Each row and column is updated by weighted least squares for eight iterations, starting from the ordinary weight-only SVD. The initial/final weighted objectives are saved for every model. Only 225,520 of 2,124,959 lag/pair cells occur in one fit sample; the floor makes unseen pairs nonzero-weight rather than pretending they are impossible. The controls are untruncated replay, rank-8 weight-only SVD, and rank-8 product-of-marginals SVD. Every truncated method has the same rank per head and lag, not necessarily the same number of numerical nonzeros or a minimal circuit size.

The first independent panel used fit/test/edit seeds 10551/10772/10773 and 512 test streams. The larger replication changed all three seeds to 11551/11772/11773, used 1,024 test streams and 1,024 edit streams, and supplied 227 supported FAQ-recall edit pairs. The table below uses that replication. Lower is better. Model KL compares the truncated model with the native model; oracle KL compares it with the exact generator; edit L1 compares its signed probability response with the exact generator's response.

| Model seed; rank 8 | Joint-weighted score MSE | In-fit Policy model KL | Held-out FAQ model KL | Held-out FAQ oracle KL | FAQ recall edit L1 |
|---|---:|---:|---:|---:|---:|
| 0; full table | 0 | ~0 | ~0 | 0.0953 | 0.2134 |
| 0; weight-only SVD | 9.53e-5 | 0.00207 | 0.01476 | 0.1173 | 0.2464 |
| 0; marginal SVD | 2.60e-5 | 0.00043 | 0.00970 | 0.1032 | 0.2155 |
| 0; joint-pair ALS | **8.06e-6** | **~0.00001** | 0.02640 | 0.1283 | 0.2611 |
| 1; full table | 0 | ~0 | ~0 | 0.1070 | 0.2232 |
| 1; weight-only SVD | 5.75e-5 | 0.00088 | **0.00240** | 0.1118 | 0.2325 |
| 1; marginal SVD | 1.65e-5 | 0.00015 | 0.00535 | 0.1147 | 0.2400 |
| 1; joint-pair ALS | **4.78e-6** | **~0.00001** | 0.00404 | **0.1083** | **0.2307** |

The joint method reduced its weighted score objective by about 12-fold from the weight-SVD initialization in each model. It also nearly eliminates in-fit Policy model KL. That gain does **not** give consistent held-out FAQ transfer: seed 0 is markedly worse than both controls on FAQ KL and causal effects; seed 1 is better than marginal SVD but worse than weight-only SVD on FAQ model KL. The smaller independent panel has the same broad pattern: joint FAQ model KL 0.01776/0.00557 versus marginal 0.00938/0.00478 and weight-only 0.01234/0.00224. Differences in exact panels and fit samples matter; this is two model seeds, not a universal ordering.

An **all-genre fit control** uses the same number of fit streams but exposes the factorization to FAQ/Guide. On the same replication test/edit panels, joint-pair FAQ model KL falls to **0.00012/0.00004**, versus 0.00318/0.00076 for the all-genre marginal fit. FAQ edit L1 returns close to the untruncated teacher: 0.2118/0.2236 versus native 0.2134/0.2232. This is strong evidence that the joint-pair method can compress the behavior on a covered distribution, while its Reports/Policy fit does not reliably protect a held-out genre. The all-genre control changes how the fixed 1,024 streams are allocated among genres, so it does not isolate genre exposure from sample count per genre.

The score-fit objective is task-blind and quadratic in the score matrix. It does not optimize downstream KL, causal selectivity, or an explicit sparse QK×OV path. A low-rank score matrix can preserve FAQ behavior without making its factors human-readable, and the exact first-layer tensor is not a unique end-to-end canonical form. The practical decision is to retain weight-only and product-marginal baselines for **whole-task transfer**, use actual pair weighting when the target distribution is intentionally specified, and require held-out causal-edit fidelity before calling a reduced component interpretable. Further fitting to the same FAQ panels would turn them into development data; a new objective should use separate fit/validation/test genres or task families.

Reproduction: [`eval_multi_genre_joint_pair_factor.py`](eval_multi_genre_joint_pair_factor.py), [first panel](multi_genre_joint_pair_factor_rank8_fresh.json), [larger independent replication](multi_genre_joint_pair_factor_rank8_replication.json), and [all-genre fit control](multi_genre_joint_pair_factor_rank8_allfit.json). The JSONs include exact seeds, fit-support counts, weighted-objective histories, prediction metrics, and edit counts. The lag-table replay has the same small bfloat16 RoPE approximation as the previous first-layer tensor work.
