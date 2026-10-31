# Can a faithful attention cut be made sparse by changing only its component basis?

The [full-model test](FULL_MODEL_BEHAVIOR_RESULTS.md) found a rank-64 first-layer QK1 cut that preserves the 546M model's next-token distribution well. A separate question is whether the *same function* has a component basis that is sparse, stable, and eventually interpretable. This experiment deliberately leaves the successful bond matrix fixed, so no fidelity allowance or additional full-model KL comparison is needed.

Let the natural sequence-fitted bond be \(X=AB^{\mathsf T}\), where \(A,B\in\mathbb R^{128\times64}\) come from its SVD. For any orthogonal \(R\in\mathbb R^{64\times64}\),

\[
X=(AR)(BR)^{\mathsf T},\qquad
z_i(u,v;R)=(u^{\mathsf T}AR)_i(v^{\mathsf T}BR)_i,
\qquad \sum_i z_i=u^{\mathsf T}Xv.
\]

Thus \(R\) can change the apparent rank-one components without changing *any* pair score, sequence head output, downstream logit, or KL in exact arithmetic. This is a restricted **orthogonal component gauge** of one fixed cut, not a canonical form for the whole attention function and not the complete family of invertible factor gauges. The largest recorded relative bond reconstruction error is `1.28e-14`; the largest orthogonality residual is `9.49e-14`. The activation-objective gradient agrees with a central difference to relative `3.8e-10`.

We compare five bases: the SVD basis; three random orthogonal rotations; rotations minimizing L1 of the *factor matrices* \(AR,BR\); rotations minimizing a pair-activation sparsity proxy on uniform and FineWeb pairs; and two variants of that proxy, one including QK2-constant pair weights and one adding rare-token pairs. The optimization uses 32,768 fit pairs per law, independent validation pairs for checkpoint selection, three minibatch seeds, and 400 CPU Adam steps. FineWeb fit, selection, and test document ranges are 0–14,999, 15,000–17,499, and 17,500–19,999. The rare pool contains tokens seen 1–10 times in fit documents; the **zero-occurrence pool is unseen in the FineWeb fit documents but can appear in synthetic uniform-token training pairs**. It is an unseen-*corpus* stratum, not a whole-token holdout.

The reported proxy is the original pair-response-energy-weighted effective number of active components,

\[
N_{\rm eff}(u,v)=\frac{(\sum_i|z_i|)^2}{\sum_i z_i^2},\qquad 1\le N_{\rm eff}\le64.
\]

Lower means a pair response is carried by fewer components. Its weight is the *unchanged original* squared pair response, including QK2 and OV. This is a concentration measure, **not** a count of semantic concepts. The table averages the three fitted seeds; SVD has one deterministic value.

| Exact component gauge | FineWeb test | WikiText-103 | TinyStories | Common IDs | Rare IDs | Unseen IDs |
|---|---:|---:|---:|---:|---:|---:|
| SVD | 23.13 | 23.01 | 22.91 | 24.05 | 25.66 | 25.70 |
| Random rotation | 24.50 | 24.32 | 23.61 | 25.67 | 25.45 | 25.66 |
| Factor-weight L1 | 23.95 | 22.44 | 19.97 | 23.13 | 24.85 | 25.04 |
| Natural-pair activation sparsity | 19.34 | 21.11 | 18.05 | 18.35 | 24.52 | 24.88 |
| Natural + QK2-constant activation sparsity | **18.56** | **20.27** | **17.01** | **16.98** | 24.19 | 24.28 |
| Natural + rare-pair activation sparsity | 19.17 | 21.06 | 18.44 | 18.34 | **23.96** | 24.29 |

The pair-activation objectives give a real, held-out concentration gain, including on the two external corpora. But it is **uneven**: common-token effective support falls from 24.05 to 16.98 under the intervention-weighted prior, while rare and unseen support only falls from about 25.7 to about 24.2. Explicit rare-pair training improves rare support to 23.96 but does not close that gap. This warns against treating average corpus sparsity as a universal property of the learned computation.

Weight and activation priors choose different gauges. Factor-weight L1 falls from **.149 in the SVD basis to .093**, yet its FineWeb effective support is **23.95**, worse than the SVD's 23.13. Activation-sparse gauges retain nearly the original factor L1 (.146–.148) while substantially lowering pair support. The factor matrices here are an SVD factorization of the *reduced bond*, not the original pre-RoPE Q/K parameter matrices; ordinary native-weight L1 is still a different, gauge-dependent proposal.

## Does sparsity identify a repeatable component basis?

For each fitted basis, we compute all 64 signed component-response vectors on the same 8,192 held-out FineWeb pairs, center and normalize them, and optimally match components between fitting seeds by absolute correlation. Mean matched correlation is **.149 for random rotations**, **.236 for natural activation sparsity**, **.263 for intervention-weighted activation sparsity**, and **.253 for rare-balanced activation sparsity**. These are above random but far from a seed-stable decomposition. Factor-weight L1 gives 1.0 only because it uses no sampled data and starts identically in every seed; that is deterministic optimization, not evidence of recovered semantics.

This experiment resolves one methodological issue: the choice of *where* to put sparsity matters even at exactly fixed model behavior. It does **not** establish human legibility. A [post-fit token-form label test](COMPONENT_LABEL_TRANSFER_RESULTS.md) now finds that the activation-sparse bases do not outperform SVD or random rotations at predicting coarse held-out token categories, despite their lower active-component count. The next gate is selective downstream causal effects on withheld contexts or known-rule language tasks. The weak cross-seed alignment means a single attractive sparse basis should not be taken as a canonical circuit. Further work can also compare nonorthogonal gauges and sparsity of the effective token-interaction function itself, which are distinct search spaces.

The script `flagship_exact_gauge_sparsity.py` reproduces the experiment. All per-seed fit traces, held-out metrics, gradient and reconstruction checks, and matching scores are in `full_width_results/head0_exact_gauge_sparsity.json`; exact rotation matrices are in `head0_exact_gauge_rotations.npz`. The external checkpoint and corpora remain referenced rather than copied into this repository.
