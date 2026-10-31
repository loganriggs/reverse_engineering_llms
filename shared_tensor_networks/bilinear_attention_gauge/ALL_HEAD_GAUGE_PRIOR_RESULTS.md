# Exact-gauge legibility priors on a shared nine-head key subspace

The [function-aware shared-key reduction](FUNCTION_AWARE_SHARED_KEY_RESULTS.md)
selects a rank-256 subspace of the 1,152 native pre-RoPE QK1 key coordinates
across nine heads. This experiment holds that **entire projection fixed**
and asks whether a particular orientation of its 256 components is easier to
read. It also repeats the comparison on the usage-weighted SVD subspace.

For an orthonormal basis \(U\) and orthogonal rotation \(R\),
\((UR)(UR)^\mathsf{T}=UU^\mathsf{T}\). Every pair response, QK2 branch edit,
and full-model logit is therefore the same function up to numerical error.
This is a comparison of **legibility priors**, with zero reconstruction
budget spent, not a comparison of reductions.

## Priors and tests

`flagship_shared_key_gauge_priors.py` compares the supplied basis, two random
rotations, a weight-entry varimax rotation, a token-response varimax rotation,
and direct normalized token-response L1 optimization. Varimax is an
orthogonal fourth-moment objective; it is only a proxy for sparsity. Direct
L1 minimizes the average sum of absolute coordinates of normalized key-token
responses \(KU R\), using 2,048 token samples from a half-uniform,
half-FineWeb-frequency training law. FineWeb documents 0–15,000 supply the
frequency measure, documents 15,000–17,500 supply validation token samples,
and documents 17,500–20,000 plus WikiText-103 and TinyStories supply test
samples. Token identities may recur across document splits, so this is
distribution transfer, not unseen-vocabulary transfer.

The measured sparsity proxy is the mean per-example effective coordinate
count \((\|z\|_1/\|z\|_2)^2\), where smaller means concentrated. For pair
outputs, each coordinate's \(z_c\) is the norm of that component's complete
nine-head OV output contribution, including cross-head interference. The
pair panels use fresh held-out token/position draws and are evaluated with
natural QK2 and a QK2-constant edit. A separate effective-head count measures
how much each basis vector spans the nine physical head blocks. None of these
scores is a semantic label or a selective full-model ablation.

## Fixed function-aware subspace

Representative seed-0 results after 600 direct-L1 steps (lower effective
coordinate counts are more concentrated):

| Gauge | Effective heads / component | FineWeb token | WikiText token | TinyStories token | FineWeb pair, natural | FineWeb pair, QK2=1 |
|---|---:|---:|---:|---:|---:|---:|
| Supplied basis | 7.6 | 81.7 | 88.1 | 74.5 | 60.0 | 67.5 |
| Random | 7.8 | 163.3 | 163.5 | 163.1 | 141.5 | 147.2 |
| Weight varimax | **5.4** | 140.0 | 141.5 | 139.1 | 104.6 | 105.5 |
| Response varimax | 7.7 | 83.4 | 92.4 | 76.5 | 64.6 | 70.1 |
| Direct response L1, native start | 7.6 | **72.5** | **81.0** | **65.9** | **50.6** | **58.2** |
| Direct response L1, random start | 7.7 | 79.0 | 87.0 | 72.1 | 55.2 | 63.3 |

The usage-SVD subspace gives nearly the same ordering and magnitudes. Direct
L1 also improves natural and edited pair-component concentration on uniform,
WikiText, and TinyStories panels; the full records include every law.

Weight sparsity makes axes more head-local, but spreads their token and pair
responses. Response varimax improves on a random orientation but does not
beat the supplied basis on these held-out concentration measures. Directly
optimizing the intended L1 objective does. Thus the choice of objective
matters even when the underlying function is fixed.

The two native-start direct-L1 runs have matched absolute component cosine
0.971 on the function-aware subspace and 0.972 on the usage-SVD subspace.
Two random-start direct-L1 runs have matches near 0.20, though they improve
their concentration scores. Both direct-L1 fits select their final step,
so these runs do **not** establish global convergence. Native-start
repeatability is a practical gauge convention, not a unique canonical
decomposition independent of initialization. The 120-iteration varimax fits
also reach their iteration cap. Extending function-aware varimax to 400
iterations leaves FineWeb token/pair concentration worse than the supplied
basis (83.2/64.8 versus 81.7/60.0); this checks the qualitative comparison,
not global optimality.

The orthogonal-projector checks pass at relative error below 1e-7 for every
gauge. The numerical pair-function probe on 64 pairs per law checks that the
actual implementation preserves output error under rotation; its maximum
difference is recorded per gauge in the JSON.

## Interpretation gate

This is **better concentration**, not established human legibility. A sparse
axis could mix unrelated token families, and a head-local axis need not be
a reusable computation. Earlier [two-bit](JOINT_FUNCTION_TOY_RESULTS.md) and
[three-bit](THREE_COUNT_CANONICAL_RESULTS.md) trained toys show that a
label-free sparse key orientation *can* recover known rules and yield
selective component deletions. Earlier [head-0 real-model tests](EXACT_GAUGE_SPARSITY_RESULTS.md)
showed that sparse activation proxies alone did not improve coarse labels or
full-model localization over random controls. The next decisive test for
this all-head dictionary is post-fit task and causal selectivity: derive
candidate behaviors from component responses, hold those labels out of
fitting, and compare matched component deletions on new contexts against
random and head-local controls.

A first [held-out token-form label test](ALL_HEAD_LABEL_TRANSFER_RESULTS.md)
shows that response varimax transfers coarse categories more strongly than
direct L1, despite worse contribution concentration. It does not yet supply
named-task or selective full-model causal validation.

Artifacts: `full_width_results/all_head_shared_key_gauge_priors_function_n2048_p2048_i120_d600.*`
and `all_head_shared_key_gauge_priors_usage_n2048_p2048_i120_d600.*`.
The longer varimax control is `all_head_shared_key_gauge_priors_function_n2048_p2048_i400_d0.*`.
