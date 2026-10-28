## H812 — Compact response subspaces preserve part of held-out causal behavior, but remain imperfect

Return to representation validation using verified copying and IOI response families. Fit uncentered SVD bases on raw L6 pre-MLP input responses, without behavioral labels in the fit. Normalize each training response to unit length; task-normalize each response matrix before the shared fit so one task's scale/count does not dominate. Compare shared and task-specific ranks 1,2,4,8,16. These are linear response subspaces, not trained BSFs, sparse overlapping gates, taxonomies, or DAGs.

The split was fixed before fitting: every third sorted lexical item is held out, grouping all contexts, source states, and doses for that item. Copy held-out item IDs are 1,5,8 (six training items); IOI held-out IDs are 0,3,7 (five training items). Only the primary surface is included. Copy doses .05/.10 and IOI doses .25/.50 are evaluated. These panels were examined previously, so this is internal held-out fitting, not fresh untouched validation.

For each held-out response, orthogonally project the raw input change, apply RMS and the exact MLP transformation against the recipient baseline, and replace only that module response during a model forward. Test own-context and swapped-context input responses. Include full-response and zero-module-response controls. The original source perturbation and other computational paths remain active.

### Reconstruction and behavioral fidelity

| Representation | Copy input relative error | Copy logprob RMS error | IOI input relative error | IOI logprob RMS error |
|---|---:|---:|---:|---:|
| Shared rank 4 | .486 | .02120 | .524 | .00317 |
| Shared rank 8 | .392 | .01062 | .459 | .00362 |
| Shared rank 16 | .371 | .01077 | .424 | .00385 |
| Task-specific rank 4 | .393 | .01057 | .469 | .00335 |
| Task-specific rank 8 | .373 | .01059 | .433 | .00339 |
| Task-specific rank 16 | .346 | .00905 | .401 | .00330 |

Input errors are mean relative vector norms; output errors are RMS differences from full-response log-probability changes, pooled over doses and arms. Shared rank r and task-specific rank r use the same active dimension, but the two specific dictionaries store twice as many columns. Shared rank 8 versus specific rank 4 matches total dictionary columns while allowing twice as many active coordinates for the shared model. Under that storage comparison, results are close; this is not evidence that atoms are semantically shared. A shared basis could simply contain separate task directions.

Increasing rank improves geometric reconstruction but does not monotonically improve behavioral fidelity. In particular, shared rank 16 reconstructs vectors better than rank 8 yet has slightly worse output error on both tasks. This directly supports measuring causal replay rather than selecting features solely by reconstruction.

### Causal-response pattern

For held-out copying items at .05, the full own-context gap is +.00403 and input-swapped gap -.02998; shared rank 8 gives +.00367 and -.02603. At .10, full gaps are +.01271/-.06875, while rank 8 gives +.00980/-.05801. The sign reversal survives, but its magnitude is underestimated: full swap contrasts -.03400/-.08146 become -.02970/-.06781.

IOI full swap contrasts are +.00595/+.01966 at .25/.50; shared rank 8 gives +.00380/+.01199. It retains the qualitative difference from copying but misses a material part of the smaller module-mediated contrast. Only three held-out lexical items per task are available, so these descriptive results are not robust population estimates.

### Avoiding an easy but misleading success criterion

Shared rank 8 has output error equal to 18.5% of total full-response RMS on copying and 6.1% on IOI. The IOI number looks especially good because the tested MLP is only one path: zeroing its response already yields just 17.1% total-response error on IOI (versus 83.0% on copying). Relative to the full-versus-zero module-response contribution, rank-8 errors are approximately 22.3% and 35.7%, respectively. Thus the representation is substantially imperfect even where total output error is small.

Zero-module-response is a causal control, not an unperturbed model: the source intervention and residual/other paths remain. Top-1 accuracy is 95.8% for copying and 100% for IOI across all dictionaries and controls, making it insensitive to these fidelity differences.

### Checks and next discriminator

1824 forwards completed. Full-response historical replay is exact in the reported log-probability metric. Train and held-out item sets are disjoint; paired projected direction mismatch is zero; basis orthogonality error is 2.34e-15. All tracked hashes unchanged and all outputs finite. The reserved test remains untouched, and the natural-document pool is 26,112.

This is evidence for partial functional compressibility of a response family, not validation of a manifold, a canonical feature basis, or shared reusable atoms. The next decisive controls are matched-rank random subspaces and cross-task-only bases: fit on copying and evaluate IOI, and vice versa, with the same causal replay metrics. These distinguish useful learned structure from generic projection and a union of task-specific spans from genuinely reusable directions. Do not promote a compact representation based only on the total-effect denominator or the three-item held-out sample.
