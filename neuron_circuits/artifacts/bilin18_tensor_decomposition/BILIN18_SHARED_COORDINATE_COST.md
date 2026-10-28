## H701 — Active-coordinate penalties disrupt collapse without recovering the structure

Repeat H700's joint label-free fits from the same random initializations and minibatch streams, adding active-support penalties0.005,0.02or0.05to the existing0.01group penalty. All16gate patterns are scored by reconstruction plus group and union-support counts. The expected cost trains the energy detector; membership remains hardTop3with straight-through gradients. Coordinate count charges selected support, matching nonzero-code count almost surely for continuous coordinates except exact zeros. No teacher labels enter optimization/validation selection. Three seeds×three costs,3000steps each. This is an exploratory sweep on the repeatedly used synthetic benchmark, not a new confirmatory test.

| Seed | Coordinate cost | Held relative MSE | Held active atoms | Distinct held gate patterns | Largest pattern fraction | Held group recovery |
|---|---:|---:|---:|---:|---:|---:|
|51|0.005|0.2267|5.980|7|71.68%|19.14%|
|51|0.02|0.3401|4.544|10|58.25%|16.21%|
|51|0.05|0.3578|3.732|16|16.41%|6.45%|
|52|0.005|0.0844|8.000|1|100.00%|19.09%|
|52|0.02|0.1091|6.692|9|85.69%|18.31%|
|52|0.05|0.2459|5.276|6|64.65%|14.99%|
|53|0.005|0.1187|6.822|6|83.74%|20.31%|
|53|0.02|0.2373|5.432|10|61.67%|16.89%|
|53|0.05|0.5814|2.940|2|98.00%|0.00%|

The sweep does not recover useful group structure. Eight of nine fits exhibit multiple heldout gate patterns, so strict constant gating is often disrupted. Yet held reconstruction errors remain8.44–58.14%, and posthoc-aligned exact group agreement is0–20.31%. Nonconstant gates are therefore not evidence of meaningful manifold or regime labels. At the largest cost, seed53 selects roughly one group and2.94atoms, with58.14%held error. The penalty can reduce representational capacity without learning the right conditional supports.

Results are not monotonically better or worse than the unpenalized H700 fits: seed52 loses its nearly exact dense reconstruction; some other arms modestly improve on a poor baseline but do not recover groups. Costs differ across arms, so their penalized objectives cannot be ranked as a shared metric. The shared reconstruction, active support, occupancy and teacher recovery diagnostics are reported separately. Posthoc alignment is a diagnostic, not a training signal or checkpoint-selection criterion.

Exact gate inference on each learned dictionary does not generally remove the reconstruction deficit. One large detector gap remains for seed53at0.05: exact inference lowers held relative MSE from0.581to0.337, still poor. This is consistent with co-adapted dictionary/membership failures as well as detector failures; it does not isolate their relative causes. A coordinate penalty alone is not a verified repair for dense-span collapse.

Next test data-only sparse-direction initialization rather than increasing penalties further: learn candidate directions from training activations without membership/gate labels, validate their recovery and heldout behavior, then test whether group learning benefits. Keep reconstruction-only/dense-span and constant-support controls. None of these results yet establishes an alternative better than BSFs on LLM activations. The broader early-layer manifold, stability, intervention and diverse-task program remains required.

Audits: vectorized/reference reconstruction agree within3.73e-9on the checked pattern; all source hashes unchanged, finite results. Exact expected-cost gradients remain those verified in H695; membership gradients remain a declared surrogate. No original-model forwards/new documents; reserved LLM test split untouched. Broad goal active.
