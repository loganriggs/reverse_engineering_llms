## H693 — Exact gate search separates inference failure from sparsity bias

Evaluate all16binary gate patterns per example, retaining H692's frozen H691 seed11 representation and exactly replayed data. Each pattern is scored by reconstruction MSE divided by the training mean-square activation scale, plus a group count penalty. Select the minimum per input without teacher labels. Compare learned gates at their original0.01penalty, and separately sweep penalties as an exploratory diagnostic. The exact optimum is conditional on the frozen dictionary and coordinate encoder; it is not a global solution to dictionary discovery.

At the original penalty, exact gate inference recovers97.56% of validation group patterns. Learned linear gates recovered at most0.10%; learned energy gates at most11.38%. Exact validation objective is0.01600 versus0.04089–0.04339 for linear and0.03545–0.03695 for energy. Thus a large inference gap exists even under the same objective and representation. This does not yet distinguish detector expressivity from straight-through optimization; both can prevent an amortized gate network from implementing the exact decisions.

| Group penalty | Validation exact recovery | Held-combination exact recovery | Held relative MSE | Held mean groups |
|---|---:|---:|---:|---:|
|0|0.39%|22.95%|3.31178e-09|3.961|
|1e-05|100.00%|99.61%|1.73691e-08|3.187|
|0.001|99.76%|92.48%|1.4634e-05|3.115|
|0.01|97.56%|74.32%|0.000536933|2.925|
|0.1|79.93%|26.86%|0.0172564|2.311|

The objective itself also changes the recovered explanation. With no group cost, redundant active groups are nearly free; tiny reconstruction differences favor mostly-open patterns. At0.01, exact inference drops weak active features: held recovery is74.32%, with0.279false-negative versus0.0127false-positive groups per example. A much smaller1e-5cost gives100%validation and99.61%held recovery in this finite, noiseless synthetic sample. This sweep is not a confirmatory heldout tuning exercise and does not prescribe that cost for LLMs. Noise, small coefficient magnitudes, dictionary error and activation scaling change the appropriate tradeoff. Exact support is a different target from the best reconstruction-versus-sparsity compromise.

For larger combinations, more groups have their shared directions already covered by other selected groups; dropping one can lose only a weak unique component. This is consistent with the stronger false-negative pattern at the same sparsity cost. A per-component marginal-error audit would be needed to quantify this proposed explanation beyond the observed counts.

Next compare distillation of training-only exact search decisions into the same detectors with direct straight-through training. This uses search-derived targets, not teacher gate labels, and can distinguish an optimization gap from detector capacity. Then test joint group discovery, hierarchy and reusable computation; the original early-layer manifold/causal program remains the destination.

Audits: all16patterns enumerated; exact objectives never exceed teacher-pattern or learned-detector objectives; direct teacher-gated errors exactly match indexed table errors; replayed H692 reconstruction metrics agree within3.73e-9. All source hashes unchanged, finite results, no original-model forwards/new documents, reserved LLM test split untouched. Broad goal active.
