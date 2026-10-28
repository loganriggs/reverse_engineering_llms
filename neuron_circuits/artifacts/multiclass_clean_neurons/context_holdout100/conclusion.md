# Round 5: locked context-holdout confirmation

All 10 preregistered seeds converged and retained 100% accuracy on fresh noisy inputs. Candidate directions were fitted and evaluated on disjoint semantic contexts:

- operation directions: fit on classes 0–49, confirm on unseen classes 75–99;
- class directions: fit on operations 0–3, confirm on unseen operations 6–7.

## Preregistered endpoints

| Counterfactual | Layer | Raw minus rotation | 95% CI | PCA minus raw | 95% CI |
|---|---:|---:|---:|---:|---:|
| Operation | 1 | 0.042 | [0.030, 0.054] | 0.469 | [0.454, 0.484] |
| Operation | 2 | 0.067 | [0.049, 0.085] | 0.283 | [0.248, 0.318] |
| Class | 1 | 0.015 | [0.006, 0.023] | 0.113 | [0.106, 0.119] |
| Class | 2 | 0.033 | [0.025, 0.040] | 0.155 | [0.148, 0.163] |

No endpoint met the strong clean-neuron criterion. All four met the preregistered distributed-subspace criterion. Both class endpoints met the preregistered falsification criterion for a practically substantial privileged-axis effect: their upper confidence bounds are below 0.05.

Operation structure generalizes better across unseen classes than class structure generalizes across unseen operations. The reusable operation PCA subspace reached 80% KL recovery with a median 24 dimensions in layer 1 and 32 in layer 2; raw neurons and rotated coordinates needed all 256. For class counterfactuals, even the PCA subspace needed the full 256 dimensions to pass 80% on these disjoint operations.

## Final interpretation

The most defensible description is:

> The trained bilinear layers develop mildly axis-aligned but strongly distributed causal representations. Computation-like operation information is more context-reusable and compressible than class identity, but it is not localized to a few clean neurons.

This is compatible with a weak “neurons are more verb-like than noun-like” tendency, but it does not isolate grammatical verbness. There are only eight operations and 100 classes, so lower intrinsic operation rank is expected. A balanced-cardinality follow-up would be required to attribute the difference to semantic type rather than entropy and task construction.

The experiment tests causal axis privilege and cross-context reuse, not the full philosophical claim of monosemanticity. A neuron could still have a human-readable feature while being redundant, and a causally privileged neuron could still be polysemantic.

Files:

- [Aggregate curves](aggregate_curves.png)
- [Endpoint comparison](endpoint_comparison.png)
- [Complete sealed statistics](locked_summary.json)

