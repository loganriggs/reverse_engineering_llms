## H692 — Reconstruction can conceal failed group discovery

Freeze H691 seed11's repaired shared dictionary, coordinate encoder and membership. Remove supplied group gates and fit a new detector solely through normalized reconstruction error plus0.01times mean active group count. Compare a linear detector (68parameters) with an even quadratic-energy detector (644parameters): learned32dimensional bias-free projection, square each coordinate, then affine4group readout. Both use hard threshold gates and straight-through gradients; counts are variable so heldout triples/all-group patterns are representable. Three initialization seeds,3000steps, best checkpoint selected using an independent2048example singleton/pair validation set. Higher combinations are evaluated only after selection. Gate labels are used for diagnostic metrics, not training or checkpoint selection.

| Seed | Detector | Validation relative MSE | Validation groups | Validation exact group recovery | Held-combination relative MSE | Held exact group recovery |
|---|---|---:|---:|---:|---:|---:|
|21|linear|0.00148|3.975|0.049%|0.00902|18.848%|
|21|energy|0.00859|2.654|11.377%|0.04726|24.414%|
|22|linear|0.00099|3.987|0.098%|0.00328|18.799%|
|22|energy|0.00449|3.230|1.416%|0.01533|31.494%|
|23|linear|0.00354|3.972|0.000%|0.01122|19.238%|
|23|energy|0.00494|3.153|5.127%|0.03034|24.707%|

Linear gates mostly select all four groups. Their low reconstruction error is not evidence of discovering the sparse generating groups: exact validation group recovery is0–0.10%. Energy gates improve the validation reconstruction-plus-group-cost objective in all three seeds and select fewer groups, but exact validation group recovery is only1.42–11.38%. Held-combination recovery is24.4–31.5%, also inadequate. Since held combinations contain more groups, comparison of raw recovery percentages across splits does not imply improved generalization.

The all-open control has held relative MSE3.31e-9; supplied teacher gates yield8.61e-9. Thus the repaired coordinate encoder and dictionary can reconstruct the data without useful gates. Reconstruction alone lets redundant groups remain active when their extra coordinates are nearly zero. A group penalty opposes this, but the current hard-gate optimization and detector family do not reliably recover the actual groups. More active gates can improve reconstruction while making the latent explanation worse.

In general, overlapping groups can yield nonunique group labels for the same atom union. Here each teacher group has a private atom, so the16binary group patterns have distinct supports; group and exact-support recovery coincide in these results. Near-zero Gaussian coefficients can still make dropping a real group cheap under a reconstruction-plus-count objective. We must distinguish objective preference, inference optimization failure, and detector capacity before interpreting poor recovery.

The energy detector satisfies exact x→−x gate invariance on all heldout examples, as intended by construction. This verifies the symmetry property, not successful discovery. Different parameter counts prevent a fair expressivity/budget claim. The fixed representation was originally learned with supplied gates; this is conditional label-free gate fitting, not an end-to-end unsupervised success.

Next diagnostic: exhaustively evaluate the16possible gate patterns per input using the same reconstruction-plus-count objective. This supplies an exact small-problem inference reference without teacher labels. Compare its recovery and objective with the learned detectors to separate a flawed objective from an inference gap. Then test joint discovery and trained hierarchy/shared computation before proceeding to early-layer LLM manifold validation.

Audits: frozen representation bitwise unchanged, all source hashes unchanged, finite results, no original-model forwards/new documents, reserved LLM test split untouched. Broad goal remains active.
