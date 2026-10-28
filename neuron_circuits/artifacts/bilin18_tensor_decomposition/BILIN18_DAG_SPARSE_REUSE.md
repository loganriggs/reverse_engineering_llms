## H698 — Sparse edges select shared nonlinear computation, with routing failures retained

Fit targets(2xy,3xy,x) using a supplied six-node candidate DAG. Nodes0/1 are learned affine input functions; nodes2/3 are duplicate candidate product routes with parents0/1 and explicit parent products; nodes4/5 can consume either route. Every node can also receive raw linear inputs. Learn continuous weights and global hard-forward straight-through edge gates. Node gates remain supplied on. Compare edge penalties0and0.01, three seeds, unrestricted readout versus a routed readout mapping nodes4/5/0 to outputs0/1/2. Train and validate outside the positive-positive quadrant, then test that held quadrant. Select checkpoints by validation normalized MSE plus edge count penalty.

| Seed | Readout | Edge cost | Active edges | Held relative MSE |
|---|---|---:|---:|---:|
|41|unrestricted|0|8|7.48794e-13|
|41|unrestricted|0.01|2|9.68682e-09|
|41|routed|0|8|0.15608|
|41|routed|0.01|4|0.155158|
|42|unrestricted|0|8|1.72564e-13|
|42|unrestricted|0.01|2|1.11411e-08|
|42|routed|0|8|5.01929e-09|
|42|routed|0.01|4|0.00133434|
|43|unrestricted|0|8|4.83462e-14|
|43|unrestricted|0.01|2|2.44572e-09|
|43|routed|0|8|1.96395e-11|
|43|routed|0.01|4|3.01775e-05|

Without edge cost, all eight candidate edges remain active. Unrestricted readout fits predict accurately, but removing individual product nodes also changes the independent linear output: normalized effect energy ranges0.011–0.647. Accurate predictions alone do not imply selective internal components; linear terms can be distributed across product nodes and cancel elsewhere.

With edge cost and unrestricted readout, all three seeds retain only edges0→2and1→2. The shared nonlinear result is read directly into both nonlinear outputs. Held relative MSE is2.45e-9–1.11e-8. Removing node2 affects both nonlinear outputs substantially (effect energy1.22–1.62times their respective target energy), while the independent output changes by at most6.54e-10times its target energy. This is evidence for selective reuse of a nonlinear intermediate, conditional on the supplied candidate operations. It is not exact recovery of a purexycoordinate: effect magnitudes need not equal the teacher contribution, since the node may also carry affine terms.

The alternate product node3 has no active parent edges under the sparse unrestricted fits, but can still carry raw affine input into readout. Removing it changes the independent output by0.00245–0.204of target energy. Thus inactive parent edges do not imply an inactive node, and node execution count/runtime savings were not demonstrated. The architecture still evaluates all nodes once; the result concerns routing and selective effects.

With routed readout and edge cost, all seeds retain the explicit shared path0/1→2→4/5. Node3 removal has zero output effect; node2 removal affects outputs0/1and leaves output2unchanged by the imposed routing. That selectivity is partly guaranteed by the control architecture. Prediction is not uniformly successful: seed41 held relative error0.155is dominated by the independent output (MSE3.84), seed42 error0.00133, seed43 error3.02e-5. The corresponding no-penalty seed41 also fails, so sparsity alone is not the demonstrated cause. Input-basis orientation/optimization and unequal output scales are candidates to test; no cause is yet established. Correct-looking edges do not certify a working computation.

This is learned edge selection inside a fixed candidate DAG, not topology discovery or a matched-parameter contest with BSFs. Readout restrictions change effective capacity. Duplicate node identities are interchangeable; retaining node2rather than node3does not identify a uniquely named primitive. Edge/whole-node interventions are both recorded, with heldout examples excluded from checkpoint selection.

Audits: all source hashes unchanged, finite results, no original-model forwards/new documents, reserved LLM test split untouched. Earlier DAG unit tests establish once-per-node evaluation and edge/node ablation mechanics; this experiment adds learned redundant-path and routing controls. Broad goal active.

Next inspect the routed failure with per-output loss scaling and fixed-edge continuous refits, retaining all seeds. Then consolidate which invariants survived the overlap/hierarchy/DAG tests and choose practical featurizers for early-layer activation experiments. Joint unsupervised group discovery, meaningful-manifold validation, matched resource budgets and diverse model-task causal evaluations remain outstanding.
