## H735 — Larger active budgets improve reconstruction but reduce atom reproducibility

We separately trained signed scalar Top-K models at K=16,32,64 for both verified heads and all three original seeds, retaining the H724 K=8 checkpoints as reference. Each model has 128 dictionary atoms in 128 write-PCA coordinates and exactly 33,024 parameters. The active budget changes; dictionary width and parameter count do not. This is not merely changing K at inference in an eight-coordinate-trained encoder.

Training matches H724: natural training/validation split, global training RMS scale, initialization seed and minibatch order, Adam learning rate .001, batch size 512, 3,000 steps, decoder-row normalization after every step, and validation-MSE checkpoint selection every 250 steps. No task labels or functional outcomes enter selection. All selected checkpoints are at step 3,000, so these are matched finite-training comparisons, not evidence of convergence.

| Active coordinates | Agreement validation relative write MSE | Capital-head validation relative write MSE |
|---|---:|---:|
| 8 | 0.3465 | 0.4420 |
| 16 | 0.2319 | 0.3365 |
| 32 | 0.1223 | 0.1927 |
| 64 | 0.0373 | 0.0565 |

Values average three seeds. Seed ranges are narrow; for K=64 they are 0.03712–0.03750 and 0.05641–0.05654. This is a substantial fidelity gain, but more active dimensions are allowed, so it is not a win at fixed sparsity or compute.

Individual decoder directions become less reproducible as sparsity is relaxed. Mean matched absolute atom cosine across seed pairs falls from 0.538 to 0.275 for agreement and 0.638 to 0.265 for the capital head between K=8 and K=64. Matching uses a Hungarian assignment on absolute decoder cosine. Support Jaccard is also recorded but is not chance corrected and mechanically depends on activation density; it cannot be compared across K as a pure semantic-stability measure. More faithful reconstruction therefore does not automatically supply more identifiable features.

Correctness: three tests pass for signed largest-magnitude selection, exact K=8 equivalence to the original implementation, dense K=128 behavior, and constant parameter count. On the full validation activations, K=8 implementation replay is exact; original validation MSEs reproduce within numerical tolerance. Every model uses the requested active budget and normalized decoder rows. All hashes remained unchanged and results were finite. Eighteen new fits completed, with reusable per-fit checkpoints. Zero language-model forwards, new natural documents, or original reserved-test evaluations were used.

Next: freeze every trained model and test the H732 cross-task panel with the H733 assay, retaining all seeds and competence-failure labels. Assess effect errors and output KL separately. If larger budgets recover the missing binding/copying effects, use them as a fidelity baseline, while testing reproducible subspaces or operations rather than assuming their unstable individual atoms are fundamental units. Selective-component and manifold validation remain outstanding.
