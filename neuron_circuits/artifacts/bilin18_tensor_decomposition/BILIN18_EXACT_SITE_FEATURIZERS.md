## H835 — Exact-site sparse pilots favor scalar features; shared membership collapses coverage

Train alternatives at the EXACT zero-indexed block1/2 raw pre-MLP state sites used by H832–834. Every model receives the same normalized-state token residual, global training RMS scale, token-mean reconstruction and one scalar recipient radius. Same32768 training states per site and512 H832 development examples; H834 confirmation is not used. This completes an initial exact-site implementation/pilot comparison, not a converged architecture search or fresh confirmation.

### Architectures and training

Two seeds8351/8352,800 Adam steps, batch128, learning rate.001. Select residual reconstruction checkpoints every200 steps including0; both seeds reported. Signed scalarTop64 and signed block8 Top8 use512 dictionary atoms with identical initial coordinate dictionaries and minibatches. Unit decoder rows, linear encoder with bias and decoder bias. The block model is not an orthonormal-within-block Grassmannian BSF and the scalar model is not a ReLU SAE.

The shared model uses the existing tested SharedGroups implementation:512 atoms,64 separately encoded gates, Top8 groups, learned Top8 membership atoms/group, exact support union and shared coordinates decoded once. Membership/gate learning uses straight-through surrogates. Fixed budgets constrain group/membership costs rather than implementing a variable-size L0 optimum. No task/response labels enter fitting. It shares the initial coordinate encoder/dictionary with the controls; it has additional gate and membership parameters.

### Development response prediction

Relative RMSE of exact local RMS/bilinear responses on H832's independent eight evaluation probes:

| Site | Signed scalar, two seeds | Block8, two seeds | Shared groups, two seeds |
|---|---|---|---|
| block1 |.585, .583|.630, .631|.726, .726|
| block2 |.619, .619|.674, .677|.788, .791|

The scalar pilot is best on this development metric. For context, H833 token-residual PCA64 scores .644/.694 and weighted residual64 .640/.687. These models have different storage and inference costs, so this is not a matched-resource scalar-over-PCA conclusion. No new full-model output fidelity is measured here, and no semantic or manifold claim follows from the ordering.

All12 fits select the final800-step checkpoint and continue improving between600 and800 steps. Training convergence has not been demonstrated. Scalar/block training losses are below development losses; shared training and development losses are both high. More steps, better gate inference or a different membership structure could change rankings. Both seeds show the same ordering, but two seeds do not establish broad optimization robustness.

### Important capacity confounds uncovered

Scalar and block models have1,181,312 trainable parameters and exactly64 active coordinates. Shared groups have1,287,872 parameters (about9% more) but only47.7–49.0 active atoms on average, with individual examples39–57. An earlier progress estimate of49–52 was approximate; these are the recorded final values. Overlap causes union support to be smaller than the sum of group sizes. The comparison therefore matches nominal dictionary width and maximum active budget, not actual active dimensions or total parameters.

More importantly, shared membership covers only120–132 unique atoms after fitting, compared with323 at random initialization. The other atoms cannot be selected by ANY group. With64 groups times8 memberships there are only512 total membership slots for512 atoms; any overlap necessarily prevents full coverage. This configuration trades coverage for reuse even before optimization, and learning concentrates memberships further. It is not a clean test of the benefit of reusable directions at equal accessible dictionary capacity.

The poor shared result is therefore inconclusive about the architecture. It reveals a concrete membership-capacity/optimization problem that must be controlled. Simply adding training steps would not remove the underlying512-slot coverage tradeoff. A degree-balanced overlapping design needs more membership slots or a smaller atom dictionary, and should be compared with learned memberships and matched actual-support baselines. Extra group-gate parameters must remain explicit.

### Correctness checks

All10 feature-structure semantic/numerical tests pass again, including union semantics, hierarchy closure and DAG checks inherited from the earlier implementation. The actual trained runs have exact binary union support (max discrepancy0) when compared with explicit selected-membership union. Decode formula error2.38e-7, normalized decoder atom checks pass, all losses/gradients checked finite, all tracked input/model/implementation hashes unchanged. All12 fits saved individually with histories and selected state; predictions and per-document/probe errors are saved. Zero original-model forwards/test evaluations or new documents; natural-document pool26,112.

Implementation correctness does not establish good learned membership or semantic recovery. The coverage audit is precisely why the shared model should not be rejected—or promoted—on reconstruction results alone.

### Next controlled comparison

Repair the experimental design before interpreting overlap: allocate enough membership slots for explicit overlap AND broad atom coverage, initialize a balanced degree-constrained graph, and retain a fixed-membership variant to separate dictionary/encoder fitting from membership discovery. Compare signed scalar supports near48 as well as64 active coordinates, and train long enough to inspect plateau behavior. Preserve all seeds and the failed-coverage pilot as evidence; do not silently replace it. Use existing development examples for this repair, then allocate NEW document-disjoint confirmation before any architecture claim. Maintain the user's eventual manifold/causal-transfer and diverse-task requirements: a better featurizer must predict reusable computation, not merely lower development reconstruction loss.
