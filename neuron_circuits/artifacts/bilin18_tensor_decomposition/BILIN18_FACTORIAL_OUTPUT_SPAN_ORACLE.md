## H956 — Output-span oracle separates activation-space limitations from coefficient error

Project exact factorial teacher MLP changes onto frozen output spans of discovery-selected1,8,64 products for every H949 penalty and seed. Each example receives teacher-informed least-squares coefficients. This is an evaluation-only diagnostic, not a featurizer that predicts unseen effects. Downstream model is fully live for isolated MLP and joint residual paths; native MLP remains intact. No checkpoint training or learned-direction changes.

SVD of row-normalized output vectors defines the spans (relative singular cutoff1e-10). All spans have their intended rank. Independent pseudoinverse solutions reproduce the projections; nested spans cannot worsen local Euclidean reconstruction. Decomposing squared actual error into out-of-span residual plus in-span coefficient error is audited by the Pythagorean identity. This identity concerns activation-space L2, not downstream KL.

Moderate-penalty (.1) results, ranges over two seeds:

| Context / changed factor | Best top8 local error | Best full64 local error | Top8 oracle joint KL ratio | Full64 oracle joint KL ratio |
|---|---|---|---|---|
| prepositional:both | 89.192–89.362% | 80.119–80.826% | 0.771–0.940 | 0.454–0.459 |
| prepositional:distractor | 86.557–86.636% | 76.820–78.031% | 0.645–0.834 | 0.373–0.404 |
| prepositional:subject | 97.965–98.033% | 89.049–89.852% | 0.908–0.983 | 0.500–0.520 |
| relative_clause:both | 88.923–89.128% | 81.365–81.934% | 0.801–0.840 | 0.415–0.504 |
| relative_clause:distractor | 92.135–92.157% | 84.563–85.093% | 0.899–0.945 | 0.525–0.634 |
| relative_clause:subject | 87.320–87.531% | 79.584–80.135% | 0.579–0.670 | 0.295–0.351 |

For prepositional subject-only changes, top-eight local error has a98% floor (actual100%); even the full64 span has89–90% floor (actual101–102%). The dominant direction alone has99.1–99.4% floor. These frozen output directions omit most of the teacher's local activation change. No encoder improvement within those spans can fix full local reconstruction.

Coefficient error nevertheless matters downstream: for the complete64 moderate fit, subject-only prepositional joint KLratio falls from.905–.966 (actual, H955) to.500–.520 with L2-oracle coefficients. Relative-clause subject-only joint task error falls from22% actual to5.1–5.3%, while KLratio falls to.295–.351. Thus incorrect coefficients and output-span limitations both contribute under the measured criteria.

Compact oracle coefficients do not uniformly improve downstream KL: some top-eight oracle cases are worse than actual predictions, and adding directions can improve local L2 while worsening KL. This is expected to be possible because downstream sensitivity and nonlinearity define a different metric. The Euclidean oracle is an upper bound on local reconstruction fidelity, not on achievable output-distribution fidelity. Remaining oracle KL does NOT prove that no coefficients in the span could do better.

All penalty weights, both seeds, three spans, six factorial cells, and both causal paths retained. 8,736 live forwards, zero additional local teacher calls (reuse H954's336), no training. Audits: identity logits2.20e-5; historical exact joint margin7.29e-6; orthogonality3.56e-15; pseudoinverse/SVD agreement2.00e-15 relative; Pythagorean error4.02e-16; independent summary aggregation<1e-12; nested-span local errors monotone; hashes unchanged and all records finite. Reserved test untouched; natural-document pool26,112.

Interpretation: the compact units are narrow approximations in activation space. Their global output directions lack substantial context-dependent teacher responses. But whether a compact, behaviorally equivalent representation exists remains open: the user's KL-based canonical-form question cannot be settled using Euclidean projection alone.

Next: a downstream-metric diagnostic within the frozen top-eight span. Estimate the output-logit Jacobian for small coefficient perturbations, use reference-distribution weighting to solve a local KL/Fisher least-squares oracle, then validate candidate coefficients with the nonlinear live model. Include finite-step checks, fixed norm/trust constraints, and the Euclidean oracle as comparator. Teacher-informed per-example coefficients are diagnostic only; no claim of learned or transferable representation. If this succeeds, investigate training a coefficient predictor on separate discovery data; if it fails, distinguish nonlinear/trust-region limits from actual span insufficiency.
