## H764 — Refined task-blind output metric is more stable; faithfulness still untested

H763's estimator is refined from8 to64 categorical samples per context, keeping the same128 hashed natural training documents, four balanced prefix lengths and sampling seed. No task outcomes, validation gradients or new documents enter this refinement. The metric is the average outer product of native sampled log-probability gradients with respect to L5H5's final-position output. It approximates local output-distribution sensitivity.

| Sampling diagnostic | Eight samples/context | Sixty-four samples/context |
|---|---:|---:|
| Independent sample-half relative Frobenius discrepancy | 80.4% | 32.1% |
| Sample-half matrix cosine | 0.724 | 0.950 |
| Metric effective rank | 61.8 | 72.8 |

The refined estimate is substantially more stable, but32% matrix discrepancy is still appreciable. These halves share the same contexts and only measure categorical-sampling uncertainty, not uncertainty over natural contexts. Effective rank is a spectral statistic, not a count of features, manifolds or independent computational units. The global metric can be broad even if context-specific computation is sparse.

Both fixed ridge settings remain:0.01 and0.1 times the mean metric eigenvalue. They yield condition numbers26.6 and18.2. All32,768 natural training head vectors are used to fit PCA in each metric, with all4,096 validation vectors used only for reconstruction summaries. No ridge or basis is chosen by copying behavior. Both fitted transforms and the8,192 gradient vectors are saved.

Numerical validation remains sound: native logits replay exactly; natural head capture error<=3.81e-5; finite-difference maximum relative derivative error2.21%, absolute error1.91e-4 under the fixed mixed tolerance; positive metric eigenvalues; inverse-transform error<=1.34e-15; metric-factor error<=4.24e-21. All gradients and fitted quantities are finite, and input/source hashes are unchanged.

This is an averaged local quadratic geometry. It ignores correlations between each activation and its own downstream sensitivity, and does not guarantee accurate KL for large reconstruction edits. The gradient sample uses final positions at four sequence lengths whereas the training covariance pools sixteen positions. The ridge also specifies a coordinate-space regularization choice. These limits matter before calling the result a data-based canonical form.

Next: measure actual native-to-reconstructed output KL on fixed natural validation contexts, report observed-token CE separately, and then test copying causal edits using the same frozen transforms. Keep both ridge settings and PCA/random controls. Improving the estimated metric objective alone will not count as better model faithfulness.

Cost/audits:320 whole-model forwards including references and finite differences;8,192 sampled-log-probability backward evaluations. Contexts are exclusively from the2,048 natural training documents. Zero original reserved-test evaluations; natural document pool26,112 unchanged. Explicit sentiment/capital wording checks remain outstanding.
