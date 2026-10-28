## H763 — Output-sensitivity geometry is numerically valid but initially noisy

A task-blind local output-distribution metric is estimated for final-position L5H5 from128 hashed natural training documents, balanced across prefix lengths16,32,64,128. For each context, eight tokens are sampled from the model's own native next-token distribution. Gradients of their log probabilities with respect to additive head-output changes are collected; their average outer product estimates a local Fisher metric. Model weights are frozen. No task labels, validation gradients or reserved-test documents enter the estimate.

Two fixed ridge strengths,0.01 and0.1 times mean metric eigenvalue, regularize the metric. PCA is fitted using all32,768 natural training head vectors in each resulting geometry. Both fits are saved without task-based selection. This is a global averaged local metric: it does not capture every context's sensitivity, correlations between sensitivity and activations, or nonlinear large-intervention KL.

The gradient implementation is verified. Native logits replay exactly; natural head captures agree within3.81e-5. Thirty-two finite-difference checks across all four lengths have maximum absolute derivative error1.91e-4 and maximum relative error2.21%, passing the fixed mixed absolute/relative tolerance. Metric eigenvalues are positive; the factorization and inverse-transform errors are below4.24e-21 and1.56e-15 respectively. All gradients are finite.

However, the first and second four-sample halves differ by80.4% of the full metric's Frobenius norm, with matrix cosine0.724. This reveals substantial Monte Carlo uncertainty despite correct differentiation. The unregularized effective rank is61.8, a numerical spectral summary, not a count of semantic units. The two ridged condition numbers are49.3 and28.8.

Decision: do not treat this pilot geometry as reliable evidence for or against downstream-sensitive featurization. Increase categorical samples per context on the same fixed training contexts before causal benchmarking. This changes estimator precision without fitting to task outcomes.

Audits:320 whole-model forward evaluations including reference and finite differences;1,024 sampled-log-probability backward evaluations; all hashes unchanged; contexts exclusively among the2,048 natural training documents. Zero reserved-test evaluations; natural document pool26,112 unchanged.
