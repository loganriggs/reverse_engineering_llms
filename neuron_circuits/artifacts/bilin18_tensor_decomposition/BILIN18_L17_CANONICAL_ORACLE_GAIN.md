## H475 — Oracle amplitude versus response direction

Use the frozen rank128/budget32 canonical map on the existing H473 512 receivers, each assigned one of the fixed 64 centered natural donor vectors, with both signed unit interventions. This is a post-hoc diagnostic, not fresh confirmation or a deployable predictor. No model maps or corpus panels are changed. Native intact residual state remains the baseline for isolating response prediction.

Let u be the true layer-write change and v the predicted change. Compare v, raw norm matching g=||u||/||v||, raw squared-error optimal signed gain g=(u·v)/(v·v), and baseline-Fisher optimal signed gain g=(Jv)^T C(Ju)/((Jv)^T C(Jv)). J differentiates the final RMS normalization, vocabulary projection and softcap; C=diag(p)-pp^T is the baseline softmax covariance. Fisher is the local quadratic metric for probability divergence. Every nontrivial gain uses the true response and is therefore an oracle. Gains are per example and sign, unclipped. Evaluate the actual finite target||prediction KL after applying each gain, rather than substituting the quadratic approximation.

| Oracle method | Negative KL recovery | Positive KL recovery | Negative raw recovery | Positive raw recovery |
|---|---:|---:|---:|---:|
|original|81.29%|77.93%|93.35%|93.27%|
|raw_norm|80.56%|78.90%|95.18%|95.29%|
|raw_optimal|80.66%|79.12%|95.47%|95.65%|
|fisher_optimal|85.08%|83.40%|81.22%|80.99%|

Recovery means one minus pooled prediction error divided by pooled native effect, not token accuracy or the mean per-case ratio. Raw uses squared write-vector error; KL uses final output distributions. These metrics preserve different aspects of the computation.

The Fisher oracle improves finite KL recovery by 3.79 and 5.48 percentage points, removing approximately 20.3% and 24.8% of the original KL error. Most error remains. This supports a mixture of gain miscalibration and directional mismatch under the baseline local metric. It does not establish that every scalar correction must leave that much finite KL error: the Fisher gain is analytically optimal only for the local quadratic objective, not certified optimal for finite KL. Local quadratic error decreases from .496045 to .383227 and .446830 to .335370, leaving 77.3% and 75.1% of local error after the best local scalar correction. This residual is the component of the true probability tangent orthogonal to the predicted tangent under the baseline metric.

Raw-optimal correction increases raw recovery from about 93.3% to 95.5–95.7%, but changes KL recovery by −.63 and +1.19 percentage points. Conversely, Fisher-optimal correction reduces raw recovery to about 81% while improving KL. Thus internal Euclidean fidelity and probability fidelity can favor different representatives, even along one fixed predicted response direction. This is direct evidence for keeping the canonical-form criterion explicit. It is not evidence that the oracle is a generally better model.

Median Fisher cosines between predicted and true probability tangents are .826 and .834; 10th percentiles .373 and .388, with minima about −.75. Median signed Fisher gains are .874 and .897, and range roughly −3.05 to 3.67. Negative gains occur when the local best scalar reverses the prediction. Finite KL worsens on 6.05% and 6.45% of cases under Fisher gain, consistent with the local/finite distinction; raw norm matching worsens about 59–60% of individual cases despite its raw-energy improvement. These are descriptive point estimates on a reused, concentrated panel, without confirmatory confidence intervals.

Audits: native intact states replay the saved baseline; unscaled per-case KL and raw errors reproduce H473 within rtol 2e−5/atol 1e−8; every Fisher oracle locally weakly improves the quadratic error; all saved gains and errors are finite. Full per-case gains, local inner products, cosines, effect denominators and finite errors are archived. Source/config hashes are frozen. The initial shell invocation found no `python` alias and did not run the experiment; using `python3` for the builder and the established venv for execution completed successfully without GPU restart.

Next: distinguish a deficient learned map from a deficient 16-dimensional output span. Fit per-case oracle coefficients in the frozen output span using the baseline Fisher metric, then evaluate finite KL. Compare with scalar correction. A large improvement would locate substantial error in coordinate prediction; a persistent ceiling would implicate output-span coverage or local-to-finite mismatch. The span oracle is also diagnostic only. Broader objective comparisons including observed-token CE, source rank1/tail tests, and transfer across heads/layers remain open. Total tracked documents remains 13504.
