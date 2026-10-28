## H1036 — Broader natural activation PCA does not repair task transfer

Previous goal turn was progress: H1035 localized projection damage to the residual stream and its interaction with cached values. This turn audited natural-data provenance, recaptured the exact current pre-attention11 boundary, and compared frozen rank128 residual bases. Other streams remained native.

Training data: first192 documents from H998's previously designated discovery subset. Every document belongs to the original BSF training partition; no reserved validation/test documents or new corpus acquisitions were used. Verified each original128-token window against its stored SHA256. Recaptured first16 tokens per document under the current facade (3072 states). This deliberately samples short natural prefixes; it does not establish coverage of all natural language. No exact evaluated task token sequence occurs within these training prefixes.

Arms: the existing family-balanced task-trained PCA128; natural PCA128 on a fixed seeded subset of1054 states matching the task state's token count; and natural PCA128 on all3072 states. Natural bases use their own training means and ordinary uniform covariance. Equal sample count is not equal effective sample size, position distribution, centering, or weighting, so differences cannot be attributed solely to semantic diversity. No evaluation-dependent rank, subset, or hyperparameter selection.

Evaluation retains the four competent historical copying/spatial cohorts (16pairs/32endpoints). These task families remain excluded from all fitting but are already opened diagnostic evaluations, not fresh validation. Whole-output reconstruction projects only residual states. Paired edits use recipient residual plus projected donor-minus-recipient residual, and exact donor differences for both other streams, consistent with H1035's residual-only condition.

| Residual basis | Whole KL | Paired relative KL | Top-answer accuracy | Centered residual MSE |
|---|---:|---:|---:|---:|
| Task-trained |2.0246|0.4787|3.1%|0.6466|
| Natural,1054 states |2.2527|0.5478|6.3%|0.6247|
| Natural,3072 states |2.4463|0.5470|6.3%|0.6087|

Lower errors are better. Behavioral metrics average four family/split cohorts equally; MSE averages per-endpoint relative errors centered using each basis's own training mean, so its denominator differs between training distributions. Natural bases modestly improve Euclidean coverage but do not improve the aggregate behavioral target. Copying whole KL improves1.0173→0.6799 with all natural states, while paired relative KL stays essentially unchanged0.3402→0.3422. Spatial whole KL worsens3.0320→4.2127 and paired error0.6173→0.7517. Neither candidate supports autonomous replacement of the residual stream.

Interpretation: broader short-prefix natural PCA is insufficient at this rank. This does not disprove richer natural training, nonlinear geometry, task-conditioned representations, or causal objectives. These are newly captured pre-attention11 states; the older natural MLP12/preMLP4 captures were not silently reused at the wrong site. The experiment directly challenges the simple hypothesis that inadequate lexical diversity alone explains the failure. It leaves variance allocation, context-dependent computations, and task-relevant change geometry as distinct hypotheses.

Verification:192 native prefix and192 suffix sequences; zero reserved BSF documents; zero exact evaluation token-subsequence overlap; PCA orthogonality error2.23e-15. Task baseline reproduces H1035 to7.37e-7 at the summary level (float32 projection scaling/order differs slightly). Saved-logit KL independently recomputed for one endpoint per arm, maximum discrepancy3.74e-14. No full independent execution rerun or multi-seed uncertainty estimate claimed.

Next discriminating experiment: fit rank128 residual bases from the same five-family discovery paired differences, compare against activation PCA with the same dictionary rank, and evaluate paired edits while preserving native recipient context. Report whole-state reconstruction separately. This tests whether variance-based geometry misses reusable changes rather than tuning selectors on these opened holdouts. Keep all task-family exclusions; obtain a separate task evaluation before claiming fresh transfer or discovery. A successful local-difference basis would be a concrete candidate dictionary for overlapping/block groups; failure would direct work toward context-conditioned rather than universal residual subspaces.
