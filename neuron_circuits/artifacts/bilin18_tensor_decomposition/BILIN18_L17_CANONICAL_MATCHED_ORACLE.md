## H480 — Matched oracle on the latest confirmation receivers

Revisit H479's512 confirmation receivers after their outcomes have been inspected. This is a post-hoc capacity diagnosis, not another fresh confirmation. Freeze the selected256-input-coordinate predictor and the same16-dimensional output basis. Compare its predicted response with a signed scalar Fisher correction, the true response's Euclidean projection into the output span, and its full-span Fisher projection. Every correction uses the native response and is an oracle, not a deployable predictor. The stored method name `original` here means H479's selected r256 predictor, not the pre-refit model.

For true write change u, prediction v and orthonormal output rows B, raw projection is u_parallel=(u B^T)B. Let F be the baseline probability metric induced by the final RMS normalization, unembedding, softcap and softmax. The full-span coefficients solve (B F B^T)c=B F u. Compute them by symmetric eigendecomposition with relative cutoff1e−10, then evaluate actual finite target||prediction KL after applying the response to the native intact baseline. This optimizes a local quadratic metric, not finite KL globally. The baseline is retained to isolate response prediction, not to claim full-layer replacement.

| Response | Negative KL recovery | Positive KL recovery | Negative raw recovery | Positive raw recovery |
|---|---:|---:|---:|---:|
|original|77.31%|72.83%|89.81%|89.24%|
|scalar_fisher|82.53%|78.35%|81.43%|73.07%|
|span_raw|79.39%|75.15%|97.94%|97.90%|
|span_fisher|92.14%|91.60%|78.74%|76.66%|

Recovery is one minus pooled prediction error divided by pooled native effect, not accuracy or average per-example recovery.

For negative interventions, the full-span oracle removes 65.37% of finite KL prediction error and 70.64% of local quadratic error. Its local recovery is 94.05%; it worsens finite KL versus the selected predictor on 0.00% of cases. The earlier H476 panel had full-span finite KL recovery 94.51%, compared with 92.14% here.

For positive interventions, the full-span oracle removes 69.07% of finite KL prediction error and 71.94% of local quadratic error. Its local recovery is 91.82%; it worsens finite KL versus the selected predictor on 0.00% of cases. The earlier H476 panel had full-span finite KL recovery 93.12%, compared with 91.60% here.

The broad capacity finding transfers descriptively: the same16 directions support~92% probability-response recovery on these receivers, despite learned recovery~73–77%. Oracle gains are large, but the remaining oracle error is not zero and the earlier93–95% result is not universal. No new confidence intervals or confirmation gates are introduced for this adaptive diagnostic.

A sharper interpretation is needed than simply calling the gap coefficient error. Euclidean projection captures~97.9% of internal response energy but only~75–79% of probability effects. The probability-optimal coefficients sacrifice internal fidelity, leaving raw recovery~77–79%, to reach~92% KL recovery. Thus the target coefficients depend on which fidelity criterion we choose. A behavior-preserving compressed representation can compensate for omitted internal directions rather than reconstruct the true coordinates of a native feature.

Algebraically, write u=u_parallel+u_perp with u_perp Euclidean-orthogonal to the fixed span. For full-rank G=B F B^T, c_F=c_raw+G^{-1}B F u_perp. Thus the difference between Fisher-optimal and raw coefficients is exactly a metric-dependent compensation for the omitted response. All per-case Gram systems have effective rank16 here, so the invertible expression applies numerically. This identity is a mathematical interpretation, not a claim that its finite-KL effect or learnability has already been isolated experimentally. The context-dependent metric also makes these behavioral coefficients different objects from fixed Euclidean feature coordinates.

Next test a matched factorial decomposition: compare learned in-span response with true raw in-span response, each with and without the actual native orthogonal remainder. Also add the oracle Fisher-minus-raw coefficient correction to the learned prediction. Evaluate finite KL and local quadratic error. The factorial distinguishes correcting native coordinates from restoring omitted writes; the compensation arm tests whether their behavioral effect can be emulated inside the compressed span. Oracle-only status and the reused panel must remain explicit. Do not infer independent additive KL contributions: the readout is nonlinear and local metric cross terms matter.

Audits: selected-r256 per-case KL/raw outcomes replay H479; basis orthogonality below1e−10; Fisher normal-equation residual below1e−6; full-span local error never exceeds scalar local error beyond numerical tolerance; all arrays finite. Full coefficients, Gram eigenvalues, error denominators and per-case outcomes archived. Sources, models and all15808 tracked documents are unchanged. The broad goal, CE comparisons, source rank1/tail behavior and cross-head/layer transfer remain open.
